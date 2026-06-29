"""
multi_deepstream.py
===================
BatchedDeepStreamCapture -- many cameras through ONE DeepStream pipeline.

Where MultiCapture runs N independent pipelines (one per camera), this runs a
SINGLE pipeline that batches all N cameras together:

    N x nvurisrcbin -> nvstreammux(batch-size=N) -> nvvideoconvert -> caps(RGB) -> fakesink
                                                             |
                                                        demux probe -> N queues

The GPU decodes and converts the whole batch in one pass, so the cameras share
the work instead of competing for it -- the efficiency win over running
MultiCapture with the deepstream backend.

How the demux works: a probe on the capsfilter fires once per batched buffer. It
walks the batch metadata, and for each frame reads
    frame_item.source_id  -> which camera (which output queue)
    frame_item.batch_id   -> the frame's index in the batch (what extract() takes)
then extracts that frame and routes it to that camera's queue. (These field names
and the extract() index were confirmed against real hardware.)

Interface mirrors MultiCapture: read_all(), read_dict(), isOpened(), release(),
len(), and context-manager use -- so it drops into the same multi-camera loop.

Output: same as the single-camera DeepStream backend -- RGB or BGR only, numpy by
default, or a zero-copy DLPack/torch CUDA tensor with output=DLPACK.

KNOWN ROUGH EDGE -- teardown: stopping a multi-source Service Maker pipeline
currently aborts inside DeepStream's C++ cleanup ("terminate called ... / core
dumped"). It happens at TEARDOWN, after all frames are delivered, so it does not
affect capture -- but it's ugly. For a run-then-exit program, call os._exit(0)
when done to skip the crashing destructor. release()
stops the pipeline but deliberately keeps the object referenced, so the crash
doesn't fire mid-program; this means release-and-recreate within one process is
not supported yet.
"""

import os
import time
import queue
import logging
import threading

from .config import StreamConfig, Backend, OutputMemory, PixelFormat, ConfigError
from .sources import SourceSpec
from .factory import BackendUnavailableError

log = logging.getLogger("stream_capture")

_QUEUE_MAXSIZE = 2          # per-camera, drop-oldest (latest frame wins)
_READ_POLL_S = 0.05
_JOIN_TIMEOUT_S = 5.0

# Set before pyservicemaker/GStreamer import (same reason as the single-cam backend).
os.environ.setdefault("USE_NEW_NVSTREAMMUX", "yes")

_DEMUX_CLS = None


def _service_maker():
    from pyservicemaker import Pipeline, Probe, BufferOperator
    return Pipeline, Probe, BufferOperator


def _demux_extractor_class():
    """Build (once) the probe operator that demuxes a batched buffer into
    per-camera frames. Defined lazily because it subclasses a pyservicemaker
    type that only exists where DeepStream is installed."""
    global _DEMUX_CLS
    if _DEMUX_CLS is not None:
        return _DEMUX_CLS

    _, _, BufferOperator = _service_maker()
    import torch

    class _BatchDemux(BufferOperator):
        """Fires once per batched buffer. For each frame in the batch, reads its
        source_id (which camera) and batch_id (its index in the batch), extracts
        that frame, and routes it to the camera's queue via `route(source_id,
        frame)`. Service Maker extracts RGB only, so we flip to BGR here if asked."""
        def __init__(self, route, to_numpy, to_bgr):
            super().__init__()
            self._route = route
            self._to_numpy = to_numpy
            self._to_bgr = to_bgr

        def handle_buffer(self, buffer) -> bool:
            for item in buffer.batch_meta.frame_items:
                source_id = item.source_id          # which camera
                batch_id = item.batch_id            # index within the batch
                tensor = torch.utils.dlpack.from_dlpack(buffer.extract(batch_id))
                if self._to_bgr:
                    tensor = tensor.flip(-1)         # RGB -> BGR on the GPU
                if self._to_numpy:
                    frame = tensor.cpu().numpy()     # .cpu() copies off the GPU buffer
                else:
                    frame = tensor.clone()           # owned GPU tensor (safe to keep)
                self._route(source_id, frame)
            return True

    _DEMUX_CLS = _BatchDemux
    return _DEMUX_CLS


class BatchedDeepStreamCapture:
    """One DeepStream pipeline batching N cameras; frames demuxed per camera."""

    def __init__(self, sources, config=None, *, names=None):
        # Fail fast with a clear error if DeepStream isn't usable here.
        try:
            import pyservicemaker  # noqa: F401
            import torch  # noqa: F401
        except Exception as e:
            raise BackendUnavailableError(
                f"DeepStream batching needs pyservicemaker + torch: {e}"
            )

        sources = list(sources)
        if not sources:
            raise ValueError("BatchedDeepStreamCapture needs at least one source.")
        self._specs = [s if isinstance(s, SourceSpec) else SourceSpec.from_string(s)
                       for s in sources]
        self._n = len(self._specs)

        if names is not None:
            names = list(names)
            if len(names) != self._n:
                raise ValueError("names must match the number of sources.")
        self._names = names

        # This class IS the DeepStream batched path; default/require that backend.
        if config is None:
            config = StreamConfig(backend=Backend.DEEPSTREAM)
        elif config.backend is not Backend.DEEPSTREAM:
            raise ConfigError(
                "BatchedDeepStreamCapture is DeepStream-only; "
                "use StreamConfig(backend=Backend.DEEPSTREAM)."
            )
        self._config = config

        # One drop-oldest queue per camera, indexed by source_id (0..N-1).
        self._queues = [queue.Queue(maxsize=_QUEUE_MAXSIZE) for _ in range(self._n)]
        self._last_frame_time = [None] * self._n

        self._pipeline = None
        self._thread = None
        self._stop_event = threading.Event()
        self._finished = False
        self._started = False
        self._start()

    # ----------------------------------------- config sanity (same as single-cam)

    def _resolve_format(self) -> str:
        pf = self._config.pixel_format
        if pf not in (PixelFormat.RGB, PixelFormat.BGR):
            raise ConfigError(
                f"the DeepStream backend can deliver rgb or bgr; "
                f"{pf.value} is not supported for tensor extraction."
            )
        return pf.value.upper()

    def _to_numpy(self) -> bool:
        return self._config.output is not OutputMemory.DLPACK

    # ----------------------------------------- pipeline construction

    def _build_pipeline(self):
        Pipeline, Probe, _ = _service_maker()
        fmt = self._resolve_format()
        to_numpy = self._to_numpy()
        to_bgr = (fmt == "BGR")
        gpu = self._config.gpu_id

        p = Pipeline(f"stream-capture-batched-{id(self)}")

        # The input half: one nvurisrcbin per source (auto-detects codec per URI).
        for i, spec in enumerate(self._specs):
            p.add("nvurisrcbin", f"src_{i}", {
                "uri": spec.uri(),
                "gpu-id": gpu,
                "latency": 100,
                "drop-on-latency": 1,
                "select-rtp-protocol": 4,        # TCP
            })

        # Batch all N sources into one buffer.
        p.add("nvstreammux", "mux", {
            "batch-size": self._n,
            "batched-push-timeout": 25000,
        })
        p.add("nvvideoconvert", "conv", {"gpu-id": gpu})

        # Always RGB in NVMM (Service Maker extracts RGB only); BGR done by flip.
        caps = "video/x-raw(memory:NVMM),format=RGB"
        if self._config.width and self._config.height:
            caps += f",width={self._config.width},height={self._config.height}"
        p.add("capsfilter", "caps", {"caps": caps})
        p.add("fakesink", "sink", {"sync": False, "silent": True})

        # Each source links to the next nvstreammux request pad (sink_0, sink_1, ...).
        for i in range(self._n):
            p.link((f"src_{i}", "mux"), ("", "sink_%u"))
        p.link("mux", "conv", "caps", "sink")

        # Attach the demux probe: it routes each frame to its camera's queue.
        extractor = _demux_extractor_class()(self._route_frame, to_numpy, to_bgr)
        p.attach("caps", Probe("demux", extractor))
        return p

    # ----------------------------------------- frame routing (the demux target)

    def _route_frame(self, source_id, frame) -> None:
        """Called by the probe for every frame, with the camera it belongs to.
        Drop-oldest per camera so each queue holds the freshest frame."""
        if not (0 <= source_id < self._n):
            return                                   # unexpected source -> ignore
        self._last_frame_time[source_id] = time.monotonic()
        q = self._queues[source_id]
        try:
            q.put_nowait(frame)
        except queue.Full:
            try:
                q.get_nowait()                       # drop the oldest
            except queue.Empty:
                pass
            try:
                q.put_nowait(frame)
            except queue.Full:
                pass

    # ----------------------------------------- worker (runs the one pipeline)

    def _start(self) -> "BatchedDeepStreamCapture":
        if self._started:
            return self
        self._started = True
        self._thread = threading.Thread(target=self._worker,
                                        name="stream-capture-batched", daemon=True)
        self._thread.start()
        return self

    def _worker(self) -> None:
        try:
            self._pipeline = self._build_pipeline()
            log.info("batched deepstream pipeline starting (%d cameras, out=%s, fmt=%s)",
                     self._n, self._config.output.value, self._config.pixel_format.value)
            self._pipeline.start()
            self._pipeline.wait()                    # blocks until EOS or stop()
        except Exception as e:
            log.warning("batched pipeline error: %s", e)
        finally:
            self._finished = True

    # ----------------------------------------- reading

    def _read_queue(self, i, timeout):
        q = self._queues[i]
        deadline = time.monotonic() + timeout
        while True:
            try:
                return (True, q.get(timeout=_READ_POLL_S))
            except queue.Empty:
                if self._finished:
                    return (False, None)
                if time.monotonic() >= deadline:
                    return (False, None)

    def read_all(self, timeout: float = 0.0) -> list:
        """Latest frame from every camera, in source order (camera i = source_id
        i). Non-blocking by default: a camera with no frame ready -> (False, None)."""
        return [self._read_queue(i, timeout) for i in range(self._n)]

    def read_dict(self, timeout: float = 0.0) -> dict:
        """Like read_all(), but keyed by camera name. Requires names=[...]."""
        if self._names is None:
            raise ValueError("read_dict() needs names; pass names=[...].")
        return {name: self._read_queue(i, timeout)
                for i, name in enumerate(self._names)}

    def isOpened(self) -> list:
        """Per-camera 'is it flowing right now?', in source order."""
        if self._finished or not self._started:
            return [False] * self._n
        now = time.monotonic()
        out = []
        for last in self._last_frame_time:
            out.append(last is not None and (now - last) < self._config.read_timeout_s)
        return out

    def __len__(self) -> int:
        return self._n

    # ----------------------------------------- shutdown

    def release(self) -> None:
        """Stop the pipeline and the worker. Safe to call more than once.

        NOTE: we keep the pipeline object referenced after stopping it -- dropping
        it would trigger DeepStream's multi-source C++ destructor, which aborts.
        So this stops frames cleanly, but the object lives until process exit; for
        a run-then-exit program, os._exit(0) when done avoids the teardown crash."""
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as e:
                log.warning("error stopping batched pipeline: %s", e)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
        self._finished = True

    def __enter__(self) -> "BatchedDeepStreamCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.release()