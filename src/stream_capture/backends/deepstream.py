"""
backends/deepstream.py
======================
DeepStream backend, built on NVIDIA's Service Maker (pyservicemaker) -- the only
backend where frames are NVMM (GPU) the whole way and can leave as a zero-copy
DLPack tensor, not just numpy.

This one is structurally the odd one out: it does NOT use gi.Gst or an appsink.
It builds a Service Maker Pipeline (nvurisrcbin -> nvstreammux -> nvvideoconvert
-> capsfilter -> fakesink) and attaches a Probe whose handle_buffer() fires for
every frame and PUSHES it to the consumer via self._emit(). So unlike the
appsink/OpenCV backends (which pull), here _run_until_failure just starts the
pipeline and blocks on wait() while frames arrive through the probe.

It still satisfies the same BaseCapture contract, so read()/isOpened()/release(),
the queue, and the reconnect loop all behave like every other backend.

Output:
  Service Maker extracts the frame as an RGB tensor only. We pin RGB in NVMM and:
  - NUMPY (default): extract -> DLPack -> .cpu().numpy()  (the .cpu() is the copy)
  - DLPACK: extract -> DLPack -> torch CUDA tensor, .clone()d so it owns its
    memory (safe once the pipeline recycles its buffer). Consume it with
    torch.from_dlpack(...) or any framework's from_dlpack.
  If the caller asked for BGR, we reverse the channels on the GPU after extract.
  Only rgb/bgr are supported here (a non-RGB buffer crashes extract()).

Notes:
  - nvurisrcbin auto-detects the codec from the URI, so config.codec is unused
    here (RTSP and file URIs both work, via source.uri()).
  - gpu_id, pixel_format, and width/height are honored; reconnect is handled by
    the BaseCapture loop (rebuild on failure), same as the other backends.
"""

import logging
import os

# The proven pipeline configures nvstreammux via properties (batch-size,
# batched-push-timeout), which only take effect under the NEW streammux. This
# env var must be set before pyservicemaker/GStreamer is imported, so we set it
# here at module import (the earliest hook in the package). setdefault respects
# an explicit override if the caller set it themselves.
os.environ.setdefault("USE_NEW_NVSTREAMMUX", "yes")

from ..base import BaseCapture
from ..config import OutputMemory, PixelFormat, ConfigError

log = logging.getLogger("stream_capture")

# Cache for the lazily-built extractor class (it subclasses a pyservicemaker type,
# so it can only be defined after that import succeeds).
_EXTRACTOR_CLS = None


def _service_maker():
    """Import Service Maker's pipeline pieces, lazily so this module loads even
    where DeepStream/pyservicemaker isn't installed."""
    from pyservicemaker import Pipeline, Probe, BufferOperator
    return Pipeline, Probe, BufferOperator


def _extractor_class():
    """Build (once) the probe operator class. Defined lazily because it inherits
    from pyservicemaker.BufferOperator, which only exists where DeepStream is."""
    global _EXTRACTOR_CLS
    if _EXTRACTOR_CLS is not None:
        return _EXTRACTOR_CLS

    _, _, BufferOperator = _service_maker()
    import torch

    class _FrameExtractor(BufferOperator):
        """Fires for every frame at the capsfilter and pushes it to the consumer.
        Service Maker extracts the tensor as RGB only (the pipeline pins RGB in
        NVMM upstream), so we reverse the channels here when BGR is requested.
        `emit` is the capture's self._emit; to_numpy/to_bgr pick the output."""
        def __init__(self, emit, to_numpy, to_bgr):
            super().__init__()
            self._emit = emit
            self._to_numpy = to_numpy
            self._to_bgr = to_bgr

        def handle_buffer(self, buffer) -> bool:
            # buffer.extract(0) yields an RGB (H,W,3) DLPack tensor on the GPU.
            tensor = torch.utils.dlpack.from_dlpack(buffer.extract(0))
            if self._to_bgr:
                tensor = tensor.flip(-1)            # RGB -> BGR (on the GPU)
            if self._to_numpy:
                frame = tensor.cpu().numpy()        # .cpu() copies off the GPU buffer
            else:
                frame = tensor.clone()              # owned GPU tensor (safe to keep)
            self._emit(frame)
            return True

    _EXTRACTOR_CLS = _FrameExtractor
    return _EXTRACTOR_CLS


class DeepStreamCapture(BaseCapture):

    def __init__(self, source, config):
        super().__init__(source, config)
        self._pipeline = None

    # ----------------------------------------- config sanity for this backend

    def _resolve_format(self) -> str:
        """DeepStream's tensor extraction only produces RGB; we can additionally
        deliver BGR by reversing channels after extraction. Anything else (RGBA,
        planar) can't be extracted as a single tensor here -- reject it loudly
        rather than let it reach extract() (which crashes on non-RGB buffers)."""
        pf = self._config.pixel_format
        if pf not in (PixelFormat.RGB, PixelFormat.BGR):
            raise ConfigError(
                f"the DeepStream backend can deliver rgb or bgr; "
                f"{pf.value} is not supported for tensor extraction."
            )
        return pf.value.upper()

    def _output_to_numpy(self) -> bool:
        return self._config.output is not OutputMemory.DLPACK

    # ----------------------------------------- pipeline construction

    def _build_pipeline(self):
        Pipeline, Probe, _ = _service_maker()
        fmt = self._resolve_format()                        # 'RGB' or 'BGR'
        to_numpy = self._output_to_numpy()
        to_bgr = (fmt == "BGR")
        gpu = self._config.gpu_id

        p = Pipeline(f"stream-capture-{id(self)}")

        # Source: opens the URI, detects codec, depays/parses, decodes on NVDEC,
        # all into GPU memory (NVMM). Works for rtsp:// and file:// URIs.
        p.add("nvurisrcbin", "src", {
            "uri": self._source.uri(),
            "gpu-id": gpu,
            "latency": 100,
            "drop-on-latency": 1,        # drop late frames to stay real-time
            "select-rtp-protocol": 4,    # TCP (more stable than UDP for RTSP)
        })

        # Mux: wraps frames in NvDsBatchMeta (batch-size=1 = single source here).
        p.add("nvstreammux", "mux", {
            "batch-size": 1,
            "batched-push-timeout": 25000,
        })

        # Convert the decoded NV12 to RGB, staying in GPU memory (caps pin RGB).
        p.add("nvvideoconvert", "conv", {"gpu-id": gpu})

        # The capsfilter is ALWAYS RGB: Service Maker can only extract RGB
        # tensors (a non-RGB buffer crashes extract()). If BGR was requested we
        # produce RGB here and reverse the channels after extraction.
        caps = "video/x-raw(memory:NVMM),format=RGB"
        if self._config.width and self._config.height:
            caps += f",width={self._config.width},height={self._config.height}"
        p.add("capsfilter", "caps", {"caps": caps})

        # Discard the frame after the probe has read it; sync=false = run flat out.
        p.add("fakesink", "sink", {"sync": False, "silent": True})

        # nvurisrcbin has dynamic pads (appear when the stream connects), so the
        # src->mux link uses the request-pad pattern; the rest is a static chain.
        p.link(("src", "mux"), ("", "sink_%u"))
        p.link("mux", "conv", "caps", "sink")

        # Attach the frame extractor to the capsfilter: handle_buffer() runs for
        # every frame and pushes it into the base queue via self._emit.
        extractor = _extractor_class()(self._emit, to_numpy, to_bgr)
        p.attach("caps", Probe("extractor", extractor))
        return p

    # ----------------------------------------- the base hooks

    def _open(self) -> None:
        self._pipeline = self._build_pipeline()
        log.info("deepstream pipeline starting (uri=%s, out=%s, fmt=%s)",
                 self._source.uri(), self._config.output.value,
                 self._config.pixel_format.value)
        self._pipeline.start()

    def _run_until_failure(self) -> None:
        # Frames flow through the probe (handle_buffer -> _emit) on the pipeline's
        # own threads; this just blocks until the stream ends or stop() is called.
        self._pipeline.wait()

    def _request_stop(self) -> None:
        # Called by release() to unblock the wait() above.
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as e:
                log.warning("error stopping pipeline: %s", e)

    def _close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as e:
                log.warning("error stopping pipeline: %s", e)
        self._pipeline = None