"""
backends/_gst_appsink.py
========================
Shared machinery for the two raw-GStreamer appsink backends (CPU and CUDA).

Both backends are identical except for ONE thing: the run of elements that
decodes and color-converts the frame (avdec + videoconvert on CPU; nvh26xdec +
the cuda* family on GPU). Everything else -- going to PLAYING, pulling samples
from the appsink, stride-correct numpy extraction, watching the bus for
ERROR/EOS, and tearing down -- is the same, so it lives here once.

A concrete backend subclasses GstAppsinkCapture and implements a single hook,
_media_chain(), returning the decode+convert elements as a list of strings. This
class wraps the source front-end before it and the appsink after it. The base
class (BaseCapture) still owns the thread, queue, reconnect loop, and the
read()/isOpened()/release() API.
"""

import logging

from ..base import BaseCapture
from ..config import Codec, SourceType, PixelFormat, ConfigError
from ..gst_utils import sample_to_ndarray

log = logging.getLogger("stream_capture")

# Planar formats can't be handed back as a single packed numpy array.
_PLANAR = (PixelFormat.NV12, PixelFormat.I420)

# How long try_pull_sample waits each loop (nanoseconds). Bounds how quickly the
# worker notices release(); when frames are flowing, pulls return immediately so
# this adds no latency.
_PULL_TIMEOUT_NS = 100_000_000  # 100 ms


def _gst():
    """Import Gst + GstApp, version-pinned so the import is deterministic and
    warning-free. GstApp is what attaches appsink's pull_sample/is_eos methods.
    Python caches both after the first call."""
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst, GstApp  # noqa: F401 (registers appsink methods)
    return Gst


class GstAppsinkCapture(BaseCapture):

    def __init__(self, source, config):
        super().__init__(source, config)
        self._pipeline = None
        self._appsink = None
        self._bus = None

    # ----------------------------------------- hook the backend fills in

    def _media_chain(self, codec: Codec, fmt: str) -> list:
        """Return the decode + color-convert elements (and any caps they need)
        as a list of pipeline-string pieces, to sit between the source front-end
        and the appsink. `fmt` is the GStreamer format name, e.g. 'BGR'."""
        raise NotImplementedError

    # ----------------------------------------- shared helpers

    def _resolve_codec(self) -> Codec:
        codec = self._config.codec
        if codec is None:
            raise ConfigError(
                "codec is not set. Auto-detection is handled by the factory "
                "(create_capture); when building a backend directly, set "
                "config.codec to Codec.H264 or Codec.H265."
            )
        return codec

    def _reject_planar(self) -> None:
        if self._config.pixel_format in _PLANAR:
            raise ConfigError(
                f"this backend delivers packed numpy frames; "
                f"{self._config.pixel_format.value} is planar -- use bgr/rgb/rgba."
            )

    def _appsink_desc(self) -> str:
        # name=sink so we can fetch it; sync=false so we don't throttle to the
        # clock. RTSP keeps only the newest frame; FILE uses a small buffer with
        # no drop so backpressure preserves every frame.
        if self._source.source_type is SourceType.RTSP:
            return "appsink name=sink sync=false max-buffers=1 drop=true"
        return "appsink name=sink sync=false max-buffers=4 drop=false"

    def _build_pipeline(self) -> str:
        self._reject_planar()
        codec = self._resolve_codec()
        fmt = self._config.pixel_format.value.upper()       # bgr -> BGR
        front = self._source.gst_chain(codec)               # source front-end
        middle = self._media_chain(codec, fmt)              # decode + convert
        parts = [front, *middle, self._appsink_desc()]
        return " ! ".join(parts)

    # ----------------------------------------- the three base hooks

    def _open(self) -> None:
        Gst = _gst()
        Gst.init(None)                          # safe to call repeatedly
        desc = self._build_pipeline()
        log.info("%s pipeline: %s", type(self).__name__, desc)
        self._pipeline = Gst.parse_launch(desc)
        self._appsink = self._pipeline.get_by_name("sink")
        if self._appsink is None:
            raise RuntimeError("appsink not found (pipeline build problem)")
        self._bus = self._pipeline.get_bus()
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("pipeline failed to start (set_state PLAYING)")

    def _run_until_failure(self) -> None:
        Gst = _gst()
        sink, bus = self._appsink, self._bus
        # Some PyGObject builds expose the timeout-aware try_pull_sample; older
        # ones only have the blocking pull_sample. Pick whichever exists.
        has_try = hasattr(sink, "try_pull_sample")
        while not self._should_stop():
            # Fatal error on the bus? Stop; the base decides whether to retry.
            err = bus.pop_filtered(Gst.MessageType.ERROR)
            if err is not None:
                e, dbg = err.parse_error()
                log.warning("gstreamer error: %s | %s", e.message, dbg)
                return
            if has_try:
                sample = sink.try_pull_sample(_PULL_TIMEOUT_NS)
            else:
                sample = None if sink.is_eos() else sink.pull_sample()
            if sample is not None:
                try:
                    frame = sample_to_ndarray(sample)
                except Exception as ex:
                    log.warning("frame conversion failed: %s", ex)
                else:
                    self._emit(frame)
                continue
            # No sample: either end-of-stream (now drained) or just a timeout.
            if sink.is_eos():
                return
            # timeout -> loop and re-check stop / bus

    def _close(self) -> None:
        if self._pipeline is not None:
            Gst = _gst()
            self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None
        self._appsink = None
        self._bus = None