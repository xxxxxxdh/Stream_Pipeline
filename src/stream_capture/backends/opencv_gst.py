"""
backends/opencv_gst.py
======================
GStreamer + OpenCV backend (Backend.OPENCV_GST). Builds a GStreamer pipeline that decodes
on the CPU (avdec) and hands the whole string to cv2.VideoCapture(...,
CAP_GSTREAMER). OpenCV drives the appsink internally and cap.read() returns BGR
numpy directly -- so unlike the cpu/cuda backends, this one pulls no samples and
does no stride handling itself (OpenCV does it).

Requires an OpenCV built WITH GStreamer support (a from-source build). The decode
path is the same as the CPU GStreamer backend; the difference is that OpenCV owns
the sink, which gives the familiar cv2.VideoCapture ergonomics over a pipeline you
control (you could, for instance, swap in a GPU decoder and still read() BGR).

This is a BaseCapture leaf (not a GstAppsinkCapture one): the base still owns the
thread, queue, reconnect loop, and read()/isOpened()/release(); this file supplies
the pipeline and the three hooks built on cv2.VideoCapture.

Pipeline handed to OpenCV:
  <source front-end> ! avdec_h26x ! videoconvert [! videoscale] ! <caps> ! appsink
"""

import logging

from ..base import BaseCapture
from ..config import Codec, SourceType, PixelFormat, ConfigError

log = logging.getLogger("stream_capture")

# CPU software decoders, by codec (this backend pins CPU decode).
_DECODER = {Codec.H264: "avdec_h264", Codec.H265: "avdec_h265"}

# Planar formats can't be handed back as a single packed numpy array.
_PLANAR = (PixelFormat.NV12, PixelFormat.I420)


def _cv2():
    """Import OpenCV lazily so this module loads even where cv2 isn't installed
    (e.g. for testing pipeline-string assembly)."""
    import cv2
    return cv2


class OpenCvGstCapture(BaseCapture):

    def __init__(self, source, config):
        super().__init__(source, config)
        self._cap = None

    # ----------------------------------------- pipeline assembly (pure string)

    def _resolve_codec(self) -> Codec:
        codec = self._config.codec
        if codec is None:
            raise ConfigError(
                "codec is not set. Auto-detection is handled by the factory "
                "(create_capture); when building a backend directly, set "
                "config.codec to Codec.H264 or Codec.H265."
            )
        return codec

    def _appsink_desc(self) -> str:
        # OpenCV attaches to the appsink itself (no name needed). RTSP keeps only
        # the newest frame; FILE keeps every frame (no drop), read sequentially.
        if self._source.source_type is SourceType.RTSP:
            return "appsink drop=true max-buffers=1 sync=false"
        return "appsink drop=false max-buffers=4 sync=false"

    def _build_pipeline(self) -> str:
        if self._config.pixel_format in _PLANAR:
            raise ConfigError(
                f"the OpenCV backend delivers packed numpy frames; "
                f"{self._config.pixel_format.value} is planar -- use bgr/rgb/rgba."
            )
        codec = self._resolve_codec()
        front = self._source.gst_chain(codec)
        fmt = self._config.pixel_format.value.upper()       # bgr -> BGR

        parts = [front, _DECODER[codec], "videoconvert"]
        caps = f"video/x-raw,format={fmt}"
        if self._config.width and self._config.height:
            parts.append("videoscale")
            caps += f",width={self._config.width},height={self._config.height}"
        parts.append(caps)
        parts.append(self._appsink_desc())
        return " ! ".join(parts)

    # ----------------------------------------- the three base hooks

    def _open(self) -> None:
        cv2 = _cv2()
        desc = self._build_pipeline()
        log.info("opencv_gst pipeline: %s", desc)
        self._cap = cv2.VideoCapture(desc, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError(
                "cv2.VideoCapture failed to open the GStreamer pipeline "
                "(is this OpenCV built WITH GStreamer support?)"
            )

    def _run_until_failure(self) -> None:
        cap = self._cap
        while not self._should_stop():
            ok, frame = cap.read()
            if not ok:
                # RTSP: a read failure means the stream dropped -> return so the
                # base reconnects. FILE: this is end-of-stream -> return so the
                # base stops. OpenCV's read() already returns an owned BGR copy.
                return
            self._emit(frame)

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None