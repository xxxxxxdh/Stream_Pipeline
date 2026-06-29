"""
factory.py
==========
create_capture(source, config) -- the one call that turns a source + config into
a ready-to-read capture. It:
  1. resolves the source (a string like "rtsp://..." or "/path.mp4" becomes a
     SourceSpec),
  2. checks the chosen backend's dependencies are actually present on this
     machine (a clear error instead of a cryptic GStreamer failure or a silent
     reconnect loop),
  3. fills in the codec for the GStreamer backends when you didn't specify one
     (probes the stream; falls back to H265 if the probe can't tell),
  4. starts the backend and hands it back.

All four backends share the same read()/isOpened()/release() API (from
BaseCapture), so the returned object is used the same way regardless of which
backend ran.
"""

import logging
from dataclasses import replace

from .config import StreamConfig, Backend, Codec, SourceType
from .sources import SourceSpec
from .backends.cpu_gst import CpuGstCapture
from .backends.cuda_gst import CudaGstCapture
from .backends.opencv_gst import OpenCvGstCapture
from .backends.deepstream import DeepStreamCapture

log = logging.getLogger("stream_capture")


class BackendUnavailableError(RuntimeError):
    """Raised when the chosen backend's dependencies aren't installed here."""


# --------------------------------------------------------------------------
# Capability checks. Each returns (ok: bool, reason: str). The reason is only
# meaningful when ok is False.
# --------------------------------------------------------------------------

def _gst_has(*elements):
    """True if Gst is importable and every named element is registered."""
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
    except Exception as e:
        return False, f"GStreamer Python bindings unavailable ({e})"
    missing = [name for name in elements if Gst.ElementFactory.find(name) is None]
    if missing:
        return False, f"missing GStreamer element(s): {', '.join(missing)}"
    return True, ""


def _check_cpu_gst():
    return _gst_has("avdec_h264", "avdec_h265", "videoconvert", "appsink")


def _check_cuda_gst():
    ok, reason = _gst_has("nvh264dec", "nvh265dec", "cudaconvert", "cudadownload")
    if not ok and reason.startswith("missing"):
        reason += " -- install the nvcodec plugin (gstreamer1.0-plugins-bad)"
    return ok, reason


def _check_opencv_gst():
    try:
        import cv2
    except Exception as e:
        return False, f"OpenCV (cv2) not importable ({e})"
    has_gst = any(("GStreamer" in ln and "YES" in ln.upper())
                  for ln in cv2.getBuildInformation().splitlines())
    if not has_gst:
        return False, ("this OpenCV build has no GStreamer support "
                       "(rebuild with WITH_GSTREAMER=ON)")
    return True, ""


def _check_deepstream():
    try:
        import pyservicemaker  # noqa: F401
    except Exception as e:
        return False, f"pyservicemaker (DeepStream Service Maker) not importable ({e})"
    try:
        import torch  # noqa: F401
        import torch.utils.dlpack  # noqa: F401
    except Exception as e:
        return False, f"torch with dlpack not importable ({e})"
    return _gst_has("nvurisrcbin", "nvstreammux", "nvvideoconvert")


# backend enum -> (friendly name, class, capability check)
_REGISTRY = {
    Backend.CPU_GST:    ("cpu_gst",    CpuGstCapture,     _check_cpu_gst),
    Backend.CUDA_GST:   ("cuda_gst",   CudaGstCapture,    _check_cuda_gst),
    Backend.OPENCV_GST: ("opencv_gst", OpenCvGstCapture,  _check_opencv_gst),
    Backend.DEEPSTREAM: ("deepstream", DeepStreamCapture, _check_deepstream),
}


def available_backends() -> dict:
    """Probe every backend's dependencies on this machine; returns
    {name: (ok, reason)}. NOTE: this imports cv2 and pyservicemaker, which have
    side effects -- call it deliberately, not in a hot path."""
    return {name: check() for (name, _cls, check) in _REGISTRY.values()}


# --------------------------------------------------------------------------
# Codec auto-detection (GStreamer backends only; DeepStream auto-detects).
# --------------------------------------------------------------------------

def _detect_codec(source, timeout_s: float = 10.0):
    """Probe the source to tell H264 from H265. Returns a Codec, or None if the
    probe can't determine it (e.g. it couldn't connect within the timeout)."""
    try:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstPbutils", "1.0")
        from gi.repository import Gst, GstPbutils
        Gst.init(None)
        disc = GstPbutils.Discoverer.new(int(timeout_s * Gst.SECOND))
        info = disc.discover_uri(source.uri())
        for stream in info.get_video_streams():
            name = stream.get_caps().get_structure(0).get_name().lower()
            if "h265" in name or "hevc" in name:
                return Codec.H265
            if "h264" in name or "avc" in name:
                return Codec.H264
    except Exception as e:
        log.warning("codec auto-detection failed (%s)", e)
    return None


def _resolve_codec(source, config: StreamConfig) -> StreamConfig:
    """Ensure the config has a concrete codec for the GStreamer backends. The
    DeepStream backend auto-detects via nvurisrcbin, so it's left untouched."""
    if config.backend is Backend.DEEPSTREAM or config.codec is not None:
        return config
    detected = _detect_codec(source)
    if detected is None:
        detected = Codec.H265
        log.warning("could not detect the codec; defaulting to %s. Pass "
                    "config.codec explicitly if that's wrong.", detected.value)
    else:
        log.info("detected codec: %s", detected.value)
    return replace(config, codec=detected)


# --------------------------------------------------------------------------
# Public entry point.
# --------------------------------------------------------------------------

def create_capture(source, config: StreamConfig | None = None):
    """Build and start a capture for `source` using `config`, returning a started
    capture that exposes read()/isOpened()/release() -- use it like a
    cv2.VideoCapture.

    `source` may be a string ("rtsp://...", "/path/clip.mp4") or a SourceSpec.
    `config` defaults to the CPU GStreamer backend (works without NVIDIA)."""
    if config is None:
        config = StreamConfig(backend=Backend.CPU_GST)

    spec = source if isinstance(source, SourceSpec) else SourceSpec.from_string(source)

    name, backend_cls, check = _REGISTRY[config.backend]
    ok, reason = check()
    if not ok:
        raise BackendUnavailableError(
            f"backend '{name}' is not usable on this machine: {reason}. "
            f"Call available_backends() to see which backends work here."
        )

    config = _resolve_codec(spec, config)

    if spec.source_type is SourceType.FILE and config.reconnect:
        log.info("reconnect is ignored for file sources (a file plays once to its end).")

    cap = backend_cls(spec, config)
    return cap.start()