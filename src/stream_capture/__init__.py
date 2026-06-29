"""
stream_capture
==============
A streaming-only, drop-in cv2.VideoCapture replacement where you pick the decode
backend. Point it at an RTSP camera or a local file, choose how the frame is
decoded (CPU, CUDA/NVDEC, OpenCV, or DeepStream), and read frames back as numpy
(or, on DeepStream, a zero-copy DLPack tensor).

Quick start
-----------
    from stream_capture import create_capture, StreamConfig, Backend

    cap = create_capture("rtsp://user:pass@10.0.0.5:554/stream",
                          StreamConfig(backend=Backend.CUDA_GST))
    while True:
        ok, frame = cap.read()       # frame is an (H, W, 3) BGR numpy array
        if not ok:
            break
        ...                          # your own inference / processing
    cap.release()

With no config it defaults to the CPU backend (no NVIDIA needed) and the codec is
auto-detected, so the smallest form is just:

    cap = create_capture("rtsp://...")

The returned object exposes read() -> (ok, frame), isOpened(), and release(), and
works as a context manager:

    with create_capture("rtsp://...") as cap:
        ok, frame = cap.read()

For several cameras at once, use MultiCapture (it auto-batches on DeepStream):

    from stream_capture import MultiCapture, StreamConfig, Backend

    with MultiCapture(urls, StreamConfig(backend=Backend.CUDA_GST)) as mc:
        for ok, frame in mc.read_all():
            ...
"""

from .config import (
    StreamConfig,
    ConfigError,
    SourceType,
    Codec,
    Backend,
    OutputMemory,
    PixelFormat,
    BatchMode,
)
from .sources import SourceSpec, RtspSource, FileSource
from .base import BaseCapture
from .factory import create_capture, available_backends, BackendUnavailableError
from .multi_deepstream import BatchedDeepStreamCapture
from .multi import MultiCapture

# StreamCapture is the constructor-style name for the factory: StreamCapture(
# source, config) reads like cv2.VideoCapture(...) and returns a started capture.
StreamCapture = create_capture

__version__ = "0.1.0"

__all__ = [
    "create_capture",
    "StreamCapture",
    "MultiCapture",
    "BatchedDeepStreamCapture",
    "available_backends",
    "StreamConfig",
    "SourceType",
    "Codec",
    "Backend",
    "OutputMemory",
    "PixelFormat",
    "BatchMode",
    "ConfigError",
    "BackendUnavailableError",
    "SourceSpec",
    "RtspSource",
    "FileSource",
    "BaseCapture",
]