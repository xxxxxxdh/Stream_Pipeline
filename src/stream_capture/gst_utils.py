"""
gst_utils.py
============
One job: turn a GStreamer appsink frame (a Gst.Sample) into a numpy array.

The tricky part is "stride" (row padding). GStreamer stores each row of a video
frame padded out to a multiple of 4 bytes. For a BGR frame that means each row
is GST_ROUND_UP_4(width * 3) bytes -- which is larger than width*3 whenever the
width is not a multiple of 4. The common shortcut...

    np.frombuffer(data, np.uint8).reshape(height, width, 3)

...assumes there is no padding, so it silently produces a skewed image for those
widths. This module reads the REAL stride and drops the padding, so every width
works.

Only the GStreamer appsink backends (CPU and CUDA) use this. It handles packed
8-bit formats (BGR/RGB/RGBA/GRAY8); planar formats (NV12/I420) are not supported
here -- ask the pipeline for a packed format instead (BGR is the default).
"""

import numpy as np


def _round_up_4(n: int) -> int:
    """Round up to the next multiple of 4 -- GStreamer's default row alignment."""
    return (n + 3) & ~3


# Packed 8-bit formats we can return as (H, W, C), with bytes-per-pixel each.
_CHANNELS = {
    "BGR": 3, "RGB": 3,
    "BGRA": 4, "RGBA": 4, "ARGB": 4, "ABGR": 4,
    "BGRx": 4, "RGBx": 4, "xRGB": 4, "xBGR": 4,
    "GRAY8": 1,
}


def _frame_from_plane(data, width: int, height: int, stride: int,
                      channels: int) -> np.ndarray:
    """Pure numpy core (no GStreamer): take the raw plane bytes plus its real
    stride, and return a contiguous (H, W, C) array with row padding removed.

    This is the part that fixes the stride bug -- a pure-numpy core with no
    GStreamer dependency, so it can be tested in isolation."""
    flat = np.frombuffer(data, dtype=np.uint8)
    needed = height * stride
    if flat.size < needed:
        raise ValueError(
            f"buffer too small: have {flat.size} bytes, need {needed} "
            f"({height} rows x {stride}-byte stride)"
        )
    rows = flat[:needed].reshape(height, stride)        # each row still has padding
    # Keep only the real pixels of each row, then copy so we OWN the memory
    # (the source buffer becomes invalid once GStreamer unmaps it). The .copy()
    # is essential even when there's no padding -- otherwise the result would
    # still point into the soon-to-be-freed buffer.
    plane = rows[:, : width * channels].copy()
    if channels == 1:
        return plane.reshape(height, width)
    return plane.reshape(height, width, channels)


def _layout(caps, buf):
    """Read width/height/format from the caps and the real stride from the
    buffer. Returns (width, height, channels, stride)."""
    s = caps.get_structure(0)
    ok_w, width = s.get_int("width")
    ok_h, height = s.get_int("height")
    if not (ok_w and ok_h):
        raise ValueError("caps is missing width/height")

    fmt = s.get_string("format") or ""
    if fmt not in _CHANNELS:
        raise NotImplementedError(
            f"format {fmt!r} is not supported for numpy extraction; ask the "
            f"pipeline for a packed format ({', '.join(sorted(_CHANNELS))}). "
            f"Planar formats like NV12/I420 are not handled here."
        )
    channels = _CHANNELS[fmt]

    # Try GstVideo for the authoritative stride (VideoMeta, then VideoInfo). If
    # anything here is unavailable, we fall back to the standard packed stride
    # below -- which is exactly what videoconvert/cudaconvert produce anyway.
    stride = None
    try:
        import gi
        gi.require_version("GstVideo", "1.0")   # pin version (no PyGIWarning)
        from gi.repository import GstVideo

        vmeta = GstVideo.buffer_get_video_meta(buf)
        if vmeta is not None:
            stride = int(vmeta.stride[0])
            if getattr(vmeta, "width", 0):
                width = int(vmeta.width)
            if getattr(vmeta, "height", 0):
                height = int(vmeta.height)
        if stride is None:
            vinfo = GstVideo.VideoInfo.new_from_caps(caps)
            if vinfo is not None:
                stride = int(vinfo.stride[0])
    except Exception:
        pass

    # Last resort: the standard packed stride (round up to a multiple of 4).
    if stride is None:
        stride = _round_up_4(width * channels)

    return width, height, channels, stride


def sample_to_ndarray(sample) -> np.ndarray:
    """Convert a GStreamer appsink Gst.Sample into a contiguous numpy array,
    correctly handling row stride. Shape is (H, W, C), or (H, W) for GRAY8."""
    from gi.repository import Gst   # lazy import (see _layout)

    caps = sample.get_caps()
    buf = sample.get_buffer()
    if caps is None or buf is None:
        raise ValueError("sample has no caps or buffer")

    width, height, channels, stride = _layout(caps, buf)

    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("could not map GStreamer buffer for reading")
    try:
        # Copy happens inside _frame_from_plane, before we unmap below.
        return _frame_from_plane(mapinfo.data, width, height, stride, channels)
    finally:
        buf.unmap(mapinfo)