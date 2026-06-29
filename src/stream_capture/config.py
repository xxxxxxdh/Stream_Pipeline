"""
config.py
=========
Everything that describes HOW to capture a stream lives here: the option enums
and the StreamConfig dataclass that ties them together.

StreamConfig is the single source of truth. Build it three ways:

    StreamConfig(backend=Backend.CUDA_GST, ...)    # directly, in code
    StreamConfig.from_dict({...})                   # from a plain dict
    StreamConfig.from_yaml("config.yaml")           # from a YAML file

All the cross-checks run in __post_init__, so an impossible combination fails 
the moment you build the config -- not later, deep inside a running pipeline.

Note: the source itself (the RTSP URL or file path) is NOT part of this config.
It is passed separately to StreamCapture(source, config). This file only knows
the decode/output options.
"""

import os
from dataclasses import dataclass, fields
from enum import StrEnum


# ---------------------------------------------------------------------------
# Option enums
# Each is a string enum, so a YAML value like `backend: cuda_gst` maps straight
# onto Backend.CUDA_GST, and the members still behave like plain strings.
# ---------------------------------------------------------------------------

class SourceType(StrEnum):
    RTSP = "rtsp"           # a live network camera (rtsp://...)
    FILE = "file"           # a local video file on disk


class Codec(StrEnum):
    H264 = "h264"
    H265 = "h265"           # aka HEVC


class Backend(StrEnum):
    CPU_GST    = "cpu_gst"      # GStreamer, CPU decode (avdec). Runs anywhere.
    CUDA_GST   = "cuda_gst"     # GStreamer, GPU decode (nvdec + cuda*). NumPy out.
    OPENCV_GST = "opencv_gst"   # cv2.VideoCapture over a GStreamer pipeline (CPU decode).
    DEEPSTREAM = "deepstream"   # DeepStream / Service Maker. Can output DLPack.


class OutputMemory(StrEnum):
    NUMPY  = "numpy"        # frames come back as numpy arrays (CPU). The default.
    DLPACK = "dlpack"       # frames come back as DLPack capsules (GPU tensors).


class PixelFormat(StrEnum):
    BGR  = "bgr"            # default -- drops straight into OpenCV
    RGB  = "rgb"
    RGBA = "rgba"
    NV12 = "nv12"           # planar -- NOT allowed on the DLPack path
    I420 = "i420"           # planar -- NOT allowed on the DLPack path


class BatchMode(StrEnum):
    # Multi-camera only (read by MultiCapture); single-stream captures ignore it.
    AUTO = "auto"           # batch when backend is DeepStream, else N independent
                            # pipelines. The default.
    ON = "on"               # force one batched pipeline (DeepStream only).
    OFF = "off"             # force N independent pipelines, even on DeepStream.


# A GPU tensor (DLPack) has to be a single packed plane. You can't hand back a
# planar NV12/I420 buffer as one clean tensor, so those are blocked on that path.
_PACKED_FORMATS = (PixelFormat.BGR, PixelFormat.RGB, PixelFormat.RGBA)


class ConfigError(ValueError):
    """Raised when the chosen options don't make sense together.
    Subclasses ValueError so callers can catch either type."""


def _coerce_enum(enum_cls, value):
    """Accept either a real enum member or a plain string, and return the
    matching member. This is what lets dicts/YAML use strings like 'cuda_gst'
    while the rest of the code works with the enum."""
    if isinstance(value, enum_cls):
        return value
    key = str(value).strip().lower()
    for member in enum_cls:
        if key == member.value or key == member.name.lower():
            return member
    valid = ", ".join(m.value for m in enum_cls)
    raise ConfigError(f"{value!r} is not a valid {enum_cls.__name__} (valid: {valid})")


# ---------------------------------------------------------------------------
# The config object
# ---------------------------------------------------------------------------

@dataclass
class StreamConfig:
    # The only required choice: which backend decodes the stream.
    backend: Backend

    # Optional. If None, the codec is auto-detected from the stream. Set it only
    # to force one (it acts as a hint/override, not a requirement).
    codec: Codec | None = None

    # How frames are handed back, and in what pixel layout.
    output: OutputMemory = OutputMemory.NUMPY
    pixel_format: PixelFormat = PixelFormat.BGR

    # Optional output size. Leave BOTH None to keep the source resolution;
    # set BOTH to resize (via videoscale / cudascale).
    width: int | None = None
    height: int | None = None

    # Which GPU to use. Honored by the DeepStream backends; the CUDA GStreamer
    # backend currently always uses GPU 0 (see README). CPU/OpenCV ignore it.
    gpu_id: int = 0

    # --- Reconnect policy (consumed by BaseCapture's retry loop) ---
    # These apply to RTSP only. A file just plays to its end, so they are
    # ignored for file sources (the factory warns if you set them anyway).
    reconnect: bool = True          # try to reconnect if the stream drops?
    max_retries: int = 0            # 0 = retry forever (the default for cameras)
    retry_delay_s: float = 2.0      # wait this long between reconnect attempts
    read_timeout_s: float = 10.0    # read() waits this long for a frame, then
                                    # returns (False, None) instead of hanging

    # Multi-camera only (MultiCapture). AUTO batches DeepStream automatically.
    batch: BatchMode = BatchMode.AUTO

    def __post_init__(self):
        # 1) Normalise: allow plain strings anywhere an enum is expected.
        self.backend = _coerce_enum(Backend, self.backend)
        self.output = _coerce_enum(OutputMemory, self.output)
        self.pixel_format = _coerce_enum(PixelFormat, self.pixel_format)
        self.batch = _coerce_enum(BatchMode, self.batch)
        if self.codec is not None:
            self.codec = _coerce_enum(Codec, self.codec)

        # 2) Cross-field rules -- anything impossible fails right here.

        # DLPack (GPU tensor) output is only wired up on DeepStream. The CUDA
        # GStreamer backend can't cleanly hand out GPU memory from Python, so it
        # must use NUMPY.
        if self.output is OutputMemory.DLPACK and self.backend is not Backend.DEEPSTREAM:
            raise ConfigError(
                "output=dlpack is only supported with backend=deepstream; "
                "use output=numpy for the other backends."
            )

        # A tensor must be one packed plane -- reject planar NV12/I420.
        if self.output is OutputMemory.DLPACK and self.pixel_format not in _PACKED_FORMATS:
            allowed = ", ".join(f.value for f in _PACKED_FORMATS)
            raise ConfigError(
                f"output=dlpack needs a packed format ({allowed}); "
                f"got pixel_format={self.pixel_format.value}."
            )

        # Width and height travel together, and must be positive if set.
        if (self.width is None) != (self.height is None):
            raise ConfigError("set width and height together, or leave both as None.")
        if self.width is not None and (self.width <= 0 or self.height <= 0):
            raise ConfigError("width and height must be positive.")

        # Simple sanity on the numeric knobs.
        if self.gpu_id < 0:
            raise ConfigError("gpu_id must be >= 0.")
        if self.max_retries < 0:
            raise ConfigError("max_retries must be >= 0 (0 means retry forever).")
        if self.retry_delay_s < 0:
            raise ConfigError("retry_delay_s must be >= 0.")
        if self.read_timeout_s <= 0:
            raise ConfigError("read_timeout_s must be > 0.")

        # Not checked here: "reconnect is ignored for FILE sources". The source
        # type isn't known to this config (the URL/path is passed separately),
        # so the factory emits that warning once it has classified the source.

    # ----- builders -----

    @classmethod
    def from_dict(cls, data: dict, *, use_env: bool = True) -> "StreamConfig":
        """Build from a plain dict (e.g. parsed YAML). Unknown keys raise on
        purpose, so a typo like `bckend:` is caught instead of silently ignored."""
        data = dict(data)  # copy so we never mutate the caller's dict

        # Catch typos: reject keys that aren't real config fields, with a
        # message that lists what IS valid.
        valid_keys = {f.name for f in fields(cls)}
        unknown = set(data) - valid_keys
        if unknown:
            raise ConfigError(
                f"unknown config key(s): {sorted(unknown)}; "
                f"valid keys: {sorted(valid_keys)}"
            )

        if use_env:
            # Tiny env-override whitelist (env wins over the dict/YAML value).
            # gpu_id is the only config-level field here; the URL and any
            # credentials live in the source string, handled in sources.py.
            env_gpu = os.environ.get("STREAM_CAP_GPU_ID")
            if env_gpu is not None:
                data["gpu_id"] = int(env_gpu)

        return cls(**data)  # __post_init__ does the coercion + validation

    @classmethod
    def from_yaml(cls, path: str, *, use_env: bool = True) -> "StreamConfig":
        """Build from a YAML file. PyYAML is imported lazily, so you only need it
        installed if you actually use this method."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data, use_env=use_env)