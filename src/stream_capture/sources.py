"""
sources.py
==========
Turns a source string (an RTSP URL or a file path) into a small object that
knows two things about that source:

  * gst_chain(codec) -> the FRONT of a GStreamer pipeline: everything up to and
    including the parser (e.g. h264parse). A GStreamer backend then appends its
    own decoder, like:   <gst_chain> ! avdec_h264 ! videoconvert ! appsink
  * uri()             -> a single URI string for DeepStream's nvurisrcbin
    (rtsp://...  for cameras,  file:///abs/path  for files).

Use SourceSpec.from_string(...) to get the right object; it looks at the string
and picks RtspSource or FileSource for you.

Only the codec-specific parser/depayloader differs between H264 and H265 -- the
backend's decoder choice (CPU vs GPU) is separate and lives in the backends.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse, unquote

from .config import Codec, SourceType, ConfigError


# Elements that depend on the codec.
_PARSER = {Codec.H264: "h264parse",    Codec.H265: "h265parse"}
_DEPAY  = {Codec.H264: "rtph264depay", Codec.H265: "rtph265depay"}

# File containers currently supported -- the ones qtdemux handles.
# .mkv/.webm (matroskademux) could be added later.
_MP4_EXTS = {".mp4", ".mov", ".m4v"}


def _quote(value: str) -> str:
    """Wrap a location value in double quotes so gst-launch parses it safely
    even when it contains spaces or odd characters. Backslashes and quotes
    inside the value are escaped."""
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _require_codec(codec: Codec) -> Codec:
    """The pipeline front-end needs to know H264 vs H265 to pick the right
    parser/depayloader. If the codec is still unknown, fail clearly here rather
    than producing a broken pipeline string."""
    if codec is None:
        raise ValueError(
            "a concrete codec (H264 or H265) is required to build the pipeline; "
            "resolve config.codec before calling gst_chain()."
        )
    return codec


class SourceSpec(ABC):
    """Base class for a capture source. Don't instantiate this directly --
    use SourceSpec.from_string(...)."""

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """RTSP or FILE."""

    @property
    @abstractmethod
    def location(self) -> str:
        """The raw value handed to GStreamer's `location=` (URL or file path).
        Handy for logging."""

    @abstractmethod
    def gst_chain(self, codec: Codec) -> str:
        """Front of the GStreamer pipeline, up to and including the parser."""

    @abstractmethod
    def uri(self) -> str:
        """A single URI for DeepStream's nvurisrcbin."""

    @classmethod
    def from_string(cls, source: str) -> "SourceSpec":
        """Inspect the string and build the right source object:
            rtsp://...        -> RtspSource
            file:///abs/path  -> FileSource
            /a/plain/path     -> FileSource
        Any other scheme (e.g. http://) is rejected."""
        s = source.strip()
        if "://" in s:
            scheme = s.split("://", 1)[0].lower()
            if scheme == "rtsp":
                return RtspSource(s)
            if scheme == "file":
                # turn a file:// URI into a plain local path
                return FileSource(unquote(urlparse(s).path))
            raise ConfigError(
                f"unsupported source scheme '{scheme}://'; "
                f"use an rtsp:// URL or a local file path."
            )
        # no scheme -> treat it as a local file path
        return FileSource(s)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.location!r})"


class RtspSource(SourceSpec):
    """A live RTSP camera, e.g. rtsp://user:pass@host:554/stream."""

    def __init__(self, url: str):
        self._url = url.strip()

    @property
    def source_type(self) -> SourceType:
        return SourceType.RTSP

    @property
    def location(self) -> str:
        return self._url

    def gst_chain(self, codec: Codec) -> str:
        # rtspsrc pulls the RTP stream; the depayloader rebuilds the elementary
        # stream; the parser frames it so the backend's decoder can take over.
        codec = _require_codec(codec)
        return (
            f"rtspsrc location={_quote(self._url)} "
            f"! {_DEPAY[codec]} ! {_PARSER[codec]}"
        )

    def uri(self) -> str:
        return self._url


class FileSource(SourceSpec):
    """A local video file. Supports mp4/mov containers (qtdemux)."""

    def __init__(self, path: str):
        # expand ~, make absolute and normalised. Doesn't need to exist yet.
        self._path = Path(path).expanduser().resolve()
        ext = self._path.suffix.lower()
        if ext not in _MP4_EXTS:
            raise ConfigError(
                f"file source supports {sorted(_MP4_EXTS)} (qtdemux); "
                f"got '{ext or 'no extension'}'. (.mkv/.webm support is planned.)"
            )

    @property
    def source_type(self) -> SourceType:
        return SourceType.FILE

    @property
    def location(self) -> str:
        return str(self._path)

    def gst_chain(self, codec: Codec) -> str:
        # filesrc reads the bytes; qtdemux splits the container; the parser
        # frames the elementary stream for the backend's decoder.
        codec = _require_codec(codec)
        return (
            f"filesrc location={_quote(self.location)} "
            f"! qtdemux ! {_PARSER[codec]}"
        )

    def uri(self) -> str:
        # file:///abs/path, with proper URL-encoding of any odd characters.
        return self._path.as_uri()