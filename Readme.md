# stream-capture

A streaming-only, drop-in `cv2.VideoCapture` replacement where **you choose the
decode backend**. Point it at an RTSP camera or a local file, pick how frames are
decoded (CPU, CUDA/NVDEC, OpenCV, or DeepStream), and read frames back as numpy —
or, on DeepStream, a zero-copy DLPack tensor.

It does **decoding only** — no inference, no tracking. You get clean frames; what
you do with them is up to you.

```python
from stream_capture import create_capture, StreamConfig, Backend

with create_capture("rtsp://...", StreamConfig(backend=Backend.CUDA_GST)) as cap:
    while True:
        ok, frame = cap.read()      # (H, W, 3) BGR numpy array
        if not ok:
            break
        ...                         # your own processing
```

## Install

From the project root (a `src/` layout, already set up):

```bash
pip install -e .
```

After this, `import stream_capture` works from anywhere — no path setup needed.
The only hard pip dependency is `numpy`; the decode backends rely on system /
NVIDIA components (see [Requirements](#requirements)).

## Backends

You pick one per capture via `StreamConfig(backend=...)`.

| Backend | `Backend` value | Decode | Default output | Needs |
|---|---|---|---|---|
| CPU GStreamer | `CPU_GST` | `avdec` (software) | numpy | system GStreamer (no NVIDIA) |
| CUDA GStreamer | `CUDA_GST` | NVDEC + `cuda*` (GPU) | numpy | NVIDIA GPU + nvcodec plugin |
| OpenCV | `OPENCV_GST` | `avdec` via `cv2` | numpy | OpenCV built **with** GStreamer |
| DeepStream | `DEEPSTREAM` | NVDEC + NVMM (GPU) | numpy / DLPack | DeepStream SDK + `pyservicemaker` + `torch` |

On a **live RTSP** stream every backend runs at the camera's frame rate — the
camera, not the decoder, is the bottleneck. The decode-speed difference shows on
**files**: the NVDEC backends (`cuda`/`deepstream`) decode far faster than
real-time, and the GPU path also frees CPU cores for your own work and scales
better across many streams.

## Requirements

- **Python 3.11+** and **numpy** (the only pip dependency).
- **GStreamer backends** (`cpu`/`cuda`/`opencv`): system GStreamer 1.x + PyGObject
  (`gi`). The `cuda` backend additionally needs the `nvcodec` plugin
  (`gstreamer1.0-plugins-bad`) and an NVIDIA GPU.
- **`opencv` backend**: an OpenCV built with `WITH_GSTREAMER=ON`. A stock
  `pip install opencv-python` does **not** include GStreamer.
- **`deepstream` backend**: the NVIDIA DeepStream SDK with `pyservicemaker`, plus
  `torch` (used for DLPack frame extraction).

Check what's actually usable on a given machine (this imports the backends, so it
reports the real state):

```python
from stream_capture import available_backends
for name, (ok, reason) in available_backends().items():
    print(f"{'OK ' if ok else 'no '} {name}  {reason}")
```

Requesting a backend that isn't installed raises `BackendUnavailableError`
immediately, with the reason — not a silent failure later.

## Single-camera usage

### Smallest form

With no config it defaults to the **CPU** backend (runs anywhere), and the codec
is auto-detected from the stream:

```python
from stream_capture import create_capture

cap = create_capture("rtsp://...")
ok, frame = cap.read()
cap.release()
```

### Choosing a backend

```python
from stream_capture import create_capture, StreamConfig, Backend

cfg = StreamConfig(backend=Backend.CUDA_GST)        # codec auto-detected
cap = create_capture("rtsp://user:pass@host:554/stream", cfg)
```

### A local file

```python
cap = create_capture("/path/clip.mp4", StreamConfig(backend=Backend.CPU_GST))
```

Supported files: `.mp4`, `.mov`, `.m4v` (opened with `qtdemux`).

### The capture object

Whatever backend you choose, the returned object behaves like `cv2.VideoCapture`:

- `read(timeout=None) -> (ok, frame)` — pops the next frame. `ok` is `False` on
  timeout or end-of-stream, and `frame` is then `None`. With no `timeout` it waits
  up to `read_timeout_s`; pass `timeout=0` for a non-blocking read.
- `isOpened() -> bool` — `True` only while frames are **actually flowing** (not
  merely "the worker thread is alive"), so a dead camera shows up as `False`.
- `release()` — stop the worker and tear down the backend. Idempotent.
- **Context manager** — `with create_capture(...) as cap:` releases automatically.

`StreamCapture` is an alias for `create_capture` if you prefer a constructor-style
name: `StreamCapture(source, config)`.

### RTSP reconnect

For RTSP sources the worker reconnects automatically when the stream drops, per
the reconnect fields below. For **file** sources there is no reconnect — a file
plays once to its end, after which `read()` returns `(False, None)`.

## Configuration

`StreamConfig` is the single source of truth for a capture. Build it directly,
from a dict (`StreamConfig.from_dict({...})`, which rejects unknown keys), or from
YAML (`StreamConfig.from_yaml("config.yaml")`). Impossible combinations are
rejected **the moment you build it**, with a clear error — not at the first frame.

| Field | Type / values | Default | Notes |
|---|---|---|---|
| `backend` | `Backend` | (required) | `CPU_GST` / `CUDA_GST` / `OPENCV_GST` / `DEEPSTREAM` |
| `codec` | `Codec` or `None` | `None` | `H264` / `H265`; `None` = auto-detect (GStreamer backends) |
| `output` | `OutputMemory` | `NUMPY` | `NUMPY` / `DLPACK`; `DLPACK` is DeepStream-only |
| `pixel_format` | `PixelFormat` | `BGR` | `BGR`/`RGB`/`RGBA`/`NV12`/`I420`; DeepStream: `RGB`/`BGR` only |
| `width`, `height` | `int` or `None` | `None` | set **both or neither**; resizes during decode |
| `gpu_id` | `int` | `0` | used by the DeepStream backends (see [limitations](#limitations--notes)) |
| `reconnect` | `bool` | `True` | RTSP only; ignored for files |
| `max_retries` | `int` | `0` | `0` = unlimited (the RTSP default) |
| `retry_delay_s` | `float` | `2.0` | wait between reconnect attempts |
| `read_timeout_s` | `float` | `10.0` | how long `read()` waits for a frame before `(False, None)` |
| `batch` | `BatchMode` | `AUTO` | multi-camera only; `auto`/`on`/`off` — see [Multiple cameras](#multiple-cameras) |

Enum fields accept the enum **or** its lowercase string, so
`StreamConfig(backend="cuda_gst", batch="off")` works, and the same strings work
in YAML.

**Validation rules** (each fails loudly at construction):

- `output=DLPACK` requires `backend=DEEPSTREAM` (only DeepStream egresses GPU
  memory cleanly).
- The DLPack / tensor path requires a **packed** format (`BGR`/`RGB`/`RGBA`);
  `NV12`/`I420` are rejected there.
- `width` and `height` must be set **together** (both, or neither).
- `gpu_id`, retry counts, and timeouts must be sane (non-negative / positive).

## Output

- **numpy** (default, every backend): an `(H, W, 3)` `uint8` array in your
  `pixel_format` (BGR by default — drops straight into OpenCV and most CV code).
- **DLPack** (`output=DLPACK`, DeepStream only): a `torch` CUDA tensor that
  already owns its memory (safe to keep), for staying on the GPU with zero copies:

```python
import torch
from stream_capture import create_capture, StreamConfig, Backend, OutputMemory

cfg = StreamConfig(backend=Backend.DEEPSTREAM, output=OutputMemory.DLPACK)
with create_capture("rtsp://...", cfg) as cap:
    ok, frame = cap.read()
    if ok:
        tensor = frame                # already a CUDA torch tensor (H, W, 3)
        # ... run your GPU model directly on `tensor`
```

(Any framework's `from_dlpack` works too; the frame is handed back ready to use.)

## Multiple cameras

`MultiCapture` runs several cameras as one group, behind the same interface as a
single capture:

```python
from stream_capture import MultiCapture, StreamConfig, Backend

cams = ["rtsp://.../1", "rtsp://.../2", "rtsp://.../3"]
with MultiCapture(cams, StreamConfig(backend=Backend.CUDA_GST)) as mc:
    while True:
        for i, (ok, frame) in enumerate(mc.read_all()):
            if ok:
                ...   # camera i's latest frame
```

### Two strategies, auto-selected

`MultiCapture` picks one of two strategies based on the `batch` setting:

- **N independent pipelines** — works with **every** backend. One full
  single-camera pipeline per source, each on its own thread, all in parallel.
- **One batched DeepStream pipeline** — **DeepStream only.** All cameras share a
  single `nvstreammux` pipeline (`batch-size=N`) processed in one GPU pass, then
  demuxed back out per camera. More efficient as the camera count grows.

The `batch` field on `StreamConfig` controls which one you get:

| `batch` | Behaviour |
|---|---|
| `auto` (default) | batch when `backend=DEEPSTREAM`, independent otherwise |
| `off` | always N independent pipelines (even on DeepStream) |
| `on` | force the batched pipeline (errors if backend isn't DeepStream) |

So `MultiCapture(cams, StreamConfig(backend=Backend.DEEPSTREAM))` batches
automatically; add `batch="off"` to force independent DeepStream pipelines.

### The MultiCapture interface

Identical no matter which strategy is running:

- `read_all(timeout=0.0) -> list[(ok, frame)]` — latest frame from each camera, in
  source order. Non-blocking by default (a camera with no frame ready →
  `(False, None)`), so one slow camera never holds up the rest.
- `read_dict(timeout=0.0) -> dict` — the same, keyed by camera name (requires
  `names=[...]`).
- `isOpened() -> list[bool]` — per-camera flow status.
- `release()` — stop every camera (idempotent); also via the context manager.
- `len(mc)` — number of cameras. `mc.is_batched()` — `True` if the batched
  pipeline is the one running.

Named cameras:

```python
mc = MultiCapture(cams, cfg, names=["front", "left", "rear"])
frames = mc.read_dict()          # {"front": (ok, frame), "left": (...), "rear": (...)}
```

In **independent** mode you can also reach the individual captures (`mc[0]`,
`for cap in mc:`); these aren't available in batched mode, where there's one
shared pipeline rather than N captures.

You can also use the batched pipeline directly via
`BatchedDeepStreamCapture(sources, config)` — same interface — but `MultiCapture`
with a DeepStream config is the simpler path.

## Public API

Everything importable from `stream_capture`:

- **Single capture:** `create_capture(source, config=None)`, `StreamCapture` (alias)
- **Multiple cameras:** `MultiCapture`, `BatchedDeepStreamCapture`
- **Config:** `StreamConfig`, and the enums `Backend`, `Codec`, `OutputMemory`,
  `PixelFormat`, `SourceType`, `BatchMode`
- **Helpers:** `available_backends()`
- **Sources** (optional — usually inferred from the source string):
  `SourceSpec`, `RtspSource`, `FileSource`
- **Errors:** `ConfigError` (bad config), `BackendUnavailableError` (backend not
  installed on this machine)
- **Base type** (for type hints): `BaseCapture`

## Limitations / notes

- **Batched DeepStream teardown.** When the batched pipeline shuts down it aborts
  the process (`terminate called ... / core dumped`). This comes from DeepStream's
  own multi-source C++ teardown, **not from this package**, and it happens *after*
  every frame has been delivered, so it never affects capture. For a
  run-then-exit program, call `os._exit(0)` when you're done. Because of it,
  releasing and re-creating a batched group inside one long-running process is not
  supported; `batch="off"` (N independent DeepStream pipelines) avoids the crash
  entirely.
- **`gpu_id`** is honored by the DeepStream backends but **not yet** by the CUDA
  GStreamer backend, which always uses **GPU 0**. Fine for single-GPU machines.
- **RTSP transport** uses GStreamer defaults; forcing TCP or a specific latency
  would become a config knob.