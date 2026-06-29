#!/usr/bin/env python3
"""
example.py -- minimal end-to-end use of stream_capture: import, point at a
source, read frames.

    python example.py "rtsp://host:8554/cam" --backend cuda
    python example.py /path/clip.mp4
    python example.py --list          # list which backends work on this machine

Backends: cpu (default), cuda, opencv, deepstream. The codec is auto-detected,
so you never pass one. For multiple cameras, see MultiCapture in the README.

(The sys.path line below lets this run from the project root before install;
after `pip install -e .` it is unnecessary -- `import stream_capture` just works.)
"""

import sys
import time
import argparse
import logging

sys.path.insert(0, "src")

# The entire public surface, from the package root:
from stream_capture import (
    create_capture, StreamConfig, Backend,
    available_backends, BackendUnavailableError,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

_BACKENDS = {
    "cpu": Backend.CPU_GST, "cuda": Backend.CUDA_GST,
    "opencv": Backend.OPENCV_GST, "deepstream": Backend.DEEPSTREAM,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?")
    ap.add_argument("--backend", choices=list(_BACKENDS), default="cpu")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, (ok, reason) in available_backends().items():
            print(f"  [{'OK' if ok else ' X'}] {name}" + (f"   ({reason})" if reason else ""))
        return
    if not args.source:
        ap.error("a source is required (or use --list)")

    # No codec set -> the factory auto-detects it. Context-manager handles release.
    config = StreamConfig(backend=_BACKENDS[args.backend])
    try:
        with create_capture(args.source, config) as cap:
            n, t0 = 0, None
            while n < args.frames:
                ok, frame = cap.read()
                if not ok:
                    if not cap.isOpened():
                        break
                    continue
                if t0 is None:
                    t0 = time.monotonic()
                    print(f"first frame: {frame.shape} {frame.dtype}")
                n += 1
            if t0:
                dt = max(time.monotonic() - t0, 1e-9)
                print(f"read {n} frames (~{n / dt:.1f} fps)")
    except BackendUnavailableError as e:
        print(e)


if __name__ == "__main__":
    main()