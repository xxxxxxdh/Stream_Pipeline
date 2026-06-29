"""
multi.py
========
MultiCapture -- drive several cameras as one group, auto-selecting the best
multi-camera strategy for the backend.

Two strategies live behind this one class:
  * N independent pipelines (one per camera) -- works with EVERY backend. Each
    camera runs on its own thread + pipeline, fully in parallel.
  * one batched DeepStream pipeline (BatchedDeepStreamCapture) -- DeepStream
    only; all cameras share a single nvstreammux pipeline and one GPU pass.

Which one you get is controlled by the config's `batch` setting (BatchMode):
  * AUTO (the default): batch when backend=deepstream, independent otherwise.
    So MultiCapture(sources, deepstream_cfg) batches for you automatically.
  * OFF: always N independent pipelines (even on DeepStream).
  * ON : force the batched pipeline (errors if the backend isn't deepstream).

Either way the interface is identical -- read_all(), read_dict(), isOpened(),
release(), len(), and context-manager use -- so your loop never changes.

    from stream_capture import MultiCapture, StreamConfig, Backend

    cams = ["rtsp://.../1", "rtsp://.../2", "rtsp://.../3"]
    with MultiCapture(cams, StreamConfig(backend=Backend.DEEPSTREAM)) as mc:
        # ^ auto-batched into one GPU pipeline
        while True:
            for i, (ok, frame) in enumerate(mc.read_all()):
                if ok:
                    ...

NOTE (batched mode only): DeepStream's multi-source teardown aborts at process
exit. For a run-then-exit program using batched mode, call os._exit(0) when done,
or use batch='off' to avoid it.
"""

import logging

from .config import StreamConfig, Backend, BatchMode, ConfigError
from .factory import create_capture
from .multi_deepstream import BatchedDeepStreamCapture

log = logging.getLogger("stream_capture")


def _wants_batching(config) -> bool:
    """Decide between one batched DeepStream pipeline (True) and N independent
    pipelines (False), from the config's backend and batch mode."""
    if not isinstance(config, StreamConfig):
        return False                       # None (cpu default) or a per-source list
    if config.batch is BatchMode.OFF:
        return False
    is_deepstream = config.backend is Backend.DEEPSTREAM
    if config.batch is BatchMode.ON:
        if not is_deepstream:
            raise ConfigError(
                "batch='on' requires backend=deepstream; "
                "use batch='off' (or 'auto') for the other backends."
            )
        return True
    return is_deepstream                   # AUTO: batch iff DeepStream


class MultiCapture:
    """Manage several cameras as one group, auto-selecting independent vs batched
    based on the config's backend + batch mode."""

    def __init__(self, sources, config=None, *, names=None):
        sources = list(sources)
        if names is not None:
            names = list(names)
            if len(names) != len(sources):
                raise ValueError("names must match the number of sources.")
        self._names = names
        self._batched = None
        self._captures = None

        if _wants_batching(config):
            # one batched DeepStream pipeline (config is a single StreamConfig here)
            log.info("MultiCapture: batched DeepStream pipeline (%d cameras)", len(sources))
            self._batched = BatchedDeepStreamCapture(sources, config, names=names)
            return

        # N independent pipelines (works with every backend)
        if isinstance(config, (list, tuple)):
            configs = list(config)
            if len(configs) != len(sources):
                raise ValueError("a config list must match the number of sources.")
        else:
            configs = [config] * len(sources)
        log.info("MultiCapture: %d independent pipelines", len(sources))
        self._captures = []
        try:
            for src, cfg in zip(sources, configs):
                self._captures.append(create_capture(src, cfg))
        except Exception:
            self.release()                 # stop any that already started
            raise

    # ------------------------------------------- introspection

    def is_batched(self) -> bool:
        """True if this group runs as one batched DeepStream pipeline."""
        return self._batched is not None

    @property
    def names(self):
        return list(self._names) if self._names is not None else None

    @property
    def captures(self) -> list:
        """The individual captures (independent mode only)."""
        if self._batched is not None:
            raise AttributeError(
                "individual captures aren't available in batched mode; use read_all()."
            )
        return list(self._captures)

    def __len__(self) -> int:
        if self._batched is not None:
            return len(self._batched)
        return len(self._captures) if self._captures is not None else 0

    def __getitem__(self, i):
        if self._batched is not None:
            raise TypeError("indexing isn't available in batched mode; use read_all().")
        return self._captures[i]

    def __iter__(self):
        if self._batched is not None:
            raise TypeError("iteration isn't available in batched mode; use read_all().")
        return iter(self._captures)

    # ------------------------------------------- group operations

    def read_all(self, timeout: float = 0.0) -> list:
        """Latest frame from every camera, in source order, as (ok, frame).
        Non-blocking by default."""
        if self._batched is not None:
            return self._batched.read_all(timeout)
        return [cap.read(timeout=timeout) for cap in self._captures]

    def read_dict(self, timeout: float = 0.0) -> dict:
        """Like read_all(), keyed by camera name. Requires names=[...]."""
        if self._batched is not None:
            return self._batched.read_dict(timeout)
        if self._names is None:
            raise ValueError("read_dict() needs names; pass names=[...] to MultiCapture.")
        return {name: cap.read(timeout=timeout)
                for name, cap in zip(self._names, self._captures)}

    def isOpened(self) -> list:
        """Per-camera 'is it flowing right now?', in source order."""
        if self._batched is not None:
            return self._batched.isOpened()
        return [cap.isOpened() for cap in self._captures]

    def release(self) -> None:
        """Stop and clean up everything. Safe to call more than once."""
        if self._batched is not None:
            self._batched.release()
        elif self._captures is not None:
            for cap in self._captures:
                try:
                    cap.release()
                except Exception as e:
                    log.warning("error releasing a capture: %s", e)

    # ------------------------------------------- context manager

    def __enter__(self) -> "MultiCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.release()