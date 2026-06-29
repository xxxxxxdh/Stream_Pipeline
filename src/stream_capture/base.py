"""
base.py
=======
BaseCapture is the shared engine behind every backend. It owns the parts that
are the same no matter how frames are actually decoded:

  * a worker thread that runs the capture in the background
  * a frame queue the worker fills and read() drains
  * the public API: read(), isOpened(), release()
  * the reconnect policy: if an RTSP stream drops, reopen and keep going

It knows NOTHING about GStreamer, OpenCV or DeepStream. Each backend fills in
three hooks:

  _open()               build and start its pipeline. Raise on hard failure.
  _run_until_failure()  block, calling self._emit(frame) for every frame.
                        Return when the stream dies OR self._should_stop().
  _close()              tear the pipeline down.

A backend that blocks inside _run_until_failure instead of polling (e.g.
DeepStream) can also override _request_stop() to interrupt itself when
release() is called.

Frame-queue behaviour differs by source on purpose:
  * RTSP -> keep only the freshest frame (drop old ones) for low latency.
  * FILE -> never drop; make the decoder wait if the consumer is slow, so you
            process every frame of a recording.
"""

import time
import queue
import logging
import threading
from abc import ABC, abstractmethod

from .config import StreamConfig, SourceType
from .sources import SourceSpec

log = logging.getLogger("stream_capture")

# Default queue sizes. RTSP stays tiny (latest frame wins); a file can buffer a
# little so the decoder isn't lock-stepped to a slow consumer.
_RTSP_QUEUE_MAXSIZE = 2
_FILE_QUEUE_MAXSIZE = 30

# How long release() waits for the worker thread to stop before giving up on it.
_JOIN_TIMEOUT_S = 5.0

# Small poll used while read() waits, so it can notice "stream finished" quickly
# instead of blocking for the whole timeout. (When frames ARE flowing, get()
# returns immediately, so this adds no latency to live capture.)
_READ_POLL_S = 0.05


class BaseCapture(ABC):

    def __init__(self, source: SourceSpec, config: StreamConfig,
                 *, maxsize: int | None = None):
        self._source = source
        self._config = config

        # RTSP drops old frames to stay real-time; FILE applies backpressure.
        self._drop_when_full = (source.source_type is SourceType.RTSP)
        if maxsize is None:
            maxsize = _RTSP_QUEUE_MAXSIZE if self._drop_when_full else _FILE_QUEUE_MAXSIZE
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._finished = False                       # no more frames will ever come
        self._last_frame_time: float | None = None   # monotonic time of last frame

    # ------------------------------------------------------------------ hooks
    # (implemented by each backend)

    @abstractmethod
    def _open(self) -> None:
        """Build and start the pipeline. Raise on a hard failure."""

    @abstractmethod
    def _run_until_failure(self) -> None:
        """Block, calling self._emit(frame) for each frame. Return when the
        stream dies or self._should_stop() becomes True."""

    @abstractmethod
    def _close(self) -> None:
        """Tear the pipeline down. Must be safe to call even after a failed or
        partial _open()."""

    def _request_stop(self) -> None:
        """Optional. Interrupt a blocking _run_until_failure(). Backends that
        poll _should_stop() can ignore this; backends that block (e.g.
        DeepStream) override it to signal EOS/quit."""

    # ----------------------------------------------- tools for the backend
    # (called from inside _run_until_failure)

    def _emit(self, frame) -> None:
        """Hand one frame to the consumer. RTSP keeps only the newest frame;
        FILE waits for room so nothing is dropped."""
        self._last_frame_time = time.monotonic()   # the stream is alive right now
        if self._drop_when_full:
            self._put_drop_oldest(frame)
        else:
            self._put_backpressure(frame)

    def _should_stop(self) -> bool:
        """True once release() has been called. The backend's loop should check
        this and return promptly."""
        return self._stop_event.is_set()

    # ------------------------------------------------------------- public API

    def start(self) -> "BaseCapture":
        """Launch the background worker. Idempotent; returns self so you can
        write `cap = Backend(...).start()`."""
        if self._started:
            return self
        self._started = True
        self._thread = threading.Thread(target=self._worker,
                                        name="stream-capture", daemon=True)
        self._thread.start()
        return self

    def read(self, timeout: float | None = None):
        """Get the next frame as (True, frame). Returns (False, None) if no frame
        arrives within the timeout, or if the stream has finished for good."""
        if not self._started:
            raise RuntimeError("capture not started; call start() first.")
        limit = self._config.read_timeout_s if timeout is None else timeout
        deadline = time.monotonic() + limit
        while True:
            try:
                item = self._queue.get(timeout=_READ_POLL_S)
            except queue.Empty:
                if self._finished:               # stream is over -> return now
                    return (False, None)
                if time.monotonic() >= deadline:  # stalled, but maybe still alive
                    return (False, None)
                continue
            return (True, item)

    def isOpened(self) -> bool:
        """True only while frames are actually flowing. False before the first
        frame, after the stream finishes, or if it stalls (no frame within
        read_timeout_s) -- NOT merely 'the worker thread is alive'."""
        if self._finished or not self._started:
            return False
        last = self._last_frame_time
        if last is None:
            return False
        return (time.monotonic() - last) < self._config.read_timeout_s

    def release(self) -> None:
        """Stop the worker and close the pipeline. Safe to call more than once."""
        if self._stop_event.is_set():
            return                       # already releasing/released
        self._stop_event.set()
        self._request_stop()             # wake a backend that blocks instead of polling
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                log.warning("worker thread did not stop within %.1fs", _JOIN_TIMEOUT_S)
        self._finished = True

    # ----------------------------------------------------- context manager

    def __enter__(self) -> "BaseCapture":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.release()

    # --------------------------------------------------------------- internals

    def _worker(self) -> None:
        """Background loop: open -> stream -> close, reconnecting for RTSP."""
        reconnects = 0
        while not self._stop_event.is_set():
            opened = False
            try:
                self._open()
                opened = True
            except Exception as e:
                log.warning("open failed: %s", e)

            if opened:
                try:
                    self._run_until_failure()     # blocks; emits frames
                except Exception as e:
                    log.warning("capture error: %s", e)

            try:
                self._close()
            except Exception as e:
                log.warning("close error: %s", e)
            self._last_frame_time = None          # not flowing while disconnected

            if self._stop_event.is_set():
                break
            # A file just reaches its end -- never retry it.
            if self._source.source_type is SourceType.FILE:
                break
            if not self._config.reconnect:
                break
            # Stop after max_retries reconnect attempts (0 = retry forever).
            if self._config.max_retries and reconnects >= self._config.max_retries:
                log.warning("giving up after %d reconnect attempt(s)", reconnects)
                break
            reconnects += 1
            log.info("reconnecting in %.1fs (attempt %d)",
                     self._config.retry_delay_s, reconnects)
            # Wait before retrying, but wake instantly if release() is called.
            self._stop_event.wait(self._config.retry_delay_s)

        # The loop is over for good -- let any reader know the stream is finished.
        self._finished = True

    def _put_drop_oldest(self, frame) -> None:
        """RTSP policy: if the queue is full, throw away the oldest frame and
        keep this newest one."""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def _put_backpressure(self, frame) -> None:
        """FILE policy: wait for room so no frame is lost, but keep checking the
        stop flag so release() stays responsive."""
        while not self._stop_event.is_set():
            try:
                self._queue.put(frame, timeout=0.1)
                return
            except queue.Full:
                continue