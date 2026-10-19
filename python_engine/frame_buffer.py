"""Bounded, latest-frame transport with independent consumer cursors."""

from dataclasses import dataclass
from collections import deque
import queue
import threading
import time


class RollingRate:
    """Count real events over time, rather than averaging reciprocal intervals."""

    def __init__(self, window=2.0):
        self.window = window
        self.times = deque()

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        self.times.append(now)
        while len(self.times) > 2 and self.times[1] < now - self.window:
            self.times.popleft()
        elapsed = now - self.times[0]
        return (len(self.times) - 1) / elapsed if elapsed > 0 else 0.0


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    frame: object
    captured_at: float
    generation: int
    capture_fps: float


class FrameBuffer:
    def __init__(self):
        self.condition = threading.Condition()
        self.packet = None
        self.closed = False
        self.sequence = 0
        self.last_monotonic = None
        self.fps = 0.0
        self.rate = RollingRate()

    def publish(self, frame, generation=0):
        now = time.monotonic()
        with self.condition:
            if self.closed:
                return
            self.fps = self.rate.tick(now)
            self.last_monotonic = now
            self.sequence += 1
            # Ownership transfers to the buffer; consumers must not modify it.
            self.packet = FramePacket(self.sequence, frame, time.time(), generation, self.fps)
            self.condition.notify_all()

    def read(self, after=0, timeout=0.5, max_age=3.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.closed or (self.packet and self.packet.sequence > after), timeout
            )
            packet = self.packet
            if (
                self.closed
                or packet is None
                or packet.sequence <= after
                or time.time() - packet.captured_at > max_age
            ):
                return None
            return packet

    def invalidate(self):
        with self.condition:
            self.packet = None
            self.last_monotonic = None
            self.fps = 0.0
            self.rate = RollingRate()

    def close(self):
        with self.condition:
            self.closed = True
            self.packet = None
            self.condition.notify_all()


class LatestCameraResults:
    """One slot per camera; a busy producer cannot evict another camera."""

    def __init__(self):
        self.condition = threading.Condition()
        self.pending = {}

    def put_nowait(self, item):
        with self.condition:
            self.pending[item[0]] = item
            self.condition.notify()

    def get(self, timeout=0.1):
        with self.condition:
            if not self.condition.wait_for(lambda: bool(self.pending), timeout):
                raise queue.Empty
            camera = next(iter(self.pending))
            return self.pending.pop(camera)
