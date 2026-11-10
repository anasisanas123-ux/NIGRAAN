"""Sightings of recognised people, for attendance and visitor history.

Recognition itself logs a person to the dashboard once per camera per engine
run. Attendance needs "seen in this room between 9:00 and 11:00 today", so this
log keeps a throttled stream of sightings: each recognised identity on each
camera at most once per SIGHTING_INTERVAL_SECONDS. The dashboard reads it with
GET /sightings?after=<seq> and stores it in MongoDB.
It only reads finished recognition results and never changes them.
"""
from collections import deque
from datetime import datetime, timezone
import threading
import time


class PresenceLog:
    def __init__(self, interval=30.0, max_events=5000):
        self.interval = max(1.0, float(interval))
        self._events = deque(maxlen=max_events)
        self._last = {}
        self._seq = 0
        self._lock = threading.Lock()
        # A new value on every engine start lets the reader notice a restart.
        self.run_id = f"{int(time.time())}"

    def record(self, camera_id, detections, now=None):
        now = time.monotonic() if now is None else float(now)
        stamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            for item in detections or ():
                if not item.get("is_known") or not item.get("student_id"):
                    continue
                key = (str(item["student_id"]), str(camera_id))
                if now - self._last.get(key, -float("inf")) < self.interval:
                    continue
                self._last[key] = now
                self._seq += 1
                self._events.append({
                    "seq": self._seq,
                    "person_id": key[0],
                    "person_name": str(item.get("name") or key[0]),
                    "role": str(item.get("role") or "student"),
                    "camera_id": key[1],
                    "confidence": round(float(item.get("confidence") or 0.0), 4),
                    "timestamp": stamp,
                })
            if len(self._last) > 10000:
                cutoff = now - self.interval
                self._last = {k: v for k, v in self._last.items() if v >= cutoff}

    def since(self, after=0, limit=1000):
        after = int(after or 0)
        with self._lock:
            events = [event for event in self._events if event["seq"] > after][:max(1, int(limit))]
            return {"run_id": self.run_id, "latest": self._seq, "events": events}
