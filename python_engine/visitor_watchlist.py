"""Alert when a checked-out or expired visitor is seen again.

When a visitor leaves the live gallery (check-out or expiry), the engine copies
their face embeddings here before they are deleted. Live recognition then sees
that person as "Unknown"; this module compares those unknown faces with the
watch list and raises an "expired_visitor_seen" review alert.

It does not change recognition: it only reads the face embedding that the
camera worker already computed for unknown people. Entries are deleted after
VISITOR_WATCHLIST_KEEP_DAYS so visitor face data is not kept indefinitely.
"""
from collections import deque
from datetime import datetime, timedelta, timezone
import json
import os
import re
import tempfile
import threading
import time

import numpy as np

_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _normalise(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    keep = np.isfinite(matrix).all(axis=1) & (norms[:, 0] > 0)
    return matrix[keep] / norms[keep]


class VisitorWatchlist:
    def __init__(self, folder, threshold=0.50, confirm=2, keep_days=7, cooldown=120.0, confirm_window=10.0):
        self.folder = folder
        self.threshold = float(threshold)
        self.confirm = max(1, int(confirm))
        self.keep_days = float(keep_days)
        self.cooldown = float(cooldown)
        self.confirm_window = float(confirm_window)
        self._lock = threading.Lock()
        self._entries = {}       # visitor_id -> {"name", "removed_at", "reason", "matrix"}
        self._hits = {}          # (camera, visitor_id) -> deque of monotonic times
        self._last_alert = {}
        self._alerts = deque(maxlen=200)
        self._loaded = False

    # ---------- storage ----------
    def _paths(self, visitor_id):
        return (os.path.join(self.folder, f"{visitor_id}.npy"),
                os.path.join(self.folder, f"{visitor_id}.json"))

    def load(self):
        with self._lock:
            self._loaded = True
            self._entries = {}
            if not os.path.isdir(self.folder):
                return
            for name in os.listdir(self.folder):
                if not name.endswith(".json"):
                    continue
                visitor_id = name[:-5]
                if not _ID.fullmatch(visitor_id):
                    continue
                npy, meta_path = self._paths(visitor_id)
                try:
                    with open(meta_path, "r", encoding="utf-8") as handle:
                        meta = json.load(handle)
                    matrix = _normalise(np.load(npy, allow_pickle=False))
                except (OSError, ValueError):
                    continue
                if len(matrix):
                    self._entries[visitor_id] = {**meta, "matrix": matrix}
        self.prune()

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def add(self, visitor_id, name, embeddings, reason="checked out"):
        visitor_id = str(visitor_id)
        if not _ID.fullmatch(visitor_id):
            raise ValueError("Invalid visitor ID.")
        matrix = _normalise(embeddings)
        if not len(matrix):
            return False
        self._ensure_loaded()
        os.makedirs(self.folder, exist_ok=True)
        npy, meta_path = self._paths(visitor_id)
        meta = {"visitor_id": visitor_id, "name": str(name or visitor_id),
                "removed_at": datetime.now(timezone.utc).isoformat(), "reason": reason}
        fd, temp = tempfile.mkstemp(prefix=".watch-", suffix=".npy", dir=self.folder)
        os.close(fd)
        try:
            np.save(temp, matrix, allow_pickle=False)
            os.replace(temp, npy)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        with self._lock:
            self._entries[visitor_id] = {**meta, "matrix": matrix}
        return True

    def remove(self, visitor_id):
        visitor_id = str(visitor_id)
        if not _ID.fullmatch(visitor_id):
            return
        with self._lock:
            self._entries.pop(visitor_id, None)
        for path in self._paths(visitor_id):
            try:
                os.unlink(path)
            except OSError:
                pass

    def prune(self, now=None):
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.keep_days)
        with self._lock:
            old = []
            for visitor_id, entry in self._entries.items():
                try:
                    removed = datetime.fromisoformat(str(entry.get("removed_at")).replace("Z", "+00:00"))
                except ValueError:
                    removed = cutoff - timedelta(seconds=1)
                if removed < cutoff:
                    old.append(visitor_id)
        for visitor_id in old:
            self.remove(visitor_id)

    # ---------- matching ----------
    def check(self, camera_id, detections, now=None):
        """Compare unknown faces on one processed frame with the watch list."""
        if not self._entries and self._loaded:
            return
        self._ensure_loaded()
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            entries = list(self._entries.items())
        if not entries:
            return
        for item in detections or ():
            if item.get("is_known"):
                continue
            embedding = item.get("face_embedding")
            if embedding is None:
                continue
            face = _normalise(embedding)
            if not len(face):
                continue
            best_id, best_score = None, -1.0
            for visitor_id, entry in entries:
                score = float((entry["matrix"] @ face[0]).max())
                if score > best_score:
                    best_id, best_score = visitor_id, score
            if best_score < self.threshold:
                continue
            self._hit(str(camera_id), best_id, best_score, now, item)

    def _hit(self, camera_id, visitor_id, score, now, item):
        with self._lock:
            hits = self._hits.setdefault((camera_id, visitor_id), deque())
            hits.append(now)
            while hits and now - hits[0] > self.confirm_window:
                hits.popleft()
            if len(hits) < self.confirm:
                return
            if now - self._last_alert.get(visitor_id, -float("inf")) < self.cooldown:
                return
            self._last_alert[visitor_id] = now
            entry = self._entries.get(visitor_id, {})
            name = entry.get("name") or visitor_id
            self._alerts.append({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "camera_id": camera_id,
                "event_type": "expired_visitor_seen",
                "visitor_id": visitor_id,
                "visitor_name": name,
                "score": round(score, 4),
                "review_only": True,
                "description": (f"{name} ({visitor_id}) may be on camera after their visit "
                                f"{'ended' if entry.get('reason') == 'checked out' else 'expired'} "
                                f"(face match {score:.2f}). Check the footage."),
            })

    def snapshot(self):
        with self._lock:
            return list(self._alerts)[::-1]

    def status(self):
        self._ensure_loaded()
        with self._lock:
            return {"watched_visitors": len(self._entries), "threshold": self.threshold,
                    "keep_days": self.keep_days}


def create_default_watchlist(config):
    return VisitorWatchlist(
        getattr(config, "VISITOR_WATCHLIST_DIR", os.path.join(config.DATA_DIR, "visitor_watchlist")),
        threshold=getattr(config, "VISITOR_WATCHLIST_THRESHOLD", 0.50),
        confirm=getattr(config, "VISITOR_WATCHLIST_CONFIRM", 2),
        keep_days=getattr(config, "VISITOR_WATCHLIST_KEEP_DAYS", 7),
        cooldown=getattr(config, "VISITOR_WATCHLIST_COOLDOWN", 120.0),
    )
