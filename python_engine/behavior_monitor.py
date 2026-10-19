"""Conservative, rule-based activity cues for operator review.

These rules describe observable motion and zone events. They do not identify
theft, aggression, intent, or a person's identity.
"""

from collections import deque
from datetime import datetime, timezone
import math
import threading
import time


class BehaviorMonitor:
    def __init__(self, max_alerts=200, track_ttl=8.0, cooldown=30.0):
        self._lock = threading.Lock()
        self._alerts = deque(maxlen=max_alerts)
        self._tracks = {}
        self._track_ttl = float(track_ttl)
        self._cooldown = float(cooldown)
        self._last_alert = {}

    @staticmethod
    def _bbox(detection):
        box = detection.get("body_bbox") or detection.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        try:
            values = tuple(float(value) for value in box)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        x1, y1, x2, y2 = values
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def _emit(self, cam_id, track_id, event_type, description, now, zone_id=None):
        key = (cam_id, track_id, event_type, zone_id or "")
        if now - self._last_alert.get(key, -float("inf")) < self._cooldown:
            return
        self._last_alert[key] = now
        self._alerts.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "camera_id": str(cam_id),
            "event_type": event_type,
            "description": description,
            "review_only": True,
            "zone_id": zone_id,
        })

    def update(self, cam_id, detections, frame_shape, zones=(), now=None):
        """Consume a frame of person tracks; zones use normalized xyxy boxes."""
        now = time.monotonic() if now is None else float(now)
        try:
            height, width = int(frame_shape[0]), int(frame_shape[1])
        except (TypeError, ValueError, IndexError):
            return
        if width <= 0 or height <= 0:
            return

        valid_zones = []
        for zone in zones or ():
            try:
                zone_id = str(zone.get("id") or "restricted")[:60]
                name = str(zone.get("name") or zone_id)[:100]
                x1, y1, x2, y2 = (float(v) for v in zone.get("box", ()))
                dwell_seconds = max(0.0, float(zone.get("dwell_seconds", 10.0)))
            except (AttributeError, TypeError, ValueError):
                continue
            if (all(math.isfinite(v) for v in (x1, y1, x2, y2, dwell_seconds))
                    and 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                valid_zones.append((zone_id, name, x1, y1, x2, y2, dwell_seconds))

        with self._lock:
            for detection in detections or ():
                if not isinstance(detection, dict):
                    continue
                track_id = detection.get("tracker_id") or detection.get("local_tracker_id")
                box = self._bbox(detection)
                if track_id is None or box is None:
                    continue
                track_id = str(track_id)[:80]
                track_key = (str(cam_id), track_id)
                x1, y1, x2, y2 = box
                center_x_px, center_y_px = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                center_x, center_y = (center_x_px / width, center_y_px / height)
                body_height = max(y2 - y1, 1.0)
                state = self._tracks.get(track_key, {})
                if now - state.get('last_seen', now) > self._track_ttl:
                    state = {}

                previous = state.get("position")
                if previous:
                    elapsed = now - previous[2]
                    if elapsed >= 0.25:
                        if elapsed <= 1.5:
                            body_lengths_per_second = math.hypot(
                                center_x_px - previous[0], center_y_px - previous[1]
                            ) / max((body_height + previous[3]) * 0.5, 1.0) / elapsed
                            if body_lengths_per_second >= 3.2:
                                self._emit(
                                    cam_id, track_id, "rapid_movement",
                                    "Rapid movement cue; review the footage for context.", now,
                                )
                        state["position"] = (center_x_px, center_y_px, now, body_height)
                else:
                    state["position"] = (center_x_px, center_y_px, now, body_height)

                active_zones = set()
                for zone_id, name, zx1, zy1, zx2, zy2, dwell_seconds in valid_zones:
                    inside = zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2
                    if not inside:
                        continue
                    active_zones.add(zone_id)
                    entered_at = state.get("zone_since", {}).get(zone_id)
                    if entered_at is None:
                        entered_at = now
                        state.setdefault("zone_since", {})[zone_id] = now
                        self._emit(
                            cam_id, track_id, "restricted_area_entry",
                            f"Track entered configured zone: {name}.", now, zone_id,
                        )
                    if now - entered_at >= dwell_seconds:
                        self._emit(
                            cam_id, track_id, "extended_zone_dwell",
                            f"Track remained in configured zone: {name}.", now, zone_id,
                        )

                for zone_id in list(state.get("zone_since", {})):
                    if zone_id not in active_zones:
                        state["zone_since"].pop(zone_id, None)

                state["last_seen"] = now
                self._tracks[track_key] = state

            expired = [
                key for key, value in self._tracks.items()
                if key[0] == str(cam_id) and now - value.get("last_seen", now) > self._track_ttl
            ]
            for key in expired:
                self._tracks.pop(key, None)
            stale_alert_keys = [key for key, emitted in self._last_alert.items()
                                if now - emitted > self._cooldown * 4]
            for key in stale_alert_keys:
                self._last_alert.pop(key, None)

    def snapshot(self, limit=100):
        with self._lock:
            return list(self._alerts)[-max(1, min(int(limit), 200)):][::-1]

    def reset_camera(self, camera):
        with self._lock:
            self._tracks = {k:v for k,v in self._tracks.items() if k[0] != camera}
            self._last_alert = {k:v for k,v in self._last_alert.items() if k[0] != camera}


behavior_monitor = BehaviorMonitor()
