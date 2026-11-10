"""Exam mode: review cues for possible cheating during an exam session.

Runs only while an exam session is active on a camera. It uses its own models
(YOLO pose for head direction and positions, the COCO YOLO model for its
"cell phone" class) and never changes face recognition. Cues:

  exam_phone            a mobile phone is visible in consecutive analysed frames
  exam_looking_around   a student's head is turned sideways/backwards for a while
  exam_left_seat        a seat learned at the start of the exam stays empty
  exam_too_close        two students' heads stay very close together

These are prompts for the invigilator, not proof of cheating. Each cue saves a
snapshot, and every session is written to EXAM_DIR as a JSON report.
"""
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
import re
import threading
import time
import uuid

import cv2
import numpy as np

# COCO keypoint indices used by YOLO pose models.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = range(7)
PHONE_CLASS = 67  # "cell phone" in COCO

EVENT_TEXT = {
    "exam_phone": "Mobile phone visible",
    "exam_looking_around": "Looking around",
    "exam_left_seat": "Left seat",
    "exam_too_close": "Students too close",
}


def _iso(ts=None):
    moment = datetime.fromtimestamp(ts or time.time(), tz=timezone.utc)
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def head_direction(kps, conf, min_conf=0.5, yaw_ratio=0.55):
    """Return "forward", "turned" or None (head not visible) from pose keypoints.

    kps: (17, 2) pixel coordinates; conf: (17,) keypoint confidences.
    Looking down at the paper keeps the nose between the eyes, so it stays
    "forward"; only sideways or backwards turns count.
    """
    seen = [bool(conf[i] >= min_conf) for i in range(7)]
    nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh = seen
    if nose and l_eye and r_eye:
        eye_dist = abs(kps[L_EYE][0] - kps[R_EYE][0])
        shoulder = abs(kps[L_SHOULDER][0] - kps[R_SHOULDER][0]) if (l_sh and r_sh) else 0.0
        if shoulder > 0 and eye_dist < 0.12 * shoulder:
            return "turned"  # eyes almost on top of each other: profile view
        if eye_dist <= 1:
            return "turned"
        mid = (kps[L_EYE][0] + kps[R_EYE][0]) / 2.0
        return "turned" if abs(kps[NOSE][0] - mid) / eye_dist > yaw_ratio else "forward"
    if nose and (l_eye != r_eye):
        return "turned"  # one eye hidden: head in profile
    if not nose and not l_eye and not r_eye and (l_ear or r_ear) and (l_sh or r_sh):
        return "turned"  # face hidden but ears/shoulders visible: facing away
    return None


def head_point(box, kps, conf, min_conf=0.5):
    for idx in (NOSE,):
        if conf[idx] >= min_conf:
            return float(kps[idx][0]), float(kps[idx][1])
    pts = [kps[i] for i in (L_EYE, R_EYE, L_EAR, R_EAR) if conf[i] >= min_conf]
    if pts:
        arr = np.asarray(pts, dtype=float)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    return (box[0] + box[2]) / 2.0, box[1] + (box[3] - box[1]) * 0.12


def shoulder_width(box, kps, conf, min_conf=0.5):
    if conf[L_SHOULDER] >= min_conf and conf[R_SHOULDER] >= min_conf:
        width = abs(kps[L_SHOULDER][0] - kps[R_SHOULDER][0])
        if width > 4:
            return float(width)
    return float(box[2] - box[0]) * 0.6


class _Session:
    def __init__(self, camera_id, name, folder, seat_learn_seconds):
        self.id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{camera_id}-{uuid.uuid4().hex[:4]}"
        self.camera_id = camera_id
        self.name = name
        self.folder = folder
        self.started_at = time.time()
        self.ended_at = None
        self.learn_until = time.monotonic() + seat_learn_seconds
        self.learning_samples = []
        self.seats = []          # {"id", "cx", "cy", "w", "h", "empty_since", "alerted"}
        self.tracks = {}         # id -> {"box", "last", "turned_since", "last_turned"}
        self.next_track = 1
        self.pairs = {}          # (a, b) -> since
        self.phone_streak = 0
        self.last_alert = {}
        self.events = []
        self.frames_analysed = 0
        self.max_people = 0
        self.overlay = {"people": [], "phones": [], "flagged": set(), "shape": None, "at": 0.0}

    def summary(self):
        counts = {}
        for event in self.events:
            counts[event["type"]] = counts.get(event["type"], 0) + 1
        return {
            "id": self.id, "camera_id": self.camera_id, "name": self.name,
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at) if self.ended_at else None,
            "active": self.ended_at is None,
            "seats": len(self.seats),
            "learning_seats": time.monotonic() < self.learn_until and self.ended_at is None,
            "max_people": self.max_people,
            "frames_analysed": self.frames_analysed,
            "counts": counts,
            "events": list(self.events),
        }


class ExamMonitor:
    def __init__(self, config):
        self.config = config
        self.folder = getattr(config, "EXAM_DIR", os.path.join(config.DATA_DIR, "exams"))
        self.interval = float(getattr(config, "EXAM_SAMPLE_INTERVAL", 0.35))
        self.phone_conf = float(getattr(config, "EXAM_PHONE_CONFIDENCE", 0.40))
        self.phone_confirm = max(1, int(getattr(config, "EXAM_PHONE_CONFIRM", 2)))
        self.yaw_ratio = float(getattr(config, "EXAM_LOOK_YAW_RATIO", 0.55))
        self.look_seconds = float(getattr(config, "EXAM_LOOK_SECONDS", 3.0))
        self.seat_learn = float(getattr(config, "EXAM_SEAT_LEARN_SECONDS", 15.0))
        self.seat_empty = float(getattr(config, "EXAM_SEAT_EMPTY_SECONDS", 45.0))
        self.close_seconds = float(getattr(config, "EXAM_CLOSE_SECONDS", 4.0))
        self.close_ratio = float(getattr(config, "EXAM_CLOSE_RATIO", 0.60))
        self.cooldown = float(getattr(config, "EXAM_ALERT_COOLDOWN", 30.0))
        self._cond = threading.Condition()
        self._sessions = {}       # camera -> _Session
        self._pending = {}        # camera -> (frame, monotonic)
        self._last_sample = {}
        self._identities = {}     # camera -> (monotonic, [(norm_box, name, id)])
        self._alerts = deque(maxlen=200)
        self._worker = None
        self._pose = None
        self._objects = None
        self._state = "idle"
        self._error = None

    # ---------- session control ----------
    def start(self, camera_id, name):
        camera_id = str(camera_id)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", camera_id):
            raise ValueError("Invalid camera ID.")
        name = re.sub(r"[\x00-\x1f]", "", str(name or "")).strip()[:100] or "Exam"
        with self._cond:
            if camera_id in self._sessions:
                return self._sessions[camera_id].summary()
            session = _Session(camera_id, name, self.folder, self.seat_learn)
            self._sessions[camera_id] = session
            if self._worker is None:
                self._worker = threading.Thread(target=self._run, name="exam-monitor", daemon=True)
                self._worker.start()
        self._save(session)
        return session.summary()

    def stop(self, camera_id):
        with self._cond:
            session = self._sessions.pop(str(camera_id), None)
            self._pending.pop(str(camera_id), None)
        if not session:
            return None
        session.ended_at = time.time()
        self._save(session)
        return session.summary()

    def relearn_seats(self, camera_id):
        with self._cond:
            session = self._sessions.get(str(camera_id))
            if not session:
                return None
            session.seats, session.learning_samples = [], []
            session.learn_until = time.monotonic() + self.seat_learn
            return session.summary()

    def active(self):
        with self._cond:
            return {camera: session.summary() for camera, session in self._sessions.items()}

    # ---------- reports ----------
    def _save(self, session):
        os.makedirs(self.folder, exist_ok=True)
        path = os.path.join(self.folder, f"{session.id}.json")
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(session.summary(), handle, indent=2)
        os.replace(temp, path)

    def reports(self, limit=50):
        if not os.path.isdir(self.folder):
            return []
        items = []
        for name in sorted(os.listdir(self.folder), reverse=True):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.folder, name), "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                continue
            data.pop("events", None)
            items.append(data)
            if len(items) >= limit:
                break
        return items

    def report(self, session_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", str(session_id)):
            return None
        with self._cond:
            for session in self._sessions.values():
                if session.id == session_id:
                    return session.summary()
        try:
            with open(os.path.join(self.folder, f"{session_id}.json"), "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None

    # ---------- inputs ----------
    def observe_identities(self, camera_id, detections, source_shape):
        """Remember where recognised students are, to name the person in a cue."""
        if str(camera_id) not in self._sessions:
            return
        height, width = source_shape[:2]
        people = []
        for item in detections or ():
            box = item.get("body_bbox") or item.get("bbox")
            if not box or len(box) != 4:
                continue
            name = item.get("name") if item.get("is_known") else None
            ident = item.get("student_id") or item.get("global_id")
            people.append(([box[0] / width, box[1] / height, box[2] / width, box[3] / height], name, ident))
        with self._cond:
            self._identities[str(camera_id)] = (time.monotonic(), people)

    def update(self, camera_id, frame):
        camera_id = str(camera_id)
        if frame is None or camera_id not in self._sessions:
            return
        now = time.monotonic()
        with self._cond:
            if now - self._last_sample.get(camera_id, -1e9) < self.interval:
                return
            self._last_sample[camera_id] = now
            self._pending[camera_id] = (frame.copy(), now)
            self._cond.notify()

    # ---------- models ----------
    def _weights(self, name):
        path = os.path.join(self.config.DATA_DIR, "models", os.path.basename(name))
        if os.path.isfile(path):
            return path
        if os.path.isfile(name):
            return name
        from ultralytics.utils.downloads import attempt_download_asset
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return str(attempt_download_asset(path))

    def _load(self):
        from ultralytics import YOLO
        import torch
        self._device = 0 if torch.cuda.is_available() else "cpu"
        self._pose = YOLO(self._weights(getattr(self.config, "EXAM_POSE_MODEL", "yolo11n-pose.pt")))
        object_name = getattr(self.config, "EXAM_OBJECT_MODEL", "yolov8s.pt")
        self._objects = YOLO(object_name if os.path.isfile(object_name) else self._weights(object_name))

    def _run(self):
        try:
            with self._cond:
                self._state = "loading"
            self._load()
            with self._cond:
                self._state = "ready"
        except Exception as error:
            with self._cond:
                self._state, self._error = "load_error", f"{type(error).__name__}: {error}"
            print(f"[EXAM] model load failed: {self._error}")
            return
        while True:
            with self._cond:
                self._cond.wait_for(lambda: bool(self._pending))
                camera_id = next(iter(self._pending))
                frame, captured = self._pending.pop(camera_id)
                session = self._sessions.get(camera_id)
            if session is None:
                continue
            try:
                self._analyse(session, frame, captured)
                with self._cond:
                    self._state, self._error = "ready", None
            except Exception as error:
                with self._cond:
                    self._state, self._error = "inference_error", f"{type(error).__name__}: {error}"
                print(f"[EXAM] {camera_id}: {self._error}")

    def _detect(self, frame):
        pose = self._pose.predict(frame, conf=0.35, classes=[0], device=self._device, verbose=False)[0]
        people = []
        if pose.boxes is not None and len(pose.boxes):
            boxes = pose.boxes.xyxy.cpu().numpy()
            kps = pose.keypoints.xy.cpu().numpy() if pose.keypoints is not None else None
            kconf = pose.keypoints.conf.cpu().numpy() if pose.keypoints is not None and pose.keypoints.conf is not None else None
            for i, box in enumerate(boxes):
                k = kps[i] if kps is not None else np.zeros((17, 2))
                c = kconf[i] if kconf is not None else np.zeros(17)
                people.append({"box": [float(v) for v in box], "kps": k, "kconf": c})
        objects = self._objects.predict(frame, conf=self.phone_conf, classes=[PHONE_CLASS],
                                        device=self._device, verbose=False)[0]
        phones = []
        if objects.boxes is not None and len(objects.boxes):
            for box, conf in zip(objects.boxes.xyxy.cpu().numpy(), objects.boxes.conf.cpu().numpy()):
                phones.append({"box": [float(v) for v in box], "conf": float(conf)})
        return people, phones

    # ---------- analysis ----------
    def _analyse(self, session, frame, now):
        people, phones = self._detect(frame)
        self.evaluate(session, frame, people, phones, now)

    def evaluate(self, session, frame, people, phones, now):
        """Apply the cue rules to one analysed frame. Separate from _detect for testing."""
        height, width = frame.shape[:2]
        session.frames_analysed += 1
        session.max_people = max(session.max_people, len(people))
        self._assign_tracks(session, people, now)
        flagged = set()

        # 1. phone
        if phones:
            session.phone_streak += 1
        else:
            session.phone_streak = 0
        if session.phone_streak >= self.phone_confirm:
            best = max(phones, key=lambda p: p["conf"])
            holder = self._nearest_person(people, best["box"])
            if self._ready(session, ("phone", holder["track"] if holder else None), now):
                self._event(session, frame, "exam_phone", now,
                            f"Possible mobile phone (score {best['conf']:.2f})",
                            person=holder, extra_box=best["box"])
                if holder:
                    flagged.add(holder["track"])

        # 2. looking around
        for person in people:
            track = session.tracks[person["track"]]
            direction = head_direction(person["kps"], person["kconf"], yaw_ratio=self.yaw_ratio)
            if direction == "turned":
                track["last_turned"] = now
                if track.get("turned_since") is None:
                    track["turned_since"] = now
            elif direction == "forward" or now - track.get("last_turned", -1e9) > 1.0:
                track["turned_since"] = None
            since = track.get("turned_since")
            if since is not None and now - since >= self.look_seconds:
                flagged.add(person["track"])
                if self._ready(session, ("look", person["track"]), now):
                    self._event(session, frame, "exam_looking_around", now,
                                f"Head turned sideways or back for {now - since:.0f} s", person=person)

        # 3. seats
        if now < session.learn_until:
            session.learning_samples.extend(
                [((p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2,
                  p["box"][2] - p["box"][0], p["box"][3] - p["box"][1]) for p in people])
        else:
            if not session.seats and session.learning_samples:
                session.seats = self._learn_seats(session.learning_samples)
                session.learning_samples = []
            for seat in session.seats:
                occupant = next((p for p in people
                                 if abs((p["box"][0] + p["box"][2]) / 2 - seat["cx"]) < seat["w"] * 0.5
                                 and abs((p["box"][1] + p["box"][3]) / 2 - seat["cy"]) < seat["h"] * 0.5), None)
                if occupant is not None:
                    # Remember who sits here, so a "left seat" cue can name them.
                    seat["name"] = self._name_for(session, occupant["box"], frame.shape) or seat.get("name")
                    event = seat.get("event")
                    if event is not None:
                        event["returned_at"] = _iso()
                        try:
                            self._save(session)
                        except OSError:
                            pass
                    seat["event"] = None
                    seat["empty_since"] = None
                    continue
                if seat.get("empty_since") is None:
                    seat["empty_since"] = now
                elif now - seat["empty_since"] >= self.seat_empty and seat.get("event") is None:
                    seat["event"] = self._event(
                        session, frame, "exam_left_seat", now,
                        f"Seat {seat['id']} empty for {now - seat['empty_since']:.0f} s",
                        seat=seat, name_hint=seat.get("name"))

        # 4. too close
        current_pairs = set()
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                a, b = people[i], people[j]
                ha = head_point(a["box"], a["kps"], a["kconf"])
                hb = head_point(b["box"], b["kps"], b["kconf"])
                avg = (shoulder_width(a["box"], a["kps"], a["kconf"]) +
                       shoulder_width(b["box"], b["kps"], b["kconf"])) / 2.0
                if avg <= 0 or math.dist(ha, hb) >= self.close_ratio * avg:
                    continue
                key = tuple(sorted((a["track"], b["track"])))
                current_pairs.add(key)
                since = session.pairs.setdefault(key, now)
                if now - since >= self.close_seconds:
                    flagged.update(key)
                    if self._ready(session, ("close",) + key, now):
                        self._event(session, frame, "exam_too_close", now,
                                    f"Two students' heads close together for {now - since:.0f} s",
                                    person=a, other=b)
        for key in list(session.pairs):
            if key not in current_pairs:
                session.pairs.pop(key, None)

        session.overlay = {
            "people": [[p["box"][0] / width, p["box"][1] / height, p["box"][2] / width, p["box"][3] / height, p["track"]]
                       for p in people],
            "phones": [[p["box"][0] / width, p["box"][1] / height, p["box"][2] / width, p["box"][3] / height]
                       for p in phones],
            "flagged": flagged,
            "seats": [[s["cx"] / width, s["cy"] / height, bool(s.get("event"))] for s in session.seats],
            "at": time.monotonic(),
        }

    def _assign_tracks(self, session, people, now):
        unused = {tid: t for tid, t in session.tracks.items() if now - t["last"] < 3.0}
        for person in sorted(people, key=lambda p: -(p["box"][2] - p["box"][0])):
            best, best_iou = None, 0.3
            for tid, track in unused.items():
                score = _iou(person["box"], track["box"])
                if score > best_iou:
                    best, best_iou = tid, score
            if best is None:
                best = session.next_track
                session.next_track += 1
                session.tracks[best] = {"box": person["box"], "last": now, "turned_since": None}
            else:
                unused.pop(best)
            session.tracks[best].update(box=person["box"], last=now)
            person["track"] = best
        for tid in [tid for tid, t in session.tracks.items() if now - t["last"] > 30.0]:
            session.tracks.pop(tid, None)

    @staticmethod
    def _learn_seats(samples):
        seats = []
        for cx, cy, w, h in sorted(samples, key=lambda s: (s[1], s[0])):
            for seat in seats:
                if abs(cx - seat["cx"]) < seat["w"] * 0.5 and abs(cy - seat["cy"]) < seat["h"] * 0.5:
                    n = seat["n"]
                    seat.update(cx=(seat["cx"] * n + cx) / (n + 1), cy=(seat["cy"] * n + cy) / (n + 1),
                                w=(seat["w"] * n + w) / (n + 1), h=(seat["h"] * n + h) / (n + 1), n=n + 1)
                    break
            else:
                seats.append({"cx": cx, "cy": cy, "w": w, "h": h, "n": 1})
        # A seat must have been seen in several samples, not a person walking past.
        seats = [s for s in seats if s["n"] >= 3]
        for index, seat in enumerate(sorted(seats, key=lambda s: (round(s["cy"] / max(s["h"], 1)), s["cx"])), 1):
            seat.update(id=index, empty_since=None, event=None)
        return seats

    @staticmethod
    def _nearest_person(people, box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best, best_d = None, float("inf")
        for person in people:
            x1, y1, x2, y2 = person["box"]
            pad = (x2 - x1) * 0.4
            if x1 - pad <= cx <= x2 + pad and y1 - pad <= cy <= y2 + pad:
                d = math.dist((cx, cy), ((x1 + x2) / 2, (y1 + y2) / 2))
                if d < best_d:
                    best, best_d = person, d
        return best

    def _ready(self, session, key, now):
        if now - session.last_alert.get(key, -1e9) < self.cooldown:
            return False
        session.last_alert[key] = now
        return True

    def _name_for(self, session, box, shape):
        with self._cond:
            seen = self._identities.get(session.camera_id)
        if not seen or time.monotonic() - seen[0] > 3.0:
            return None
        height, width = shape[:2]
        norm = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
        best, best_iou = None, 0.3
        for other, name, ident in seen[1]:
            score = _iou(norm, other)
            if score > best_iou and name:
                best, best_iou = f"{name} ({ident})" if ident else name, score
        return best

    def _event(self, session, frame, event_type, now, detail, person=None, other=None, seat=None,
               extra_box=None, name_hint=None):
        names = []
        for p in (person, other):
            if p is not None:
                names.append(self._name_for(session, p["box"], frame.shape) or f"student #{p['track']}")
        if name_hint:
            names.append(name_hint)
        who = " and ".join(names)
        number = len(session.events) + 1
        snapshot = self._snapshot(session, frame, number, [p["box"] for p in (person, other) if p] +
                                  ([extra_box] if extra_box else []) +
                                  ([[seat["cx"] - seat["w"] / 2, seat["cy"] - seat["h"] / 2,
                                     seat["cx"] + seat["w"] / 2, seat["cy"] + seat["h"] / 2]] if seat else []))
        title = EVENT_TEXT[event_type]
        event = {
            "number": number,
            "type": event_type,
            "title": title,
            "who": who or None,
            "detail": detail,
            "timestamp": _iso(),
            "snapshot": snapshot,
            "seat": seat["id"] if seat else None,
        }
        session.events.append(event)
        self._alerts.append({
            "timestamp": event["timestamp"],
            "camera_id": session.camera_id,
            "event_type": event_type,
            "exam_session": session.id,
            "exam_name": session.name,
            "review_only": True,
            "description": f"Exam '{session.name}': {title.lower()}{' - ' + who if who else ''}. {detail}.",
        })
        try:
            self._save(session)
        except OSError:
            pass
        return event

    def _snapshot(self, session, frame, number, boxes):
        try:
            folder = os.path.join(self.folder, session.id)
            os.makedirs(folder, exist_ok=True)
            image = frame.copy()
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            scale = min(1.0, 960 / image.shape[1])
            if scale < 1:
                image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
            name = f"{number:04d}.jpg"
            cv2.imwrite(os.path.join(folder, name), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return f"{session.id}/{name}"
        except Exception:
            return None

    # ---------- output ----------
    def annotate(self, camera_id, frame):
        session = self._sessions.get(str(camera_id))
        if session is None:
            return frame
        frame = frame.copy()
        height, width = frame.shape[:2]
        overlay = session.overlay
        fresh = time.monotonic() - overlay.get("at", 0) < 2.0
        if fresh:
            for x1, y1, x2, y2, track in overlay.get("people", []):
                colour = (0, 0, 255) if track in overlay.get("flagged", ()) else (255, 200, 0)
                cv2.rectangle(frame, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), colour, 1)
            for x1, y1, x2, y2 in overlay.get("phones", []):
                cv2.rectangle(frame, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), (0, 0, 255), 2)
                cv2.putText(frame, "phone?", (int(x1 * width), max(12, int(y1 * height) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            for cx, cy, empty in overlay.get("seats", []):
                cv2.circle(frame, (int(cx * width), int(cy * height)), 4, (0, 0, 255) if empty else (0, 200, 0), -1)
        learning = time.monotonic() < session.learn_until
        text = f"EXAM: {session.name}" + (" (learning seats)" if learning else "") + f"  cues: {len(session.events)}"
        cv2.putText(frame, text, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, text, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        return frame

    def snapshot(self):
        with self._cond:
            return list(self._alerts)[::-1]

    def status(self):
        with self._cond:
            return {"state": self._state, "error": self._error,
                    "active": {camera: session.summary() for camera, session in self._sessions.items()}}


def create_default_exam_monitor(config):
    return ExamMonitor(config)
