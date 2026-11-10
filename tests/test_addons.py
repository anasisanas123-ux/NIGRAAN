"""Offline tests for the add-on modules: sightings, visitor watch list,
exam-mode cue rules and the recorder. No camera or model is needed."""
import json
import os
import shutil
import sys
import tempfile
import time
import types
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_engine.presence_log import PresenceLog
from python_engine.visitor_watchlist import VisitorWatchlist
from python_engine import exam_monitor as exam


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class PresenceLogTests(unittest.TestCase):
    def test_known_people_are_logged_once_per_interval(self):
        log = PresenceLog(interval=30)
        ali = {"is_known": True, "student_id": "S1", "name": "Ali", "confidence": 0.8}
        stranger = {"is_known": False, "global_id": "UNK-1"}
        log.record("cam_a", [ali, stranger], now=0)
        log.record("cam_a", [ali], now=10)
        log.record("cam_b", [ali], now=10)
        log.record("cam_a", [ali], now=31)
        events = log.since(0)["events"]
        self.assertEqual([(e["person_id"], e["camera_id"]) for e in events],
                         [("S1", "cam_a"), ("S1", "cam_b"), ("S1", "cam_a")])
        self.assertEqual(log.since(events[1]["seq"])["events"][0]["seq"], 3)


class VisitorWatchlistTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_expired_visitor_seen_twice_raises_one_alert(self):
        face = unit(np.arange(1, 513))
        watch = VisitorWatchlist(self.folder, threshold=0.5, confirm=2, cooldown=60)
        watch.add("V100", "Sara", face[None], reason="expired")
        unknown = {"is_known": False, "face_embedding": face}
        other = {"is_known": False, "face_embedding": unit(np.r_[np.ones(256), -np.ones(256)])}
        watch.check("cam_a", [unknown, other], now=0)
        self.assertEqual(watch.snapshot(), [])
        watch.check("cam_a", [unknown], now=1)
        watch.check("cam_a", [unknown], now=2)
        alerts = watch.snapshot()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["event_type"], "expired_visitor_seen")
        self.assertIn("Sara", alerts[0]["description"])
        self.assertIn("expired", alerts[0]["description"])

    def test_known_people_and_faceless_results_are_ignored(self):
        face = unit(np.ones(512))
        watch = VisitorWatchlist(self.folder, threshold=0.5, confirm=1)
        watch.add("V1", "A", face[None])
        watch.check("cam_a", [{"is_known": True, "face_embedding": face}, {"is_known": False}], now=0)
        self.assertEqual(watch.snapshot(), [])

    def test_entries_survive_restart_and_are_deleted_after_keep_days(self):
        watch = VisitorWatchlist(self.folder, keep_days=7)
        watch.add("V1", "A", unit(np.ones(512))[None])
        reloaded = VisitorWatchlist(self.folder, keep_days=7)
        reloaded.load()
        self.assertEqual(reloaded.status()["watched_visitors"], 1)
        reloaded.prune(now=datetime.now(timezone.utc) + timedelta(days=8))
        self.assertEqual(reloaded.status()["watched_visitors"], 0)
        self.assertEqual(os.listdir(self.folder), [])


def person(box, nose=None, eyes=None, ears=None, shoulders=None):
    """Pose keypoints for one person; unspecified points are invisible."""
    kps, conf = np.zeros((17, 2)), np.zeros(17)
    for index, point in ((exam.NOSE, nose),):
        if point:
            kps[index], conf[index] = point, 0.9
    if eyes:
        for index, point in zip((exam.L_EYE, exam.R_EYE), eyes):
            if point:
                kps[index], conf[index] = point, 0.9
    if ears:
        for index, point in zip((exam.L_EAR, exam.R_EAR), ears):
            if point:
                kps[index], conf[index] = point, 0.9
    if shoulders:
        for index, point in zip((exam.L_SHOULDER, exam.R_SHOULDER), shoulders):
            kps[index], conf[index] = point, 0.9
    return {"box": list(box), "kps": kps, "kconf": conf}


def facing(x, y=100):
    """A student facing the camera, head at (x, y)."""
    return person((x - 40, y - 30, x + 40, y + 150), nose=(x, y), eyes=((x + 10, y - 8), (x - 10, y - 8)),
                  shoulders=((x + 35, y + 40), (x - 35, y + 40)))


def turned(x, y=100):
    """Head turned sideways: nose far off the centre between the eyes."""
    return person((x - 40, y - 30, x + 40, y + 150), nose=(x + 18, y), eyes=((x + 10, y - 8), (x - 4, y - 8)),
                  shoulders=((x + 35, y + 40), (x - 35, y + 40)))


class HeadDirectionTests(unittest.TestCase):
    def test_directions(self):
        p = facing(200)
        self.assertEqual(exam.head_direction(p["kps"], p["kconf"]), "forward")
        p = turned(200)
        self.assertEqual(exam.head_direction(p["kps"], p["kconf"]), "turned")
        p = person((0, 0, 80, 180), nose=(40, 20), eyes=((45, 12), None))
        self.assertEqual(exam.head_direction(p["kps"], p["kconf"]), "turned")
        p = person((0, 0, 80, 180), ears=((60, 20), (20, 20)), shoulders=((70, 60), (10, 60)))
        self.assertEqual(exam.head_direction(p["kps"], p["kconf"]), "turned")
        p = person((0, 0, 80, 180))
        self.assertIsNone(exam.head_direction(p["kps"], p["kconf"]))


class ExamRuleTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        config = types.SimpleNamespace(DATA_DIR=self.folder, EXAM_DIR=self.folder, EXAM_SAMPLE_INTERVAL=0.1,
                                       EXAM_SEAT_LEARN_SECONDS=2, EXAM_SEAT_EMPTY_SECONDS=5,
                                       EXAM_LOOK_SECONDS=3, EXAM_CLOSE_SECONDS=4, EXAM_ALERT_COOLDOWN=30)
        self.monitor = exam.ExamMonitor(config)
        self.session = exam._Session("cam_a", "Maths", self.folder, 0)
        self.session.learn_until = 2.0
        self.frame = np.zeros((480, 640, 3), np.uint8)

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def run_frames(self, frames):
        for now, people, phones in frames:
            self.monitor.evaluate(self.session, self.frame, people, phones, now)
        return [event["type"] for event in self.session.events]

    def test_phone_needs_two_frames(self):
        phone = [{"box": [100, 150, 120, 180], "conf": 0.7}]
        self.assertEqual(self.run_frames([(0, [facing(100)], phone), (0.4, [facing(100)], [])]), [])
        self.assertEqual(self.run_frames([(0.8, [facing(100)], phone), (1.2, [facing(100)], phone)]),
                         ["exam_phone"])
        snapshot = self.session.events[0]["snapshot"]
        self.assertTrue(os.path.isfile(os.path.join(self.folder, snapshot)))
        alert = self.monitor.snapshot()[0]
        self.assertEqual(alert["event_type"], "exam_phone")
        self.assertIn("Maths", alert["description"])

    def test_looking_around_only_after_three_seconds(self):
        frames = [(t / 2, [turned(100), facing(400)], []) for t in range(0, 6)]
        self.assertEqual(self.run_frames(frames), [])
        self.assertEqual(self.run_frames([(3.1, [turned(100), facing(400)], [])]), ["exam_looking_around"])
        self.assertIn("student #1", self.session.events[0]["who"])

    def test_left_seat_after_learning(self):
        both = [facing(100), facing(400)]
        frames = [(t / 2, both, []) for t in range(0, 5)]           # learn seats for 2 s
        frames += [(2.5 + t / 2, [facing(400)], []) for t in range(0, 12)]  # left student gone 5.5 s
        types_seen = self.run_frames(frames)
        self.assertEqual(types_seen, ["exam_left_seat"])
        self.assertEqual(len(self.session.seats), 2)
        self.run_frames([(9.0, both, [])])
        self.assertIsNotNone(self.session.events[0]["returned_at"])

    def test_heads_close_together(self):
        pair = [facing(200), facing(230)]
        self.assertEqual(self.run_frames([(t, pair, []) for t in (0, 1, 2, 3)]), [])
        self.assertEqual(self.run_frames([(4.1, pair, [])]), ["exam_too_close"])
        apart = [facing(100), facing(400)]
        self.monitor.evaluate(self.session, self.frame, apart, [], 5)
        self.assertEqual(self.session.pairs, {})

    def test_report_is_written_and_listed(self):
        self.monitor._sessions["cam_a"] = self.session
        self.session.events.append({"type": "exam_phone"})
        result = self.monitor.stop("cam_a")
        self.assertEqual(result["counts"], {"exam_phone": 1})
        listed = self.monitor.reports()
        self.assertEqual(listed[0]["id"], self.session.id)
        self.assertNotIn("events", listed[0])
        self.assertEqual(self.monitor.report(self.session.id)["counts"], {"exam_phone": 1})


class RecorderTests(unittest.TestCase):
    def test_recording_writes_playable_mp4_and_sidecar(self):
        from python_engine.recorder import CameraRecorder
        import av

        folder = tempfile.mkdtemp()
        try:
            finished = []
            recorder = CameraRecorder(folder, fps=10, max_width=640, encoder="libx264",
                                      on_finished=finished.append)
            recorder.start("cam_a", "Lab test")
            for index in range(12):
                frame = np.full((720, 1280, 3), index * 20 % 255, np.uint8)
                recorder.update("cam_a", frame)
                time.sleep(0.11)
            self.assertTrue(recorder.status()["cameras"]["cam_a"]["recording"])
            result = recorder.stop("cam_a")
            self.assertEqual(len(result["files"]), 1)
            meta_path = os.path.join(folder, result["files"][0][:-4] + ".json")
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertEqual((meta["status"], meta["label"], meta["width"]), ("complete", "Lab test", 640))
            with av.open(os.path.join(folder, result["files"][0])) as container:
                frames = sum(1 for _ in container.decode(video=0))
            self.assertGreaterEqual(frames, 8)
            self.assertEqual(finished[0]["id"], meta["id"])
            self.assertEqual(recorder.status()["cameras"], {})
        finally:
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
