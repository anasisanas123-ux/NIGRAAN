"""Offline regressions; no cameras, MongoDB, or model inference are started."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from python_engine.arcface_recognizer import ArcFaceRecognizer
from python_engine.identity_cache import invalidate_inactive, needs_face_check
from python_engine.behavior_monitor import BehaviorMonitor
from python_engine.violence_classifier import ViolenceClassifier
from python_engine import engine_api
from python_engine.face_quality import FaceEvidence


def clear_image(height=100, width=100):
    return np.random.default_rng(7).integers(50, 205, (height, width, 3), dtype=np.uint8)


def recognizer():
    result = ArcFaceRecognizer.__new__(ArcFaceRecognizer)
    result.gallery_lock = threading.RLock()
    result.gallery_version = 0
    result.inference_lock = threading.RLock()
    result.evidence = FaceEvidence()
    result.crop_times = {}
    result.known_embeddings = {}
    result.known_names = {}
    result.known_roles = {}
    result.valid_until = {}
    result.emb_ids = []
    result.emb_matrix = None
    face = Mock(det_score=0.99, bbox=np.array([0, 0, 100, 100]),
                embedding=np.array([1., 0., 0., 0.]))
    result.app = Mock()
    result.app.get.return_value = [face]
    return result


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.paths = patch.object(config, 'EMBEDDINGS_DIR', self.directory.name)
        self.paths.start()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self.paths.stop)
        self.rec = recognizer()

    def enroll(self, identity='S001', role='student', **kwargs):
        return self.rec.add_identity_from_images(identity, 'Test Person',
            [clear_image()], role=role, **kwargs)

    def test_deleted_identity_disappears_from_both_track_caches_and_disk(self):
        self.enroll()
        confirmed = {'cam_a_1': {'student_id': 'S001'}}
        locked = {'cam_a_1': {'student_id': 'S001', 'is_known': True}}
        self.rec.remove_identity('S001')
        invalidate_inactive(self.rec, confirmed, locked)
        self.assertEqual((confirmed, locked), ({}, {}))
        self.assertFalse((Path(self.directory.name) / 'S001.npy').exists())
        restored = recognizer()
        restored.load_database()
        self.assertFalse(restored.is_identity_active('S001'))

    def test_new_visitor_updates_live_gallery_and_survives_reload_with_role(self):
        expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        result = self.enroll('V123', 'visitor', valid_until=expiry)
        self.assertEqual(result['gallery_version'], 1)
        self.assertTrue(self.rec.is_identity_active('V123'))
        match = self.rec.detect_and_recognize(clear_image(120, 120))[0]
        self.assertTrue(match['is_known'])
        self.assertEqual(match['role'], 'visitor')
        restored = recognizer()
        restored.load_database()
        self.assertTrue(restored.is_identity_active('V123'))
        self.assertEqual(restored.known_roles['V123'], 'visitor')

    def test_visitor_expiry_invalidates_cached_label_without_gallery_change(self):
        self.enroll('V123', 'visitor', valid_until=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
        self.rec.valid_until['V123'] = (datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        confirmed = {'cam_a_1': {'student_id': 'V123'}}
        locked = {'cam_a_1': {'student_id': 'V123'}}
        invalidate_inactive(self.rec, confirmed, locked)
        self.assertFalse(confirmed or locked)

    def test_unknown_retries_after_enrollment_and_again_when_face_was_hidden(self):
        entry = {'global_id': 'UNK-001', 'gallery_version': 1, 'last_face_check': 10.}
        self.assertTrue(needs_face_check(entry, 2, 10.1, .8))
        entry.update(gallery_version=2, last_face_check=10.1)
        self.assertFalse(needs_face_check(entry, 2, 10.5, .8))
        self.assertTrue(needs_face_check(entry, 2, 11., .8))
        self.assertEqual(entry['global_id'], 'UNK-001')


    def test_live_camera_waits_for_three_matches_and_resets_after_gallery_change(self):
        self.enroll()
        image = clear_image(120, 120)
        first = self.rec.detect_and_recognize(image, cam_id='cam_a')[0]
        self.assertFalse(first['is_known'])
        self.assertTrue(first['identity_pending'])
        self.assertIsNone(first['student_id'])
        self.assertFalse(self.rec.detect_and_recognize(image, cam_id='cam_a')[0]['is_known'])
        self.assertTrue(self.rec.detect_and_recognize(image, cam_id='cam_a')[0]['is_known'])
        self.rec.gallery_version += 1
        self.assertFalse(self.rec.detect_and_recognize(image, cam_id='cam_a')[0]['is_known'])


    def test_soft_camera_frame_can_confirm_strong_visitor_match(self):
        self.enroll('V123','visitor',valid_until=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
        face = self.rec.app.get.return_value[0]
        face.embedding = np.array([.75,np.sqrt(1-.75**2),0.,0.])
        soft = cv2.GaussianBlur(clear_image(120,120),(9,9),1.4)
        self.assertFalse(self.rec.detect_and_recognize(soft,cam_id='cam_a')[0]['is_known'])
        self.assertFalse(self.rec.detect_and_recognize(soft,cam_id='cam_a')[0]['is_known'])
        result = self.rec.detect_and_recognize(soft,cam_id='cam_a')[0]
        self.assertTrue(result['is_known'])
        self.assertEqual(result['role'],'visitor')
        self.assertEqual(result['student_id'],'V123')

    def test_soft_frame_weak_match_is_not_used_for_identity_or_anonymous_gallery(self):
        self.enroll()
        face = self.rec.app.get.return_value[0]
        face.embedding = np.array([.65,np.sqrt(1-.65**2),0.,0.])
        soft = cv2.GaussianBlur(clear_image(120,120),(9,9),1.4)
        self.assertEqual(self.rec.detect_and_recognize(soft,cam_id='cam_a'),[])
        with self.assertRaises(ValueError):
            self.rec.add_identity_from_images('S002','Soft enrollment',[soft])


class FeedTests(unittest.TestCase):
    def setUp(self):
        self.client = engine_api.app.test_client()
        for state in (engine_api.feeds, engine_api.raw_feeds, engine_api.camera_state):
            state.clear()

    def test_snapshot_uses_unannotated_engine_frame(self):
        raw = np.full((120, 160, 3), 220, np.uint8)
        with patch.object(engine_api.violence_classifier, 'update') as classifier, \
                patch.object(engine_api.weapon_detector, 'update'):
            engine_api.update_feed('cam_a', np.zeros_like(raw), activity_frame=raw)
            classifier.assert_called_once()
        response = self.client.get('/snapshot/cam_a')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        decoded = cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(float(decoded.mean()), 210)

    def test_snapshot_refuses_offline_or_unknown_camera(self):
        self.assertEqual(self.client.get('/snapshot/cam_a').status_code, 503)
        self.assertEqual(self.client.get('/snapshot/does-not-exist').status_code, 404)
        engine_api.raw_feeds['cam_a'] = np.zeros((10, 10, 3), np.uint8)
        engine_api.camera_state['cam_a'] = {'updated_epoch': 0}
        self.assertEqual(self.client.get('/snapshot/cam_a').status_code, 503)

    def test_enrollment_and_removal_api_use_the_live_recognizer(self):
        identity = Mock()
        identity.add_identity_from_images.return_value = {'identity_id': 'V123', 'role': 'visitor'}
        identity.remove_identity.return_value = {'identity_id': 'V123'}
        ok, jpeg = cv2.imencode('.jpg', np.zeros((100, 100, 3), np.uint8))
        self.assertTrue(ok)
        import base64
        photo = 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode('ascii')
        with patch.object(engine_api, 'identity_recognizer', identity):
            response = self.client.post('/enroll', json={
                'identity_id': 'V123', 'name': 'Visitor', 'role': 'visitor',
                'valid_until': '2099-01-01T00:00:00Z', 'images': [photo],
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(identity.add_identity_from_images.call_args.kwargs['role'], 'visitor')
            self.assertEqual(self.client.delete('/identity/V123').status_code, 200)
            identity.remove_identity.assert_called_once_with('V123')


class BehaviorTests(unittest.TestCase):
    def test_x3d_probability_is_not_softmaxed_a_second_time(self):
        import torch
        score = ViolenceClassifier._violence_probability(torch.tensor([[.9, .1]]))
        self.assertAlmostEqual(score, .1, places=6)
        with self.assertRaises(ValueError):
            ViolenceClassifier._violence_probability(torch.tensor([[float('nan'), .1]]))
        with self.assertRaises(ValueError):
            ViolenceClassifier._violence_probability(torch.tensor([[2., -1.]]))

    def test_violence_cue_requires_consecutive_positive_windows_and_cooldown(self):
        classifier = ViolenceClassifier('unused.pt', threshold=.4, confirm_windows=2)
        classifier._record_score('cam_a', .8)
        classifier._record_score('cam_a', .1)
        classifier._record_score('cam_a', .8)
        self.assertFalse(classifier.snapshot())
        classifier._record_score('cam_a', .8)
        self.assertEqual(len(classifier.snapshot()), 1)
        classifier._record_score('cam_a', .9)
        self.assertEqual(len(classifier.snapshot()), 1)

    def test_camera_outage_restarts_clip_collection_and_confirmation(self):
        classifier = ViolenceClassifier('unused.pt', sample_interval=.25)
        classifier._worker_started = True
        classifier._state = 'ready'
        frame = np.zeros((20, 20, 3), np.uint8)
        with patch('python_engine.violence_classifier.os.path.isfile', return_value=True):
            for index in range(16):
                classifier.update('cam_a', frame, now=index * .25)
            self.assertEqual(classifier.status()['cameras']['cam_a']['sampled_frames'], 16)
            classifier._record_score('cam_a', .8)
            self.assertEqual(classifier.status()['cameras']['cam_a']['positive_windows'], 1)
            classifier.update('cam_a', frame, now=20)
        status = classifier.status()['cameras']['cam_a']
        self.assertEqual(status['sampled_frames'], 1)
        self.assertNotIn('last_inference_at', status)
        self.assertEqual(classifier._streaks['cam_a'], 0)

    def test_entry_dwell_and_rapid_motion_cues(self):
        monitor = BehaviorMonitor()
        zones = [{'id': 'door', 'name': 'Door', 'box': [0, 0, 1, 1], 'dwell_seconds': 1}]
        monitor.update('cam_a', [{'tracker_id': '1', 'bbox': [10, 10, 30, 30]}], (200, 200), zones, now=0)
        monitor.update('cam_a', [{'tracker_id': '1', 'bbox': [100, 10, 120, 30]}], (200, 200), zones, now=1)
        types = {alert['event_type'] for alert in monitor.snapshot()}
        self.assertEqual(types, {'restricted_area_entry', 'extended_zone_dwell', 'rapid_movement'})

    def test_reused_track_does_not_inherit_dwell_after_gap(self):
        monitor = BehaviorMonitor(track_ttl=8)
        zones = [{'box': [0, 0, 1, 1], 'dwell_seconds': 10}]
        detection = [{'tracker_id': '1', 'bbox': [10, 10, 30, 30]}]
        monitor.update('cam_a', detection, (200, 200), zones, now=0)
        monitor.update('cam_a', detection, (200, 200), zones, now=20)
        self.assertNotIn('extended_zone_dwell', {a['event_type'] for a in monitor.snapshot()})


if __name__ == '__main__':
    unittest.main()
