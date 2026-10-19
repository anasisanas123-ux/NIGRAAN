from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from python_engine.weapon_detector import WeaponDetector


def detection(label='gun', score=.9):
    return {'weapon_type': label, 'confidence': score, 'box': [.1, .1, .3, .3]}


class WeaponTests(unittest.TestCase):
    def setUp(self):
        self.detector = WeaponDetector('unused.onnx')

    def test_two_frames_are_required_and_cooldown_is_per_weapon_and_camera(self):
        d = self.detector
        d._record_result('cam_a', [detection()], 1, now=1)
        self.assertEqual(d.snapshot(), [])
        d._record_result('cam_a', [detection()], 1.5, now=1.5)
        d._record_result('cam_a', [detection()], 2, now=2)
        self.assertEqual(len(d.snapshot()), 1)
        for moment in (2.5, 3):
            d._record_result('cam_a', [detection('knife')], moment, now=moment)
        self.assertEqual({x['weapon_type'] for x in d.snapshot()}, {'gun', 'knife'})
        for moment in (3, 3.5):
            d._record_result('cam_b', [detection()], moment, now=moment)
        self.assertEqual(len(d.snapshot()), 3)

    def test_absent_object_and_camera_gap_reset_confirmation(self):
        d = self.detector
        for time, items in ((1, [detection()]), (1.5, []), (2, [detection()]), (6, [detection()])):
            d._record_result('cam_a', items, time, now=time)
        self.assertEqual(d.snapshot(), [])

    def test_stale_result_does_not_alert_or_remain_on_video(self):
        d = self.detector
        d._record_result('cam_a', [detection()], 1, now=1)
        d._record_result('cam_a', [detection()], 1.5, now=10)
        self.assertEqual(d.snapshot(), [])
        image = np.zeros((100, 100, 3), np.uint8)
        self.assertIs(d.annotate('cam_a', image, now=10), image)

    def test_decode_letterboxing_per_class_thresholds_and_duplicate_boxes(self):
        # Raw view is 1280x720; letterboxed image is 640x640 with 140 top pad.
        output = np.array([[
            [320, 320, 50], [320, 320, 320], [100, 100, 20], [100, 100, 20],
            [.9, .8, .1], [.1, .1, .59],
        ]], dtype=np.float32)
        found = self.detector._decode(output, 1280, 720, .5, 0, 140)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['weapon_type'], 'gun')
        np.testing.assert_allclose(found[0]['box'], [540/1280, 260/720, 740/1280, 460/720])
        output[0, 5, 2] = .8
        self.assertEqual(len(self.detector._decode(output, 1280, 720, .5, 0, 140)), 2)

    def test_invalid_model_output_is_rejected(self):
        for output in (np.zeros((1, 84, 8400)), np.full((1, 6, 1), np.nan)):
            with self.assertRaises(ValueError):
                self.detector._decode(output, 640, 640, 1, 0, 0)

    def test_pending_frames_are_replaced_and_the_queue_is_bounded(self):
        d = self.detector
        d._started = True
        frame = np.zeros((20, 20, 3), np.uint8)
        with patch('python_engine.weapon_detector.os.path.isfile', return_value=True):
            for index in range(10):
                d.update('cam_a', frame, now=index)
            self.assertEqual(len(d._pending), 1)
            for index in range(10):
                d.update(f'other_{index}', frame, now=20)
            self.assertLessEqual(len(d._pending), 4)

    def test_draw_uses_normalized_boxes_and_does_not_modify_original(self):
        self.detector._record_result('cam_a', [detection()], 1, now=1)
        frame = np.zeros((100, 200, 3), np.uint8)
        drawn = self.detector.annotate('cam_a', frame, now=1.2)
        self.assertEqual(int(frame.sum()), 0)
        self.assertGreater(int(drawn.sum()), 0)


if __name__ == '__main__':
    unittest.main()
