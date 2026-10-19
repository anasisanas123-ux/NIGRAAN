"""Frame freshness, bounded queues, cached JPEGs and asynchronous previews."""

from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
import unittest
from unittest.mock import Mock, patch
import numpy as np

import config
from python_engine.frame_buffer import FrameBuffer, FramePacket, LatestCameraResults, RollingRate
from python_engine.preview import CameraPreview
from python_engine import engine_api


class BufferTests(unittest.TestCase):
    def test_fps_counts_real_frames_despite_bursty_delivery(self):
        rate = RollingRate()
        self.assertEqual(rate.tick(0), 0)
        rate.tick(0.01)
        self.assertEqual(rate.tick(1), 2)
        self.assertEqual(rate.tick(2), 1.5)

    def test_independent_consumers_never_repeat_frames_or_accumulate_backlog(self):
        buffer = FrameBuffer()
        first = np.zeros((4, 4, 3), np.uint8)
        buffer.publish(first)
        a = buffer.read(timeout=0)
        b = buffer.read(timeout=0)
        self.assertIs(a, b)
        self.assertIsNone(buffer.read(after=a.sequence, timeout=0))
        for _ in range(100):
            buffer.publish(first)
        latest = buffer.read(after=a.sequence, timeout=0)
        self.assertEqual(latest.sequence, 101)
        self.assertIs(buffer.packet, latest)
        buffer.invalidate()
        self.assertIsNone(buffer.read(timeout=0))
        buffer.publish(first, generation=1)
        self.assertEqual(buffer.read(timeout=0).generation, 1)

    def test_close_wakes_waiters_and_stale_capture_is_rejected(self):
        buffer = FrameBuffer()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(buffer.read, 0, 2)
            buffer.close()
            self.assertIsNone(future.result(timeout=0.5))
        buffer = FrameBuffer()
        buffer.publish(np.zeros((1, 1)))
        with patch("python_engine.frame_buffer.time.time", return_value=time.time() + 10):
            self.assertIsNone(buffer.read(timeout=0))

    def test_camera_slots_replace_old_results_without_starving_other_camera(self):
        results = LatestCameraResults()
        results.put_nowait(("a", 0))
        results.put_nowait(("b", 9))
        for i in range(100):
            results.put_nowait(("a", i))
        self.assertEqual(len(results.pending), 2)
        self.assertEqual(results.get(), ("a", 99))
        self.assertEqual(results.get(), ("b", 9))
        with self.assertRaises(queue.Empty):
            results.get(timeout=0)


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((720, 1280, 3), np.uint8)
        self.packet = FramePacket(1, self.frame, time.time(), 0, 25)
        self.annotator = Mock()
        self.annotator.draw.side_effect = lambda image, detections: image
        self.model = Mock()
        self.model.annotate.side_effect = lambda camera, image: image
        self.zones = Mock()
        self.zones.annotate.side_effect = lambda camera, image: image
        self.recognizer = Mock()
        self.recognizer.is_identity_active.return_value = True
        self.publish = Mock()
        self.preview = CameraPreview(
            "cam_a",
            Mock(),
            LatestCameraResults(),
            self.annotator,
            self.recognizer,
            self.publish,
            (self.model,),
            self.zones,
        )

    def test_preview_without_any_inference_is_fresh_and_preserves_aspect_ratio(self):
        self.preview.render(self.packet)
        image, detections = self.annotator.draw.call_args.args
        self.assertEqual(image.shape, (360, 640, 3))
        self.assertEqual(detections, [])
        self.model.update.assert_called_once_with("cam_a", self.frame)
        self.assertIs(self.publish.call_args.args[2], self.frame)
        np.testing.assert_array_equal(self.frame, np.zeros_like(self.frame))

    def test_scaled_boxes_and_stale_deleted_or_reconnected_labels(self):
        detection = {"bbox": [100, 100, 300, 500], "is_known": True, "student_id": "S1"}
        self.preview.submit(self.packet, [detection])
        self.preview.render(self.packet)
        scaled = self.annotator.draw.call_args.args[1]
        self.assertEqual(scaled[0]["bbox"], [50, 50, 150, 250])
        self.assertEqual(detection["bbox"], [100, 100, 300, 500])
        self.recognizer.is_identity_active.return_value = False
        self.assertEqual(self.preview.current_detections(self.packet)[0], [])
        stale = FramePacket(2, self.frame, self.packet.captured_at + 3, 0, 25)
        self.assertEqual(self.preview.current_detections(stale)[0], [])
        reconnect = FramePacket(2, self.frame, self.packet.captured_at, 1, 25)
        self.assertEqual(self.preview.current_detections(reconnect)[0], [])


class EncodingTests(unittest.TestCase):
    def setUp(self):
        with engine_api.lock:
            for mapping in (
                engine_api.feeds,
                engine_api.feed_versions,
                engine_api.encoded_frames,
                engine_api.raw_feeds,
                engine_api.camera_state,
            ):
                mapping.clear()

    def test_concurrent_viewers_encode_once_per_frame_version(self):
        engine_api.publish_preview("cam_a", np.zeros((10, 10, 3), np.uint8))
        real_encode = engine_api.cv2.imencode
        with patch.object(engine_api.cv2, "imencode", wraps=real_encode) as encode:
            with ThreadPoolExecutor(max_workers=5) as pool:
                results = list(pool.map(lambda _: engine_api.encoded_frame("cam_a"), range(10)))
            self.assertEqual(encode.call_count, 1)
            self.assertTrue(all(value == results[0] for value in results))
            a, b = engine_api.generate("cam_a"), engine_api.generate("cam_a")
            self.assertEqual(next(a), next(b))
            self.assertEqual(encode.call_count, 1)
            engine_api.publish_preview("cam_a", np.full((10, 10, 3), 255, np.uint8))
            self.assertEqual(next(a), next(b))
            self.assertEqual(encode.call_count, 2)
            a.close()
            b.close()

    def test_preview_does_not_replay_analysis_and_health_tracks_capture_age(self):
        with patch.object(engine_api.behavior_monitor, "update") as behavior:
            engine_api.publish_preview(
                "cam_a", np.zeros((10, 10, 3), np.uint8), captured_at=time.time() - 10
            )
            behavior.assert_not_called()
        health = engine_api.app.test_client().get("/health").json["cameras"]["cam_a"]
        self.assertEqual(health["status"], "offline")
        engine_api.update_analysis("cam_a", [], (10, 10), {"processed_fps": 12, "inference_ms": 80})
        health = engine_api.app.test_client().get("/health").json["cameras"]["cam_a"]
        self.assertEqual(health["processed_fps"], 12)
        self.assertEqual(health["inference_ms"], 80)

    def test_worker_runs_once_on_a_unique_packet(self):
        from python_engine import camera_worker as module
        from python_engine.runtime_context import RuntimeContext

        stream = Mock(running=True)
        packet = FramePacket(1, np.zeros((720, 1280, 3), np.uint8), time.time(), 0, 25)
        calls = []

        def read(after=0, timeout=0.5):
            calls.append(after)
            if after == 0:
                return packet
            stream.running = False
            return None

        stream.read_packet.side_effect = read
        detector = Mock()
        detector.detect.return_value = []
        context = RuntimeContext(Mock(), detector, Mock(), Mock(), Mock(), Mock())
        with (
            patch.object(module, "CameraStream", return_value=stream),
            patch.object(module, "CameraPreview") as preview,
            patch.object(module, "update_analysis") as analysis,
        ):
            module.camera_worker("cam_a", 0, LatestCameraResults(), context)
        detector.detect.assert_called_once()
        self.assertEqual(detector.detect.call_args.kwargs["color_order"], "bgr")
        self.assertEqual(calls, [0, 0, 1])
        analysis.assert_called_once()
        preview.return_value.submit.assert_called_once()


class ColorContractTests(unittest.TestCase):
    def test_bgr_fast_path_preserves_existing_rgb_person_and_face_crops(self):
        import torch
        from python_engine.body_detector import BodyDetector

        detector = BodyDetector.__new__(BodyDetector)
        detector.device = "cpu"
        boxes = Mock()
        boxes.__len__ = Mock(return_value=1)
        boxes.xyxy = torch.tensor([[2, 2, 40, 80]])
        boxes.conf = torch.tensor([0.9])
        boxes.id = torch.tensor([3.0])
        model = Mock()
        model.track.return_value = [Mock(boxes=boxes)]
        detector._get_model = Mock(return_value=model)
        bgr = np.full((96, 64, 3), [10, 60, 200], dtype=np.uint8)
        original_rgb = bgr[:, :, ::-1].copy()
        legacy = detector.detect(original_rgb, cam_id="cam_a")
        optimized = detector.detect(bgr, cam_id="cam_a", color_order="bgr")
        self.assertEqual(len(legacy), 1)
        self.assertEqual(len(optimized), 1)
        np.testing.assert_array_equal(legacy[0]["body_crop"], optimized[0]["body_crop"])
        np.testing.assert_array_equal(legacy[0]["face_crop"], optimized[0]["face_crop"])
        self.assertIs(model.track.call_args.kwargs["source"], bgr)


if __name__ == "__main__":
    unittest.main()
