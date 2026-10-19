"""Asynchronous, scene-level gun/knife review cues from a separate ONNX model."""
from collections import deque
import ast
import os
import threading
import time

import cv2
import numpy as np


class WeaponDetector:
    LABELS = ('gun', 'knife')

    def __init__(self, model_path, enabled=True, thresholds=None, interval=.5,
                 confirm_frames=2, cooldown=20., result_ttl=2.):
        self.model_path = os.path.abspath(model_path)
        self.enabled = bool(enabled)
        self.thresholds = dict(thresholds or {'gun': .55, 'knife': .60})
        self.interval = max(.2, float(interval))
        self.confirm_frames = max(2, int(confirm_frames))
        self.cooldown = max(1., float(cooldown))
        self.result_ttl = float(result_ttl)
        self._condition = threading.Condition()
        self._pending = {}
        self._last_sample = {}
        self._cameras = {}
        self._streaks = {}
        self._last_alert = {}
        self._alerts = deque(maxlen=200)
        self._started = False
        self._session = None
        self._error = None
        self._state = ('disabled' if not enabled else
                       'waiting_frames' if os.path.isfile(self.model_path) else 'weights_missing')

    def status(self):
        with self._condition:
            return {'state': self._state, 'error': self._error, 'device': 'cpu',
                    'classes': list(self.LABELS), 'thresholds': dict(self.thresholds),
                    'cameras': {cam: {key: value for key, value in info.items()
                                     if key != 'captured_at'} for cam, info in self._cameras.items()}}

    def snapshot(self):
        with self._condition:
            return list(self._alerts)[::-1]

    def update(self, camera_id, frame, now=None):
        if not self.enabled or frame is None:
            return
        now = time.monotonic() if now is None else float(now)
        camera_id = str(camera_id)
        with self._condition:
            if self._state == 'load_error':
                return
            if not os.path.isfile(self.model_path):
                self._state = 'weights_missing'
                return
            if now - self._last_sample.get(camera_id, -float('inf')) < self.interval:
                return
            if not self._started:
                self._started = True
                self._state = 'loading'
                threading.Thread(target=self._worker, name='weapon-detector', daemon=True).start()
            self._last_sample[camera_id] = now
            if camera_id not in self._pending and len(self._pending) >= 4:
                self._pending.pop(next(iter(self._pending)))
            self._pending[camera_id] = (frame.copy(), now)
            self._condition.notify()

    def _load_session(self):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(self.model_path, sess_options=options,
                                       providers=['CPUExecutionProvider'])
        names = ast.literal_eval(session.get_modelmeta().custom_metadata_map.get('names', '{}'))
        if names != {0: 'gun', 1: 'knife'}:
            raise ValueError('Weapon model must contain exactly gun and knife classes.')
        if session.get_inputs()[0].shape != [1, 3, 640, 640]:
            raise ValueError('Weapon model must accept one 640x640 RGB image.')
        return session

    def _infer(self, frame):
        height, width = frame.shape[:2]
        ratio = min(640 / width, 640 / height)
        resized_width, resized_height = round(width * ratio), round(height * ratio)
        left, top = (640 - resized_width) // 2, (640 - resized_height) // 2
        image = np.full((640, 640, 3), 114, np.uint8)
        image[top:top + resized_height, left:left + resized_width] = cv2.resize(
            frame, (resized_width, resized_height))
        tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.
        output = self._session.run(None, {self._session.get_inputs()[0].name: tensor})[0]
        return self._decode(output, width, height, ratio, left, top)

    def _decode(self, output, width, height, ratio, left, top):
        output = np.asarray(output)
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 6:
            raise ValueError('Unexpected gun/knife model output shape.')
        if not np.isfinite(output).all():
            raise ValueError('Weapon model returned non-finite predictions.')
        predictions = output[0].T
        classes = predictions[:, 4:].argmax(axis=1)
        scores = predictions[:, 4:].max(axis=1)
        results = []
        for class_id, label in enumerate(self.LABELS):
            selected = predictions[(classes == class_id) & (scores >= self.thresholds[label]) & (scores <= 1.)]
            boxes, confidences = [], []
            for row in selected:
                cx, cy, bw, bh = row[:4]
                x1 = float(np.clip((cx - bw / 2 - left) / ratio, 0, width))
                y1 = float(np.clip((cy - bh / 2 - top) / ratio, 0, height))
                x2 = float(np.clip((cx + bw / 2 - left) / ratio, 0, width))
                y2 = float(np.clip((cy + bh / 2 - top) / ratio, 0, height))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(float(row[4 + class_id]))
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0., .45) if boxes else []
            for index in np.asarray(indices).reshape(-1):
                x, y, w, h = boxes[int(index)]
                results.append({'weapon_type': label, 'confidence': confidences[int(index)],
                                'box': [x / width, y / height, (x + w) / width, (y + h) / height]})
        return sorted(results, key=lambda item: item['confidence'], reverse=True)[:10]

    def _record_result(self, camera_id, detections, captured_at, now=None):
        now = time.monotonic() if now is None else float(now)
        with self._condition:
            previous = self._cameras.get(camera_id, {})
            stale = now - captured_at > self.result_ttl
            gap = captured_at - previous.get('captured_at', captured_at) > self.result_ttl
            if stale:
                detections = []
            for label in self.LABELS:
                key = (camera_id, label)
                found = [item for item in detections if item['weapon_type'] == label]
                if gap:
                    self._streaks[key] = 0
                count = min(self.confirm_frames, self._streaks.get(key, 0) + 1) if found else 0
                self._streaks[key] = count
                if count < self.confirm_frames or now - self._last_alert.get(key, -float('inf')) < self.cooldown:
                    continue
                self._last_alert[key] = now
                score = max(item['confidence'] for item in found)
                self._alerts.append({
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'camera_id': camera_id, 'event_type': 'possible_weapon',
                    'weapon_type': label, 'score': round(score, 4), 'review_only': True,
                    'description': f'Possible {label} visible (score {score:.2f}). Review the camera footage.',
                })
            self._cameras[camera_id] = {
                'captured_at': captured_at, 'detections': detections,
                'last_inference_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            }
            self._state, self._error = 'ready', None

    def annotate(self, camera_id, frame, now=None):
        now = time.monotonic() if now is None else float(now)
        with self._condition:
            info = self._cameras.get(camera_id, {})
            detections = list(info.get('detections', [])) if now - info.get('captured_at', 0) <= self.result_ttl else []
        if not detections:
            return frame
        frame = frame.copy()
        height, width = frame.shape[:2]
        for item in detections:
            x1, y1, x2, y2 = [int(value * size) for value, size in zip(item['box'], (width, height, width, height))]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, f"Possible {item['weapon_type']} {item['confidence']:.2f}",
                        (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 165, 255), 2)
        return frame

    def _worker(self):
        try:
            self._session = self._load_session()
        except Exception as error:
            with self._condition:
                self._state, self._error = 'load_error', f'{type(error).__name__}: {error}'
            return
        while True:
            with self._condition:
                self._condition.wait_for(lambda: bool(self._pending))
                camera_id = next(iter(self._pending))
                frame, captured_at = self._pending.pop(camera_id)
            try:
                self._record_result(camera_id, self._infer(frame), captured_at)
            except Exception as error:
                with self._condition:
                    self._state, self._error = 'inference_error', f'{type(error).__name__}: {error}'
                    self._cameras.pop(camera_id, None)
                    for label in self.LABELS:
                        self._streaks[(camera_id, label)] = 0


def create_default_weapon_detector(config):
    return WeaponDetector(
        config.WEAPON_MODEL_PATH, enabled=config.WEAPON_MODEL_ENABLED,
        thresholds=config.WEAPON_MODEL_THRESHOLDS, interval=config.WEAPON_MODEL_INTERVAL,
        confirm_frames=config.WEAPON_MODEL_CONFIRM_FRAMES, cooldown=config.WEAPON_MODEL_COOLDOWN)
