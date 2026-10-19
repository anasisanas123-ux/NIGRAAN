"""Publish fresh camera frames independently of recognition latency."""

import threading
import time
import cv2
import config


class CameraPreview:
    def __init__(
        self,
        camera,
        stream,
        display_results,
        annotator,
        recognizer,
        publish,
        activity_models,
        zones,
    ):
        self.camera, self.stream = camera, stream
        self.display_results, self.annotator = display_results, annotator
        self.recognizer, self.publish = recognizer, publish
        self.activity_models, self.zones = activity_models, zones
        self.lock = threading.Lock()
        self.result = None
        self.thread = threading.Thread(target=self.run, daemon=True, name=f"preview-{camera}")

    def submit(self, packet, detections):
        with self.lock:
            self.result = (packet.captured_at, packet.generation, [{**d} for d in detections])

    def current_detections(self, packet):
        with self.lock:
            result = self.result
        if result is None or result[1] != packet.generation:
            return [], None
        age = packet.captured_at - result[0]
        if not 0 <= age <= getattr(config, "OVERLAY_MAX_AGE", 1.0):
            return [], age
        # A deleted or expired enrollment must disappear even between AI runs.
        return [
            d
            for d in result[2]
            if not d.get("is_known") or self.recognizer.is_identity_active(d.get("student_id"))
        ], age

    def render(self, packet):
        detections, age = self.current_detections(packet)
        source_height, source_width = packet.frame.shape[:2]
        max_width, max_height = getattr(config, "PREVIEW_SIZE", (640, 480))
        ratio = min(max_width / source_width, max_height / source_height, 1.0)
        size = (max(1, round(source_width * ratio)), max(1, round(source_height * ratio)))
        small = cv2.resize(packet.frame, size) if ratio != 1 else packet.frame.copy()
        scaled = []
        for detection in detections:
            item = dict(detection)
            for key in ("bbox", "body_bbox", "face_bbox"):
                if item.get(key) is not None:
                    item[key] = [round(v * ratio) for v in item[key]]
            scaled.append(item)
        output = self.annotator.draw(small, scaled)
        for model in self.activity_models:
            model.update(self.camera, packet.frame)
            annotate = getattr(model, "annotate", None)
            if annotate:
                output = annotate(self.camera, output)
        output = self.zones.annotate(self.camera, output)
        self.publish(
            self.camera,
            output,
            packet.frame,
            packet.captured_at,
            {
                "capture_fps": round(packet.capture_fps, 1),
                "capture_sequence": packet.sequence,
                "source_width": source_width,
                "source_height": source_height,
                "overlay_age_seconds": round(age, 3) if age is not None else None,
            },
        )
        self.display_results.put_nowait((self.camera, output, [], packet.capture_fps))

    def run(self):
        sequence, last_publish = 0, 0.0
        interval = 1 / max(1, getattr(config, "PREVIEW_FPS", 20))
        while self.stream.running:
            delay = interval - (time.monotonic() - last_publish)
            if delay > 0:
                time.sleep(delay)
            packet = self.stream.read_packet(sequence, timeout=0.25)
            if packet is None:
                continue
            sequence = packet.sequence
            last_publish = time.monotonic()
            try:
                self.render(packet)
            except Exception as error:
                # Report the camera ID only; source URLs may contain passwords.
                print(f"[PREVIEW ERROR] {self.camera}: {type(error).__name__}: {error}")
