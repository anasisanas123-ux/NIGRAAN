"""Explicit shared dependencies for per-camera identity workers."""

from dataclasses import dataclass, field
import threading


@dataclass
class RuntimeContext:
    recognizer: object
    body_detector: object
    body_reid: object
    reid: object
    anonymous_face_reid: object
    annotator: object
    global_logged_students: set = field(default_factory=set)
    global_logged_unknowns: set = field(default_factory=set)
    log_lock: object = field(default_factory=threading.Lock)
    cross_cam_fired: set = field(default_factory=set)
    cross_cam_lock: object = field(default_factory=threading.Lock)
    streams: dict = field(default_factory=dict)
    stream_lock: object = field(default_factory=threading.Lock)

    def register_stream(self, camera, stream):
        with self.stream_lock:
            self.streams[camera] = stream

    def stop(self):
        with self.stream_lock:
            streams = list(self.streams.values())
        for stream in streams:
            stream.stop()
