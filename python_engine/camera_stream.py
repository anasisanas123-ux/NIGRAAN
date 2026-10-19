"""Camera capture and reconnection, independent of model inference."""

import os
import threading
import time
import cv2
import config
from python_engine.frame_buffer import FrameBuffer


class CameraStream:
    def __init__(self, source):
        self._source = source
        self.generation = 0
        self.running = True
        self.prefer_mjpeg = bool(getattr(config, "USB_CAMERA_MJPEG", False))
        self.buffer = FrameBuffer()
        self.cap = self._open_capture()
        self.thread = threading.Thread(target=self._reader, daemon=True, name="camera-capture")
        self.thread.start()

    def _open_capture(self):
        if isinstance(self._source, str) and self._source.startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|framedrop;1|max_delay;0|reorder_queue_size;0"
            )
            cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self._source)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if isinstance(self._source, int) and self.prefer_mjpeg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, getattr(config, "CAPTURE_FPS", 25))
        except Exception:
            pass
        return cap

    def _reader(self):
        fails, reconnect_pending = 0, False
        while self.running:
            try:
                success, frame = self.cap.read()
            except Exception:
                success, frame = False, None
            if not success or frame is None:
                fails += 1
                if fails > 30:
                    # Invalidate the old session before opening a new one.
                    self.buffer.invalidate()
                    self.cap.release()
                    # Fall back to the driver's default format if MJPEG fails.
                    self.prefer_mjpeg = False
                    time.sleep(2)
                    if not self.running:
                        break
                    self.cap = self._open_capture()
                    reconnect_pending = True
                    fails = 0
                time.sleep(0.01)
                continue
            fails = 0
            if reconnect_pending:
                self.generation += 1
                reconnect_pending = False
            self.buffer.publish(frame, self.generation)

    def read_packet(self, after=0, timeout=0.5):
        return self.buffer.read(after, timeout)

    def read(self):
        """Compatibility for connection checks; inference uses read_packet."""
        packet = self.read_packet(timeout=0)
        return (False, None) if packet is None else (True, packet.frame.copy())

    def stop(self):
        self.running = False
        self.buffer.close()
        self.cap.release()
