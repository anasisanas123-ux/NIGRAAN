"""On-demand camera recording to H.264 MP4 files.

The dashboard starts and stops recording per camera. Frames come from the same
preview loop that feeds the activity models (raw camera frames, no overlays), are
queued, and encoded on a separate thread so the live preview never waits for the
encoder. Each file has a JSON sidecar with its camera, label and start/end times,
which the dashboard uses to list recordings and the appearance search uses to
turn a position in the video into a wall-clock time.
"""
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
import queue
import re
import threading
import time

import cv2

_CAMERA = re.compile(r"[A-Za-z0-9_-]{1,40}")


def _utc_now():
    return datetime.now(timezone.utc)


def _iso(moment):
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path, data):
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(temp, path)


class _Segment:
    """One open MP4 file."""

    def __init__(self, path, width, height, fps, encoder):
        import av

        self.path = path
        self.container = av.open(path, mode="w", options={"movflags": "+faststart"})
        last_error = None
        for codec in encoder:
            try:
                stream = self.container.add_stream(codec, rate=fps)
                stream.width, stream.height = width, height
                stream.pix_fmt = "yuv420p"
                stream.time_base = Fraction(1, 1000)
                if codec == "libx264":
                    stream.options = {"preset": "veryfast", "crf": "24"}
                elif codec == "h264_nvenc":
                    stream.options = {"preset": "p4", "cq": "26"}
                # Opening the codec here makes a missing GPU encoder fail now,
                # not on the first frame.
                stream.codec_context.open()
                self.stream, self.codec = stream, codec
                break
            except Exception as error:  # try the next encoder
                last_error = error
                self.container.close()
                self.container = av.open(path, mode="w", options={"movflags": "+faststart"})
        else:
            self.container.close()
            raise RuntimeError(f"No H.264 encoder could be opened: {last_error}")
        self.width, self.height = width, height
        self.frames = 0
        self.last_pts = -1

    def write(self, frame_bgr, pts_ms):
        import av

        pts = max(int(pts_ms), self.last_pts + 1)
        self.last_pts = pts
        frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        frame.pts = pts
        frame.time_base = Fraction(1, 1000)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


class _Recording(threading.Thread):
    def __init__(self, recorder, camera, label):
        super().__init__(daemon=True, name=f"recorder-{camera}")
        self.recorder, self.camera, self.label = recorder, camera, label
        self.queue = queue.Queue(maxsize=max(10, recorder.fps * 4))
        self.stop_event = threading.Event()
        self.started_at = _utc_now()
        self.segment = None
        self.segment_meta = None
        self.segment_started = None
        self.files = []
        self.dropped = 0
        self.error = None
        self.last_sample = -float("inf")

    # Called from the preview thread; must never block.
    def offer(self, frame, wall_time):
        if wall_time - self.last_sample < 1.0 / self.recorder.fps:
            return
        self.last_sample = wall_time
        try:
            self.queue.put_nowait((frame.copy(), wall_time))
        except queue.Full:
            self.dropped += 1

    def _resize(self, frame):
        height, width = frame.shape[:2]
        scale = min(1.0, self.recorder.max_width / max(1, width))
        new_w = max(2, int(width * scale) // 2 * 2)
        new_h = max(2, int(height * scale) // 2 * 2)
        if (new_w, new_h) != (width, height):
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame

    def _open_segment(self, frame, wall_time):
        folder = os.path.join(self.recorder.folder, self.camera)
        os.makedirs(folder, exist_ok=True)
        start = datetime.fromtimestamp(wall_time, tz=timezone.utc)
        local = datetime.fromtimestamp(wall_time).strftime("%Y%m%d-%H%M%S")
        base = f"{self.camera}_{local}"
        path = os.path.join(folder, base + ".mp4")
        suffix = 1
        while os.path.exists(path):
            suffix += 1
            path = os.path.join(folder, f"{base}-{suffix}.mp4")
        height, width = frame.shape[:2]
        self.segment = _Segment(path, width, height, self.recorder.fps, self.recorder.encoders())
        self.segment_started = wall_time
        recording_id = os.path.splitext(os.path.basename(path))[0]
        self.segment_meta = {
            "id": recording_id,
            "camera_id": self.camera,
            "label": self.label,
            "file": os.path.relpath(path, self.recorder.folder).replace("\\", "/"),
            "started_at": _iso(start),
            "ended_at": None,
            "duration_seconds": 0,
            "width": width,
            "height": height,
            "codec": self.segment.codec,
            "status": "recording",
        }
        _write_json(path[:-4] + ".json", self.segment_meta)
        self.files.append(self.segment_meta["file"])

    def _close_segment(self, wall_time, status="complete"):
        if not self.segment:
            return
        segment, meta = self.segment, self.segment_meta
        self.segment = None
        try:
            segment.close()
        except Exception as error:
            status, self.error = "damaged", f"{type(error).__name__}: {error}"
        meta.update({
            "ended_at": _iso(datetime.fromtimestamp(wall_time, tz=timezone.utc)),
            "duration_seconds": round(max(0.0, wall_time - self.segment_started), 1),
            "frames": segment.frames,
            "status": status if segment.frames else "empty",
        })
        _write_json(segment.path[:-4] + ".json", meta)
        if not segment.frames:
            for path in (segment.path, segment.path[:-4] + ".json"):
                try:
                    os.unlink(path)
                except OSError:
                    pass
        else:
            self.recorder._finished(meta)

    def run(self):
        last_time = time.time()
        try:
            while not (self.stop_event.is_set() and self.queue.empty()):
                try:
                    frame, wall_time = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                last_time = wall_time
                frame = self._resize(frame)
                height, width = frame.shape[:2]
                too_long = (self.segment and wall_time - self.segment_started
                            >= self.recorder.segment_seconds)
                new_size = self.segment and (width, height) != (self.segment.width, self.segment.height)
                if too_long or new_size:
                    self._close_segment(wall_time)
                if not self.segment:
                    self._open_segment(frame, wall_time)
                self.segment.write(frame, (wall_time - self.segment_started) * 1000.0)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            print(f"[RECORDER ERROR] {self.camera}: {self.error}")
        finally:
            self._close_segment(max(last_time, time.time()) if self.segment else last_time)
            self.recorder._ended(self.camera, self)

    def status(self):
        return {
            "recording": self.is_alive() and not self.stop_event.is_set(),
            "label": self.label,
            "started_at": _iso(self.started_at),
            "elapsed_seconds": round((_utc_now() - self.started_at).total_seconds(), 1),
            "current_file": self.segment_meta["file"] if self.segment else None,
            "frames": self.segment.frames if self.segment else 0,
            "dropped_frames": self.dropped,
            "error": self.error,
        }


class CameraRecorder:
    """Activity model interface: update(camera, frame) is called for every preview frame."""

    def __init__(self, folder, fps=15, max_width=1280, segment_minutes=10, encoder="auto", on_finished=None):
        self.folder = folder
        self.fps = max(1, min(30, int(fps)))
        self.max_width = max(320, int(max_width))
        self.segment_seconds = max(60, float(segment_minutes) * 60)
        self.encoder = encoder
        self.on_finished = on_finished
        self._lock = threading.Lock()
        self._active = {}
        self._last_error = {}
        self._recover_interrupted()

    def encoders(self):
        if self.encoder in ("h264_nvenc", "libx264"):
            return [self.encoder]
        # libx264 first: opening NVENC holds the Python lock for ~2 s, which
        # would pause the live preview at the start of every file.
        return ["libx264", "h264_nvenc"]

    def _recover_interrupted(self):
        """Mark files left 'recording' by a crash, so the list shows them honestly."""
        if not os.path.isdir(self.folder):
            return
        for root, _dirs, files in os.walk(self.folder):
            for name in files:
                if not name.endswith(".json"):
                    continue
                path = os.path.join(root, name)
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        meta = json.load(handle)
                    if meta.get("status") == "recording":
                        meta["status"] = "interrupted"
                        _write_json(path, meta)
                except (OSError, ValueError):
                    continue

    def start(self, camera, label=""):
        camera = str(camera)
        if not _CAMERA.fullmatch(camera):
            raise ValueError("Invalid camera ID.")
        label = re.sub(r"[\x00-\x1f]", "", str(label or ""))[:80]
        with self._lock:
            current = self._active.get(camera)
            if current and current.is_alive() and not current.stop_event.is_set():
                return current.status()
            recording = _Recording(self, camera, label)
            self._active[camera] = recording
            self._last_error.pop(camera, None)
        recording.start()
        return recording.status()

    def stop(self, camera):
        with self._lock:
            recording = self._active.get(str(camera))
        if not recording:
            return {"recording": False}
        recording.stop_event.set()
        recording.join(timeout=15)
        return {"recording": False, "files": list(recording.files), "error": recording.error}

    def stop_all(self):
        with self._lock:
            cameras = list(self._active)
        for camera in cameras:
            self.stop(camera)

    def _ended(self, camera, recording):
        with self._lock:
            if self._active.get(camera) is recording:
                self._active.pop(camera, None)
                if recording.error:
                    self._last_error[camera] = recording.error

    def _finished(self, meta):
        if self.on_finished:
            try:
                self.on_finished(dict(meta))
            except Exception:
                pass

    def update(self, camera, frame):
        recording = self._active.get(camera)
        if recording is not None and frame is not None and not recording.stop_event.is_set():
            recording.offer(frame, time.time())

    def annotate(self, camera, frame):
        recording = self._active.get(camera)
        if recording is None or recording.stop_event.is_set():
            return frame
        frame = frame.copy()
        blink = int(time.time() * 2) % 2 == 0
        if blink:
            cv2.circle(frame, (18, 18), 7, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (30, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

    def status(self):
        with self._lock:
            active = {camera: rec.status() for camera, rec in self._active.items()}
            errors = dict(self._last_error)
        return {"cameras": active, "last_errors": errors, "folder": self.folder}


def create_default_recorder(config):
    return CameraRecorder(
        getattr(config, "RECORDINGS_DIR", os.path.join(config.DATA_DIR, "recordings")),
        fps=getattr(config, "RECORDING_FPS", 15),
        max_width=getattr(config, "RECORDING_MAX_WIDTH", 1280),
        segment_minutes=getattr(config, "RECORDING_SEGMENT_MINUTES", 10),
        encoder=getattr(config, "RECORDING_ENCODER", "auto"),
    )
