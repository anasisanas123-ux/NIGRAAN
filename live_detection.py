import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import time
import threading
import queue
import requests
import torch
import config
from python_engine.face_detector   import FaceDetector
from python_engine.face_recognizer import FaceRecognizer
from python_engine.tracker         import Tracker
from python_engine.annotator       import Annotator

print("Loading models...")
detector   = FaceDetector()
recognizer = FaceRecognizer()
tracker    = Tracker()
annotator  = Annotator()
print(f"Device : {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"Students loaded: {len(recognizer.known_embeddings)}")
print(f"Threshold: {config.RECOGNITION_THRESHOLD}")
print("All models loaded.\n")

# ── Thread 1: Camera reader ───────────────────────────────
class CameraStream:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.frame   = None
        self.ret     = False
        self.lock    = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret   = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()


# ── Thread 2: Recognition worker ─────────────────────────
class RecognitionWorker:
    def __init__(self):
        self.in_queue  = queue.Queue(maxsize=1)
        self.out_queue = queue.Queue(maxsize=1)
        self.running   = True
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while self.running:
            try:
                detections = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            results = []
            for det in detections:
                r = recognizer.recognize(det['face_crop'])
                r['bbox']       = det['bbox']
                r['confidence'] = det['confidence']
                r['tracker_id'] = None
                results.append(r)

            # replace old result, don't queue up
            try:
                self.out_queue.get_nowait()
            except queue.Empty:
                pass
            self.out_queue.put(results)

    def submit(self, detections):
        try:
            self.out_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.in_queue.put_nowait(detections)
        except queue.Full:
            pass

    def get_results(self):
        try:
            return self.out_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False


def post_detection(data):
    try:
        requests.post(config.NODE_DETECTION_URL,
                      json=data, timeout=0.5)
    except Exception:
        pass


def run_camera(cam_id, source):
    print(f"Connecting to camera: {source}")
    stream = CameraStream(source)
    worker = RecognitionWorker()
    time.sleep(2)

    ret, test = stream.read()
    if not ret or test is None:
        print("Cannot connect to camera.")
        stream.stop()
        return

    print(f"Camera {cam_id} ready. Press Q to quit.\n")

    frame_count  = 0
    last_log     = time.time()
    last_results = []
    fps_time     = time.time()
    fps          = 0
    detect_every = config.FRAME_SKIP

    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        frame_count += 1
        now      = time.time()
        elapsed  = max(now - fps_time, 0.001)
        fps      = 0.92 * fps + 0.08 * (1.0 / elapsed)
        fps_time = now

        # ── run detection every N frames ──────────────────
        if frame_count % detect_every == 0:
            frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect(frame_rgb)
            tracked    = tracker.update(detections, frame_rgb)

            if detections:
                worker.submit(detections)
            else:
                last_results = []

        # ── get recognition results (non-blocking) ────────
        new_results = worker.get_results()
        if new_results is not None:
            # match tracker IDs
            results = []
            for r in new_results:
                dx = (r['bbox'][0] + r['bbox'][2]) // 2
                dy = (r['bbox'][1] + r['bbox'][3]) // 2
                r['tracker_id'] = None
                for t in tracked:
                    tx1,ty1,tx2,ty2 = t['bbox']
                    if tx1 <= dx <= tx2 and ty1 <= dy <= ty2:
                        r['tracker_id'] = t['tracker_id']
                        break
                results.append(r)
            last_results = results

            # log to Node.js
            if now - last_log > 3:
                for r in last_results:
                    threading.Thread(
                        target=post_detection,
                        args=({
                            'camera_id':    cam_id,
                            'student_id':   r.get('student_id'),
                            'student_name': r.get('name', 'Unknown'),
                            'is_known':     r.get('is_known', False),
                            'confidence':   round(r.get('confidence',0),4),
                            'tracker_id':   r.get('tracker_id')
                        },),
                        daemon=True
                    ).start()
                last_log = now

        # ── draw and show ─────────────────────────────────
        display = annotator.draw(frame.copy(), last_results)
        cv2.putText(display, f"FPS: {fps:.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.putText(display,
                    f"Students: {len(recognizer.known_embeddings)}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
        cv2.imshow(f"Campus Surveillance — {cam_id}", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    worker.stop()
    cv2.destroyAllWindows()
    print("System stopped.")


if __name__ == "__main__":
    emb_dir   = config.EMBEDDINGS_DIR
    npy_files = [f for f in os.listdir(emb_dir)
                 if f.endswith('.npy')] if os.path.exists(emb_dir) else []

    if not npy_files:
        print("No embeddings found! Run embedding_builder.py first.")
        sys.exit(1)

    cam_id = list(config.CAMERA_SOURCES.keys())[0]
    source = list(config.CAMERA_SOURCES.values())[0]
    run_camera(cam_id, source)