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
from python_engine.reid_manager    import ReIDManager
from python_engine.engine_api      import update_feed, start_engine

# maximize GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False

print("Loading models...")
detector   = FaceDetector()
recognizer = FaceRecognizer()
annotator  = Annotator()
reid       = ReIDManager()

trackers = {
    cam_id: Tracker()
    for cam_id in config.CAMERA_SOURCES.keys()
}

print(f"Device  : {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"Students: {len(recognizer.known_embeddings)}")
print(f"Cameras : {list(config.CAMERA_SOURCES.keys())}")
print("All models loaded.\n")

# start video feed server in background
threading.Thread(target=start_engine, daemon=True).start()
print(f"Video feed server started on port {config.ENGINE_PORT}")


class CameraStream:
    def __init__(self, source):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = \
            'rtsp_transport;udp|fflags;nobuffer|flags;low_delay|framedrop;1'
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        self.frame   = None
        self.ret     = False
        self.lock    = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
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


def post_detection(data):
    try:
        requests.post(config.NODE_DETECTION_URL,
                      json=data, timeout=0.5)
    except Exception:
        pass


def camera_worker(cam_id, source, result_queue):
    print(f"Connecting to {cam_id}: {source}")
    stream = CameraStream(source)
    time.sleep(1)

    ret, test = stream.read()
    if not ret or test is None:
        print(f"Cannot connect to {cam_id}")
        stream.stop()
        return

    print(f"{cam_id} connected.")

    frame_count = 0
    last_log    = time.time()
    last_result = []
    fps_time    = time.time()
    fps         = 0
    tracked     = []

    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(0.001)
            continue

        frame_count += 1
        now      = time.time()
        fps      = 0.92 * fps + 0.08 / max(now - fps_time, 0.001)
        fps_time = now

        # skip frames — just display last result
        if frame_count % config.FRAME_SKIP != 0:
            try:
                result_queue.put_nowait((cam_id, frame.copy(),
                                         last_result, fps))
            except queue.Full:
                pass
            continue

        frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect(frame_rgb)
        tracked    = trackers[cam_id].update(detections, frame_rgb)

        results = []
        for det in detections:
            rec = recognizer.recognize(det['face_crop'])
            rec['bbox']       = det['bbox']
            rec['confidence'] = det['confidence']
            rec['tracker_id'] = None
            rec['global_id']  = None
            rec['cameras']    = [cam_id]

            # match tracker id
            dx = (det['bbox'][0] + det['bbox'][2]) // 2
            dy = (det['bbox'][1] + det['bbox'][3]) // 2
            for t in tracked:
                tx1, ty1, tx2, ty2 = t['bbox']
                if tx1 <= dx <= tx2 and ty1 <= dy <= ty2:
                    rec['tracker_id'] = t['tracker_id']
                    break

            # cross-camera reid — only every 5th detection frame
            if not rec['is_known']:
                if frame_count % (config.FRAME_SKIP * 5) == 0:
                    emb = recognizer.get_embedding(det['face_crop'])
                    if emb is not None:
                        global_id, score = reid.find_or_create(
                            emb, cam_id
                        )
                        rec['global_id'] = global_id
                        rec['cameras']   = reid.get_cameras_for(global_id)
                        rec['name']      = global_id

                        # notify Node.js about cross-camera match
                        if len(rec['cameras']) > 1:
                            threading.Thread(
                                target=lambda: requests.post(
                                    f"{config.NODE_API_URL}/api/cross_camera",
                                    json={
                                        'global_id': global_id,
                                        'cameras':   rec['cameras']
                                    },
                                    timeout=0.5
                                ),
                                daemon=True
                            ).start()
                else:
                    rec['name'] = 'Unknown'

            results.append(rec)

        last_result = results

        # log to Node.js
        if results and now - last_log > 3:
            for r in results:
                threading.Thread(
                    target=post_detection,
                    args=({
                        'camera_id':    cam_id,
                        'student_id':   r.get('student_id'),
                        'student_name': r.get('name', 'Unknown'),
                        'is_known':     r.get('is_known', False),
                        'confidence':   round(r.get('confidence', 0), 4),
                        'tracker_id':   r.get('global_id') or
                                        r.get('tracker_id')
                    },),
                    daemon=True
                ).start()
            last_log = now

        try:
            result_queue.put_nowait((cam_id, frame.copy(), results, fps))
        except queue.Full:
            pass

    stream.stop()


def run_display(result_queue):
    latest          = {}
    win_size        = (640, 480)
    windows_created = set()

    while True:
        try:
            cam_id, frame, detections, fps = result_queue.get(timeout=0.1)
            latest[cam_id] = (frame, detections, fps)
        except queue.Empty:
            pass

        for cam_id, (frame, detections, fps) in latest.items():
            display = annotator.draw(frame.copy(), detections)

            # camera label
            cv2.putText(display, f"{cam_id.upper()}",
                        (display.shape[1] - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)

            # FPS
            cv2.putText(display, f"FPS: {fps:.0f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

            # cross camera alert
            for det in detections:
                if not det.get('is_known') and \
                   len(det.get('cameras', [])) > 1:
                    cv2.putText(
                        display,
                        f"CROSS-CAM: {det.get('global_id')}",
                        (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2
                    )

            # resize to fixed window
            display  = cv2.resize(display, win_size)
            win_name = f"Campus Surveillance — {cam_id}"

            if cam_id not in windows_created:
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, win_size[0], win_size[1])
                windows_created.add(cam_id)

            cv2.imshow(win_name, display)

            # push annotated frame to web dashboard
            update_feed(cam_id, display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    emb_dir   = config.EMBEDDINGS_DIR
    npy_files = [f for f in os.listdir(emb_dir)
                 if f.endswith('.npy')] if os.path.exists(emb_dir) else []

    if not npy_files:
        print("No embeddings found! Run embedding_builder.py first.")
        sys.exit(1)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    result_queue = queue.Queue(maxsize=2)

    for cam_id, source in config.CAMERA_SOURCES.items():
        t = threading.Thread(
            target=camera_worker,
            args=(cam_id, source, result_queue),
            daemon=True
        )
        t.start()
        print(f"Started thread for {cam_id}")

    time.sleep(2)
    print("\nBoth cameras running. Press Q to quit.\n")
    run_display(result_queue)
    print("System stopped.")