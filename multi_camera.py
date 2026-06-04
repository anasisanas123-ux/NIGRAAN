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

    frame_count      = 0
    last_result      = []
    fps_time         = time.time()
    fps              = 0
    tracked          = []
    logged_ids       = set()
    tracker_to_uid   = {}
    pending_log_time = {}

    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(0.001)
            continue

        frame_count += 1
        now      = time.time()
        fps      = 0.92 * fps + 0.08 / max(now - fps_time, 0.001)
        fps_time = now

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

            # match tracker id
            temp_tracker = None
            dx = (det['bbox'][0] + det['bbox'][2]) // 2
            dy = (det['bbox'][1] + det['bbox'][3]) // 2
            for t in tracked:
                tx1, ty1, tx2, ty2 = t['bbox']
                if tx1 <= dx <= tx2 and ty1 <= dy <= ty2:
                    temp_tracker = t['tracker_id']
                    break

            # recognize with voting
            rec = recognizer.recognize(
                det['face_crop'],
                tracker_id=f"{cam_id}_{temp_tracker}"
            )
            rec['bbox']       = det['bbox']
            rec['confidence'] = det['confidence']
            rec['tracker_id'] = temp_tracker
            rec['global_id']  = None
            rec['cameras']    = [cam_id]

            if rec['is_known']:
                # ── known person ──────────────────────────
                stable_uid = rec['student_id']
                rec['global_id'] = stable_uid

                if stable_uid not in logged_ids:
                    if stable_uid not in pending_log_time:
                        # first time seen — start timer
                        pending_log_time[stable_uid] = now
                    elif now - pending_log_time[stable_uid] >= 1.0:
                        # stable for 1 second — log once
                        logged_ids.add(stable_uid)
                        pending_log_time.pop(stable_uid, None)
                        threading.Thread(
                            target=post_detection,
                            args=({
                                'camera_id':    cam_id,
                                'student_id':   rec.get('student_id'),
                                'student_name': rec.get('name', 'Unknown'),
                                'is_known':     True,
                                'confidence':   round(rec.get('confidence', 0), 4),
                                'tracker_id':   temp_tracker
                            },),
                            daemon=True
                        ).start()

            else:
                # ── unknown person ────────────────────────
                emb = recognizer.get_embedding(det['face_crop'])

                if emb is not None:
                    global_id, score = reid.find_or_create(emb, cam_id)
                    rec['global_id'] = global_id
                    rec['cameras']   = reid.get_cameras_for(global_id)
                    rec['name']      = global_id

                    if temp_tracker:
                        tracker_to_uid[f"{cam_id}_{temp_tracker}"] = global_id

                    if global_id not in logged_ids:
                        if global_id not in pending_log_time:
                            # first time seen — start timer
                            pending_log_time[global_id] = now
                        elif now - pending_log_time[global_id] >= 2.0:
                            # stable for 2 seconds — log once
                            logged_ids.add(global_id)
                            pending_log_time.pop(global_id, None)
                            threading.Thread(
                                target=post_detection,
                                args=({
                                    'camera_id':    cam_id,
                                    'student_id':   None,
                                    'student_name': global_id,
                                    'is_known':     False,
                                    'confidence':   0.0,
                                    'tracker_id':   global_id
                                },),
                                daemon=True
                            ).start()

                    # cross-camera alert
                    if len(rec['cameras']) > 1:
                        threading.Thread(
                            target=lambda gid=global_id, cams=rec['cameras']:
                                requests.post(
                                    f"{config.NODE_API_URL}/api/cross_camera",
                                    json={
                                        'global_id': gid,
                                        'cameras':   cams
                                    },
                                    timeout=0.5
                                ),
                            daemon=True
                        ).start()

                else:
                    cached = tracker_to_uid.get(f"{cam_id}_{temp_tracker}")
                    if cached:
                        rec['global_id'] = cached
                        rec['name']      = cached
                        rec['cameras']   = reid.get_cameras_for(cached)
                    else:
                        rec['name'] = 'Unknown'

            results.append(rec)

        last_result = results

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

            cv2.putText(display, f"{cam_id.upper()}",
                        (display.shape[1] - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)

            cv2.putText(display, f"FPS: {fps:.0f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

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

            display  = cv2.resize(display, win_size)
            win_name = f"Campus Surveillance — {cam_id}"

            if cam_id not in windows_created:
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, win_size[0], win_size[1])
                windows_created.add(cam_id)

            cv2.imshow(win_name, display)
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