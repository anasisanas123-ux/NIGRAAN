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
from python_engine.face_detector      import FaceDetector
from python_engine.arcface_recognizer import ArcFaceRecognizer
from python_engine.body_detector      import BodyDetector
from python_engine.body_reid          import BodyReID
from python_engine.tracker            import Tracker
from python_engine.annotator          import Annotator
from python_engine.reid_manager       import ReIDManager
from python_engine.engine_api         import update_feed, start_engine

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False

print("Loading models...")
detector      = FaceDetector()
recognizer    = ArcFaceRecognizer()
body_detector = BodyDetector()
body_reid     = BodyReID()
annotator     = Annotator()
reid          = ReIDManager()

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
        if isinstance(source, str) and source.startswith('rtsp'):
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = \
                'rtsp_transport;udp|fflags;nobuffer|flags;low_delay|framedrop;1'
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                time.sleep(1)
                self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                time.sleep(1)
                self.cap = cv2.VideoCapture(source)

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

    connected = False
    for attempt in range(5):
        time.sleep(1)
        ret, test = stream.read()
        if ret and test is not None:
            connected = True
            break
        print(f"Retrying {cam_id} - attempt {attempt+1}/5")

    if not connected:
        print(f"Cannot connect to {cam_id} after 5 attempts")
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
    locked_ids       = {}

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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_bgr = frame

        # ArcFace detection + recognition with zone filter
        arc_results = recognizer.detect_and_recognize(
            frame_bgr, cam_id=cam_id
        )

        # YOLOv8 body detection with zone filter
        body_dets = body_detector.detect(frame_rgb, cam_id=cam_id)

        # build final detections
        detections = []

        for fd in arc_results:
            fx1, fy1, fx2, fy2 = fd['bbox']
            fcx = (fx1 + fx2) // 2
            fcy = (fy1 + fy2) // 2
            for bd in body_dets:
                bx1, by1, bx2, by2 = bd['bbox']
                if bx1 <= fcx <= bx2 and by1 <= fcy <= by2:
                    fd['body_crop'] = bd.get('body_crop')
                    fd['bbox']      = bd['bbox']
                    break
            if 'body_crop' not in fd:
                fd['body_crop'] = None
            detections.append(fd)

        # body-only detections (no face visible)
        for bd in body_dets:
            bx1, by1, bx2, by2 = bd['bbox']
            bcx = (bx1+bx2)//2
            bcy = (by1+by2)//2
            has_face = any(
                det['bbox'][0] <= bcx <= det['bbox'][2] and
                det['bbox'][1] <= bcy <= det['bbox'][3]
                for det in detections
            )
            if not has_face:
                detections.append({
                    'bbox':       bd['bbox'],
                    'confidence': bd['confidence'],
                    'face_crop':  None,
                    'body_crop':  bd.get('body_crop'),
                    'is_known':   False,
                    'student_id': None,
                    'name':       'Unknown',
                })

        tracked = trackers[cam_id].update(detections, frame_rgb)

        active_keys = {f"{cam_id}_{t['tracker_id']}" for t in tracked}
        for key in list(locked_ids.keys()):
            if key not in active_keys:
                del locked_ids[key]

        results = []
        for det in detections:

            temp_tracker = None
            dx = (det['bbox'][0] + det['bbox'][2]) // 2
            dy = (det['bbox'][1] + det['bbox'][3]) // 2
            for t in tracked:
                tx1, ty1, tx2, ty2 = t['bbox']
                if tx1 <= dx <= tx2 and ty1 <= dy <= ty2:
                    temp_tracker = t['tracker_id']
                    break

            tracker_key = f"{cam_id}_{temp_tracker}"

            # keep known identity if face temporarily not visible
            prev_lock = locked_ids.get(tracker_key, {})
            if prev_lock.get('is_known') and not det.get('is_known', False):
                rec = {
                    'is_known':   True,
                    'student_id': prev_lock['student_id'],
                    'name':       prev_lock['name'],
                    'confidence': prev_lock['confidence'],
                    'bbox':       det['bbox'],
                    'tracker_id': temp_tracker,
                    'global_id':  prev_lock['student_id'],
                    'cameras':    [cam_id]
                }
                results.append(rec)
                continue

            if tracker_key in locked_ids:
                locked = locked_ids[tracker_key]
                rec = {
                    'is_known':   locked['is_known'],
                    'student_id': locked.get('student_id'),
                    'name':       locked['name'],
                    'confidence': locked['confidence']
                }
            else:
                rec = {
                    'is_known':   det.get('is_known', False),
                    'student_id': det.get('student_id'),
                    'name':       det.get('name', 'Unknown'),
                    'confidence': det.get('confidence', 0.0)
                }
                if rec['is_known']:
                    locked_ids[tracker_key] = {
                        'is_known':   True,
                        'student_id': rec['student_id'],
                        'name':       rec['name'],
                        'confidence': rec['confidence']
                    }

            rec['bbox']       = det['bbox']
            rec['confidence'] = rec.get('confidence', det.get('confidence', 0))
            rec['tracker_id'] = temp_tracker
            rec['global_id']  = None
            rec['cameras']    = [cam_id]

            if rec['is_known']:
                stable_uid       = rec['student_id']
                rec['global_id'] = stable_uid

                cached_uid = tracker_to_uid.get(tracker_key)
                if cached_uid and cached_uid in pending_log_time:
                    pending_log_time.pop(cached_uid, None)

                if stable_uid not in logged_ids:
                    if stable_uid not in pending_log_time:
                        pending_log_time[stable_uid] = now
                    elif now - pending_log_time[stable_uid] >= 0.5:
                        logged_ids.add(stable_uid)
                        pending_log_time.pop(stable_uid, None)
                        threading.Thread(
                            target=post_detection,
                            args=({
                                'camera_id':    cam_id,
                                'student_id':   rec.get('student_id'),
                                'student_name': rec.get('name', 'Unknown'),
                                'is_known':     True,
                                'confidence':   round(
                                    rec.get('confidence', 0), 4
                                ),
                                'tracker_id':   temp_tracker
                            },),
                            daemon=True
                        ).start()

            else:
                body_crop = det.get('body_crop')
                snap_bgr  = cv2.cvtColor(body_crop, cv2.COLOR_RGB2BGR) \
                            if body_crop is not None else None
                body_emb  = body_reid.get_embedding(body_crop) \
                            if body_crop is not None else None

                if body_emb is not None:
                    global_id, score = reid.find_or_create(
                        body_emb, cam_id, frame=snap_bgr
                    )
                    rec['global_id'] = global_id
                    rec['cameras']   = reid.get_cameras_for(global_id)
                    rec['name']      = global_id

                    if temp_tracker:
                        tracker_to_uid[tracker_key] = global_id

                    if tracker_key not in locked_ids:
                        locked_ids[tracker_key] = {
                            'is_known':   False,
                            'student_id': None,
                            'name':       global_id,
                            'confidence': 0.0
                        }

                    was_known = prev_lock.get('is_known', False)

                    if not was_known and global_id not in logged_ids:
                        if global_id not in pending_log_time:
                            pending_log_time[global_id] = now
                        elif now - pending_log_time[global_id] >= 5.0:
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
                                    'tracker_id':   global_id,
                                    'snapshot_path': reid.get_snapshot_path(
                                        global_id
                                    )
                                },),
                                daemon=True
                            ).start()

                    if len(rec['cameras']) > 1:
                        threading.Thread(
                            target=lambda gid=global_id,
                            cams=rec['cameras']:
                                requests.post(
                                    f"{config.NODE_API_URL}"
                                    f"/api/cross_camera",
                                    json={
                                        'global_id': gid,
                                        'cameras':   cams
                                    },
                                    timeout=0.5
                                ),
                            daemon=True
                        ).start()

                else:
                    cached = tracker_to_uid.get(tracker_key)
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
    latest   = {}
    win_size = (640, 480)

    while True:
        try:
            cam_id, frame, detections, fps = result_queue.get(timeout=0.1)
            latest[cam_id] = (frame, detections, fps)
        except queue.Empty:
            pass

        for cam_id, (frame, detections, fps) in latest.items():
            display = annotator.draw(frame.copy(), detections)

            # draw detection zone
            if hasattr(config, 'DETECTION_ZONE'):
                z = config.DETECTION_ZONE.get(cam_id)
                if z:
                    h_d, w_d = display.shape[:2]
                    zx1 = int(z[0]*w_d)
                    zy1 = int(z[1]*h_d)
                    zx2 = int(z[2]*w_d)
                    zy2 = int(z[3]*h_d)
                    cv2.rectangle(display,
                                  (zx1, zy1), (zx2, zy2),
                                  (0, 255, 0), 2)
                    cv2.putText(display, "Detection Zone",
                                (zx1+5, zy1+20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)

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
            win_name = f"Campus Surveillance - {cam_id}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, win_size[0], win_size[1])
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

    time.sleep(3)
    print("\nBoth cameras running. Press Q to quit.\n")
    run_display(result_queue)
    print("System stopped.")