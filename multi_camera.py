import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import time
import threading
import queue
import requests
import torch
import numpy as np
import config
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
        self._source = source
        if isinstance(source, str) and source.startswith('rtsp'):
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = \
                'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay' \
                '|framedrop;1|max_delay;0|reorder_queue_size;0'
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                time.sleep(1)
                self.cap = cv2.VideoCapture(source)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          25)
        self.frame           = None
        self.ret             = False
        self.lock            = threading.Lock()
        self.running         = True
        self.last_frame_time = time.time()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        fails = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                fails += 1
                if fails > 30:
                    print(f"Reconnecting {self._source[:40]}...")
                    self.cap.release()
                    time.sleep(2)
                    if isinstance(self._source, str) and \
                       self._source.startswith('rtsp'):
                        self.cap = cv2.VideoCapture(
                            self._source, cv2.CAP_FFMPEG
                        )
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    fails = 0
                time.sleep(0.01)
                continue
            fails = 0
            with self.lock:
                self.ret             = ret
                self.frame           = frame
                self.last_frame_time = time.time()

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            if time.time() - self.last_frame_time > 3.0:
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


def emb_valid(emb):
    return (emb is not None and
            isinstance(emb, np.ndarray) and
            emb.size > 0)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1   = max(ax1, bx1)
    iy1   = max(ay1, by1)
    ix2   = min(ax2, bx2)
    iy2   = min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / max(union, 1)


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
        print(f"Retrying {cam_id} attempt {attempt+1}/5")

    if not connected:
        print(f"Cannot connect to {cam_id}")
        stream.stop()
        return

    print(f"{cam_id} connected.")

    frame_count          = 0
    last_result          = []
    fps_time             = time.time()
    fps                  = 0
    logged_ids           = set()
    pending_unknown_time = {}
    cross_cam_fired      = set()
    locked_ids           = {}
    LOCK_EXPIRE          = 15.0

    # identity cache — tracker_key -> identity dict
    # once a tracker is confirmed known we skip ArcFace for them
    # and use cached identity directly until they leave frame
    confirmed_ids = {}

    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        now      = time.time()
        fps      = 0.92 * fps + 0.08 / max(now - fps_time, 0.001)
        fps_time = now

        # expire old locks and confirmed identities
        expired = [k for k, v in locked_ids.items()
                   if now - v.get('time', 0) > LOCK_EXPIRE]
        for k in expired:
            del locked_ids[k]
            confirmed_ids.pop(k, None)

        if frame_count % config.FRAME_SKIP != 0:
            try:
                result_queue.put_nowait(
                    (cam_id, frame.copy(), last_result, fps)
                )
            except queue.Full:
                pass
            continue

        frame_bgr = frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # body detection always runs — lightweight, needed for tracking
        body_dets = body_detector.detect(frame_rgb, cam_id=cam_id)

        # get current tracker IDs from DeepSORT using body detections
        # we need tracker IDs before deciding who needs ArcFace
        # build temp detections for tracking only
        temp_dets = []
        for bd in body_dets:
            temp_dets.append({
                'face_bbox':  None,
                'body_bbox':  bd['bbox'],
                'bbox':       bd['bbox'],
                'is_known':   False,
                'student_id': None,
                'name':       'Unknown',
                'confidence': 0.0,
                'face_crop':  None,
                'body_crop':  bd.get('body_crop'),
            })

        tracked = trackers[cam_id].update(temp_dets, frame_rgb)

        def get_tracker_id(bbox):
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            best_tid  = None
            best_dist = 9999
            for t in tracked:
                tx1, ty1, tx2, ty2 = t['bbox']
                dist = abs(cx-(tx1+tx2)//2) + abs(cy-(ty1+ty2)//2)
                if dist < best_dist:
                    best_dist = dist
                    best_tid  = t['tracker_id']
            return best_tid if best_dist < 200 else None

        # determine which body regions need ArcFace
        # skip ArcFace for trackers already confirmed as known
        need_arcface_regions = []
        confirmed_results    = []

        for bd in body_dets:
            tid         = get_tracker_id(bd['bbox'])
            tracker_key = f"{cam_id}_{tid}"

            if tracker_key in confirmed_ids:
                # already confirmed — use cached identity, skip ArcFace
                cached = confirmed_ids[tracker_key]
                locked_ids[tracker_key]['time'] = now
                confirmed_results.append({
                    'face_bbox':  None,
                    'body_bbox':  bd['bbox'],
                    'bbox':       bd['bbox'],
                    'is_known':   True,
                    'student_id': cached['student_id'],
                    'name':       cached['name'],
                    'confidence': cached['confidence'],
                    'tracker_id': tid,
                    'global_id':  cached['student_id'],
                    'cameras':    [cam_id],
                    '_cached':    True,
                })
            else:
                need_arcface_regions.append((tid, tracker_key, bd))

        # run ArcFace only on faces that are NOT yet confirmed
        # build a mask image containing only unconfirmed body regions
        arc_results = []
        if need_arcface_regions:
            arc_results = recognizer.detect_and_recognize(
                frame_bgr, cam_id=cam_id
            )

        # match ArcFace results to unconfirmed body regions
        detections  = list(confirmed_results)
        used_faces  = set()
        used_bodies = set()

        for i, (tid, tracker_key, bd) in enumerate(need_arcface_regions):
            bx1, by1, bx2, by2 = bd['bbox']
            bcx = (bx1+bx2)//2
            bcy = (by1+by2)//2

            matched_face = None
            for fi, fd in enumerate(arc_results):
                if fi in used_faces:
                    continue
                fx1, fy1, fx2, fy2 = fd['bbox']
                fcx = (fx1+fx2)//2
                fcy = (fy1+fy2)//2
                in_body = (bx1 <= fcx <= bx2 and
                           by1 <= fcy <= by1 + (by2-by1)*0.6)
                if in_body:
                    matched_face = (fi, fd)
                    used_faces.add(fi)
                    break

            if matched_face is not None:
                fi, fd = matched_face
                detections.append({
                    'face_bbox':  fd['bbox'],
                    'body_bbox':  bd['bbox'],
                    'bbox':       bd['bbox'],
                    'is_known':   fd['is_known'],
                    'student_id': fd['student_id'],
                    'name':       fd['name'],
                    'confidence': fd['confidence'],
                    'face_crop':  fd.get('face_crop'),
                    'body_crop':  bd.get('body_crop'),
                    'tracker_id': tid,
                    '_cached':    False,
                })
            else:
                # body visible but no face matched
                prev = locked_ids.get(tracker_key, {})
                detections.append({
                    'face_bbox':  None,
                    'body_bbox':  bd['bbox'],
                    'bbox':       bd['bbox'],
                    'is_known':   prev.get('is_known', False),
                    'student_id': prev.get('student_id'),
                    'name':       prev.get('name', 'Unknown'),
                    'confidence': prev.get('confidence', 0.0),
                    'face_crop':  None,
                    'body_crop':  bd.get('body_crop'),
                    'tracker_id': tid,
                    '_cached':    False,
                })

        # process results and handle logging
        results = []

        for det in detections:
            tid         = det.get('tracker_id') or \
                          get_tracker_id(det['bbox'])
            tracker_key = f"{cam_id}_{tid}"
            prev        = locked_ids.get(tracker_key, {})

            if det.get('_cached'):
                # already confirmed known — just append
                results.append(det)
                continue

            if det['is_known']:
                # new confirmation — add to confirmed cache
                pending_unknown_time.pop(tracker_key, None)
                pending_unknown_time.pop(
                    det.get('student_id', ''), None
                )

                confirmed_ids[tracker_key] = {
                    'student_id': det['student_id'],
                    'name':       det['name'],
                    'confidence': det['confidence'],
                }
                locked_ids[tracker_key] = {
                    'is_known':   True,
                    'student_id': det['student_id'],
                    'name':       det['name'],
                    'confidence': det['confidence'],
                    'time':       now,
                }

                rec = {
                    'face_bbox':  det['face_bbox'],
                    'body_bbox':  det['body_bbox'],
                    'bbox':       det['bbox'],
                    'is_known':   True,
                    'student_id': det['student_id'],
                    'name':       det['name'],
                    'confidence': det['confidence'],
                    'tracker_id': tid,
                    'global_id':  det['student_id'],
                    'cameras':    [cam_id],
                }

            elif prev.get('is_known'):
                # was known before — keep lock
                locked_ids[tracker_key]['time'] = now
                rec = {
                    'face_bbox':  det['face_bbox'],
                    'body_bbox':  det['body_bbox'],
                    'bbox':       det['bbox'],
                    'is_known':   True,
                    'student_id': prev['student_id'],
                    'name':       prev['name'],
                    'confidence': prev['confidence'],
                    'tracker_id': tid,
                    'global_id':  prev['student_id'],
                    'cameras':    [cam_id],
                }

            else:
                # unknown — run body ReID
                body_crop = det.get('body_crop')
                body_emb  = None

                if body_crop is not None:
                    try:
                        body_emb = body_reid.get_embedding(body_crop)
                    except Exception:
                        body_emb = None

                global_id    = None
                reid_cameras = [cam_id]

                if emb_valid(body_emb):
                    try:
                        snap = None
                        if body_crop is not None:
                            snap = cv2.cvtColor(
                                body_crop, cv2.COLOR_RGB2BGR
                            )
                        global_id, _ = reid.find_or_create(
                            body_emb, cam_id, frame=snap
                        )
                        cams = reid.get_cameras_for(global_id)
                        reid_cameras = list(cams) if cams else [cam_id]
                    except Exception as e:
                        print(f"ReID error: {e}")
                        global_id = None

                if global_id is None:
                    global_id = prev.get('name') or \
                                f"UNK-{abs(hash(tracker_key))%900+100}"

                locked_ids[tracker_key] = {
                    'is_known':   False,
                    'student_id': None,
                    'name':       global_id,
                    'confidence': 0.0,
                    'time':       now,
                }

                rec = {
                    'face_bbox':  det['face_bbox'],
                    'body_bbox':  det['body_bbox'],
                    'bbox':       det['bbox'],
                    'is_known':   False,
                    'student_id': None,
                    'name':       global_id,
                    'confidence': 0.0,
                    'tracker_id': tid,
                    'global_id':  global_id,
                    'cameras':    reid_cameras,
                }

                if len(reid_cameras) > 1:
                    cross_key = global_id + '_' + \
                                '_'.join(sorted(reid_cameras))
                    if cross_key not in cross_cam_fired:
                        cross_cam_fired.add(cross_key)
                        threading.Thread(
                            target=lambda g=global_id,
                            c=reid_cameras:
                                requests.post(
                                    f"{config.NODE_API_URL}"
                                    f"/api/cross_camera",
                                    json={'global_id': g,
                                          'cameras':   c},
                                    timeout=0.5
                                ),
                            daemon=True
                        ).start()

            # ── LOGGING ──────────────────────────────────────────
            log_uid  = rec.get('student_id') or rec.get('global_id')
            has_face = det.get('face_bbox') is not None

            if log_uid and log_uid not in logged_ids and has_face:

                if rec['is_known']:
                    logged_ids.add(log_uid)
                    threading.Thread(
                        target=post_detection,
                        args=({
                            'camera_id':     cam_id,
                            'student_id':    rec.get('student_id'),
                            'student_name':  rec.get('name', 'Unknown'),
                            'is_known':      True,
                            'confidence':    round(
                                rec.get('confidence', 0), 4
                            ),
                            'tracker_id':    rec.get('global_id'),
                            'snapshot_path': None
                        },),
                        daemon=True
                    ).start()

                else:
                    if log_uid not in pending_unknown_time:
                        pending_unknown_time[log_uid] = now
                    elif now - pending_unknown_time[log_uid] >= 8.0:
                        logged_ids.add(log_uid)
                        pending_unknown_time.pop(log_uid, None)
                        snap_path = None
                        try:
                            snap_path = reid.get_snapshot_path(log_uid)
                        except Exception:
                            pass
                        threading.Thread(
                            target=post_detection,
                            args=({
                                'camera_id':     cam_id,
                                'student_id':    None,
                                'student_name':  rec.get('name', 'Unknown'),
                                'is_known':      False,
                                'confidence':    0.0,
                                'tracker_id':    log_uid,
                                'snapshot_path': snap_path
                            },),
                            daemon=True
                        ).start()

            results.append(rec)

        last_result = results

        try:
            result_queue.put_nowait(
                (cam_id, frame.copy(), results, fps)
            )
        except queue.Full:
            pass

    stream.stop()


def run_display(result_queue):
    latest   = {}
    win_size = (640, 480)

    while True:
        try:
            cam_id, frame, detections, fps = \
                result_queue.get(timeout=0.1)
            latest[cam_id] = (frame, detections, fps)
        except queue.Empty:
            pass

        for cam_id, (frame, detections, fps) in latest.items():
            display = annotator.draw(frame.copy(), detections)

            if hasattr(config, 'DETECTION_ZONE'):
                z = config.DETECTION_ZONE.get(cam_id)
                if z:
                    h_d, w_d = display.shape[:2]
                    cv2.rectangle(
                        display,
                        (int(z[0]*w_d), int(z[1]*h_d)),
                        (int(z[2]*w_d), int(z[3]*h_d)),
                        (0, 255, 0), 1
                    )

            cv2.putText(
                display,
                f"{cam_id.upper()}  FPS:{fps:.0f}  "
                f"Det:{len(detections)}  "
                f"Cached:{sum(1 for d in detections if d.get('_cached'))}",
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2
            )

            display  = cv2.resize(display, win_size)
            win_name = f"Campus Surveillance - {cam_id}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, *win_size)
            cv2.imshow(win_name, display)
            update_feed(cam_id, display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    emb_dir   = config.EMBEDDINGS_DIR
    npy_files = [f for f in os.listdir(emb_dir)
                 if f.endswith('.npy')] \
                if os.path.exists(emb_dir) else []

    if not npy_files:
        print("No embeddings found. Run arcface_builder.py first.")
        sys.exit(1)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    result_queue = queue.Queue(maxsize=2)

    for cam_id, source in config.CAMERA_SOURCES.items():
        threading.Thread(
            target=camera_worker,
            args=(cam_id, source, result_queue),
            daemon=True
        ).start()
        print(f"Started thread for {cam_id}")

    time.sleep(3)
    print("\nBoth cameras running. Press Q to quit.\n")
    run_display(result_queue)
    print("System stopped.")