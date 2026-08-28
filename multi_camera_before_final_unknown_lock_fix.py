import sys
import os

# ============================================================
# CUDA / cuDNN DLL FIX FOR WINDOWS
# Must execute before importing ONNX Runtime / InsightFace
# ============================================================

if sys.platform == 'win32':
    import ctypes

    ctypes.windll.kernel32.SetDllDirectoryW(
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin'
    )


sys.path.insert(
    0,
    os.path.abspath(
        os.path.dirname(__file__)
    )
)


# ============================================================
# IMPORTS
# ============================================================

import cv2
import time
import threading
import queue
import requests
import torch
import numpy as np

import config

from python_engine.arcface_recognizer import ArcFaceRecognizer
from python_engine.body_detector import BodyDetector
from python_engine.body_reid import BodyReID
from python_engine.annotator import Annotator
from python_engine.reid_manager import ReIDManager
from python_engine.anonymous_face_reid import AnonymousFaceReID
from python_engine.engine_api import update_feed, start_engine


# ============================================================
# CUDA SETTINGS
# ============================================================

if torch.cuda.is_available():

    torch.backends.cudnn.benchmark = True

    torch.backends.cudnn.deterministic = False


# ============================================================
# GLOBAL LOGGING STATE
# ============================================================

global_logged_students = set()

global_logged_unknowns = set()

log_lock = threading.Lock()


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")


recognizer = ArcFaceRecognizer()

body_detector = BodyDetector()

body_reid = BodyReID()

annotator = Annotator()

reid = ReIDManager()
anonymous_face_reid = AnonymousFaceReID()

# Shared across ALL camera threads
cross_cam_fired = set()
cross_cam_lock = threading.Lock()

# YOLO may use a low threshold for tracking, but a weak
# body-only detection must not create a global UNK identity.
NEW_UNK_MIN_BODY_CONF = 0.50





print(
    f"Device  : "
    f"{'CUDA GPU' if torch.cuda.is_available() else 'CPU'}"
)

print(
    f"Students: "
    f"{len(recognizer.known_embeddings)}"
)

print(
    f"Cameras : "
    f"{list(config.CAMERA_SOURCES.keys())}"
)

print(
    "All models loaded.\n"
)


# ============================================================
# START PYTHON VIDEO API
# ============================================================

threading.Thread(
    target=start_engine,
    daemon=True
).start()


print(
    f"Video feed server started "
    f"on port {config.ENGINE_PORT}"
)


# ============================================================
# CAMERA STREAM
# ============================================================

class CameraStream:

    def __init__(self, source):

        self._source = source


        # ----------------------------------------------------
        # RTSP CAMERA
        # ----------------------------------------------------

        if (
            isinstance(source, str)
            and
            source.startswith('rtsp')
        ):

            os.environ[
                'OPENCV_FFMPEG_CAPTURE_OPTIONS'
            ] = (
                'rtsp_transport;tcp'
                '|fflags;nobuffer'
                '|flags;low_delay'
                '|framedrop;1'
                '|max_delay;0'
                '|reorder_queue_size;0'
            )


            self.cap = cv2.VideoCapture(
                source,
                cv2.CAP_FFMPEG
            )


        # ----------------------------------------------------
        # USB / WEBCAM
        # ----------------------------------------------------

        else:

            self.cap = cv2.VideoCapture(
                source,
                cv2.CAP_DSHOW
            )


            if not self.cap.isOpened():

                time.sleep(1)

                self.cap = cv2.VideoCapture(
                    source
                )


        # ----------------------------------------------------
        # CAMERA SETTINGS
        # ----------------------------------------------------

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            config.FRAME_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            config.FRAME_HEIGHT
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            25
        )


        self.frame = None

        self.ret = False

        self.lock = threading.Lock()

        self.running = True

        self.last_frame_time = time.time()


        threading.Thread(
            target=self._reader,
            daemon=True
        ).start()


    # ========================================================
    # CAMERA READER THREAD
    # ========================================================

    def _reader(self):

        fails = 0


        while self.running:

            ret, frame = self.cap.read()


            if not ret:

                fails += 1


                if fails > 30:

                    print(
                        f"Reconnecting "
                        f"{str(self._source)[:40]}..."
                    )


                    self.cap.release()

                    time.sleep(2)


                    if (
                        isinstance(self._source, str)
                        and
                        self._source.startswith('rtsp')
                    ):

                        self.cap = cv2.VideoCapture(
                            self._source,
                            cv2.CAP_FFMPEG
                        )

                        self.cap.set(
                            cv2.CAP_PROP_BUFFERSIZE,
                            1
                        )


                    fails = 0


                time.sleep(0.01)

                continue


            fails = 0


            with self.lock:

                self.ret = ret

                self.frame = frame

                self.last_frame_time = (
                    time.time()
                )


    # ========================================================
    # READ LATEST FRAME
    # ========================================================

    def read(self):

        with self.lock:

            if self.frame is None:

                return False, None


            if (
                time.time()
                -
                self.last_frame_time
                >
                3.0
            ):

                return False, None


            return (
                self.ret,
                self.frame.copy()
            )


    # ========================================================
    # STOP STREAM
    # ========================================================

    def stop(self):

        self.running = False

        self.cap.release()


# ============================================================
# SEND DETECTION TO NODE.JS
# ============================================================

def post_detection(data):

    try:

        response = requests.post(

            config.NODE_DETECTION_URL,

            json=data,

            timeout=1.0
        )


        if response.status_code not in (
            200,
            201
        ):

            print(
                "[LOG ERROR] "
                f"Node returned "
                f"{response.status_code}: "
                f"{response.text[:200]}"
            )


    except Exception as e:

        print(
            f"[LOG ERROR] "
            f"{e}"
        )


# ============================================================
# EMBEDDING VALIDATION
# ============================================================

def emb_valid(emb):

    return (

        emb is not None

        and

        isinstance(
            emb,
            np.ndarray
        )

        and

        emb.size > 0
    )


# ============================================================
# IOU
# ============================================================

def iou(a, b):

    ax1, ay1, ax2, ay2 = a

    bx1, by1, bx2, by2 = b


    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )


    inter = (

        max(
            0,
            ix2 - ix1
        )

        *

        max(
            0,
            iy2 - iy1
        )
    )


    if inter == 0:

        return 0.0


    union = (

        (ax2 - ax1)
        *
        (ay2 - ay1)

        +

        (bx2 - bx1)
        *
        (by2 - by1)

        -

        inter
    )


    return (
        inter /
        max(
            union,
            1
        )
    )


# ============================================================
# CAMERA WORKER
# ============================================================

def camera_worker(
    cam_id,
    source,
    result_queue
):

    print(
        f"Connecting to "
        f"{cam_id}: {source}"
    )


    stream = CameraStream(
        source
    )


    # ========================================================
    # WAIT FOR CAMERA CONNECTION
    # ========================================================

    connected = False


    for attempt in range(5):

        time.sleep(1)


        ret, test = stream.read()


        if (
            ret
            and
            test is not None
        ):

            connected = True

            break


        print(
            f"Retrying "
            f"{cam_id} "
            f"attempt "
            f"{attempt + 1}/5"
        )


    if not connected:

        print(
            f"Cannot connect to "
            f"{cam_id}"
        )

        stream.stop()

        return


    print(
        f"{cam_id} connected."
    )


    # ========================================================
    # CAMERA STATE
    # ========================================================

    frame_count = 0

    last_result = []

    fps_time = time.time()

    fps = 0


    # --------------------------------------------------------
    # Logging state
    # --------------------------------------------------------

    pending_unknown_time = {}


    # --------------------------------------------------------
    # Identity state
    # --------------------------------------------------------

    locked_ids = {}

    LOCK_EXPIRE = 15.0


    # Known identities cached by BoT-SORT track
    confirmed_ids = {}


    # Last known bounding boxes
    last_known_boxes = {}

    # Last detected face position for each BoT-SORT track
    last_face_boxes = {}

    # Keep face box briefly when SCRFD misses during movement
    FACE_BOX_TTL = 0.8


    # Bounding box smoothing
    smoothed_boxes = {}


    SMOOTH_ALPHA = 0.55


    # --------------------------------------------------------
    # Confirmed unknown IDs
    #
    # tracker_key ->
    # {
    #     global_id,
    #     cameras,
    #     score,
    #     time
    # }
    # --------------------------------------------------------

    confirmed_unknown_ids = {}


    UNKNOWN_RECHECK = 0.8


    # ========================================================
    # CAMERA-LOCAL UNKNOWN IDENTITY LOCK
    #
    # Purpose:
    #
    # BoT-SORT local ID may change while the same physical
    # person is still inside this camera.
    #
    # Example:
    #
    # cam_b_13 -> UNK-001
    # cam_b_16 -> ?
    #
    # If the new track strongly overlaps the last position of
    # UNK-001, temporarily inherit UNK-001.
    #
    # IMPORTANT:
    # This dictionary lives inside camera_worker(), therefore
    # each camera has its OWN lock state.
    # ========================================================

    CAM_ID_LOCK_TTL = 0.0

    # Body-only inheritance must still be reasonably strong.
    # A visible face does NOT use this shortcut; ArcFace gets
    # priority instead.
    CAM_ID_LOCK_MIN_BODY_CONF = 2.0

    CAM_ID_LOCK_MIN_IOU = 0.25
    CAM_ID_LOCK_MAX_CENTER = 0.30
    CAM_ID_LOCK_MIN_AREA_RATIO = 0.50
    CAM_ID_LOCK_MIN_CONTAINMENT = 0.55

    # If two existing people are nearly equally plausible,
    # don't guess.
    CAM_ID_LOCK_AMBIGUITY_MARGIN = 0.10

    # global_id -> {
    #     bbox,
    #     tracker_key,
    #     time
    # }
    camera_identity_locks = {}


    def prune_camera_identity_locks(
        current_time
    ):

        expired = []

        for gid, info in list(
            camera_identity_locks.items()
        ):

            age = (
                current_time
                -
                float(
                    info.get(
                        'time',
                        0.0
                    )
                )
            )

            if age > CAM_ID_LOCK_TTL:

                expired.append(
                    gid
                )


        for gid in expired:

            camera_identity_locks.pop(
                gid,
                None
            )


    def camera_lock_metrics(
        box_a,
        box_b
    ):

        if (
            box_a is None
            or
            box_b is None
        ):

            return None


        try:

            ax1, ay1, ax2, ay2 = [
                float(v)
                for v in box_a
            ]

            bx1, by1, bx2, by2 = [
                float(v)
                for v in box_b
            ]


            aw = max(
                1.0,
                ax2 - ax1
            )

            ah = max(
                1.0,
                ay2 - ay1
            )

            bw = max(
                1.0,
                bx2 - bx1
            )

            bh = max(
                1.0,
                by2 - by1
            )


            area_a = (
                aw
                *
                ah
            )

            area_b = (
                bw
                *
                bh
            )


            ix1 = max(
                ax1,
                bx1
            )

            iy1 = max(
                ay1,
                by1
            )

            ix2 = min(
                ax2,
                bx2
            )

            iy2 = min(
                ay2,
                by2
            )


            iw = max(
                0.0,
                ix2 - ix1
            )

            ih = max(
                0.0,
                iy2 - iy1
            )


            intersection = (
                iw
                *
                ih
            )


            union = (
                area_a
                +
                area_b
                -
                intersection
            )


            if union > 0:

                iou = (
                    intersection
                    /
                    union
                )

            else:

                iou = 0.0


            smaller_area = min(
                area_a,
                area_b
            )

            larger_area = max(
                area_a,
                area_b
            )


            if smaller_area > 0:

                containment = (
                    intersection
                    /
                    smaller_area
                )

            else:

                containment = 0.0


            if larger_area > 0:

                area_ratio = (
                    smaller_area
                    /
                    larger_area
                )

            else:

                area_ratio = 0.0


            acx = (
                ax1 + ax2
            ) / 2.0

            acy = (
                ay1 + ay2
            ) / 2.0

            bcx = (
                bx1 + bx2
            ) / 2.0

            bcy = (
                by1 + by2
            ) / 2.0


            center_distance = (

                (
                    acx - bcx
                ) ** 2

                +

                (
                    acy - bcy
                ) ** 2

            ) ** 0.5


            diag_a = (
                aw ** 2
                +
                ah ** 2
            ) ** 0.5

            diag_b = (
                bw ** 2
                +
                bh ** 2
            ) ** 0.5


            normalization = max(

                1.0,

                (
                    diag_a
                    +
                    diag_b
                ) / 2.0
            )


            normalized_center = (
                center_distance
                /
                normalization
            )


            center_quality = max(

                0.0,

                1.0
                -
                min(
                    normalized_center,
                    1.0
                )
            )


            geometry_score = (

                0.40
                *
                iou

                +

                0.25
                *
                containment

                +

                0.20
                *
                area_ratio

                +

                0.15
                *
                center_quality
            )


            return {

                'iou':
                    float(
                        iou
                    ),

                'containment':
                    float(
                        containment
                    ),

                'area_ratio':
                    float(
                        area_ratio
                    ),

                'center':
                    float(
                        normalized_center
                    ),

                'score':
                    float(
                        geometry_score
                    )
            }


        except Exception:

            return None


    def match_camera_identity_lock(
        tracker_key,
        current_bbox,
        current_time
    ):

        prune_camera_identity_locks(
            current_time
        )


        candidates = []


        for (
            global_id,
            info
        ) in camera_identity_locks.items():

            # The lock exists to bridge a LOCAL TRACK CHANGE.
            # If BoT-SORT ID did not change, normal cached ReID
            # already handles the person.
            if (
                str(
                    info.get(
                        'tracker_key'
                    )
                )
                ==
                str(
                    tracker_key
                )
            ):

                continue


            previous_bbox = info.get(
                'bbox'
            )


            metrics = camera_lock_metrics(

                current_bbox,

                previous_bbox
            )


            if metrics is None:

                continue


            # Prevent a tiny false detection inside a real
            # person's box from inheriting the person's ID.
            if (
                metrics[
                    'area_ratio'
                ]
                <
                CAM_ID_LOCK_MIN_AREA_RATIO
            ):

                continue


            geometry_ok = (

                metrics[
                    'iou'
                ]
                >=
                CAM_ID_LOCK_MIN_IOU

                or

                (
                    metrics[
                        'center'
                    ]
                    <=
                    CAM_ID_LOCK_MAX_CENTER

                    and

                    metrics[
                        'containment'
                    ]
                    >=
                    CAM_ID_LOCK_MIN_CONTAINMENT
                )
            )


            if not geometry_ok:

                continue


            candidates.append(
                (
                    global_id,
                    metrics
                )
            )


        if not candidates:

            return (
                None,
                None
            )


        candidates.sort(

            key=lambda item:
                item[
                    1
                ][
                    'score'
                ],

            reverse=True
        )


        best_gid = candidates[
            0
        ][
            0
        ]

        best_metrics = candidates[
            0
        ][
            1
        ]


        # Multiple people may be standing close together.
        # Don't guess if two locks are almost equally good.
        if len(
            candidates
        ) > 1:

            second_metrics = candidates[
                1
            ][
                1
            ]


            difference = (

                best_metrics[
                    'score'
                ]

                -

                second_metrics[
                    'score'
                ]
            )


            if (
                difference
                <
                CAM_ID_LOCK_AMBIGUITY_MARGIN
            ):

                print(
                    f"[CAM LOCK AMBIGUOUS] "
                    f"{cam_id} "
                    f"{tracker_key} "
                    f"best={best_gid}:"
                    f"{best_metrics['score']:.3f} "
                    f"margin={difference:.3f}"
                )

                return (
                    None,
                    None
                )


        return (
            best_gid,
            best_metrics
        )


    def refresh_camera_identity_lock(
        global_id,
        bbox,
        tracker_key,
        current_time
    ):

        if (
            global_id is None
            or
            bbox is None
        ):

            return


        global_id = str(
            global_id
        )


        # This mechanism is only for anonymous global IDs.
        if not global_id.startswith(
            'UNK-'
        ):

            return


        prune_camera_identity_locks(
            current_time
        )


        camera_identity_locks[
            global_id
        ] = {

            'bbox':
                list(
                    bbox
                ),

            'tracker_key':
                str(
                    tracker_key
                ),

            'time':
                float(
                    current_time
                )
        }


    def get_cached_face_bbox(
        tracker_key,
        current_body_bbox,
        current_time
    ):
        cached = last_face_boxes.get(
            tracker_key
        )

        if cached is None:
            return None

        if (
            current_time
            -
            cached.get('time', 0)
            >
            FACE_BOX_TTL
        ):
            return None

        old_face = cached.get(
            'face_bbox'
        )

        old_body = cached.get(
            'body_bbox'
        )

        if (
            old_face is None
            or
            old_body is None
            or
            current_body_bbox is None
        ):
            return None

        obx1, oby1, obx2, oby2 = [
            float(v)
            for v in old_body
        ]

        fx1, fy1, fx2, fy2 = [
            float(v)
            for v in old_face
        ]

        nbx1, nby1, nbx2, nby2 = [
            float(v)
            for v in current_body_bbox
        ]

        old_w = max(
            obx2 - obx1,
            1.0
        )

        old_h = max(
            oby2 - oby1,
            1.0
        )

        new_w = max(
            nbx2 - nbx1,
            1.0
        )

        new_h = max(
            nby2 - nby1,
            1.0
        )

        # Face coordinates relative to old body box
        rx1 = (fx1 - obx1) / old_w
        ry1 = (fy1 - oby1) / old_h
        rx2 = (fx2 - obx1) / old_w
        ry2 = (fy2 - oby1) / old_h

        # Apply them to the current BoT-SORT body box
        new_fx1 = int(
            nbx1 + rx1 * new_w
        )

        new_fy1 = int(
            nby1 + ry1 * new_h
        )

        new_fx2 = int(
            nbx1 + rx2 * new_w
        )

        new_fy2 = int(
            nby1 + ry2 * new_h
        )

        if (
            new_fx2 <= new_fx1
            or
            new_fy2 <= new_fy1
        ):
            return None

        return [
            new_fx1,
            new_fy1,
            new_fx2,
            new_fy2
        ]



    # ========================================================
    # MAIN CAMERA LOOP
    # ========================================================

    while True:

        ret, frame = stream.read()


        if (
            not ret
            or
            frame is None
        ):

            time.sleep(0.01)

            continue


        frame_count += 1


        now = time.time()


        fps = (

            0.92 * fps

            +

            0.08 /
            max(
                now - fps_time,
                0.001
            )
        )


        fps_time = now


        # ====================================================
        # EXPIRE OLD TRACK LOCKS
        # ====================================================

        expired = [

            key

            for key, value
            in locked_ids.items()

            if (
                now
                -
                value.get(
                    'time',
                    0
                )
                >
                LOCK_EXPIRE
            )
        ]


        for key in expired:

            last_face_boxes.pop(key, None)

            locked_ids.pop(
                key,
                None
            )

            confirmed_ids.pop(
                key,
                None
            )

            confirmed_unknown_ids.pop(
                key,
                None
            )

            last_known_boxes.pop(
                key,
                None
            )

            smoothed_boxes.pop(
                key,
                None
            )


        # ====================================================
        # FRAME SKIPPING
        # ====================================================

        if (
            frame_count
            %
            config.FRAME_SKIP
            !=
            0
        ):

            try:

                result_queue.put_nowait(

                    (
                        cam_id,
                        frame.copy(),
                        last_result,
                        fps
                    )
                )

            except queue.Full:

                pass


            continue


        frame_bgr = frame


        frame_rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # PERSON DETECTION
        # ====================================================

        body_dets = body_detector.detect(

            frame_rgb,

            cam_id=cam_id
        )


        # ====================================================
        # SHORT DETECTION FLICKER FALLBACK
        # ====================================================

        if not body_dets:

            for (
                tracker_key,
                info
            ) in list(
                confirmed_ids.items()
            ):

                if not tracker_key.startswith(
                    cam_id + "_"
                ):

                    continue


                last = last_known_boxes.get(
                    tracker_key
                )


                if (
                    last
                    and
                    (
                        now -
                        last['time']
                    )
                    <
                    0.6
                ):

                    body_dets.append({

                        'bbox':
                            last['bbox'],

                        'confidence':
                            0.0,

                        'body_crop':
                            None,

                        'face_crop':
                            None,

                        'tracker_id':
                            tracker_key.rsplit('_', 1)[-1]
                    })


        # ====================================================
        # BOT-SORT TRACK IDS COME DIRECTLY FROM BODY DETECTOR
        # ====================================================

        # ====================================================
        # SPLIT CONFIRMED / UNCONFIRMED PEOPLE
        # ====================================================

        need_arcface_regions = []

        confirmed_results = []


        for bd in body_dets:

            tid = bd.get('tracker_id')


            if tid is None:

                continue


            tracker_key = (
                f"{cam_id}_{tid}"
            )


            if (
                bd.get(
                    'confidence',
                    0
                )
                >
                0
            ):

                last_known_boxes[
                    tracker_key
                ] = {

                    'bbox':
                        bd['bbox'],

                    'time':
                        now
                }


            # ------------------------------------------------
            # Already confirmed known student
            # ------------------------------------------------

            if (
                tracker_key
                in
                confirmed_ids
            ):

                cached = confirmed_ids[
                    tracker_key
                ]


                if tracker_key in locked_ids:

                    locked_ids[
                        tracker_key
                    ]['time'] = now


                confirmed_results.append({

                    'face_bbox':
                        get_cached_face_bbox(tracker_key, bd['bbox'], now),

                    'body_bbox':
                        bd['bbox'],

                    'bbox':
                        bd['bbox'],

                    'is_known':
                        True,

                    'student_id':
                        cached[
                            'student_id'
                        ],

                    'name':
                        cached[
                            'name'
                        ],

                    'confidence':
                        cached[
                            'confidence'
                        ],

                    'tracker_id':
                        tid,

                    'global_id':
                        cached[
                            'student_id'
                        ],

                    'cameras':
                        [cam_id],

                    '_cached':
                        True
                })


            else:

                need_arcface_regions.append(

                    (
                        tid,
                        tracker_key,
                        bd
                    )
                )


        # ====================================================
        # CONFIRMED UNKNOWN TRACK LOCK
        #
        # Same idea as confirmed known students:
        #
        # Once:
        #     cam_a_2 -> UNK-001
        #
        # has been confirmed, keep UNK-001 attached to
        # cam_a_2 while BoT-SORT continues returning that
        # tracker ID.
        #
        # Therefore:
        #
        # face visible        -> identify UNK-001
        # face turns away     -> still UNK-001
        # face hidden         -> still UNK-001
        # body remains tracked-> still UNK-001
        #
        # ArcFace/OSNet only run again if the local tracker
        # actually changes or disappears.
        # ====================================================

        if need_arcface_regions:

            remaining_arcface_regions = []


            for (
                tid,
                tracker_key,
                bd
            ) in need_arcface_regions:

                cached_unknown = (
                    confirmed_unknown_ids.get(
                        tracker_key
                    )
                )


                if (
                    cached_unknown is None

                    or

                    cached_unknown.get(
                        'global_id'
                    )
                    is None
                ):

                    remaining_arcface_regions.append(
                        (
                            tid,
                            tracker_key,
                            bd
                        )
                    )

                    continue


                # --------------------------------------------
                # SAME BOT-SORT TRACK IS STILL ALIVE
                # --------------------------------------------

                global_id = cached_unknown.get(
                    'global_id'
                )


                cached_unknown[
                    'time'
                ] = now


                # Keep local display/identity lock alive too.
                if tracker_key in locked_ids:

                    locked_ids[
                        tracker_key
                    ][
                        'time'
                    ] = now

                    locked_ids[
                        tracker_key
                    ][
                        'name'
                    ] = global_id


                else:

                    locked_ids[
                        tracker_key
                    ] = {

                        'is_known':
                            False,

                        'student_id':
                            None,

                        'name':
                            global_id,

                        'confidence':
                            0.0,

                        'time':
                            now
                    }


                cameras = cached_unknown.get(
                    'cameras',
                    [cam_id]
                )


                if not cameras:

                    cameras = [
                        cam_id
                    ]


                try:

                    cached_score = float(
                        cached_unknown.get(
                            'score',
                            0.0
                        )
                        or
                        0.0
                    )

                except Exception:

                    cached_score = 0.0


                # --------------------------------------------
                # BUILD RESULT DIRECTLY
                #
                # No ArcFace required here.
                # No OSNet global identity decision required.
                #
                # BoT-SORT is maintaining the person.
                # --------------------------------------------

                confirmed_results.append({

                    'face_bbox':
                        get_cached_face_bbox(
                            tracker_key,
                            bd['bbox'],
                            now
                        ),

                    'body_bbox':
                        bd['bbox'],

                    'bbox':
                        bd['bbox'],

                    'is_known':
                        False,

                    'student_id':
                        None,

                    'name':
                        global_id,

                    'confidence':
                        0.0,

                    'reid_confidence':
                        cached_score,

                    'body_confidence':
                        bd.get(
                            'confidence',
                            0.0
                        ),

                    'tracker_id':
                        tid,

                    'local_tracker_id':
                        str(
                            tid
                        ),

                    'global_id':
                        global_id,

                    'cameras':
                        cameras,

                    # Existing downstream cached-result path
                    # will append this directly.
                    '_cached':
                        True
                })


            # Only tracks which DO NOT already have a locked
            # unknown identity continue to ArcFace.
            need_arcface_regions = (
                remaining_arcface_regions
            )


        # ====================================================
        # RUN ARCFACE
        # ====================================================

        arc_results = []


        if need_arcface_regions:

            arc_results = (
                recognizer
                .detect_and_recognize(

                    frame_bgr,

                    cam_id=cam_id
                )
            )


        # ====================================================
        # FACE/BODY MATCHING
        # ====================================================

        detections = list(
            confirmed_results
        )


        used_faces = set()


        def _iou(a, b):

            ax1, ay1, ax2, ay2 = a

            bx1, by1, bx2, by2 = b


            ix1 = max(
                ax1,
                bx1
            )

            iy1 = max(
                ay1,
                by1
            )

            ix2 = min(
                ax2,
                bx2
            )

            iy2 = min(
                ay2,
                by2
            )


            iw = max(
                0,
                ix2 - ix1
            )

            ih = max(
                0,
                iy2 - iy1
            )


            inter = (
                iw *
                ih
            )


            area_a = (

                max(
                    0,
                    ax2 - ax1
                )

                *

                max(
                    0,
                    ay2 - ay1
                )
            )


            area_b = (

                max(
                    0,
                    bx2 - bx1
                )

                *

                max(
                    0,
                    by2 - by1
                )
            )


            union = (

                area_a

                +

                area_b

                -

                inter
            )


            return (

                inter /
                union

                if union > 0

                else 0.0
            )


        # ====================================================
        # BUILD BODY/FACE PAIRS
        # ====================================================

        pair_scores = []


        for (
            body_index,
            (
                tid,
                tracker_key,
                bd
            )
        ) in enumerate(
            need_arcface_regions
        ):

            bx1, by1, bx2, by2 = (
                bd['bbox']
            )


            face_region = (

                bx1,

                by1,

                bx2,

                by1
                +
                (
                    by2 - by1
                )
                *
                0.6
            )


            for (
                face_index,
                fd
            ) in enumerate(
                arc_results
            ):

                fx1, fy1, fx2, fy2 = (
                    fd['bbox']
                )


                fcx = (
                    fx1 +
                    fx2
                ) // 2

                fcy = (
                    fy1 +
                    fy2
                ) // 2


                in_body = (

                    bx1
                    <=
                    fcx
                    <=
                    bx2

                    and

                    by1
                    <=
                    fcy
                    <=
                    by1
                    +
                    (
                        by2 - by1
                    )
                    *
                    0.6
                )


                if not in_body:

                    continue


                score = _iou(

                    fd['bbox'],

                    face_region
                )


                pair_scores.append(

                    (
                        score,
                        body_index,
                        face_index
                    )
                )


        # ====================================================
        # GREEDY BIPARTITE ASSIGNMENT
        # ====================================================

        pair_scores.sort(

            key=lambda x: x[0],

            reverse=True
        )


        body_to_face = {}

        used_face_idx = set()

        used_body_idx = set()


        for (
            score,
            body_index,
            face_index
        ) in pair_scores:

            if (
                body_index
                in
                used_body_idx

                or

                face_index
                in
                used_face_idx
            ):

                continue


            body_to_face[
                body_index
            ] = face_index


            used_face_idx.add(
                face_index
            )

            used_body_idx.add(
                body_index
            )


        # ====================================================
        # PREVENT DUPLICATE KNOWN IDENTITY IN SAME FRAME
        # ====================================================

        claimed_student_ids = set()


        for (
            i,
            (
                tid,
                tracker_key,
                bd
            )
        ) in enumerate(
            need_arcface_regions
        ):

            matched_face = None


            face_index = (
                body_to_face.get(i)
            )


            if face_index is not None:

                fd = arc_results[
                    face_index
                ]


                sid = fd.get(
                    'student_id'
                )


                if (
                    fd.get(
                        'is_known'
                    )
                    and
                    sid
                    in
                    claimed_student_ids
                ):

                    matched_face = None


                else:

                    matched_face = (
                        face_index,
                        fd
                    )


                    used_faces.add(
                        face_index
                    )


                    if (
                        fd.get(
                            'is_known'
                        )
                        and
                        sid
                    ):

                        claimed_student_ids.add(
                            sid
                        )


            # ------------------------------------------------
            # FACE MATCHED TO BODY
            # ------------------------------------------------

            if matched_face is not None:

                (
                    face_index,
                    fd
                ) = matched_face


                last_face_boxes[tracker_key] = {
                    'face_bbox': list(fd['bbox']),
                    'body_bbox': list(bd['bbox']),
                    'time': now
                }

                detections.append({

                    'face_bbox':
                        fd['bbox'],

                    'body_bbox':
                        bd['bbox'],

                    'bbox':
                        bd['bbox'],

                    'is_known':
                        fd[
                            'is_known'
                        ],

                    'student_id':
                        fd[
                            'student_id'
                        ],

                    'name':
                        fd[
                            'name'
                        ],

                    'confidence':
                        fd[
                            'confidence'
                        ],

                    'body_confidence':
                        bd.get(
                            'confidence',
                            0.0
                        ),

                    'face_crop':
                        fd.get(
                            'face_crop'
                        ),

                    'face_embedding':
                        fd.get(
                            'embedding'
                        ),

                    'body_crop':
                        bd.get(
                            'body_crop'
                        ),

                    'tracker_id':
                        tid,

                    '_cached':
                        False
                })


            # ------------------------------------------------
            # BODY EXISTS BUT FACE WAS NOT MATCHED
            # ------------------------------------------------

            else:

                prev = locked_ids.get(

                    tracker_key,

                    {}
                )


                detections.append({

                    'face_bbox':
                        get_cached_face_bbox(tracker_key, bd['bbox'], now),

                    'body_bbox':
                        bd['bbox'],

                    'bbox':
                        bd['bbox'],

                    'is_known':
                        prev.get(
                            'is_known',
                            False
                        ),

                    'student_id':
                        prev.get(
                            'student_id'
                        ),

                    'name':
                        prev.get(
                            'name',
                            'Unknown'
                        ),

                    'confidence':
                        prev.get(
                            'confidence',
                            0.0
                        ),

                    'body_confidence':
                        bd.get(
                            'confidence',
                            0.0
                        ),

                    'face_crop':
                        None,

                    'body_crop':
                        bd.get(
                            'body_crop'
                        ),

                    'tracker_id':
                        tid,

                    '_cached':
                        False
                })


        # ====================================================
        # PROCESS IDENTITIES
        # ====================================================

        results = []


        for det in detections:

            tid = det.get('tracker_id')


            if tid is None:

                continue


            tracker_key = (
                f"{cam_id}_{tid}"
            )


            prev = locked_ids.get(

                tracker_key,

                {}
            )


            # =================================================
            # ALREADY-CACHED KNOWN PERSON
            # =================================================

            if det.get(
                '_cached'
            ):

                results.append(
                    det
                )

                continue


            # =================================================
            # KNOWN PERSON
            # =================================================

            if det[
                'is_known'
            ]:

                pending_unknown_time.pop(

                    tracker_key,

                    None
                )


                confirmed_unknown_ids.pop(

                    tracker_key,

                    None
                )


                confirmed_ids[
                    tracker_key
                ] = {

                    'student_id':
                        det[
                            'student_id'
                        ],

                    'name':
                        det[
                            'name'
                        ],

                    'confidence':
                        det[
                            'confidence'
                        ]
                }


                locked_ids[
                    tracker_key
                ] = {

                    'is_known':
                        True,

                    'student_id':
                        det[
                            'student_id'
                        ],

                    'name':
                        det[
                            'name'
                        ],

                    'confidence':
                        det[
                            'confidence'
                        ],

                    'time':
                        now
                }


                rec = {

                    'face_bbox':
                        det[
                            'face_bbox'
                        ],

                    'body_bbox':
                        det[
                            'body_bbox'
                        ],

                    'bbox':
                        det[
                            'bbox'
                        ],

                    'is_known':
                        True,

                    'student_id':
                        det[
                            'student_id'
                        ],

                    'name':
                        det[
                            'name'
                        ],

                    'confidence':
                        det[
                            'confidence'
                        ],

                    'tracker_id':
                        tid,

                    'local_tracker_id':
                        str(tid),

                    'global_id':
                        det[
                            'student_id'
                        ],

                    'cameras':
                        [cam_id]
                }


            # =================================================
            # PREVIOUSLY KNOWN PERSON
            # =================================================

            elif prev.get(
                'is_known'
            ):

                locked_ids[
                    tracker_key
                ][
                    'time'
                ] = now


                rec = {

                    'face_bbox':
                        det[
                            'face_bbox'
                        ],

                    'body_bbox':
                        det[
                            'body_bbox'
                        ],

                    'bbox':
                        det[
                            'bbox'
                        ],

                    'is_known':
                        True,

                    'student_id':
                        prev[
                            'student_id'
                        ],

                    'name':
                        prev[
                            'name'
                        ],

                    'confidence':
                        prev[
                            'confidence'
                        ],

                    'tracker_id':
                        tid,

                    'local_tracker_id':
                        str(tid),

                    'global_id':
                        prev[
                            'student_id'
                        ],

                    'cameras':
                        [cam_id]
                }


            # =================================================
            # UNKNOWN PERSON
            # =================================================

            else:

                cached_unk = (
                    confirmed_unknown_ids.get(
                        tracker_key
                    )
                )


                skip_reid = (

                    cached_unk
                    is not None

                    and

                    (
                        now
                        -
                        cached_unk[
                            'time'
                        ]
                    )
                    <
                    UNKNOWN_RECHECK
                )


                body_crop = det.get(
                    'body_crop'
                )


                face_embedding = det.get(
                    'face_embedding'
                )


                body_confidence = float(
                    det.get(
                        'body_confidence',
                        0.0
                    )
                    or
                    0.0
                )


                body_emb = None

                global_id = None

                reid_score = 0.0

                reid_cameras = [
                    cam_id
                ]


                # =============================================
                # USE RECENT CONFIRMED UNKNOWN CACHE
                # =============================================

                if skip_reid:

                    global_id = (
                        cached_unk[
                            'global_id'
                        ]
                    )


                    reid_score = (
                        cached_unk.get(
                            'score',
                            0.0
                        )
                    )


                    reid_cameras = (
                        cached_unk.get(
                            'cameras',
                            [cam_id]
                        )
                    )


                # =============================================
                # RUN OSNET / GLOBAL RE-ID
                # =============================================

                else:

                    allow_body_reid = (

                        face_embedding
                        is not None

                        or

                        body_confidence
                        >=
                        NEW_UNK_MIN_BODY_CONF
                    )


                    if (
                        not allow_body_reid
                    ):

                        print(
                            f"[REID PERSON GATE] "
                            f"{tracker_key} "
                            f"blocked weak body-only detection "
                            f"conf={body_confidence:.3f}"
                        )


                    if (
                        body_crop
                        is not None

                        and

                        allow_body_reid
                    ):

                        try:

                            body_emb = (
                                body_reid
                                .get_embedding(
                                    body_crop
                                )
                            )


                        except Exception as e:

                            print(
                                f"[BODY REID ERROR] "
                                f"{tracker_key}: "
                                f"{e}"
                            )

                            body_emb = None


                    # =========================================
                    # CAMERA-LOCAL TRACK -> GLOBAL ID INHERIT
                    #
                    # Only used when:
                    #
                    # 1. no face embedding is available
                    # 2. body detection is reasonably strong
                    # 3. a previous confirmed UNK was at almost
                    #    the same physical location
                    #
                    # ArcFace always remains higher priority.
                    # =========================================

                    current_body_bbox = det.get(
                        'body_bbox'
                    )

                    if current_body_bbox is None:

                        current_body_bbox = det.get(
                            'bbox'
                        )


                    if (
                        global_id is None

                        and

                        face_embedding is None

                        and

                        current_body_bbox is not None

                        and

                        body_confidence
                        >=
                        CAM_ID_LOCK_MIN_BODY_CONF
                    ):

                        (
                            lock_global_id,
                            lock_metrics
                        ) = match_camera_identity_lock(

                            tracker_key,

                            current_body_bbox,

                            now
                        )


                        if lock_global_id is not None:

                            try:

                                lock_bound = (
                                    reid.bind_track_identity(

                                        track_key=
                                            tracker_key,

                                        global_id=
                                            lock_global_id,

                                        camera_id=
                                            cam_id,

                                        body_embedding=
                                            body_emb,

                                        bbox=
                                            current_body_bbox
                                    )
                                )


                            except Exception as e:

                                print(
                                    f"[CAM LOCK ERROR] "
                                    f"{tracker_key}: "
                                    f"{e}"
                                )

                                lock_bound = False


                            if lock_bound:

                                global_id = (
                                    lock_global_id
                                )

                                reid_score = float(
                                    lock_metrics[
                                        'score'
                                    ]
                                )


                                print(
                                    f"[CAM LOCK BIND] "
                                    f"{tracker_key} "
                                    f"-> {global_id} "
                                    f"geom="
                                    f"{lock_metrics['score']:.3f} "
                                    f"iou="
                                    f"{lock_metrics['iou']:.3f} "
                                    f"center="
                                    f"{lock_metrics['center']:.3f} "
                                    f"area="
                                    f"{lock_metrics['area_ratio']:.3f}"
                                )


                    # =========================================
                    # ANONYMOUS FACE RE-ID
                    #
                    # Face is the strongest signal for an
                    # already-known UNKNOWN identity.
                    # =========================================

                    current_body_bbox = det.get(
                        'body_bbox'
                    )

                    if current_body_bbox is None:

                        current_body_bbox = det.get(
                            'bbox'
                        )


                    if face_embedding is not None:

                        try:

                            (
                                face_global_id,
                                face_reid_score,
                                face_reid_margin
                            ) = anonymous_face_reid.match(
                                face_embedding
                            )


                            if face_global_id is not None:

                                face_bound = (
                                    reid.bind_track_identity(

                                        track_key=
                                            tracker_key,

                                        global_id=
                                            face_global_id,

                                        camera_id=
                                            cam_id,

                                        body_embedding=(
                                            body_emb
                                            if emb_valid(
                                                body_emb
                                            )
                                            else None
                                        ),

                                        bbox=
                                            current_body_bbox
                                    )
                                )


                                if face_bound:

                                    global_id = (
                                        face_global_id
                                    )

                                    reid_score = (
                                        face_reid_score
                                    )


                                    cams = (
                                        reid
                                        .get_cameras_for(
                                            global_id
                                        )
                                    )


                                    reid_cameras = (

                                        list(cams)

                                        if cams

                                        else [cam_id]
                                    )


                                    margin_text = (

                                        "n/a"

                                        if face_reid_margin
                                        is None

                                        else
                                        f"{face_reid_margin:.3f}"
                                    )


                                    print(
                                        f"[ANON FACE MATCH] "
                                        f"{tracker_key} "
                                        f"-> {global_id} "
                                        f"score="
                                        f"{face_reid_score:.3f} "
                                        f"margin="
                                        f"{margin_text}"
                                    )


                            elif face_reid_score > 0.0:

                                margin_text = (

                                    "n/a"

                                    if face_reid_margin
                                    is None

                                    else
                                    f"{face_reid_margin:.3f}"
                                )


                                print(
                                    f"[ANON FACE NO MATCH] "
                                    f"{tracker_key} "
                                    f"best="
                                    f"{face_reid_score:.3f} "
                                    f"margin="
                                    f"{margin_text}"
                                )


                        except Exception as e:

                            print(
                                f"[ANON FACE ERROR] "
                                f"{tracker_key}: "
                                f"{e}"
                            )


                    # =========================================
                    # FALL BACK TO OSNET / BODY RE-ID
                    #
                    # Only run body identity creation if
                    # anonymous face did NOT already restore
                    # a global UNK identity.
                    # =========================================

                    if (
                        global_id
                        is None

                        and

                        emb_valid(
                            body_emb
                        )
                    ):

                        try:

                            snap = None


                            if (
                                body_crop
                                is not None
                            ):

                                snap = cv2.cvtColor(

                                    body_crop,

                                    cv2.COLOR_RGB2BGR
                                )


                            # =================================
                            # IMPORTANT:
                            # Uses new ReIDManager API
                            # =================================

                            current_body_bbox = det.get(
                                'body_bbox'
                            )

                            if current_body_bbox is None:

                                current_body_bbox = det.get(
                                    'bbox'
                                )


                            (
                                global_id,
                                reid_score
                            ) = reid.find_or_create(

                                embedding=body_emb,

                                camera_id=cam_id,

                                track_key=tracker_key,

                                frame=snap,

                                bbox=current_body_bbox
                            )


                            if (
                                global_id
                                is not None
                            ):

                                cams = (
                                    reid
                                    .get_cameras_for(
                                        global_id
                                    )
                                )


                                reid_cameras = (

                                    list(cams)

                                    if cams

                                    else [cam_id]
                                )


                                print(
                                    f"[REID RESULT] "
                                    f"{tracker_key} "
                                    f"-> "
                                    f"{global_id} "
                                    f"score="
                                    f"{reid_score:.3f}"
                                )


                        except Exception as e:

                            print(
                                f"[REID ERROR] "
                                f"{tracker_key}: "
                                f"{e}"
                            )


                            global_id = None

                            reid_score = 0.0

                            reid_cameras = [
                                cam_id
                            ]


                    # =========================================
                    # ONLY CACHE CONFIRMED GLOBAL IDS
                    # =========================================

                    if (
                        global_id
                        is not None
                    ):

                        confirmed_unknown_ids[
                            tracker_key
                        ] = {

                            'global_id':
                                global_id,

                            'cameras':
                                reid_cameras,

                            'score':
                                reid_score,

                            'time':
                                now
                        }


                # =============================================
                # PERSIST ANONYMOUS FACE FOR CONFIRMED UNK
                # =============================================

                if (
                    global_id
                    is not None

                    and

                    face_embedding
                    is not None
                ):

                    try:

                        # If this track is using the short-term
                        # unknown cache, only add the new face
                        # sample when ArcFace itself agrees with
                        # the cached global identity.
                        existing_face_samples = (
                            anonymous_face_reid
                            .get_sample_count(
                                global_id
                            )
                        )


                        # A brand-new global UNK has no face
                        # gallery yet, so its first sample
                        # is allowed.
                        if existing_face_samples == 0:

                            allow_face_add = True


                        # Existing UNK: only save the face if
                        # ArcFace independently agrees that it
                        # belongs to this global identity.
                        else:

                            (
                                verified_face_id,
                                verified_face_score,
                                verified_face_margin
                            ) = anonymous_face_reid.match(
                                face_embedding
                            )


                            allow_face_add = (
                                verified_face_id
                                ==
                                global_id
                            )

                        if allow_face_add:

                            anonymous_face_reid.add(
                                global_id,
                                face_embedding
                            )


                    except Exception as e:

                        print(
                            f"[ANON FACE SAVE ERROR] "
                            f"{tracker_key}: "
                            f"{e}"
                        )


                # =============================================
                # DISPLAY UNKNOWN UNTIL GLOBAL ID CONFIRMED
                # =============================================

                display_name = (

                    global_id

                    if (
                        global_id
                        is not None
                    )

                    else

                    "Unknown"
                )


                # =============================================
                # UPDATE LOCAL IDENTITY LOCK
                # =============================================

                locked_ids[
                    tracker_key
                ] = {

                    'is_known':
                        False,

                    'student_id':
                        None,

                    'name':
                        display_name,

                    'confidence':
                        0.0,

                    'time':
                        now
                }


                # =============================================
                # BUILD UNKNOWN RESULT
                # =============================================

                rec = {

                    'face_bbox':
                        det.get(
                            'face_bbox'
                        ),

                    'body_bbox':
                        det.get(
                            'body_bbox'
                        ),

                    'bbox':
                        det.get(
                            'bbox'
                        ),

                    'is_known':
                        False,

                    'student_id':
                        None,

                    'name':
                        display_name,

                    'confidence':
                        0.0,

                    'reid_confidence':
                        reid_score,

                    # Local DeepSORT ID
                    'tracker_id':
                        tid,

                    'local_tracker_id':
                        str(tid),

                    # Persistent identity
                    'global_id':
                        global_id,

                    'cameras':
                        reid_cameras
                }


                # =============================================
                # CROSS-CAMERA EVENT
                # =============================================

                if (
                    global_id
                    is not None

                    and

                    len(
                        reid_cameras
                    )
                    >
                    1
                ):

                    cross_key = (

                        global_id

                        +

                        '_'

                        +

                        '_'.join(
                            sorted(
                                reid_cameras
                            )
                        )
                    )


                    with cross_cam_lock:

                        should_fire_cross = (
                            cross_key
                            not in
                            cross_cam_fired
                        )

                        if should_fire_cross:

                            cross_cam_fired.add(
                                cross_key
                            )


                    if should_fire_cross:

                        def send_cross_camera(
                            gid,
                            cameras
                        ):

                            try:

                                requests.post(

                                    (
                                        f"{config.NODE_API_URL}"
                                        f"/api/cross_camera"
                                    ),

                                    json={

                                        'global_id':
                                            gid,

                                        'cameras':
                                            cameras
                                    },

                                    timeout=1.0
                                )

                            except Exception as e:

                                print(
                                    "[CROSS CAM ERROR] "
                                    f"{e}"
                                )


                        threading.Thread(

                            target=send_cross_camera,

                            args=(
                                global_id,
                                reid_cameras
                            ),

                            daemon=True
                        ).start()


            # =================================================
            # LOGGING
            # =================================================

            log_uid = (

                rec.get(
                    'student_id'
                )

                or

                rec.get(
                    'global_id'
                )
            )


            # =================================================
            # KNOWN PERSON LOGGING
            # =================================================

            if (
                rec[
                    'is_known'
                ]

                and

                log_uid
            ):

                with log_lock:

                    if (
                        log_uid
                        not in
                        global_logged_students
                    ):

                        global_logged_students.add(
                            log_uid
                        )


                        payload = {

                            'camera_id':
                                cam_id,

                            'person_id':
                                rec.get(
                                    'student_id'
                                ),

                            'global_person_id':
                                rec.get(
                                    'student_id'
                                ),

                            'student_id':
                                rec.get(
                                    'student_id'
                                ),

                            'student_name':
                                rec.get(
                                    'name',
                                    'Unknown'
                                ),

                            'person_name':
                                rec.get(
                                    'name',
                                    'Unknown'
                                ),

                            'role':
                                'student',

                            'is_known':
                                True,

                            'confidence':
                                round(
                                    rec.get(
                                        'confidence',
                                        0
                                    ),
                                    4
                                ),

                            'reid_confidence':
                                0.0,

                            'tracker_id':
                                str(tid),

                            'local_tracker_id':
                                str(tid),

                            'event_type':
                                'entered',

                            'snapshot_path':
                                None
                        }


                        threading.Thread(

                            target=post_detection,

                            args=(
                                payload,
                            ),

                            daemon=True
                        ).start()


            # =================================================
            # UNKNOWN PERSON LOGGING
            # =================================================
            #
            # IMPORTANT:
            # NO FACE REQUIREMENT.
            #
            # Unknown body Re-ID must work when face is not
            # visible.
            # =================================================

            elif (
                not rec[
                    'is_known'
                ]

                and

                log_uid
            ):

                if (
                    log_uid
                    not in
                    pending_unknown_time
                ):

                    pending_unknown_time[
                        log_uid
                    ] = now


                elif (

                    now
                    -
                    pending_unknown_time[
                        log_uid
                    ]

                    >=
                    0.3
                ):

                    # -----------------------------------------
                    # Only log once per global ID during this
                    # process run.
                    # -----------------------------------------

                    should_log = False


                    with log_lock:

                        if (
                            log_uid
                            not in
                            global_logged_unknowns
                        ):

                            global_logged_unknowns.add(
                                log_uid
                            )

                            should_log = True


                    pending_unknown_time.pop(
                        log_uid,
                        None
                    )


                    if should_log:

                        snap_path = None


                        try:

                            snap_path = (
                                reid
                                .get_snapshot_path(
                                    log_uid
                                )
                            )

                        except Exception:

                            pass


                        payload = {

                            'camera_id':
                                cam_id,

                            'person_id':
                                log_uid,

                            'global_person_id':
                                log_uid,

                            'student_id':
                                None,

                            'student_name':
                                log_uid,

                            'person_name':
                                log_uid,

                            'role':
                                'unknown',

                            'is_known':
                                False,

                            'confidence':
                                0.0,

                            'reid_confidence':
                                round(
                                    rec.get(
                                        'reid_confidence',
                                        0.0
                                    ),
                                    4
                                ),

                            # Local DeepSORT ID
                            'tracker_id':
                                str(tid),

                            'local_tracker_id':
                                str(tid),

                            'event_type':
                                (
                                    'reidentified'

                                    if len(
                                        rec.get(
                                            'cameras',
                                            []
                                        )
                                    )
                                    >
                                    1

                                    else

                                    'first_seen'
                                ),

                            'snapshot_path':
                                snap_path
                        }


                        print(
                            f"[LOG UNKNOWN] "
                            f"{log_uid} "
                            f"camera={cam_id} "
                            f"track={tid}"
                        )


                        threading.Thread(

                            target=post_detection,

                            args=(
                                payload,
                            ),

                            daemon=True
                        ).start()


            # =================================================
            # APPEND RESULT
            # =================================================

            results.append(
                rec
            )


        # ====================================================
        # FINAL PER-FRAME IDENTITY DEDUP
        #
        # BoT-SORT may temporarily represent one physical
        # person with several local track IDs.
        #
        # Example:
        #     cam_b_3 -> UNK-001
        #     cam_b_6 -> UNK-001
        #
        # Keep only ONE final result for the stable identity.
        # This does not modify ArcFace, OSNet or ReIDManager.
        # ====================================================

        if results:

            def result_bbox(
                item
            ):

                bbox = item.get(
                    'body_bbox'
                )

                if bbox is None:

                    bbox = item.get(
                        'bbox'
                    )

                return bbox


            def bbox_area(
                bbox
            ):

                if (
                    bbox is None
                    or
                    len(bbox) != 4
                ):

                    return 0.0

                try:

                    x1, y1, x2, y2 = [
                        float(v)
                        for v in bbox
                    ]

                    return max(
                        0.0,
                        x2 - x1
                    ) * max(
                        0.0,
                        y2 - y1
                    )

                except Exception:

                    return 0.0


            def containment_overlap(
                box_a,
                box_b
            ):

                """
                Intersection / smaller-box area.

                Better than normal IoU for detecting a small
                duplicate box sitting inside a larger body box.
                """

                if (
                    box_a is None
                    or
                    box_b is None
                ):

                    return 0.0

                try:

                    ax1, ay1, ax2, ay2 = [
                        float(v)
                        for v in box_a
                    ]

                    bx1, by1, bx2, by2 = [
                        float(v)
                        for v in box_b
                    ]


                    ix1 = max(
                        ax1,
                        bx1
                    )

                    iy1 = max(
                        ay1,
                        by1
                    )

                    ix2 = min(
                        ax2,
                        bx2
                    )

                    iy2 = min(
                        ay2,
                        by2
                    )


                    intersection = (

                        max(
                            0.0,
                            ix2 - ix1
                        )

                        *

                        max(
                            0.0,
                            iy2 - iy1
                        )
                    )


                    area_a = bbox_area(
                        box_a
                    )

                    area_b = bbox_area(
                        box_b
                    )


                    smaller = min(
                        area_a,
                        area_b
                    )


                    if smaller <= 0.0:

                        return 0.0


                    return float(
                        intersection
                        /
                        smaller
                    )

                except Exception:

                    return 0.0


            def result_quality(
                item
            ):

                score = 0.0


                # Registered student is strongest.
                if item.get(
                    'student_id'
                ) is not None:

                    score += 100.0


                # Confirmed global unknown.
                if item.get(
                    'global_id'
                ) is not None:

                    score += 50.0


                # Prefer observation with an actual face.
                if item.get(
                    'face_bbox'
                ) is not None:

                    score += 20.0


                try:

                    score += (
                        float(
                            item.get(
                                'reid_confidence',
                                0.0
                            )
                            or
                            0.0
                        )
                        *
                        10.0
                    )

                except Exception:

                    pass


                # Prefer fuller/larger body observation when
                # everything else is equal.
                area = bbox_area(
                    result_bbox(
                        item
                    )
                )

                score += min(
                    area / 100000.0,
                    5.0
                )


                return score


            # ------------------------------------------------
            # PASS 1:
            # One result per stable student/global identity
            # ------------------------------------------------

            identity_results = []

            identity_positions = {}


            for item in results:

                student_id = item.get(
                    'student_id'
                )

                global_id = item.get(
                    'global_id'
                )


                if student_id is not None:

                    stable_key = (
                        'student',
                        str(student_id)
                    )


                elif global_id is not None:

                    stable_key = (
                        'global',
                        str(global_id)
                    )


                else:

                    stable_key = None


                if stable_key is None:

                    identity_results.append(
                        item
                    )

                    continue


                existing_index = (
                    identity_positions.get(
                        stable_key
                    )
                )


                if existing_index is None:

                    identity_positions[
                        stable_key
                    ] = len(
                        identity_results
                    )

                    identity_results.append(
                        item
                    )

                    continue


                existing = identity_results[
                    existing_index
                ]


                if (
                    result_quality(
                        item
                    )
                    >
                    result_quality(
                        existing
                    )
                ):

                    identity_results[
                        existing_index
                    ] = item


            # ------------------------------------------------
            # PASS 2:
            # Suppress an unconfirmed temporary track if its
            # body box substantially sits inside/overlaps a
            # confirmed identity's body box.
            #
            # This handles:
            #
            # UNK-001 full box
            # + temporary "Unknown" partial duplicate box
            # ------------------------------------------------

            confirmed_results = [

                item

                for item
                in identity_results

                if (
                    item.get(
                        'student_id'
                    )
                    is not None

                    or

                    item.get(
                        'global_id'
                    )
                    is not None
                )
            ]


            final_results = []


            for item in identity_results:

                is_confirmed = (

                    item.get(
                        'student_id'
                    )
                    is not None

                    or

                    item.get(
                        'global_id'
                    )
                    is not None
                )


                if is_confirmed:

                    final_results.append(
                        item
                    )

                    continue


                current_bbox = result_bbox(
                    item
                )


                overlaps_confirmed = False


                for confirmed_item in confirmed_results:

                    overlap = containment_overlap(

                        current_bbox,

                        result_bbox(
                            confirmed_item
                        )
                    )


                    if overlap >= 0.70:

                        overlaps_confirmed = True

                        break


                if overlaps_confirmed:

                    continue


                final_results.append(
                    item
                )


            if len(final_results) < len(results):

                print(
                    f"[RESULT DEDUP] "
                    f"{cam_id} "
                    f"{len(results)} -> "
                    f"{len(final_results)}"
                )


            results = final_results


        # ====================================================
        # REFRESH CAMERA IDENTITY LOCKS
        #
        # Run AFTER final result dedup so a duplicated local
        # track cannot overwrite the lock with a second box.
        # ====================================================

        for lock_item in results:

            if lock_item.get(
                'is_known',
                False
            ):

                continue


            lock_global_id = lock_item.get(
                'global_id'
            )


            if lock_global_id is None:

                continue


            lock_bbox = lock_item.get(
                'body_bbox'
            )


            if lock_bbox is None:

                lock_bbox = lock_item.get(
                    'bbox'
                )


            if lock_bbox is None:

                continue


            lock_tid = lock_item.get(
                'tracker_id'
            )


            lock_tracker_key = (
                f"{cam_id}_{lock_tid}"
            )


            refresh_camera_identity_lock(

                lock_global_id,

                lock_bbox,

                lock_tracker_key,

                now
            )


        # ====================================================
        # SMOOTH BOUNDING BOXES
        # ====================================================

        for rec_item in results:

            tracker_key = (

                f"{cam_id}_"
                f"{rec_item.get('tracker_id')}"
            )


            raw_bbox = rec_item.get(
                'bbox'
            )


            if raw_bbox is None:

                continue


            if (
                tracker_key
                in
                smoothed_boxes
            ):

                previous_box = (
                    smoothed_boxes[
                        tracker_key
                    ]
                )


                new_box = [

                    previous_box[index]
                    *
                    (
                        1 -
                        SMOOTH_ALPHA
                    )

                    +

                    raw_bbox[index]
                    *
                    SMOOTH_ALPHA

                    for index
                    in range(4)
                ]


            else:

                new_box = list(
                    raw_bbox
                )


            smoothed_boxes[
                tracker_key
            ] = new_box


            smoothed_int = [

                int(value)

                for value
                in new_box
            ]


            rec_item[
                'bbox'
            ] = smoothed_int


            if (
                rec_item.get(
                    'body_bbox'
                )
                is not None
            ):

                rec_item[
                    'body_bbox'
                ] = smoothed_int


        # ====================================================
        # SAVE LAST RESULTS
        # ====================================================

        last_result = results


        # ====================================================
        # PUSH TO DISPLAY THREAD
        # ====================================================

        try:

            result_queue.put_nowait(

                (
                    cam_id,
                    frame.copy(),
                    results,
                    fps
                )
            )


        except queue.Full:

            pass


    stream.stop()


# ============================================================
# DISPLAY
# ============================================================

def run_display(
    result_queue
):

    latest = {}

    win_size = (
        640,
        480
    )


    while True:

        try:

            (
                cam_id,
                frame,
                detections,
                fps
            ) = result_queue.get(
                timeout=0.1
            )


            latest[
                cam_id
            ] = (

                frame,
                detections,
                fps
            )


        except queue.Empty:

            pass


        # ====================================================
        # DRAW EACH CAMERA
        # ====================================================

        for (
            cam_id,
            (
                frame,
                detections,
                fps
            )
        ) in latest.items():

            display = annotator.draw(

                frame.copy(),

                detections
            )


            # =================================================
            # DETECTION ZONE
            # =================================================

            if hasattr(
                config,
                'DETECTION_ZONE'
            ):

                zone = (
                    config
                    .DETECTION_ZONE
                    .get(
                        cam_id
                    )
                )


                if zone:

                    height, width = (
                        display.shape[:2]
                    )


                    cv2.rectangle(

                        display,

                        (
                            int(
                                zone[0]
                                *
                                width
                            ),

                            int(
                                zone[1]
                                *
                                height
                            )
                        ),

                        (
                            int(
                                zone[2]
                                *
                                width
                            ),

                            int(
                                zone[3]
                                *
                                height
                            )
                        ),

                        (
                            0,
                            255,
                            0
                        ),

                        1
                    )


            # =================================================
            # DEBUG / FPS TEXT
            # =================================================

            cached_count = sum(

                1

                for detection
                in detections

                if detection.get(
                    '_cached'
                )
            )


            cv2.putText(

                display,

                (
                    f"{cam_id.upper()}  "
                    f"FPS:{fps:.0f}  "
                    f"Det:{len(detections)}  "
                    f"Cached:{cached_count}"
                ),

                (
                    10,
                    26
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (
                    0,
                    255,
                    255
                ),

                2
            )


            # =================================================
            # DISPLAY WINDOW
            # =================================================

            display = cv2.resize(

                display,

                win_size
            )


            win_name = (

                f"Campus Surveillance "
                f"- {cam_id}"
            )


            cv2.namedWindow(

                win_name,

                cv2.WINDOW_NORMAL
            )


            cv2.resizeWindow(

                win_name,

                *win_size
            )


            cv2.imshow(

                win_name,

                display
            )


            # Dashboard MJPEG / API feed
            update_feed(

                cam_id,

                display
            )


        # ====================================================
        # EXIT
        # ====================================================

        if (
            cv2.waitKey(1)
            &
            0xFF
            ==
            ord('q')
        ):

            break


    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    emb_dir = (
        config.EMBEDDINGS_DIR
    )


    npy_files = (

        [

            filename

            for filename
            in os.listdir(
                emb_dir
            )

            if filename.endswith(
                '.npy'
            )
        ]

        if os.path.exists(
            emb_dir
        )

        else []
    )


    if not npy_files:

        print(
            "No embeddings found. "
            "Run arcface_builder.py first."
        )

        sys.exit(1)


    # ========================================================
    # GPU INFORMATION
    # ========================================================

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )


    # ========================================================
    # RESULT QUEUE
    # ========================================================

    result_queue = queue.Queue(
        maxsize=2
    )


    # ========================================================
    # START CAMERA THREADS
    # ========================================================

    for (
        cam_id,
        source
    ) in config.CAMERA_SOURCES.items():

        threading.Thread(

            target=camera_worker,

            args=(

                cam_id,

                source,

                result_queue
            ),

            daemon=True
        ).start()


        print(
            f"Started thread for "
            f"{cam_id}"
        )


    time.sleep(3)


    print(
        "\nBoth cameras running. "
        "Press Q to quit.\n"
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    run_display(
        result_queue
    )


    print(
        "System stopped."
    )

