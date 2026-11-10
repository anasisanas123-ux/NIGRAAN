import sys
import os

from dotenv import load_dotenv


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    )
)


# ============================================================
# CUDA / cuDNN DLL FIX FOR WINDOWS
# Must execute before ONNX Runtime / InsightFace imports
# ============================================================

if sys.platform == "win32":

    import ctypes

    ctypes.windll.kernel32.SetDllDirectoryW(
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
    )


# ============================================================
# DATA PATHS
# ============================================================

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

RAW_IMAGES_DIR = os.path.join(
    DATA_DIR,
    "raw_images"
)

PROCESSED_DIR = os.path.join(
    DATA_DIR,
    "processed"
)

EMBEDDINGS_DIR = os.path.join(
    DATA_DIR,
    "embeddings"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database",
    "surveillance.db"
)

SCHEMA_PATH = os.path.join(
    BASE_DIR,
    "database",
    "schema.sql"
)


# ============================================================
# CAMERA SOURCES
# ============================================================

CAMERA_SOURCES = {
    "cam_a": 0,
    "cam_b": os.getenv("CAM_B_RTSP"),
}


# ============================================================
# FACE RECOGNITION SETTINGS
# ============================================================

RECOGNITION_THRESHOLD = 0.55
FACE_MATCH_MIN_SIZE = 60
FACE_ENROLL_MIN_SIZE = 80
FACE_QUALITY_CONFIDENCE = 0.65
FACE_MIN_SHARPNESS = 8.0
FACE_ENROLL_MIN_SHARPNESS = 35.0
FACE_SOFT_SHARPNESS = 35.0
FACE_SOFT_MATCH_THRESHOLD = 0.70
FACE_MIN_BRIGHTNESS = 25
FACE_MAX_BRIGHTNESS = 235
FACE_MAX_POSE = 45
FACE_CONFIRM_OBSERVATIONS = 3
FACE_CROP_MAX_REGIONS = 4
FACE_CROP_INTERVAL = 0.8
FACE_INFERENCE_INTERVAL = 0.2

FACE_DETECTION_CONFIDENCE = 0.75

EMBEDDING_SIZE = 512

MIN_FACE_SIZE = 20


# ============================================================
# BODY / UNKNOWN REID SETTINGS
# ============================================================

REID_SAME_CAMERA_THRESHOLD = 0.85

REID_CROSS_CAMERA_THRESHOLD = 0.88
# Backward-compatible name for older call sites.
REID_THRESHOLD = REID_CROSS_CAMERA_THRESHOLD

REID_MIN_MARGIN = 0.12

REID_CONFIRM_FRAMES = 3

REID_NEW_ID_FRAMES = 8

REID_MAX_GALLERY_EMBEDDINGS = 20

REID_TRACK_TIMEOUT = 8.0

REID_GALLERY_UPDATE_THRESHOLD = 0.72

REID_NEW_PERSON_THRESHOLD = 0.50

REID_SAMPLE_INTERVAL = 0.40

REID_REVALIDATE_THRESHOLD = 0.55

REID_REVALIDATE_FRAMES = 3

# OSNet inference options. Flip-test averages original and mirrored body
# embeddings; blur filtering rejects only near-unusable crops before they enter
# the track or identity gallery. Validate these on held-out camera footage.
REID_FLIP_TEST = True
REID_MIN_CROP_BLUR = 2.0

# Optional normalized camera ROIs: (x1, y1, x2, y2), each in [0, 1].
# Define these per camera to enable live occupancy and dwell summaries.
DETECTION_ZONE = {}
DETECTION_ZONE_NAMES = {}
ZONE_TRACK_GRACE_SECONDS = 1.0

# Optional review-only activity cues. Define restricted ROIs as normalized
# boxes per camera, for example:
# BEHAVIOR_RESTRICTED_ZONES = {"cam_a": [{"id": "staff_only",
#     "name": "Staff-only door", "box": [0.72, 0.15, 0.98, 0.92],
#     "dwell_seconds": 10}]}
# Cues describe zone entry, extended dwell, or rapid movement only. They do
# not classify theft, aggression, intent, or identity.
BEHAVIOR_MONITOR_ENABLED = True
BEHAVIOR_RESTRICTED_ZONES = {}

# RWF-2000 violence model. The engine reads a checkpoint from DATA_DIR/models
# when present; it never downloads model files automatically at startup.
VIOLENCE_MODEL_ENABLED = True
VIOLENCE_MODEL_PATH = os.path.join(
    DATA_DIR,
    "models",
    "violence_rwf2000_x3d_m.pt",
)
VIOLENCE_MODEL_THRESHOLD = 0.40
VIOLENCE_MODEL_CONFIRM_WINDOWS = 2
VIOLENCE_MODEL_SAMPLE_INTERVAL = 0.25
VIOLENCE_MODEL_INFERENCE_INTERVAL = 1.0
VIOLENCE_MODEL_ALERT_COOLDOWN = 20.0

# Separate local gun/knife detector; does not replace any existing model.
WEAPON_MODEL_ENABLED = True
WEAPON_MODEL_PATH = os.path.join(DATA_DIR, 'models', 'weapon_gun_knife_yolo11n.onnx')
WEAPON_MODEL_THRESHOLDS = {'gun': 0.55, 'knife': 0.60}
WEAPON_MODEL_INTERVAL = 0.5
WEAPON_MODEL_CONFIRM_FRAMES = 2
WEAPON_MODEL_COOLDOWN = 20.0


# ============================================================
# EVENT SETTINGS
# ============================================================

EVENT_REENTRY_GAP = 10.0


# ============================================================
# TRACKER SETTINGS
# ============================================================

MAX_AGE = 30

N_INIT = 2

MAX_IOU_DIST = 0.7


# ============================================================
# FRAME PROCESSING SETTINGS
# ============================================================

FRAME_SKIP = 1
CAPTURE_FPS = 25
USB_CAMERA_MJPEG = True
PREVIEW_FPS = 20
PREVIEW_SIZE = (640, 480)
PREVIEW_JPEG_QUALITY = 70
OVERLAY_MAX_AGE = 1.0
SHOW_DESKTOP_WINDOWS = os.getenv('SHOW_DESKTOP_WINDOWS', '1').lower() not in ('0', 'false', 'no')

FRAME_WIDTH = 1920

FRAME_HEIGHT = 1080


# ============================================================
# NODE.JS API
# ============================================================

NODE_API_URL = "http://localhost:3000"

NODE_DETECTION_URL = (
    f"{NODE_API_URL}/api/detections/ingest"
)

NODE_UNKNOWN_URL = (
    f"{NODE_API_URL}/api/unknowns/ingest"
)


# ============================================================
# PYTHON ENGINE
# ============================================================

ENGINE_HOST = "0.0.0.0"

ENGINE_PORT = 5001


# ============================================================
# ANNOTATION COLOURS - BGR
# ============================================================

COLOR_KNOWN = (
    0,
    255,
    0
)

COLOR_UNKNOWN = (
    0,
    0,
    255
)

FONT_SCALE = 0.6

FONT_THICKNESS = 2

# ============================================================
# ADD-ON FEATURES (recording, sightings, visitor watch list, exam mode)
# None of these change face recognition, Re-ID or tracking.
# ============================================================
# Recording starts and stops from the dashboard (Recordings tab).
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
RECORDING_FPS = 15                  # frames written per second
RECORDING_MAX_WIDTH = 1280          # larger frames are scaled down before encoding
RECORDING_SEGMENT_MINUTES = 10      # long recordings are split; a crash only loses the open file
RECORDING_ENCODER = "auto"          # "auto" (CPU libx264, GPU h264_nvenc if that fails), or one of them

# Sightings: who was seen on which camera and when (used by attendance).
# A person is logged again on the same camera at most once per interval.
SIGHTING_INTERVAL_SECONDS = 30

# Expired / checked-out visitor watch list.
VISITOR_WATCHLIST_DIR = os.path.join(DATA_DIR, "visitor_watchlist")
VISITOR_WATCHLIST_THRESHOLD = 0.50  # ArcFace cosine similarity needed for a match
VISITOR_WATCHLIST_CONFIRM = 2       # matches within 10 s before an alert
VISITOR_WATCHLIST_KEEP_DAYS = 7     # face data is deleted after this many days
VISITOR_WATCHLIST_COOLDOWN = 120.0  # seconds between alerts for the same visitor

# Exam mode (cheating cues). Runs only while an exam session is active.
EXAM_DIR = os.path.join(DATA_DIR, "exams")
EXAM_POSE_MODEL = "yolo11n-pose.pt"   # downloaded by ultralytics on first use
EXAM_OBJECT_MODEL = "yolov8s.pt"      # COCO model, used only for its "cell phone" class
EXAM_SAMPLE_INTERVAL = 0.35           # seconds between analysed frames per camera
EXAM_PHONE_CONFIDENCE = 0.40
EXAM_PHONE_CONFIRM = 2                # consecutive analysed frames with a phone
EXAM_LOOK_YAW_RATIO = 0.55            # how far the nose is off-centre between the eyes
EXAM_LOOK_SECONDS = 3.0               # sideways/backwards this long before a cue
EXAM_SEAT_LEARN_SECONDS = 15.0        # seats are learned from where people sit at the start
EXAM_SEAT_EMPTY_SECONDS = 45.0        # a learned seat empty this long gives a cue
EXAM_CLOSE_SECONDS = 4.0              # two students this close for this long gives a cue
EXAM_CLOSE_RATIO = 0.60               # head distance / average shoulder width
EXAM_ALERT_COOLDOWN = 30.0            # seconds between repeated cues for the same student/seat
