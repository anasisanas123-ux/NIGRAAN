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

FACE_DETECTION_CONFIDENCE = 0.75

EMBEDDING_SIZE = 512

MIN_FACE_SIZE = 20


# ============================================================
# BODY / UNKNOWN REID SETTINGS
# ============================================================

REID_THRESHOLD = 0.68

REID_SAME_CAMERA_THRESHOLD = 0.68

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

FRAME_WIDTH = 1280

FRAME_HEIGHT = 720


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