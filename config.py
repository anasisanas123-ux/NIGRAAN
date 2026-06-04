import os

# ── Base paths ──────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_IMAGES_DIR  = os.path.join(DATA_DIR, "raw_images")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
EMBEDDINGS_DIR  = os.path.join(DATA_DIR, "embeddings")
DATABASE_PATH   = os.path.join(BASE_DIR, "database", "surveillance.db")
SCHEMA_PATH     = os.path.join(BASE_DIR, "database", "schema.sql")

# ── Camera sources ───────────────────────────────────────
CAMERA_SOURCES = {
    "cam_a": "RTSP_URL_REDACTED",
    "cam_b": "RTSP_URL_REDACTED",
}

# ── Face recognition settings ────────────────────────────
RECOGNITION_THRESHOLD     = 0.60
REID_THRESHOLD            = 0.25
FACE_DETECTION_CONFIDENCE = 0.92
EMBEDDING_SIZE            = 512
MIN_FACE_SIZE             = 60

# ── Tracker settings ─────────────────────────────────────
MAX_AGE      = 120
N_INIT       = 2
MAX_IOU_DIST = 0.7

# ── Processing settings ──────────────────────────────────
FRAME_SKIP   = 2
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ── Node.js API ──────────────────────────────────────────
NODE_API_URL       = "http://localhost:3000"
NODE_DETECTION_URL = f"{NODE_API_URL}/api/detections/ingest"
NODE_UNKNOWN_URL   = f"{NODE_API_URL}/api/unknowns/ingest"

# ── Python engine ────────────────────────────────────────
ENGINE_HOST = "0.0.0.0"
ENGINE_PORT = 5001

# ── Annotation colours (BGR) ─────────────────────────────
COLOR_KNOWN    = (0, 255, 0)
COLOR_UNKNOWN  = (0, 0, 255)
FONT_SCALE     = 0.6
FONT_THICKNESS = 2