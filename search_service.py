"""Appearance search worker. The dashboard starts it on demand; you can also run
    .\\venv\\Scripts\\python.exe search_service.py
It listens on 127.0.0.1 only (port SEARCH_SERVICE_PORT, default 5002):

  GET  /health                       {"ready": true}
  POST /index   {job_id, video, out, sample_fps}   queue a video for indexing
  GET  /jobs/<job_id>                {"state": "queued|running|done|error", "progress", "result"|"error"}
  POST /jobs/<job_id>/cancel
  POST /search  {sources: [{id, folder}], text, photo, upper, lower, limit}

It is a separate process from the live camera engine, so indexing a long video
does not slow recognition down, and recognition is not changed in any way.
(A stdin/stdout protocol was avoided on purpose: on Windows a blocked read of a
stdin pipe can deadlock DLL loading in other threads.)
"""
import os
import queue
import sys
import threading
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import config  # noqa: E402

# Use the cached CLIP weights without asking the internet each start.
if os.path.isdir(os.path.join(config.DATA_DIR, "models", "clip")):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from flask import Flask, request  # noqa: E402

from python_engine.appearance_search import Models, index_video, load_index, photo_query, search  # noqa: E402

app = Flask(__name__)
models = None
jobs = {}
jobs_lock = threading.Lock()
job_queue = queue.Queue()
index_cache = {}


def _job(job_id, **changes):
    with jobs_lock:
        job = jobs.setdefault(job_id, {"state": "queued", "progress": 0.0})
        job.update(changes)
        return dict(job)


def index_worker():
    while True:
        job_id, request_body = job_queue.get()
        if jobs.get(job_id, {}).get("cancel"):
            _job(job_id, state="error", error="Indexing was cancelled.")
            continue
        _job(job_id, state="running", progress=0.0)
        try:
            result = index_video(
                models, request_body["video"], request_body["out"],
                sample_fps=float(request_body.get("sample_fps") or 3.0),
                progress=lambda value: _job(job_id, progress=round(value, 3)),
                should_stop=lambda: bool(jobs.get(job_id, {}).get("cancel")),
            )
            _job(job_id, state="done", progress=1.0, result=result)
        except Exception as error:
            traceback.print_exc()
            _job(job_id, state="error", error=f"{type(error).__name__}: {error}")


def cached_index(folder):
    stamp = os.path.getmtime(os.path.join(folder, "tracks.json"))
    hit = index_cache.get(folder)
    if hit and hit[0] == stamp:
        return hit[1]
    value = load_index(folder)
    index_cache[folder] = (stamp, value)
    return value


@app.get("/health")
def health():
    with jobs_lock:
        running = [k for k, v in jobs.items() if v["state"] in ("queued", "running")]
    return {"ready": models is not None, "device": models.device if models else None, "pending_jobs": running}


@app.post("/index")
def queue_index():
    body = request.get_json(silent=True) or {}
    job_id = str(body.get("job_id") or "")
    if not job_id or not body.get("video") or not body.get("out"):
        return {"error": "job_id, video and out are required."}, 400
    if not os.path.isfile(body["video"]):
        return {"error": "Video file not found."}, 404
    _job(job_id, state="queued", progress=0.0, error=None, result=None, cancel=False)
    job_queue.put((job_id, body))
    return {"queued": True}


@app.get("/jobs/<job_id>")
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        return ({k: v for k, v in job.items() if k != "cancel"}, 200) if job else ({"error": "Unknown job."}, 404)


@app.post("/jobs/<job_id>/cancel")
def cancel_job(job_id):
    _job(job_id, cancel=True)
    return {"cancelled": True}


@app.post("/search")
def run_search():
    body = request.get_json(silent=True) or {}
    try:
        text = models.text_embedding(body["text"]) if (body.get("text") or "").strip() else None
        photo = None
        if body.get("photo"):
            image = cv2.imdecode(np.fromfile(body["photo"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return {"error": "The search photo could not be read."}, 400
            photo = photo_query(models, image)
            if photo["face"] is None and not photo["person_found"]:
                return {"error": "No person or face was found in the search photo."}, 400
        indexes = []
        for source in body.get("sources") or []:
            folder = source.get("folder")
            if folder and os.path.isfile(os.path.join(folder, "tracks.json")):
                tracks, arrays = cached_index(folder)
                indexes.append((source["id"], folder, tracks, arrays))
        rows = search(indexes, text=text, photo=photo, upper=body.get("upper") or None,
                      lower=body.get("lower") or None, limit=int(body.get("limit") or 60),
                      text_query=body.get("text"))
        return {"matches": rows, "face_in_photo": bool(photo and photo["face"] is not None),
                "searched": len(indexes)}
    except Exception as error:
        traceback.print_exc()
        return {"error": f"{type(error).__name__}: {error}"}, 500


def main():
    global models
    port = int(os.getenv("SEARCH_SERVICE_PORT", "5002"))
    print("Loading appearance search models...", flush=True)
    loaded = Models(config.DATA_DIR)
    loaded.faces()             # load now, not during the first search
    loaded.person_detector()
    models = loaded
    threading.Thread(target=index_worker, daemon=True, name="indexer").start()
    print(f"Appearance search ready on http://127.0.0.1:{port} ({models.device})", flush=True)
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
