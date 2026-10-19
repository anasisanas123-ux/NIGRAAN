import os
import sys
import threading
import time
import base64
import binascii
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, Response, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from python_engine.behavior_monitor import behavior_monitor
from python_engine.violence_classifier import create_default_classifier
from python_engine.weapon_detector import create_default_weapon_detector
from python_engine.zone_store import ZoneStore
from python_engine.frame_buffer import RollingRate


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
feeds = {}
raw_feeds = {}
feed_versions = {}
camera_state = {}
zone_entry_times = {}
zone_last_seen = {}
zone_dwell_samples = {}
lock = threading.Lock()
feed_condition = threading.Condition(lock)
encoded_frames = {}
encoding_locks = {}
preview_rates = {}
violence_classifier = create_default_classifier(config)
weapon_detector = create_default_weapon_detector(config)
identity_recognizer = None
zone_store = ZoneStore(config)


@app.route("/zones/<cam_id>", methods=["GET", "PUT"])
def camera_zones(cam_id):
    if not _is_local_request():
        return {"success": False, "error": "Use the local dashboard for camera settings."}, 403
    try:
        if request.method == "PUT":
            value = zone_store.save(cam_id, request.get_json(silent=True))
            with lock:
                for mapping in (zone_entry_times, zone_last_seen, zone_dwell_samples):
                    for key in list(mapping):
                        if key[0] == cam_id:
                            mapping.pop(key, None)
            behavior_monitor.reset_camera(cam_id)
        else:
            value = zone_store.get(cam_id)
        return {"success": True, "data": value}
    except ValueError as error:
        return {"success": False, "error": str(error)}, 400
    except OSError:
        return {"success": False, "error": "Could not save camera zones."}, 500


def set_identity_recognizer(recognizer):
    global identity_recognizer
    identity_recognizer = recognizer


def _is_local_request():
    return request.remote_addr in ("127.0.0.1", "::1")


@app.route("/enroll", methods=["POST"])
def enroll_identity():
    if not _is_local_request():
        return {
            "success": False,
            "error": "Enrollment is available through the local dashboard only.",
        }, 403
    if identity_recognizer is None:
        return {"success": False, "error": "Recognition engine is not ready."}, 503
    body = request.get_json(silent=True) or {}
    data_urls = body.get("images")
    if not isinstance(data_urls, list) or not data_urls or len(data_urls) > 12:
        return {"success": False, "error": "Provide between 1 and 12 enrollment images."}, 400
    images = []
    for data_url in data_urls:
        if not isinstance(data_url, str) or "," not in data_url:
            return {"success": False, "error": "An enrollment image is invalid."}, 400
        header, encoded = data_url.split(",", 1)
        if header not in (
            "data:image/jpeg;base64",
            "data:image/png;base64",
            "data:image/webp;base64",
        ):
            return {"success": False, "error": "Use JPEG, PNG, or WebP enrollment photos."}, 400
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return {"success": False, "error": "An enrollment image is not valid base64."}, 400
        if not raw or len(raw) > 1024 * 1024:
            return {"success": False, "error": "Each enrollment photo must be under 1 MB."}, 400
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"success": False, "error": "An enrollment photo could not be decoded."}, 400
        images.append(image)
    try:
        result = identity_recognizer.add_identity_from_images(
            body.get("identity_id"),
            body.get("name"),
            images,
            role=body.get("role", "student"),
            valid_until=body.get("valid_until"),
        )
        return {"success": True, "data": result}
    except ValueError as error:
        return {"success": False, "error": str(error)}, 400
    except Exception as error:
        app.logger.exception("Identity enrollment failed")
        return {"success": False, "error": f"Could not enroll identity: {error}"}, 500


@app.route("/identity/<identity_id>", methods=["DELETE"])
def remove_identity(identity_id):
    if not _is_local_request():
        return {
            "success": False,
            "error": "Identity management is available through the local dashboard only.",
        }, 403
    if identity_recognizer is None:
        return {"success": False, "error": "Recognition engine is not ready."}, 503
    try:
        return {"success": True, "data": identity_recognizer.remove_identity(identity_id)}
    except Exception as error:
        app.logger.exception("Identity removal failed")
        return {"success": False, "error": f"Could not remove identity: {error}"}, 500


@app.route("/rebuild-embeddings", methods=["POST"])
def rebuild_embeddings():
    if not _is_local_request():
        return {
            "success": False,
            "error": "Embedding rebuild is available through the local dashboard only.",
        }, 403
    if identity_recognizer is None:
        return {"success": False, "error": "Recognition engine is not ready."}, 503
    try:
        result = identity_recognizer.rebuild_from_saved_photos()
        return {"success": True, "data": result}, 200
    except Exception as error:
        app.logger.exception("Embedding rebuild failed")
        return {"success": False, "error": f"Could not rebuild embeddings: {error}"}, 500


def update_analysis(cam_id, detections, source_shape, telemetry=None):
    """Only new, processed frames drive activity/occupancy decisions."""
    now = time.time()
    if getattr(config, "BEHAVIOR_MONITOR_ENABLED", True):
        behavior_monitor.update(
            cam_id,
            detections,
            source_shape,
            getattr(config, "BEHAVIOR_RESTRICTED_ZONES", {}).get(cam_id, []),
        )
    with lock:
        zone_metrics = []
        roi = getattr(config, "DETECTION_ZONE", {}).get(cam_id)
        if roi and len(roi) == 4:
            frame_height, frame_width = source_shape
            x1n, y1n, x2n, y2n = [float(value) for value in roi]
            inside = set()

            for detection in detections:
                bbox = detection.get("body_bbox") or detection.get("bbox")
                if bbox is None or len(bbox) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = [float(value) for value in bbox]
                except (TypeError, ValueError):
                    continue
                center_x = ((x1 + x2) * 0.5) / max(frame_width, 1)
                center_y = ((y1 + y2) * 0.5) / max(frame_height, 1)
                if x1n <= center_x <= x2n and y1n <= center_y <= y2n:
                    identity = (
                        detection.get("global_person_id")
                        or detection.get("global_id")
                        or detection.get("student_id")
                        or detection.get("person_id")
                        or detection.get("tracker_id")
                    )
                    if identity is not None:
                        inside.add(str(identity))

            zone_key = (cam_id, "primary")
            monotonic_now = time.monotonic()
            grace = float(getattr(config, "ZONE_TRACK_GRACE_SECONDS", 1.0))
            expired = [
                key
                for key in zone_entry_times
                if key[0] == cam_id
                and key[1] not in inside
                and monotonic_now - zone_last_seen.get(key, 0.0) > grace
            ]
            for key in expired:
                last_seen = zone_last_seen.pop(key, monotonic_now)
                ended_at = min(monotonic_now, last_seen + grace)
                duration = max(0.0, ended_at - zone_entry_times.pop(key))
                zone_last_seen.pop(key, None)
                samples = zone_dwell_samples.setdefault(zone_key, [])
                samples.append(duration)
                del samples[:-50]

            for identity in inside:
                key = (cam_id, identity)
                zone_entry_times.setdefault(key, monotonic_now)
                zone_last_seen[key] = monotonic_now
            current_dwell = [
                max(0.0, monotonic_now - zone_entry_times[(cam_id, identity)])
                for identity in inside
            ]
            dwell_samples = zone_dwell_samples.get(zone_key, []) + current_dwell
            zone_metrics.append(
                {
                    "zone_id": "primary",
                    "name": getattr(config, "DETECTION_ZONE_NAMES", {}).get(
                        cam_id, f"{cam_id} zone"
                    ),
                    "occupancy": len(inside),
                    "average_dwell_seconds": (
                        sum(dwell_samples) / len(dwell_samples) if dwell_samples else 0.0
                    ),
                }
            )

        previous = camera_state.get(cam_id, {})
        camera_state[cam_id] = {
            **previous,
            "last_analysis_epoch": now,
            "active_tracks": len(detections),
            "zones": zone_metrics,
            **(telemetry or {}),
        }


def publish_preview(cam_id, frame, raw_frame=None, captured_at=None, telemetry=None):
    """Publish one rendered frame, shared by browser and desktop viewers."""
    now = time.time()
    captured_at = now if captured_at is None else captured_at
    with feed_condition:
        feeds[cam_id] = frame
        if raw_frame is not None:
            # Camera frames are immutable after publication; snapshot copies on use.
            raw_feeds[cam_id] = raw_frame
        feed_versions[cam_id] = feed_versions.get(cam_id, 0) + 1
        previous = camera_state.get(cam_id, {})
        fps = preview_rates.setdefault(cam_id, RollingRate()).tick()
        camera_state[cam_id] = {
            **previous,
            "updated_epoch": captured_at,
            "last_preview_epoch": now,
            "last_frame_at": datetime.fromtimestamp(captured_at, timezone.utc).isoformat(),
            "fps": round(fps, 1),
            **(telemetry or {}),
        }
        feed_condition.notify_all()


def update_feed(cam_id, frame, detections=None, source_shape=None, activity_frame=None):
    """Compatibility for callers that publish analysis and preview together."""
    detections = [] if detections is None else detections
    update_analysis(cam_id, detections, source_shape or frame.shape[:2])
    if activity_frame is not None:
        violence_classifier.update(cam_id, activity_frame)
        weapon_detector.update(cam_id, activity_frame)
    frame = weapon_detector.annotate(cam_id, frame)
    if cam_id in config.CAMERA_SOURCES:
        frame = zone_store.annotate(cam_id, frame)
    publish_preview(cam_id, frame, activity_frame)


def encoded_frame(cam_id):
    """Lazy encoding, at most once per version across concurrent viewers."""
    with lock:
        encoding_lock = encoding_locks.setdefault(cam_id, threading.Lock())
    with encoding_lock:
        with lock:
            version, frame = feed_versions.get(cam_id, 0), feeds.get(cam_id)
            cached = encoded_frames.get(cam_id)
            if frame is None:
                return None
            if cached and cached[0] == version:
                return cached
        success, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, getattr(config, "PREVIEW_JPEG_QUALITY", 70)]
        )
        if not success:
            return None
        result = (version, buffer.tobytes())
        with lock:
            encoded_frames[cam_id] = result
        return result


def generate(cam_id):
    last_version = -1
    while True:
        with feed_condition:
            fresh = feed_condition.wait_for(
                lambda: cam_id in feeds and feed_versions.get(cam_id, 0) > last_version, timeout=1
            )
        if not fresh:
            continue
        result = encoded_frame(cam_id)
        if result is None:
            time.sleep(0.02)
            continue
        last_version, jpeg = result
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


@app.route("/video_feed/<cam_id>")
def video_feed(cam_id):
    if cam_id not in config.CAMERA_SOURCES:
        return {"success": False, "error": "Unknown camera."}, 404
    return Response(
        generate(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/health")
def health():
    now = time.time()
    with lock:
        cameras = {}
        for cam_id in config.CAMERA_SOURCES:
            state = camera_state.get(cam_id)
            age = (
                max(0.0, now - state["updated_epoch"])
                if state and "updated_epoch" in state
                else None
            )
            analysis_age = now - state.get("last_analysis_epoch", now) if state else None
            roi = getattr(config, "DETECTION_ZONE", {}).get(cam_id)
            zone_name = getattr(config, "DETECTION_ZONE_NAMES", {}).get(cam_id, f"{cam_id} zone")
            zones = state.get("zones", []) if state else []
            if roi and not zones:
                zones = [
                    {
                        "zone_id": "primary",
                        "name": zone_name,
                        "occupancy": 0,
                        "average_dwell_seconds": 0.0,
                    }
                ]
            cameras[cam_id] = {
                "status": "online" if age is not None and age <= 5.0 else "offline",
                "last_frame_at": state.get("last_frame_at") if state else None,
                "frame_age_seconds": round(age, 1) if age is not None else None,
                "fps": state.get("fps", 0.0) if state else 0.0,
                "active_tracks": (
                    state.get("active_tracks", 0)
                    if state and age is not None and age <= 5 and analysis_age <= 2
                    else 0
                ),
                "capture_fps": state.get("capture_fps", 0) if state else 0,
                "source_width": state.get("source_width") if state else None,
                "source_height": state.get("source_height") if state else None,
                "processed_fps": state.get("processed_fps", 0) if state else 0,
                "inference_ms": state.get("inference_ms") if state else None,
                "person_ms": state.get("person_ms") if state else None,
                "face_ms": state.get("face_ms") if state else None,
                "dropped_inference_frames": (
                    state.get("dropped_inference_frames", 0) if state else 0
                ),
                "inference_age_seconds": (
                    round(now - state["analysis_frame_at"], 3)
                    if state and "analysis_frame_at" in state
                    else None
                ),
                "overlay_age_seconds": state.get("overlay_age_seconds") if state else None,
                "zones": zones,
            }
    return {"status": "ok", "cameras": cameras}


@app.route("/snapshot/<cam_id>")
def snapshot(cam_id):
    """Reuse an open engine camera for enrollment without opening it twice."""
    if cam_id not in config.CAMERA_SOURCES:
        return {"success": False, "error": "Unknown camera."}, 404
    with lock:
        frame = raw_feeds.get(cam_id)
        updated = camera_state.get(cam_id, {}).get("updated_epoch", 0)
        if frame is None or time.time() - updated > 5:
            return {
                "success": False,
                "error": "Camera is offline. Start the Python engine or select another camera.",
            }, 503
        frame = frame.copy()
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return {"success": False, "error": "Could not capture camera frame."}, 503
    return Response(encoded.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/behavior-alerts")
def behavior_alerts():
    """Return recent, non-identifying review cues and model readiness."""
    alerts = (
        behavior_monitor.snapshot() + violence_classifier.snapshot() + weapon_detector.snapshot()
    )
    alerts.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    return {
        "status": "ok",
        "alerts": alerts[:200],
        "violence_model": violence_classifier.status(),
        "weapon_model": weapon_detector.status(),
        "monitor": {
            "enabled": bool(getattr(config, "BEHAVIOR_MONITOR_ENABLED", True)),
            "restricted_zone_cameras": [
                cam_id
                for cam_id, zones in getattr(config, "BEHAVIOR_RESTRICTED_ZONES", {}).items()
                if zones
            ],
        },
    }


def start_engine():
    app.run(
        host=config.ENGINE_HOST,
        port=config.ENGINE_PORT,
        threaded=True,
        debug=False,
        use_reloader=False,
    )
