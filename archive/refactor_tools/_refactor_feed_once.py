"""Separate analysis updates from preview transport without changing zone math."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
path = root/'python_engine/engine_api.py'
source = path.read_text(encoding='utf-8')
start = source.index('def update_feed(')
end = source.index('@app.route("/video_feed/<cam_id>")',start)
old = source[start:end]
zone_start = old.index('        zone_metrics = []')
zone_end = old.index('        camera_state[cam_id] = {',zone_start)
zone_math = old[zone_start:zone_end].replace('source_shape or frame.shape[:2]', 'source_shape')
replacement = '''def update_analysis(cam_id, detections, source_shape, telemetry=None):
    """Only new, processed frames drive activity/occupancy decisions."""
    now = time.time()
    if getattr(config, 'BEHAVIOR_MONITOR_ENABLED', True):
        behavior_monitor.update(cam_id, detections, source_shape,
            getattr(config,'BEHAVIOR_RESTRICTED_ZONES',{}).get(cam_id,[]))
    with lock:
'''+zone_math+'''
        previous = camera_state.get(cam_id,{})
        camera_state[cam_id] = {**previous, 'last_analysis_epoch':now,
            'active_tracks':len(detections), 'zones':zone_metrics, **(telemetry or {})}


def publish_preview(cam_id, frame, raw_frame=None, captured_at=None, telemetry=None):
    """Publish one rendered frame, shared by browser and desktop viewers."""
    now = time.time()
    captured_at = now if captured_at is None else captured_at
    with feed_condition:
        feeds[cam_id] = frame
        if raw_frame is not None:
            # Camera frames are immutable after publication; snapshot copies on use.
            raw_feeds[cam_id] = raw_frame
        feed_versions[cam_id] = feed_versions.get(cam_id,0)+1
        previous = camera_state.get(cam_id,{})
        elapsed = now-previous.get('last_preview_epoch',now)
        instant = 1/elapsed if elapsed > 0 else 0
        old_fps = previous.get('fps',0)
        fps = .75*old_fps+.25*instant if old_fps else instant
        camera_state[cam_id] = {**previous, 'updated_epoch':captured_at,
            'last_preview_epoch':now,
            'last_frame_at':datetime.fromtimestamp(captured_at,timezone.utc).isoformat(),
            'fps':round(fps,1), **(telemetry or {})}
        feed_condition.notify_all()


def update_feed(cam_id, frame, detections=None, source_shape=None, activity_frame=None):
    """Compatibility for callers that publish analysis and preview together."""
    detections = [] if detections is None else detections
    update_analysis(cam_id,detections,source_shape or frame.shape[:2])
    if activity_frame is not None:
        violence_classifier.update(cam_id,activity_frame)
        weapon_detector.update(cam_id,activity_frame)
    frame = weapon_detector.annotate(cam_id,frame)
    if cam_id in config.CAMERA_SOURCES:
        frame = zone_store.annotate(cam_id,frame)
    publish_preview(cam_id,frame,activity_frame)


def encoded_frame(cam_id):
    """Lazy encoding, at most once per version across concurrent viewers."""
    with lock:
        encoding_lock = encoding_locks.setdefault(cam_id,threading.Lock())
    with encoding_lock:
        with lock:
            version, frame = feed_versions.get(cam_id,0), feeds.get(cam_id)
            cached = encoded_frames.get(cam_id)
            if frame is None:
                return None
            if cached and cached[0] == version:
                return cached
        success, buffer = cv2.imencode('.jpg',frame,
            [cv2.IMWRITE_JPEG_QUALITY,getattr(config,'PREVIEW_JPEG_QUALITY',70)])
        if not success:
            return None
        result = (version,buffer.tobytes())
        with lock:
            encoded_frames[cam_id] = result
        return result


def generate(cam_id):
    last_version = -1
    while True:
        with feed_condition:
            fresh = feed_condition.wait_for(
                lambda: cam_id in feeds and feed_versions.get(cam_id,0) > last_version,timeout=1)
        if not fresh:
            continue
        result = encoded_frame(cam_id)
        if result is None:
            time.sleep(.02)
            continue
        last_version, jpeg = result
        yield b'--frame\\r\\nContent-Type: image/jpeg\\r\\n\\r\\n'+jpeg+b'\\r\\n'


'''
source = source[:start]+replacement+source[end:]
source = source.replace('lock = threading.Lock()','lock = threading.Lock()\nfeed_condition = threading.Condition(lock)\nencoded_frames = {}\nencoding_locks = {}',1)
source = source.replace('age = max(0.0, now - state["updated_epoch"]) if state else None','age = max(0.0, now - state["updated_epoch"]) if state and "updated_epoch" in state else None')
source = source.replace('"active_tracks": state.get("active_tracks", 0) if state else 0,','''"active_tracks": state.get("active_tracks", 0) if state and age is not None and age <= 5 else 0,
                "capture_fps": state.get('capture_fps',0) if state else 0,
                "processed_fps": state.get('processed_fps',0) if state else 0,
                "inference_ms": state.get('inference_ms') if state else None,
                "person_ms": state.get('person_ms') if state else None,
                "face_ms": state.get('face_ms') if state else None,
                "dropped_inference_frames": state.get('dropped_inference_frames',0) if state else 0,
                "inference_age_seconds": round(now-state['analysis_frame_at'],3) if state and 'analysis_frame_at' in state else None,
                "overlay_age_seconds": state.get('overlay_age_seconds') if state else None,''')
path.write_text(source,encoding='utf-8')
print('Separated processed-frame analysis from fresh previews; JPEGs cached per frame.')
