"""Per-camera tracking and identity decisions; models are injected explicitly."""

import time
import threading
import requests
import cv2
import config
from python_engine.camera_stream import CameraStream
from python_engine.identity_cache import (
    contested_track_keys,
    invalidate_inactive,
    is_rejected_weak_detection,
    needs_face_check,
    update_pending_hold,
)
from python_engine.gait_pose import GaitPoseExtractor
from python_engine.gait_signature import GaitSignatureEngine
from python_engine.runtime_helpers import post_detection, emb_valid
from python_engine.runtime_settings import (
    NEW_UNK_MIN_BODY_CONF,
    ANON_FACE_HOLD_MIN_SCORE,
    ANON_FACE_HOLD_MIN_MARGIN,
    ANON_FACE_HOLD_STRONG_SCORE,
    ANON_FACE_HOLD_SECONDS,
    ANON_FACE_HOLD_STALE_SECONDS,
    FAR_PERSON_MIN_HEIGHT_RATIO,
    SAME_CAM_CONTINUITY_TTL,
    SAME_CAM_CONTINUITY_MIN_BODY_SCORE,
    SAME_CAM_CONTINUITY_MIN_BODY_MARGIN,
    SAME_CAM_CONTINUITY_MAX_CENTER,
    SAME_CAM_CONTINUITY_MIN_AREA_RATIO,
)
from python_engine.preview import CameraPreview
from python_engine.frame_buffer import RollingRate
from python_engine.engine_api import (
    update_analysis,
    publish_preview,
    violence_classifier,
    weapon_detector,
    zone_store,
    exam_monitor,
    recorder,
)


def camera_worker(cam_id, source, result_queue, context):
    recognizer = context.recognizer
    body_detector = context.body_detector
    body_reid = context.body_reid
    reid = context.reid
    anonymous_face_reid = context.anonymous_face_reid
    annotator = context.annotator
    global_logged_students = context.global_logged_students
    global_logged_unknowns = context.global_logged_unknowns
    log_lock = context.log_lock
    cross_cam_fired = context.cross_cam_fired
    cross_cam_lock = context.cross_cam_lock
    print(f"Connecting to " f"{cam_id}")
    stream = CameraStream(source)

    # ========================================================
    # PASSIVE GAIT OBSERVATION
    #
    # IMPORTANT:
    #
    # Gait is NOT allowed to assign/change an identity here.
    #
    # Current ArcFace + anonymous face + OSNet + camera locks
    # remain the authority.
    #
    # Gait only observes their final UNK identity and reports
    # whether the gait signature agrees with previous samples.
    # ========================================================
    GAIT_OBSERVATION_ENABLED = True
    GAIT_SEQUENCE_FRAMES = 32

    # About 5-6 gait samples / second.
    # Pose does NOT run on every video frame.
    GAIT_POSE_INTERVAL = 0.18

    # Do not collect multiple copies of nearly identical
    # frames too quickly.
    GAIT_SAMPLE_INTERVAL = 0.15

    # Current global identity must remain stable before its
    # sequence may be written to the observation gallery.
    GAIT_ID_STABLE_TIME = 3.0

    # Drop abandoned local-track buffers.
    GAIT_TRACK_TTL = 5.0

    # Conservative body <-> pose association.
    GAIT_MIN_ASSOC_IOU = 0.45

    # Lazy initialization:
    #
    # cam_b being offline will NOT load another pose model.
    # The model loads only after a confirmed unknown actually
    # appears on this camera.
    gait_pose = None
    gait_signature = None
    gait_init_attempted = False
    gait_last_pose_run = 0.0
    gait_track_state = {}

    def same_camera_continuity_geometry(previous_bbox, current_bbox):
        """
        Returns:
            center_distance,
            area_ratio

        center_distance is normalized by the average person
        box diagonal rather than full-frame size.

        This lets a far person move somewhat while still
        preventing arbitrary identity jumps across the frame.
        """
        try:
            (px1, py1, px2, py2) = [float(v) for v in previous_bbox]
            (cx1, cy1, cx2, cy2) = [float(v) for v in current_bbox]
            prev_w = max(px2 - px1, 1.0)
            prev_h = max(py2 - py1, 1.0)
            curr_w = max(cx2 - cx1, 1.0)
            curr_h = max(cy2 - cy1, 1.0)
            prev_center_x = (px1 + px2) / 2.0
            prev_center_y = (py1 + py2) / 2.0
            curr_center_x = (cx1 + cx2) / 2.0
            curr_center_y = (cy1 + cy2) / 2.0
            dx = curr_center_x - prev_center_x
            dy = curr_center_y - prev_center_y
            center_distance_pixels = (dx * dx + dy * dy) ** 0.5
            prev_diag = (prev_w * prev_w + prev_h * prev_h) ** 0.5
            curr_diag = (curr_w * curr_w + curr_h * curr_h) ** 0.5
            normalizer = max((prev_diag + curr_diag) / 2.0, 1.0)
            center_distance = center_distance_pixels / normalizer
            prev_area = prev_w * prev_h
            curr_area = curr_w * curr_h
            area_ratio = min(prev_area, curr_area) / max(prev_area, curr_area, 1.0)
            return (float(center_distance), float(area_ratio))
        except Exception:
            return (999.0, 0.0)

    def gait_bbox_iou(box_a, box_b):
        try:
            ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
            bx1, by1, bx2, by2 = [float(v) for v in box_b]
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            intersection = iw * ih
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - intersection
            if union <= 0:
                return 0.0
            return float(intersection / union)
        except Exception:
            return 0.0

    context.register_stream(cam_id, stream)
    preview = CameraPreview(
        cam_id,
        stream,
        result_queue,
        annotator,
        recognizer,
        publish_preview,
        # Recorder last, so the REC marker is drawn on top. Both receive raw frames.
        (violence_classifier, weapon_detector, exam_monitor, recorder),
        zone_store,
    )
    preview.thread.start()

    # ========================================================
    # WAIT FOR CAMERA CONNECTION
    # ========================================================
    while stream.running:
        packet = stream.read_packet(timeout=1)
        if packet is not None:
            break
    else:
        return
    last_sequence = 0
    dropped_frames = 0

    # Current physical camera session.
    camera_generation = packet.generation
    print(f"{cam_id} connected.")

    # ========================================================
    # CAMERA STATE
    # ========================================================
    frame_count = 0
    last_result = []
    inference_rate = RollingRate()
    last_face_inference = -float("inf")
    fps = 0.0

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

    # Temporary possible-existing-face states.
    #
    # tracker_key -> {
    #     candidate_id,
    #     start_time,
    #     last_seen,
    #     max_score,
    #     last_log
    # }
    near_face_holds = {}
    far_person_last_log = {}

    # global_id -> most recent confirmed observation
    #
    # {
    #     'bbox': ...,
    #     'tracker_key': ...,
    #     'time': ...
    # }
    recent_same_camera_identities = {}
    UNKNOWN_RECHECK = 0.8

    # tracker_key -> time until an enrolled-face match under confirmation
    # keeps the track out of unknown identity creation, gallery and logs.
    pending_identity_holds = {}
    # Matches the FaceEvidence window: a longer gap restarts confirmation.
    IDENTITY_PENDING_HOLD = 2.0

    # Overlap (intersection over the smaller box) at which a named track may
    # have been swapped with another person by the tracker.
    CONTESTED_MIN_OVERLAP = 0.25
    # A face match this recent keeps the name through an overlap.
    KNOWN_FACE_FRESH_SECONDS = 1.0
    # How long a released track waits for its face before it may be treated
    # as a new unknown person.
    CONTESTED_IDENTITY_HOLD = 5.0

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
    CAM_ID_LOCK_TTL = 4.0

    # Body-only inheritance must still be reasonably strong.
    # A visible face does NOT use this shortcut; ArcFace gets
    # priority instead.
    CAM_ID_LOCK_MIN_BODY_CONF = 0.50
    CAM_ID_LOCK_MIN_IOU = 0.60
    CAM_ID_LOCK_MAX_CENTER = 0.18
    CAM_ID_LOCK_MIN_AREA_RATIO = 0.65
    CAM_ID_LOCK_MIN_CONTAINMENT = 0.75

    # Camera-lock transfer is NOT allowed from geometry alone.
    #
    # The new local track must also look like the SAME global
    # identity according to OSNet.
    CAM_ID_LOCK_MIN_BODY_SCORE = 0.52
    CAM_ID_LOCK_MIN_BODY_MARGIN = 0.08

    # If two existing people are nearly equally plausible,
    # don't guess.
    CAM_ID_LOCK_AMBIGUITY_MARGIN = 0.12

    # global_id -> {
    #     bbox,
    #     tracker_key,
    #     time
    # }
    camera_identity_locks = {}

    def prune_camera_identity_locks(current_time):
        expired = []
        for gid, info in list(camera_identity_locks.items()):
            age = current_time - float(info.get("time", 0.0))
            if age > CAM_ID_LOCK_TTL:
                expired.append(gid)
        for gid in expired:
            camera_identity_locks.pop(gid, None)

    def camera_lock_metrics(box_a, box_b):
        if box_a is None or box_b is None:
            return None
        try:
            ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
            bx1, by1, bx2, by2 = [float(v) for v in box_b]
            aw = max(1.0, ax2 - ax1)
            ah = max(1.0, ay2 - ay1)
            bw = max(1.0, bx2 - bx1)
            bh = max(1.0, by2 - by1)
            area_a = aw * ah
            area_b = bw * bh
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            intersection = iw * ih
            union = area_a + area_b - intersection
            if union > 0:
                iou = intersection / union
            else:
                iou = 0.0
            smaller_area = min(area_a, area_b)
            larger_area = max(area_a, area_b)
            if smaller_area > 0:
                containment = intersection / smaller_area
            else:
                containment = 0.0
            if larger_area > 0:
                area_ratio = smaller_area / larger_area
            else:
                area_ratio = 0.0
            acx = (ax1 + ax2) / 2.0
            acy = (ay1 + ay2) / 2.0
            bcx = (bx1 + bx2) / 2.0
            bcy = (by1 + by2) / 2.0
            center_distance = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
            diag_a = (aw**2 + ah**2) ** 0.5
            diag_b = (bw**2 + bh**2) ** 0.5
            normalization = max(1.0, (diag_a + diag_b) / 2.0)
            normalized_center = center_distance / normalization
            center_quality = max(0.0, 1.0 - min(normalized_center, 1.0))
            geometry_score = (
                0.40 * iou + 0.25 * containment + 0.20 * area_ratio + 0.15 * center_quality
            )
            return {
                "iou": float(iou),
                "containment": float(containment),
                "area_ratio": float(area_ratio),
                "center": float(normalized_center),
                "score": float(geometry_score),
            }
        except Exception:
            return None

    def match_camera_identity_lock(tracker_key, current_bbox, current_time, body_embedding=None):
        prune_camera_identity_locks(current_time)

        # ====================================================
        # APPEARANCE VERIFICATION
        #
        # Never transfer a camera lock using geometry alone.
        # ====================================================
        if body_embedding is None:
            return (None, None)
        try:
            rankings = reid._rank_gallery(body_embedding)
        except Exception:

            # Fallback in case internal helper changes.
            rankings = []
            try:
                for gid, identity in reid.unknown_gallery.items():
                    score = reid._identity_score(body_embedding, identity)
                    rankings.append((gid, float(score)))
                rankings.sort(key=lambda item: item[1], reverse=True)
            except Exception:
                rankings = []
        if not rankings:
            return (None, None)
        best_body_id = str(rankings[0][0])
        best_body_score = float(rankings[0][1])
        if len(rankings) > 1:
            second_body_score = float(rankings[1][1])
            body_margin = best_body_score - second_body_score
        else:
            second_body_score = None
            body_margin = 1.0

        # Weak/ambiguous OSNet evidence:
        # do NOT inherit any camera lock.
        if best_body_score < CAM_ID_LOCK_MIN_BODY_SCORE:
            print(
                f"[CAM LOCK BODY REJECT] "
                f"{tracker_key} "
                f"best={best_body_id}:"
                f"{best_body_score:.3f} "
                f"reason=weak"
            )
            return (None, None)
        if len(rankings) > 1 and body_margin < CAM_ID_LOCK_MIN_BODY_MARGIN:
            print(
                f"[CAM LOCK BODY REJECT] "
                f"{tracker_key} "
                f"best={best_body_id}:"
                f"{best_body_score:.3f} "
                f"margin={body_margin:.3f} "
                f"reason=ambiguous"
            )
            return (None, None)
        candidates = []
        for global_id, info in camera_identity_locks.items():
            global_id = str(global_id)

            # =================================================
            # CRITICAL RULE:
            #
            # Spatial lock can ONLY transfer the identity which
            # OSNet independently ranked #1.
            # =================================================
            if global_id != best_body_id:
                continue

            # The lock exists to bridge a LOCAL TRACK CHANGE.
            if str(info.get("tracker_key")) == str(tracker_key):
                continue
            previous_bbox = info.get("bbox")
            metrics = camera_lock_metrics(current_bbox, previous_bbox)
            if metrics is None:
                continue
            if metrics["area_ratio"] < CAM_ID_LOCK_MIN_AREA_RATIO:
                continue
            geometry_ok = metrics["iou"] >= CAM_ID_LOCK_MIN_IOU or (
                metrics["center"] <= CAM_ID_LOCK_MAX_CENTER
                and metrics["containment"] >= CAM_ID_LOCK_MIN_CONTAINMENT
            )
            if not geometry_ok:
                continue
            metrics = dict(metrics)
            metrics["body_score"] = best_body_score
            metrics["body_margin"] = body_margin
            candidates.append((global_id, metrics))
        if not candidates:
            return (None, None)
        candidates.sort(key=lambda item: item[1]["score"], reverse=True)
        best_gid = candidates[0][0]
        best_metrics = candidates[0][1]
        return (best_gid, best_metrics)

    def refresh_camera_identity_lock(global_id, bbox, tracker_key, current_time):
        if global_id is None or bbox is None:
            return
        global_id = str(global_id)

        # This mechanism is only for anonymous global IDs.
        if not global_id.startswith("UNK-"):
            return
        prune_camera_identity_locks(current_time)
        camera_identity_locks[global_id] = {
            "bbox": list(bbox),
            "tracker_key": str(tracker_key),
            "time": float(current_time),
        }

    def get_cached_face_bbox(tracker_key, current_body_bbox, current_time):
        cached = last_face_boxes.get(tracker_key)
        if cached is None:
            return None
        if current_time - cached.get("time", 0) > FACE_BOX_TTL:
            return None
        old_face = cached.get("face_bbox")
        old_body = cached.get("body_bbox")
        if old_face is None or old_body is None or current_body_bbox is None:
            return None
        obx1, oby1, obx2, oby2 = [float(v) for v in old_body]
        fx1, fy1, fx2, fy2 = [float(v) for v in old_face]
        nbx1, nby1, nbx2, nby2 = [float(v) for v in current_body_bbox]
        old_w = max(obx2 - obx1, 1.0)
        old_h = max(oby2 - oby1, 1.0)
        new_w = max(nbx2 - nbx1, 1.0)
        new_h = max(nby2 - nby1, 1.0)

        # Face coordinates relative to old body box
        rx1 = (fx1 - obx1) / old_w
        ry1 = (fy1 - oby1) / old_h
        rx2 = (fx2 - obx1) / old_w
        ry2 = (fy2 - oby1) / old_h

        # Apply them to the current BoT-SORT body box
        new_fx1 = int(nbx1 + rx1 * new_w)
        new_fy1 = int(nby1 + ry1 * new_h)
        new_fx2 = int(nbx1 + rx2 * new_w)
        new_fy2 = int(nby1 + ry2 * new_h)
        if new_fx2 <= new_fx1 or new_fy2 <= new_fy1:
            return None
        return [new_fx1, new_fy1, new_fx2, new_fy2]

    # ========================================================
    # MAIN CAMERA LOOP
    # ========================================================
    while stream.running:
        packet = stream.read_packet(last_sequence, timeout=0.5)
        if packet is None:
            continue
        if last_sequence:
            dropped_frames += max(0, packet.sequence - last_sequence - 1)
        last_sequence = packet.sequence
        frame = packet.frame
        inference_started = time.perf_counter()
        face_due = time.monotonic() - last_face_inference >= config.FACE_INFERENCE_INTERVAL
        face_ms = 0.0
        frame_count += 1
        now = time.time()
        invalidate_inactive(recognizer, confirmed_ids, locked_ids)

        # =====================================================
        # CAMERA SESSION RESET AFTER PHYSICAL RECONNECT
        # =====================================================
        if packet.generation != camera_generation:
            camera_generation = packet.generation

            # Fresh YOLO object = fresh BoT-SORT tracker.
            body_detector.reset_camera(cam_id)

            # Drop temporary ReID track states only.
            if hasattr(reid, "reset_camera_tracks"):
                reid.reset_camera_tracks(cam_id)

            # Clear only local/session caches.
            # Persistent UNK galleries are untouched.
            for _state_name in (
                "locked_ids",
                "confirmed_ids",
                "confirmed_unknown_ids",
                "last_known_boxes",
                "last_face_boxes",
                "smoothed_boxes",
                "near_face_holds",
                "far_person_last_log",
                "recent_same_camera_identities",
                "camera_identity_locks",
                "gait_track_state",
                "pending_unknown_time",
                "unknown_last_sample",
                "last_result",
            ):
                _state = locals().get(_state_name)
                if hasattr(_state, "clear"):
                    _state.clear()
            print(f"[CAMERA SESSION RESET] " f"{cam_id} " f"generation={camera_generation}")

        # ====================================================
        # EXPIRE OLD TRACK LOCKS
        # ====================================================
        expired = [
            key for key, value in locked_ids.items() if (now - value.get("time", 0) > LOCK_EXPIRE)
        ]
        for key in expired:
            last_face_boxes.pop(key, None)
            locked_ids.pop(key, None)
            confirmed_ids.pop(key, None)
            confirmed_unknown_ids.pop(key, None)
            last_known_boxes.pop(key, None)
            smoothed_boxes.pop(key, None)

        # ====================================================
        # FRAME SKIPPING
        # ====================================================
        if frame_count % max(1, config.FRAME_SKIP) != 0:
            continue
        frame_bgr = frame
        stage_started = time.perf_counter()
        body_dets = body_detector.detect(frame_bgr, cam_id=cam_id, color_order="bgr")
        person_ms = (time.perf_counter() - stage_started) * 1000

        # ====================================================
        # SHORT DETECTION FLICKER FALLBACK
        # ====================================================
        if not body_dets:
            for tracker_key, info in list(confirmed_ids.items()):
                if not tracker_key.startswith(cam_id + "_"):
                    continue
                last = last_known_boxes.get(tracker_key)
                if last and (now - last["time"]) < 0.6:
                    body_dets.append(
                        {
                            "bbox": last["bbox"],
                            "confidence": 0.0,
                            "body_crop": None,
                            "face_crop": None,
                            "tracker_id": tracker_key.rsplit("_", 1)[-1],
                        }
                    )

        # ====================================================
        # RELEASE NAMES ON OVERLAPPING TRACKS
        #
        # When two people's boxes overlap, the tracker can swap
        # their IDs, and a cached name would follow the wrong
        # person. Drop a cached name unless this track's own face
        # matched it very recently, and hold the track as
        # "being confirmed" so it is neither named nor logged as
        # a new unknown until a face is seen again.
        # ====================================================
        overlap_boxes = {}
        overlap_identities = {}
        for bd in body_dets:
            if bd.get("tracker_id") is None or bd.get("bbox") is None:
                continue
            key = f"{cam_id}_{bd['tracker_id']}"
            named = confirmed_ids.get(key)
            if named is None and locked_ids.get(key, {}).get("is_known"):
                named = locked_ids[key]
            identity = named.get("student_id") if named else None
            # Weak, unidentified boxes (a chair, a coat) are not treated as
            # another person; only credible or identified tracks count.
            if (
                identity is None
                and key not in confirmed_unknown_ids
                and float(bd.get("confidence", 0.0) or 0.0) < NEW_UNK_MIN_BODY_CONF
            ):
                continue
            overlap_boxes[key] = bd["bbox"]
            overlap_identities[key] = identity
        for tracker_key in contested_track_keys(
            overlap_boxes, CONTESTED_MIN_OVERLAP, overlap_identities
        ):
            known = confirmed_ids.get(tracker_key)
            if known is None and locked_ids.get(tracker_key, {}).get("is_known"):
                known = locked_ids[tracker_key]
            if known is None or now - known.get("face_time", 0.0) <= KNOWN_FACE_FRESH_SECONDS:
                continue
            confirmed_ids.pop(tracker_key, None)
            locked_ids.pop(tracker_key, None)
            pending_identity_holds[tracker_key] = now + CONTESTED_IDENTITY_HOLD
            print(
                f"[ID CONTESTED] {tracker_key} released "
                f"{known.get('name')} after overlapping another person"
            )

        # ====================================================
        # BOT-SORT TRACK IDS COME DIRECTLY FROM BODY DETECTOR
        # ====================================================

        # ====================================================
        # SPLIT CONFIRMED / UNCONFIRMED PEOPLE
        # ====================================================
        need_arcface_regions = []
        confirmed_results = []
        for bd in body_dets:
            tid = bd.get("tracker_id")
            if tid is None:
                continue
            tracker_key = f"{cam_id}_{tid}"
            if bd.get("confidence", 0) > 0:
                last_known_boxes[tracker_key] = {"bbox": bd["bbox"], "time": now}

            # ------------------------------------------------
            # Already confirmed known student
            # ------------------------------------------------
            if tracker_key in confirmed_ids:
                cached = confirmed_ids[tracker_key]
                if cached.get("gallery_version", 0) < recognizer.gallery_version:
                    identity_id = cached.get("student_id")
                    if identity_id not in recognizer.known_embeddings:
                        confirmed_ids.pop(tracker_key, None)
                        locked_ids.pop(tracker_key, None)
                        need_arcface_regions.append((tid, tracker_key, bd))
                        continue
                    cached["gallery_version"] = recognizer.gallery_version
                if tracker_key in locked_ids:
                    locked_ids[tracker_key]["time"] = now
                confirmed_results.append(
                    {
                        "face_bbox": get_cached_face_bbox(tracker_key, bd["bbox"], now),
                        "body_bbox": bd["bbox"],
                        "bbox": bd["bbox"],
                        "is_known": True,
                        "student_id": cached["student_id"],
                        "role": cached.get("role", "student"),
                        "name": cached["name"],
                        "confidence": cached["confidence"],
                        "tracker_id": tid,
                        "global_id": cached["student_id"],
                        "cameras": [cam_id],
                        "_cached": True,
                    }
                )
            else:
                need_arcface_regions.append((tid, tracker_key, bd))

        # ====================================================
        # CONFIRMED UNKNOWN TRACK LOCK
        # Keep the body identity stable, but periodically retry face recognition
        # so enrollment can promote an existing unknown track to a known person.
        # ====================================================
        if need_arcface_regions:
            remaining_arcface_regions = []
            for tid, tracker_key, bd in need_arcface_regions:
                cached_unknown = confirmed_unknown_ids.get(tracker_key)
                if cached_unknown is None or cached_unknown.get("global_id") is None:
                    remaining_arcface_regions.append((tid, tracker_key, bd))
                    continue
                if face_due and needs_face_check(
                    cached_unknown, recognizer.gallery_version, now, UNKNOWN_RECHECK
                ):
                    cached_unknown["gallery_version"] = recognizer.gallery_version
                    cached_unknown["last_face_check"] = now
                    remaining_arcface_regions.append((tid, tracker_key, bd))
                    continue

                # --------------------------------------------
                # SAME BOT-SORT TRACK IS STILL ALIVE
                # --------------------------------------------
                global_id = cached_unknown.get("global_id")
                cached_unknown["time"] = now

                # Keep local display/identity lock alive too.
                if tracker_key in locked_ids:
                    locked_ids[tracker_key]["time"] = now
                    locked_ids[tracker_key]["name"] = global_id
                else:
                    locked_ids[tracker_key] = {
                        "is_known": False,
                        "student_id": None,
                        "name": global_id,
                        "confidence": 0.0,
                        "time": now,
                    }
                cameras = cached_unknown.get("cameras", [cam_id])
                if not cameras:
                    cameras = [cam_id]
                try:
                    cached_score = float(cached_unknown.get("score", 0.0) or 0.0)
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
                confirmed_results.append(
                    {
                        "face_bbox": get_cached_face_bbox(tracker_key, bd["bbox"], now),
                        "body_bbox": bd["bbox"],
                        "bbox": bd["bbox"],
                        "is_known": False,
                        "role": "unknown",
                        "student_id": None,
                        "name": global_id,
                        "confidence": 0.0,
                        "reid_confidence": cached_score,
                        "body_confidence": bd.get("confidence", 0.0),
                        "tracker_id": tid,
                        "local_tracker_id": str(tid),
                        "global_id": global_id,
                        "cameras": cameras,
                        # Existing downstream cached-result path
                        # will append this directly.
                        "_cached": True,
                    }
                )

            # Only tracks which DO NOT already have a locked
            # unknown identity continue to ArcFace.
            need_arcface_regions = remaining_arcface_regions

        # ====================================================
        # RUN ARCFACE
        # ====================================================
        arc_results = []
        if need_arcface_regions and face_due:
            last_face_inference = time.monotonic()
            stage_started = time.perf_counter()
            arc_results = recognizer.detect_and_recognize(
                frame_bgr, cam_id=cam_id, regions=[item[2]["bbox"] for item in need_arcface_regions]
            )
        face_ms = (
            (time.perf_counter() - stage_started) * 1000
            if need_arcface_regions and face_due
            else 0.0
        )

        # ====================================================
        # FACE/BODY MATCHING
        # ====================================================
        detections = list(confirmed_results)
        used_faces = set()

        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        # ====================================================
        # BUILD BODY/FACE PAIRS
        # ====================================================
        pair_scores = []
        for body_index, (tid, tracker_key, bd) in enumerate(need_arcface_regions):
            bx1, by1, bx2, by2 = bd["bbox"]
            face_region = (bx1, by1, bx2, by1 + (by2 - by1) * 0.6)
            for face_index, fd in enumerate(arc_results):
                fx1, fy1, fx2, fy2 = fd["bbox"]
                fcx = (fx1 + fx2) // 2
                fcy = (fy1 + fy2) // 2
                in_body = bx1 <= fcx <= bx2 and by1 <= fcy <= by1 + (by2 - by1) * 0.6
                if not in_body:
                    continue
                score = _iou(fd["bbox"], face_region)
                pair_scores.append((score, body_index, face_index))

        # ====================================================
        # GREEDY BIPARTITE ASSIGNMENT
        # ====================================================
        pair_scores.sort(key=lambda x: x[0], reverse=True)
        body_to_face = {}
        used_face_idx = set()
        used_body_idx = set()
        for score, body_index, face_index in pair_scores:
            if body_index in used_body_idx or face_index in used_face_idx:
                continue
            body_to_face[body_index] = face_index
            used_face_idx.add(face_index)
            used_body_idx.add(body_index)

        # ====================================================
        # PREVENT DUPLICATE KNOWN IDENTITY IN SAME FRAME
        # ====================================================
        claimed_student_ids = set()
        for i, (tid, tracker_key, bd) in enumerate(need_arcface_regions):
            matched_face = None
            face_index = body_to_face.get(i)
            if face_index is not None:
                fd = arc_results[face_index]
                sid = fd.get("student_id")
                if fd.get("is_known") and sid in claimed_student_ids:
                    matched_face = None
                else:
                    matched_face = (face_index, fd)
                    used_faces.add(face_index)
                    if fd.get("is_known") and sid:
                        claimed_student_ids.add(sid)

            # ------------------------------------------------
            # FACE MATCHED TO BODY
            # ------------------------------------------------
            if matched_face is not None:
                (face_index, fd) = matched_face
                last_face_boxes[tracker_key] = {
                    "face_bbox": list(fd["bbox"]),
                    "body_bbox": list(bd["bbox"]),
                    "time": now,
                }
                detections.append(
                    {
                        "face_bbox": fd["bbox"],
                        "body_bbox": bd["bbox"],
                        "bbox": bd["bbox"],
                        "is_known": fd["is_known"],
                        "student_id": fd["student_id"],
                        "role": fd.get("role", "student" if fd.get("is_known") else "unknown"),
                        "name": fd["name"],
                        "confidence": fd["confidence"],
                        "body_confidence": bd.get("confidence", 0.0),
                        "face_crop": fd.get("face_crop"),
                        "face_embedding": fd.get("embedding"),
                        "body_crop": bd.get("body_crop"),
                        "tracker_id": tid,
                        "face_confirmed": bool(fd.get("is_known")),
                        "identity_pending": update_pending_hold(
                            pending_identity_holds,
                            tracker_key,
                            bool(fd.get("identity_pending")),
                            now,
                            IDENTITY_PENDING_HOLD,
                        ),
                        "_cached": False,
                    }
                )

            # ------------------------------------------------
            # BODY EXISTS BUT FACE WAS NOT MATCHED
            # ------------------------------------------------
            else:
                prev = locked_ids.get(tracker_key, {})
                detections.append(
                    {
                        "face_bbox": get_cached_face_bbox(tracker_key, bd["bbox"], now),
                        "body_bbox": bd["bbox"],
                        "bbox": bd["bbox"],
                        "is_known": prev.get("is_known", False),
                        "student_id": prev.get("student_id"),
                        "role": prev.get("role", "unknown"),
                        "name": prev.get("name", "Unknown"),
                        "confidence": prev.get("confidence", 0.0),
                        "body_confidence": bd.get("confidence", 0.0),
                        "face_crop": None,
                        "body_crop": bd.get("body_crop"),
                        "tracker_id": tid,
                        # No face this frame: an active hold still applies.
                        "identity_pending": update_pending_hold(
                            pending_identity_holds,
                            tracker_key,
                            None,
                            now,
                            IDENTITY_PENDING_HOLD,
                        ),
                        "_cached": False,
                    }
                )

        # ====================================================
        # PROCESS IDENTITIES
        # ====================================================
        results = []
        for det in detections:
            tid = det.get("tracker_id")
            if tid is None:
                continue
            tracker_key = f"{cam_id}_{tid}"
            prev = locked_ids.get(tracker_key, {})

            # =================================================
            # ALREADY-CACHED KNOWN PERSON
            # =================================================
            if det.get("_cached"):
                results.append(det)
                continue

            # =================================================
            # KNOWN PERSON
            # =================================================
            if det["is_known"]:
                pending_unknown_time.pop(tracker_key, None)
                confirmed_unknown_ids.pop(tracker_key, None)
                previous_face_time = confirmed_ids.get(tracker_key, {}).get("face_time", 0.0)
                confirmed_ids[tracker_key] = {
                    "student_id": det["student_id"],
                    "name": det["name"],
                    "confidence": det["confidence"],
                    "role": det.get("role", "student"),
                    "gallery_version": recognizer.gallery_version,
                    # Last time this track's face itself matched the name.
                    "face_time": now if det.get("face_confirmed") else previous_face_time,
                }
                locked_ids[tracker_key] = {
                    "is_known": True,
                    "role": det.get("role", "student"),
                    "student_id": det["student_id"],
                    "name": det["name"],
                    "confidence": det["confidence"],
                    "time": now,
                }
                rec = {
                    "face_bbox": det["face_bbox"],
                    "body_bbox": det["body_bbox"],
                    "bbox": det["bbox"],
                    "is_known": True,
                    "role": det.get("role", "student"),
                    "student_id": det["student_id"],
                    "name": det["name"],
                    "confidence": det["confidence"],
                    "tracker_id": tid,
                    "local_tracker_id": str(tid),
                    "global_id": det["student_id"],
                    "cameras": [cam_id],
                }

            # =================================================
            # PREVIOUSLY KNOWN PERSON
            # =================================================
            elif prev.get("is_known"):
                locked_ids[tracker_key]["time"] = now
                rec = {
                    "face_bbox": det["face_bbox"],
                    "body_bbox": det["body_bbox"],
                    "bbox": det["bbox"],
                    "is_known": True,
                    "student_id": prev["student_id"],
                    "role": prev.get("role", "student"),
                    "name": prev["name"],
                    "confidence": prev["confidence"],
                    "tracker_id": tid,
                    "local_tracker_id": str(tid),
                    "global_id": prev["student_id"],
                    "cameras": [cam_id],
                }

            # =================================================
            # ENROLLED FACE STILL BEING CONFIRMED
            #
            # Not known yet, but not a stranger either: skip unknown
            # identity creation, anonymous gallery updates and logs.
            # A track already holding an UNK keeps it for display.
            # =================================================
            elif det.get("identity_pending"):
                cached_unk = confirmed_unknown_ids.get(tracker_key) or {}
                if cached_unk:
                    cached_unk["time"] = now
                held_global_id = cached_unk.get("global_id")
                rec = {
                    "face_bbox": det.get("face_bbox"),
                    "body_bbox": det.get("body_bbox"),
                    "bbox": det.get("bbox"),
                    "is_known": False,
                    "student_id": None,
                    "name": held_global_id or "Unknown",
                    "confidence": 0.0,
                    "reid_confidence": cached_unk.get("score", 0.0),
                    "tracker_id": tid,
                    "local_tracker_id": str(tid),
                    "global_id": held_global_id,
                    "cameras": cached_unk.get("cameras", [cam_id]),
                }

            # =================================================
            # UNKNOWN PERSON
            # =================================================
            else:
                cached_unk = confirmed_unknown_ids.get(tracker_key)

                # =============================================
                # HARD UNKNOWN TRACK LOCK
                #
                # Same behavior as known students:
                # while the SAME BoT-SORT tracker is alive,
                # do not recalculate its identity just because
                # the face disappears.
                # =============================================
                skip_reid = cached_unk is not None and cached_unk.get("global_id") is not None
                if skip_reid:

                    # Refresh continuously while the person's
                    # body remains tracked.
                    cached_unk["time"] = now
                body_crop = det.get("body_crop")
                face_embedding = det.get("face_embedding")
                body_confidence = float(det.get("body_confidence", 0.0) or 0.0)
                body_emb = None
                global_id = None
                reid_score = 0.0
                reid_cameras = [cam_id]

                # =============================================
                # USE RECENT CONFIRMED UNKNOWN CACHE
                # =============================================
                if skip_reid:
                    global_id = cached_unk["global_id"]
                    reid_score = cached_unk.get("score", 0.0)
                    reid_cameras = cached_unk.get("cameras", [cam_id])

                # =============================================
                # RUN OSNET / GLOBAL RE-ID
                # =============================================
                else:
                    allow_body_reid = (
                        face_embedding is not None or body_confidence >= NEW_UNK_MIN_BODY_CONF
                    )
                    if not allow_body_reid:
                        print(
                            f"[REID PERSON GATE] "
                            f"{tracker_key} "
                            f"blocked weak body-only detection "
                            f"conf={body_confidence:.3f}"
                        )
                    if body_crop is not None and allow_body_reid:
                        try:
                            body_emb = body_reid.get_embedding(body_crop)
                        except Exception as e:
                            print(f"[BODY REID ERROR] " f"{tracker_key}: " f"{e}")
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
                    current_body_bbox = det.get("body_bbox")
                    if current_body_bbox is None:
                        current_body_bbox = det.get("bbox")
                    if (
                        global_id is None
                        and face_embedding is None
                        and current_body_bbox is not None
                        and body_confidence >= CAM_ID_LOCK_MIN_BODY_CONF
                    ):
                        (lock_global_id, lock_metrics) = match_camera_identity_lock(
                            tracker_key, current_body_bbox, now, body_emb
                        )
                        if lock_global_id is not None:
                            try:
                                lock_bound = reid.bind_track_identity(
                                    track_key=tracker_key,
                                    global_id=lock_global_id,
                                    camera_id=cam_id,
                                    body_embedding=body_emb,
                                    bbox=current_body_bbox,
                                )
                            except Exception as e:
                                print(f"[CAM LOCK ERROR] " f"{tracker_key}: " f"{e}")
                                lock_bound = False
                            if lock_bound:
                                global_id = lock_global_id
                                reid_score = float(lock_metrics["score"])
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
                                    f"{lock_metrics['area_ratio']:.3f} "
                                    f"body="
                                    f"{lock_metrics.get('body_score', 0.0):.3f} "
                                    f"margin="
                                    f"{lock_metrics.get('body_margin', 0.0):.3f}"
                                )

                    # =========================================
                    # ANONYMOUS FACE RE-ID
                    #
                    # Face is the strongest signal for an
                    # already-known UNKNOWN identity.
                    # =========================================
                    current_body_bbox = det.get("body_bbox")
                    if current_body_bbox is None:
                        current_body_bbox = det.get("bbox")

                    # =================================
                    # FAR PERSON QUALITY CHECK
                    #
                    # Use bbox height relative to the complete
                    # frame rather than an absolute pixel size.
                    # =================================
                    far_person = False
                    body_height_ratio = 1.0
                    try:
                        if current_body_bbox is not None:
                            (far_x1, far_y1, far_x2, far_y2) = [float(v) for v in current_body_bbox]
                            frame_height = float(frame_bgr.shape[0])
                            body_height = max(0.0, far_y2 - far_y1)
                            if frame_height > 0:
                                body_height_ratio = body_height / frame_height
                                far_person = body_height_ratio < FAR_PERSON_MIN_HEIGHT_RATIO
                    except Exception:
                        far_person = False
                        body_height_ratio = 1.0
                    near_face_hold_active = False
                    near_face_hold_candidate = None

                    # Existing hold can survive a brief
                    # missed-face frame.
                    previous_face_hold = near_face_holds.get(tracker_key)
                    if previous_face_hold is not None:
                        hold_age = now - float(previous_face_hold.get("start_time", now))
                        hold_stale = now - float(previous_face_hold.get("last_seen", 0.0))
                        if (
                            hold_age <= ANON_FACE_HOLD_SECONDS
                            and hold_stale <= ANON_FACE_HOLD_STALE_SECONDS
                        ):
                            near_face_hold_active = True
                            near_face_hold_candidate = previous_face_hold.get("candidate_id")
                        else:
                            near_face_holds.pop(tracker_key, None)

                    # =================================
                    # SAME-CAMERA TRACK CONTINUITY BRIDGE
                    #
                    # Runs only when:
                    #
                    # 1. current local tracker has NO identity
                    # 2. no usable face is available
                    # 3. no near-face hold is active
                    # 4. OSNet strongly prefers one identity
                    # 5. that SAME identity was just seen on
                    #    THIS camera
                    # 6. position/scale remain plausible
                    #
                    # Multiple-person safe:
                    #
                    # we do NOT simply inherit the closest
                    # global ID.
                    # =================================
                    if (
                        global_id is None
                        and face_embedding is None
                        and not near_face_hold_active
                        and emb_valid(body_emb)
                        and current_body_bbox is not None
                    ):
                        try:
                            continuity_rankings = reid._rank_gallery(body_emb)
                        except Exception:
                            continuity_rankings = []
                        if continuity_rankings:
                            try:
                                continuity_best_id = str(continuity_rankings[0][0])
                                continuity_best_score = float(continuity_rankings[0][1])
                                continuity_second_score = (
                                    float(continuity_rankings[1][1])
                                    if len(continuity_rankings) > 1
                                    else 0.0
                                )
                                continuity_margin = continuity_best_score - continuity_second_score
                            except Exception:
                                continuity_best_id = None
                                continuity_best_score = 0.0
                                continuity_margin = 0.0
                            recent_identity = (
                                recent_same_camera_identities.get(continuity_best_id)
                                if continuity_best_id is not None
                                else None
                            )
                            if recent_identity is not None:
                                continuity_age = now - float(recent_identity.get("time", 0.0))
                                previous_tracker_key = recent_identity.get("tracker_key")
                                previous_bbox = recent_identity.get("bbox")
                                (continuity_center, continuity_area) = (
                                    same_camera_continuity_geometry(
                                        previous_bbox, current_body_bbox
                                    )
                                    if previous_bbox is not None
                                    else (999.0, 0.0)
                                )
                                continuity_pass = (
                                    continuity_age <= SAME_CAM_CONTINUITY_TTL
                                    and previous_tracker_key != tracker_key
                                    and continuity_best_score >= SAME_CAM_CONTINUITY_MIN_BODY_SCORE
                                    and continuity_margin >= SAME_CAM_CONTINUITY_MIN_BODY_MARGIN
                                    and continuity_center <= SAME_CAM_CONTINUITY_MAX_CENTER
                                    and continuity_area >= SAME_CAM_CONTINUITY_MIN_AREA_RATIO
                                )
                                if continuity_pass:
                                    try:
                                        continuity_bound = reid.bind_track_identity(
                                            track_key=tracker_key,
                                            global_id=continuity_best_id,
                                            camera_id=cam_id,
                                            body_embedding=body_emb,
                                            bbox=current_body_bbox,
                                        )
                                    except Exception as e:
                                        continuity_bound = False
                                        print(f"[REID CONTINUITY ERROR] " f"{tracker_key}: {e}")
                                    if continuity_bound:
                                        global_id = continuity_best_id
                                        reid_score = continuity_best_score
                                        cams = reid.get_cameras_for(global_id)
                                        reid_cameras = list(cams) if cams else [cam_id]
                                        print(
                                            f"[REID CONTINUITY BIND] "
                                            f"{tracker_key} "
                                            f"-> {global_id} "
                                            f"body="
                                            f"{continuity_best_score:.3f} "
                                            f"margin="
                                            f"{continuity_margin:.3f} "
                                            f"age="
                                            f"{continuity_age:.2f}s "
                                            f"center="
                                            f"{continuity_center:.3f} "
                                            f"area="
                                            f"{continuity_area:.3f}"
                                        )
                    if face_embedding is not None:
                        try:
                            (face_global_id, face_reid_score, face_reid_margin) = (
                                anonymous_face_reid.match(face_embedding)
                            )
                            (raw_face_id, raw_face_score, raw_face_margin) = (
                                anonymous_face_reid.best_candidate(face_embedding)
                            )

                            # =================================
                            # STRICT MATCH WON
                            # =================================
                            if face_global_id is not None:
                                near_face_holds.pop(tracker_key, None)
                                near_face_hold_active = False
                                near_face_hold_candidate = None
                                face_bound = reid.bind_track_identity(
                                    track_key=tracker_key,
                                    global_id=face_global_id,
                                    camera_id=cam_id,
                                    body_embedding=(body_emb if emb_valid(body_emb) else None),
                                    bbox=current_body_bbox,
                                )
                                if face_bound:
                                    global_id = face_global_id
                                    reid_score = face_reid_score
                                    cams = reid.get_cameras_for(global_id)
                                    reid_cameras = list(cams) if cams else [cam_id]
                                    margin_text = (
                                        "n/a"
                                        if face_reid_margin is None
                                        else f"{face_reid_margin:.3f}"
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
                            else:

                                # =================================
                                # NEAR-FACE HOLD
                                #
                                # This does NOT assign raw_face_id.
                                #
                                # It only prevents ReIDManager from
                                # creating another global UNK for a
                                # few seconds while we wait for a
                                # clearer face frame.
                                # =================================
                                margin_ok = (
                                    raw_face_margin is None
                                    or raw_face_margin >= ANON_FACE_HOLD_MIN_MARGIN
                                )
                                strong_score = raw_face_score >= ANON_FACE_HOLD_STRONG_SCORE
                                near_score = raw_face_score >= ANON_FACE_HOLD_MIN_SCORE
                                qualifies_for_hold = (
                                    raw_face_id is not None
                                    and near_score
                                    and (margin_ok or strong_score)
                                )
                                if qualifies_for_hold:
                                    hold = near_face_holds.get(tracker_key)

                                    # New candidate or previous
                                    # state became stale -> restart
                                    # the bounded timer.
                                    if (
                                        hold is None
                                        or hold.get("candidate_id") != raw_face_id
                                        or (
                                            now - float(hold.get("last_seen", 0.0))
                                            > ANON_FACE_HOLD_STALE_SECONDS
                                        )
                                    ):
                                        hold = {
                                            "candidate_id": raw_face_id,
                                            "start_time": now,
                                            "last_seen": now,
                                            "max_score": raw_face_score,
                                            "last_log": 0.0,
                                        }
                                        near_face_holds[tracker_key] = hold
                                    else:
                                        hold["last_seen"] = now
                                        hold["max_score"] = max(
                                            float(hold.get("max_score", 0.0)), raw_face_score
                                        )
                                    hold_age = now - float(hold["start_time"])
                                    if hold_age <= ANON_FACE_HOLD_SECONDS:
                                        near_face_hold_active = True
                                        near_face_hold_candidate = raw_face_id
                                        if now - float(hold.get("last_log", 0.0)) >= 0.75:
                                            hold["last_log"] = now
                                            hold_margin_text = (
                                                "n/a"
                                                if raw_face_margin is None
                                                else f"{raw_face_margin:.3f}"
                                            )
                                            print(
                                                f"[ANON FACE HOLD] "
                                                f"{tracker_key} "
                                                f"candidate="
                                                f"{raw_face_id} "
                                                f"score="
                                                f"{raw_face_score:.3f} "
                                                f"margin="
                                                f"{hold_margin_text} "
                                                f"age="
                                                f"{hold_age:.1f}/"
                                                f"{ANON_FACE_HOLD_SECONDS:.1f}s"
                                            )
                                    else:
                                        near_face_hold_active = False
                                        near_face_hold_candidate = None
                                        print(
                                            f"[ANON FACE HOLD EXPIRED] "
                                            f"{tracker_key} "
                                            f"candidate="
                                            f"{raw_face_id} "
                                            f"max_score="
                                            f"{float(hold.get('max_score', 0.0)):.3f}"
                                        )
                                        near_face_holds.pop(tracker_key, None)

                                # Existing original diagnostic
                                # output remains below.
                                if face_reid_score > 0.0:
                                    margin_text = (
                                        "n/a"
                                        if face_reid_margin is None
                                        else f"{face_reid_margin:.3f}"
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
                            print(f"[ANON FACE ERROR] " f"{tracker_key}: " f"{e}")
                    if global_id is None and near_face_hold_active:

                        # No assignment here.
                        #
                        # Keep the current person unresolved
                        # rather than creating a duplicate UNK.
                        pass
                    if global_id is None and far_person:
                        previous_far_log = float(far_person_last_log.get(tracker_key, 0.0))
                        if now - previous_far_log >= 1.0:
                            far_person_last_log[tracker_key] = now
                            print(
                                f"[REID FAR HOLD] "
                                f"{tracker_key} "
                                f"height_ratio="
                                f"{body_height_ratio:.3f} "
                                f"conf="
                                f"{body_confidence:.3f} "
                                f"no_new_id"
                            )

                    # =========================================
                    # FALL BACK TO OSNET / BODY RE-ID
                    #
                    # Only run body identity creation if
                    # anonymous face did NOT already restore
                    # a global UNK identity.
                    # =========================================
                    if (
                        global_id is None
                        and not near_face_hold_active
                        and not far_person
                        and emb_valid(body_emb)
                    ):
                        try:
                            snap = None
                            if body_crop is not None:
                                snap = cv2.cvtColor(body_crop, cv2.COLOR_RGB2BGR)

                            # =================================
                            # IMPORTANT:
                            # Uses new ReIDManager API
                            # =================================
                            current_body_bbox = det.get("body_bbox")
                            if current_body_bbox is None:
                                current_body_bbox = det.get("bbox")
                            (global_id, reid_score) = reid.find_or_create(
                                embedding=body_emb,
                                camera_id=cam_id,
                                track_key=tracker_key,
                                frame=snap,
                                bbox=current_body_bbox,
                            )
                            if global_id is not None:
                                cams = reid.get_cameras_for(global_id)
                                reid_cameras = list(cams) if cams else [cam_id]
                                print(
                                    f"[REID RESULT] "
                                    f"{tracker_key} "
                                    f"-> "
                                    f"{global_id} "
                                    f"score="
                                    f"{reid_score:.3f}"
                                )
                        except Exception as e:
                            print(f"[REID ERROR] " f"{tracker_key}: " f"{e}")
                            global_id = None
                            reid_score = 0.0
                            reid_cameras = [cam_id]

                    # =========================================
                    # ONLY CACHE CONFIRMED GLOBAL IDS
                    # =========================================
                    if global_id is not None:
                        confirmed_unknown_ids[tracker_key] = {
                            "global_id": global_id,
                            "cameras": reid_cameras,
                            "score": reid_score,
                            "time": now,
                            "last_face_check": now,
                            "gallery_version": recognizer.gallery_version,
                        }

                # =============================================
                # PERSIST ANONYMOUS FACE FOR CONFIRMED UNK
                # =============================================
                if global_id is not None and face_embedding is not None:
                    try:

                        # If this track is using the short-term
                        # unknown cache, only add the new face
                        # sample when ArcFace itself agrees with
                        # the cached global identity.
                        existing_face_samples = anonymous_face_reid.get_sample_count(global_id)

                        # A brand-new global UNK has no face
                        # gallery yet, so its first sample
                        # is allowed.
                        if existing_face_samples == 0:
                            allow_face_add = True

                        # Existing UNK: only save the face if
                        # ArcFace independently agrees that it
                        # belongs to this global identity.
                        else:
                            (verified_face_id, verified_face_score, verified_face_margin) = (
                                anonymous_face_reid.match(face_embedding)
                            )
                            allow_face_add = verified_face_id == global_id
                        if allow_face_add:
                            anonymous_face_reid.add(global_id, face_embedding)
                    except Exception as e:
                        print(f"[ANON FACE SAVE ERROR] " f"{tracker_key}: " f"{e}")

                # =============================================
                # DISPLAY UNKNOWN UNTIL GLOBAL ID CONFIRMED
                # =============================================
                display_name = global_id if (global_id is not None) else "Unknown"

                # =============================================
                # UPDATE LOCAL IDENTITY LOCK
                # =============================================
                locked_ids[tracker_key] = {
                    "is_known": False,
                    "student_id": None,
                    "name": display_name,
                    "confidence": 0.0,
                    "time": now,
                }

                # =============================================
                # HIDE REJECTED WEAK DETECTIONS
                #
                # The detector runs at a low threshold so distant
                # people keep their tracks. A track with no face, no
                # identity and a body score below the new-identity
                # gate was rejected above; do not publish it as a
                # person box. Tracking state is unchanged.
                # =============================================
                if is_rejected_weak_detection(
                    global_id, face_embedding, body_confidence, NEW_UNK_MIN_BODY_CONF
                ):
                    continue

                # =============================================
                # BUILD UNKNOWN RESULT
                # =============================================
                rec = {
                    "face_bbox": det.get("face_bbox"),
                    "body_bbox": det.get("body_bbox"),
                    "bbox": det.get("bbox"),
                    "is_known": False,
                    "student_id": None,
                    "name": display_name,
                    "confidence": 0.0,
                    "reid_confidence": reid_score,
                    # Local DeepSORT ID
                    "tracker_id": tid,
                    "local_tracker_id": str(tid),
                    # Persistent identity
                    "global_id": global_id,
                    "cameras": reid_cameras,
                    # Read-only, for the expired-visitor watch list.
                    "face_embedding": face_embedding,
                }

                # =============================================
                # CROSS-CAMERA EVENT
                # =============================================
                if global_id is not None and len(reid_cameras) > 1:
                    cross_key = global_id + "_" + "_".join(sorted(reid_cameras))
                    with cross_cam_lock:
                        should_fire_cross = cross_key not in cross_cam_fired
                        if should_fire_cross:
                            cross_cam_fired.add(cross_key)
                    if should_fire_cross:

                        def send_cross_camera(gid, cameras):
                            try:
                                requests.post(
                                    (f"{config.NODE_API_URL}" f"/api/cross_camera"),
                                    json={"global_id": gid, "cameras": cameras},
                                    timeout=1.0,
                                )
                            except Exception as e:
                                print("[CROSS CAM ERROR] " f"{e}")

                        threading.Thread(
                            target=send_cross_camera, args=(global_id, reid_cameras), daemon=True
                        ).start()

            # =================================================
            # LOGGING
            # =================================================
            log_uid = rec.get("student_id") or rec.get("global_id")

            # =================================================
            # KNOWN PERSON LOGGING
            # =================================================
            if rec["is_known"] and log_uid:
                log_key = (str(log_uid), str(cam_id))
                with log_lock:
                    should_log = log_key not in global_logged_students
                    if should_log:
                        global_logged_students.add(log_key)
                    seen_on_other_camera = any(
                        identity == str(log_uid) and camera != str(cam_id)
                        for identity, camera in global_logged_students
                    )
                if should_log:
                    payload = {
                        "camera_id": cam_id,
                        "person_id": rec.get("student_id"),
                        "global_person_id": rec.get("student_id"),
                        "student_id": rec.get("student_id"),
                        "student_name": rec.get("name", "Unknown"),
                        "person_name": rec.get("name", "Unknown"),
                        "role": rec.get("role", "student"),
                        "is_known": True,
                        "confidence": round(rec.get("confidence", 0), 4),
                        "reid_confidence": 0.0,
                        "tracker_id": str(tid),
                        "local_tracker_id": str(tid),
                        "event_type": "camera_transition" if seen_on_other_camera else "entered",
                        "snapshot_path": None,
                    }
                    threading.Thread(target=post_detection, args=(payload,), daemon=True).start()

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
            elif not rec["is_known"] and log_uid:
                log_key = (str(log_uid), str(cam_id))
                with log_lock:
                    should_log = log_key not in global_logged_unknowns
                    if should_log:
                        global_logged_unknowns.add(log_key)
                    seen_on_other_camera = any(
                        identity == str(log_uid) and camera != str(cam_id)
                        for identity, camera in global_logged_unknowns
                    )
                if should_log:
                    snap_path = None
                    try:
                        snap_path = reid.get_snapshot_path(log_uid)
                    except Exception:
                        pass
                    payload = {
                        "camera_id": cam_id,
                        "person_id": log_uid,
                        "global_person_id": log_uid,
                        "student_id": None,
                        "student_name": log_uid,
                        "person_name": log_uid,
                        "role": "unknown",
                        "is_known": False,
                        "confidence": 0.0,
                        "reid_confidence": round(rec.get("reid_confidence", 0.0), 4),
                        # Local DeepSORT ID
                        "tracker_id": str(tid),
                        "local_tracker_id": str(tid),
                        "event_type": "camera_transition" if seen_on_other_camera else "first_seen",
                        "snapshot_path": snap_path,
                    }
                    print(f"[LOG UNKNOWN] " f"{log_uid} " f"camera={cam_id} " f"track={tid}")
                    threading.Thread(target=post_detection, args=(payload,), daemon=True).start()

            # =================================================
            # APPEND RESULT
            # =================================================
            results.append(rec)

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

            def result_bbox(item):
                bbox = item.get("body_bbox")
                if bbox is None:
                    bbox = item.get("bbox")
                return bbox

            def bbox_area(bbox):
                if bbox is None or len(bbox) != 4:
                    return 0.0
                try:
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
                except Exception:
                    return 0.0

            def containment_overlap(box_a, box_b):
                """
                Intersection / smaller-box area.

                Better than normal IoU for detecting a small
                duplicate box sitting inside a larger body box.
                """
                if box_a is None or box_b is None:
                    return 0.0
                try:
                    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
                    bx1, by1, bx2, by2 = [float(v) for v in box_b]
                    ix1 = max(ax1, bx1)
                    iy1 = max(ay1, by1)
                    ix2 = min(ax2, bx2)
                    iy2 = min(ay2, by2)
                    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    area_a = bbox_area(box_a)
                    area_b = bbox_area(box_b)
                    smaller = min(area_a, area_b)
                    if smaller <= 0.0:
                        return 0.0
                    return float(intersection / smaller)
                except Exception:
                    return 0.0

            def result_quality(item):
                score = 0.0

                # Registered student is strongest.
                if item.get("student_id") is not None:
                    score += 100.0

                # Confirmed global unknown.
                if item.get("global_id") is not None:
                    score += 50.0

                # Prefer observation with an actual face.
                if item.get("face_bbox") is not None:
                    score += 20.0
                try:
                    score += float(item.get("reid_confidence", 0.0) or 0.0) * 10.0
                except Exception:
                    pass

                # Prefer fuller/larger body observation when
                # everything else is equal.
                area = bbox_area(result_bbox(item))
                score += min(area / 100000.0, 5.0)
                return score

            # ------------------------------------------------
            # PASS 1:
            # One result per stable student/global identity
            # ------------------------------------------------
            identity_results = []
            identity_positions = {}
            for item in results:
                student_id = item.get("student_id")
                global_id = item.get("global_id")
                if student_id is not None:
                    stable_key = ("student", str(student_id))
                elif global_id is not None:
                    stable_key = ("global", str(global_id))
                else:
                    stable_key = None
                if stable_key is None:
                    identity_results.append(item)
                    continue
                existing_index = identity_positions.get(stable_key)
                if existing_index is None:
                    identity_positions[stable_key] = len(identity_results)
                    identity_results.append(item)
                    continue
                existing = identity_results[existing_index]
                if result_quality(item) > result_quality(existing):
                    identity_results[existing_index] = item

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
                for item in identity_results
                if (item.get("student_id") is not None or item.get("global_id") is not None)
            ]
            final_results = []
            for item in identity_results:
                is_confirmed = (
                    item.get("student_id") is not None or item.get("global_id") is not None
                )
                if is_confirmed:
                    final_results.append(item)
                    continue
                current_bbox = result_bbox(item)
                overlaps_confirmed = False
                for confirmed_item in confirmed_results:
                    overlap = containment_overlap(current_bbox, result_bbox(confirmed_item))
                    if overlap >= 0.70:
                        overlaps_confirmed = True
                        break
                if overlaps_confirmed:
                    continue
                final_results.append(item)
            if len(final_results) < len(results):
                print(f"[RESULT DEDUP] " f"{cam_id} " f"{len(results)} -> " f"{len(final_results)}")
            results = final_results

            # =================================================
            # REFRESH SAME-CAMERA CONTINUITY MEMORY
            #
            # Only confirmed anonymous identities enter this
            # memory.
            #
            # Known students are not handled here.
            # =================================================
            for continuity_item in results:
                if continuity_item.get("is_known", False):
                    continue
                continuity_gid = continuity_item.get("global_id")
                if continuity_gid is None:
                    continue
                continuity_gid = str(continuity_gid)
                if not continuity_gid.startswith("UNK-"):
                    continue
                continuity_tid = continuity_item.get("tracker_id")
                if continuity_tid is None:
                    continue
                continuity_bbox = continuity_item.get("body_bbox") or continuity_item.get("bbox")
                if continuity_bbox is None:
                    continue
                recent_same_camera_identities[continuity_gid] = {
                    "bbox": list(continuity_bbox),
                    "tracker_key": f"{cam_id}_{continuity_tid}",
                    "time": now,
                }

            # Remove stale continuity memories.
            stale_continuity_ids = [
                gid
                for gid, value in recent_same_camera_identities.items()
                if (now - float(value.get("time", 0.0)) > (SAME_CAM_CONTINUITY_TTL + 1.0))
            ]
            for stale_gid in stale_continuity_ids:
                recent_same_camera_identities.pop(stale_gid, None)

            # =================================================
            # PASSIVE GAIT OBSERVATION
            #
            # This executes AFTER final identity dedup.
            #
            # It NEVER writes to:
            #
            #     global_id
            #     locked_ids
            #     confirmed_unknown_ids
            #     ReIDManager
            #     anonymous_face_reid
            #
            # It is diagnostic only.
            # =================================================
            if GAIT_OBSERVATION_ENABLED:
                gait_candidates = []

                # ---------------------------------------------
                # USE CONFIRMED UNKNOWN PEOPLE ONLY
                # ---------------------------------------------
                for gait_item in results:
                    if gait_item.get("is_known", False):
                        continue
                    gait_gid = gait_item.get("global_id")
                    if gait_gid is None:
                        continue
                    gait_gid = str(gait_gid)
                    if not gait_gid.startswith("UNK-"):
                        continue
                    gait_tid = gait_item.get("tracker_id")
                    if gait_tid is None:
                        continue
                    gait_bbox = gait_item.get("body_bbox") or gait_item.get("bbox")
                    if gait_bbox is None:
                        continue
                    gait_candidates.append(
                        {
                            "item": gait_item,
                            "global_id": gait_gid,
                            "tracker_id": gait_tid,
                            "tracker_key": f"{cam_id}_{gait_tid}",
                            "bbox": gait_bbox,
                        }
                    )

                # ---------------------------------------------
                # CLEAN ABANDONED TRACK BUFFERS
                # ---------------------------------------------
                stale_gait_keys = [
                    key
                    for key, value in gait_track_state.items()
                    if (now - float(value.get("last_seen", 0.0)) > GAIT_TRACK_TTL)
                ]
                for stale_key in stale_gait_keys:
                    gait_track_state.pop(stale_key, None)

                # ---------------------------------------------
                # THROTTLED POSE INFERENCE
                # ---------------------------------------------
                should_run_gait_pose = (
                    bool(gait_candidates) and (now - gait_last_pose_run) >= GAIT_POSE_INTERVAL
                )
                if should_run_gait_pose:
                    gait_last_pose_run = now

                    # -----------------------------------------
                    # LAZY MODEL INITIALIZATION
                    # -----------------------------------------
                    if not gait_init_attempted:
                        gait_init_attempted = True
                        try:
                            gait_pose = GaitPoseExtractor()
                            gait_signature = GaitSignatureEngine(
                                gallery_dir=("data/" "gait_gallery_" "runtime_observation"),
                                sequence_length=(GAIT_SEQUENCE_FRAMES),
                            )
                            print(f"[GAIT OBS MODE] " f"{cam_id} ready")
                        except Exception as e:
                            gait_pose = None
                            gait_signature = None
                            print(f"[GAIT OBS INIT ERROR] " f"{cam_id}: {e}")

                    # -----------------------------------------
                    # POSE EXTRACTION
                    # -----------------------------------------
                    gait_people = []
                    if gait_pose is not None and gait_signature is not None:
                        try:
                            gait_people = gait_pose.extract(frame_bgr)
                        except Exception as e:
                            print(f"[GAIT OBS ERROR] " f"{cam_id}: {e}")
                            gait_people = []

                    # -----------------------------------------
                    # ONE-TO-ONE BODY <-> POSE ASSOCIATION
                    #
                    # Important when multiple people exist.
                    #
                    # One pose cannot be assigned to two
                    # BoT-SORT results.
                    # -----------------------------------------
                    gait_pairs = []
                    for candidate_index, candidate in enumerate(gait_candidates):
                        for pose_index, pose_person in enumerate(gait_people):
                            pose_bbox = pose_person.get("bbox")
                            if pose_bbox is None:
                                continue
                            overlap = gait_bbox_iou(candidate["bbox"], pose_bbox)
                            if overlap >= GAIT_MIN_ASSOC_IOU:
                                gait_pairs.append((overlap, candidate_index, pose_index))
                    gait_pairs.sort(key=lambda value: value[0], reverse=True)
                    used_gait_candidates = set()
                    used_gait_poses = set()
                    for association_iou, candidate_index, pose_index in gait_pairs:
                        if candidate_index in used_gait_candidates:
                            continue
                        if pose_index in used_gait_poses:
                            continue
                        used_gait_candidates.add(candidate_index)
                        used_gait_poses.add(pose_index)
                        candidate = gait_candidates[candidate_index]
                        pose_person = gait_people[pose_index]
                        tracker_key = candidate["tracker_key"]
                        global_id = candidate["global_id"]

                        # -------------------------------------
                        # TRACK STATE
                        # -------------------------------------
                        state = gait_track_state.get(tracker_key)
                        if state is None:
                            state = {
                                "global_id": global_id,
                                "gid_since": now,
                                "features": [],
                                "sample_times": [],
                                "last_sample": 0.0,
                                "last_seen": now,
                            }
                            gait_track_state[tracker_key] = state

                        # -------------------------------------
                        # IF CURRENT SYSTEM CHANGES THE ID,
                        # DISCARD THE PARTIAL GAIT SEQUENCE.
                        #
                        # Prevents:
                        #
                        # UNK-001 frames
                        # +
                        # temporary UNK-002 frames
                        #
                        # being mixed together.
                        # -------------------------------------
                        if state.get("global_id") != global_id:
                            print(
                                f"[GAIT OBS RESET] "
                                f"{tracker_key} "
                                f"{state.get('global_id')} "
                                f"-> {global_id}"
                            )
                            state["global_id"] = global_id
                            state["gid_since"] = now
                            state["features"] = []
                            state["sample_times"] = []
                            state["last_sample"] = 0.0
                        state["last_seen"] = now

                        # -------------------------------------
                        # CREATE ONE FRAME GAIT FEATURE
                        # -------------------------------------
                        try:
                            gait_feature = gait_signature.frame_feature(
                                pose_person["keypoints"], pose_person["bbox"]
                            )
                        except Exception:
                            gait_feature = None
                        if gait_feature is None:
                            continue
                        if now - float(state.get("last_sample", 0.0)) < GAIT_SAMPLE_INTERVAL:
                            continue
                        state["last_sample"] = now
                        state["features"].append(gait_feature)
                        state.setdefault("sample_times", []).append(now)
                        feature_count = len(state["features"])

                        # -------------------------------------
                        # SMALL PROGRESS LOG
                        # -------------------------------------
                        if feature_count in (8, 16, 24):
                            print(
                                f"[GAIT COLLECT] "
                                f"{tracker_key} "
                                f"{global_id} "
                                f"{feature_count}/"
                                f"{GAIT_SEQUENCE_FRAMES}"
                            )
                        if feature_count < GAIT_SEQUENCE_FRAMES:
                            continue

                        # -------------------------------------
                        # CURRENT GLOBAL ID MUST HAVE BEEN
                        # STABLE LONG ENOUGH
                        # -------------------------------------
                        identity_age = now - float(state.get("gid_since", now))
                        if identity_age < GAIT_ID_STABLE_TIME:

                            # Keep latest frames, but don't
                            # write a signature yet.
                            state["features"] = state["features"][-GAIT_SEQUENCE_FRAMES:]
                            state["sample_times"] = state.get("sample_times", [])[
                                -GAIT_SEQUENCE_FRAMES:
                            ]
                            continue

                        # -------------------------------------
                        # BUILD GAIT SIGNATURE
                        # -------------------------------------
                        try:
                            signature = gait_signature.build_signature(
                                state["features"][-GAIT_SEQUENCE_FRAMES:],
                                timestamps=state.get("sample_times", [])[-GAIT_SEQUENCE_FRAMES:],
                            )
                        except Exception as e:
                            print(f"[GAIT SIGNATURE ERROR] " f"{tracker_key}: {e}")
                            signature = None

                        # Start a fresh non-overlapping
                        # walking sequence.
                        state["features"] = []
                        state["sample_times"] = []
                        if signature is None:
                            continue

                        # -------------------------------------
                        # MATCH EXISTING GAIT OBSERVATIONS
                        # BEFORE SAVING CURRENT SAMPLE
                        # -------------------------------------
                        try:
                            (gait_best, gait_score, gait_margin) = gait_signature.match(signature)
                        except Exception as e:
                            print(f"[GAIT MATCH ERROR] " f"{tracker_key}: {e}")
                            gait_best = None
                            gait_score = 0.0
                            gait_margin = 0.0
                        if gait_best is None:
                            agreement = "N/A"
                            best_display = "none"
                        else:
                            agreement = "YES" if str(gait_best) == str(global_id) else "NO"
                            best_display = str(gait_best)
                        print(
                            f"[GAIT OBSERVE] "
                            f"{tracker_key} "
                            f"assigned={global_id} "
                            f"best={best_display} "
                            f"score={gait_score:.3f} "
                            f"margin={gait_margin:.3f} "
                            f"agree={agreement} "
                            f"iou={association_iou:.3f}"
                        )

                        # -------------------------------------
                        # OBSERVATION GALLERY ONLY
                        #
                        # This is intentionally separate from:
                        #
                        # data/reid_gallery
                        # data/anonymous_face_gallery
                        #
                        # Deleting this gait folder cannot
                        # affect your current recognition.
                        # -------------------------------------
                        try:
                            gait_signature.save(global_id, signature)
                        except Exception as e:
                            print(f"[GAIT SAVE ERROR] " f"{tracker_key}: {e}")

        # ====================================================
        # REFRESH CAMERA IDENTITY LOCKS
        #
        # Run AFTER final result dedup so a duplicated local
        # track cannot overwrite the lock with a second box.
        # ====================================================
        for lock_item in results:
            if lock_item.get("is_known", False):
                continue
            lock_global_id = lock_item.get("global_id")
            if lock_global_id is None:
                continue
            lock_bbox = lock_item.get("body_bbox")
            if lock_bbox is None:
                lock_bbox = lock_item.get("bbox")
            if lock_bbox is None:
                continue
            lock_tid = lock_item.get("tracker_id")
            lock_tracker_key = f"{cam_id}_{lock_tid}"
            refresh_camera_identity_lock(lock_global_id, lock_bbox, lock_tracker_key, now)

        # ====================================================
        # SMOOTH BOUNDING BOXES
        # ====================================================
        for rec_item in results:
            tracker_key = f"{cam_id}_" f"{rec_item.get('tracker_id')}"
            raw_bbox = rec_item.get("bbox")
            if raw_bbox is None:
                continue
            if tracker_key in smoothed_boxes:
                previous_box = smoothed_boxes[tracker_key]
                new_box = [
                    previous_box[index] * (1 - SMOOTH_ALPHA) + raw_bbox[index] * SMOOTH_ALPHA
                    for index in range(4)
                ]
            else:
                new_box = list(raw_bbox)
            smoothed_boxes[tracker_key] = new_box
            smoothed_int = [int(value) for value in new_box]
            rec_item["bbox"] = smoothed_int
            if rec_item.get("body_bbox") is not None:
                rec_item["body_bbox"] = smoothed_int

        # ====================================================
        # SAVE LAST RESULTS
        # ====================================================

        # A removal can arrive while inference is in flight.
        results = [
            item
            for item in results
            if not item.get("is_known") or recognizer.is_identity_active(item.get("student_id"))
        ]
        last_result = results
        completed = time.perf_counter()
        fps = inference_rate.tick(completed)
        telemetry = {
            "processed_fps": round(fps, 1),
            "inference_ms": round((time.perf_counter() - inference_started) * 1000, 1),
            "person_ms": round(person_ms, 1),
            "face_ms": round(face_ms, 1),
            "dropped_inference_frames": dropped_frames,
            "analysis_frame_at": packet.captured_at,
        }
        update_analysis(cam_id, results, frame.shape[:2], telemetry)
        preview.submit(packet, results)
    stream.stop()
