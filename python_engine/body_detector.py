import sys
import os
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch

from ultralytics import YOLO

import config


class BodyDetector:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # Each camera needs its own YOLO/BoT-SORT state.
        #
        # cam_a tracker history must never be mixed with cam_b.
        # -----------------------------------------------------
        self.models = {}
        self.model_lock = threading.Lock()
        print(f"BodyDetector + BoT-SORT ready " f"on {self.device.upper()}")

    # =========================================================
    # GET SEPARATE TRACKER MODEL FOR EACH CAMERA
    # =========================================================

    def _get_model(self, cam_id):
        camera_key = cam_id if cam_id is not None else "default"
        with self.model_lock:
            if camera_key not in self.models:
                print(f"[BOT-SORT] Creating tracker " f"for {camera_key}")
                model = YOLO("yolov8s.pt")
                model.to(self.device)
                self.models[camera_key] = model
        return self.models[camera_key]

    # =========================================================
    # DETECT + TRACK PERSONS
    # =========================================================

    # =========================================================
    # CAMERA / BOT-SORT SESSION RESET
    # =========================================================
    def reset_camera(self, cam_id):
        """
        Remove one camera's YOLO model instance.

        The next detect() call recreates YOLO and therefore
        starts a fresh BoT-SORT tracker for that camera.
        """
        camera_key = cam_id if cam_id is not None else "default"
        models = getattr(self, "models", None)
        model_lock = getattr(self, "model_lock", None)
        if models is None or model_lock is None:
            print(f"[BOT-SORT RESET] " f"{camera_key} skipped")
            return False
        with model_lock:
            old_model = models.pop(camera_key, None)
        removed = old_model is not None
        if removed:
            del old_model
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[BOT-SORT RESET] " f"{camera_key}")
        return removed

    def detect(self, frame_rgb, cam_id=None, color_order="rgb"):
        results = []
        if frame_rgb is None or frame_rgb.size == 0:
            return results
        try:

            # -------------------------------------------------
            # Our pipeline uses RGB.
            #
            # Ultralytics receives the OpenCV-style BGR frame.
            # -------------------------------------------------
            frame_bgr = (
                frame_rgb if color_order == "bgr" else cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            )
            model = self._get_model(cam_id)

            # -------------------------------------------------
            # YOLO PERSON DETECTION + BoT-SORT
            #
            # persist=True:
            # preserve BoT-SORT state between consecutive
            # frames from THIS camera.
            # -------------------------------------------------
            tracked_results = model.track(
                source=frame_bgr,
                persist=True,
                tracker="botsort_nigraan.yaml",
                classes=[0],
                conf=0.10,
                iou=0.45,
                imgsz=1280,
                verbose=False,
                device=self.device,
            )
            if not tracked_results:
                return results
            detection_result = tracked_results[0]
            boxes = detection_result.boxes
            if boxes is None or len(boxes) == 0:
                return results

            # -------------------------------------------------
            # BoT-SORT IDs
            # -------------------------------------------------
            track_ids = boxes.id
            h, w = frame_rgb.shape[:2]

            # -------------------------------------------------
            # OPTIONAL DETECTION ZONE
            # -------------------------------------------------
            zone = None
            if cam_id and hasattr(config, "DETECTION_ZONE"):
                z = config.DETECTION_ZONE.get(cam_id)
                if z:
                    zone = (int(z[0] * w), int(z[1] * h), int(z[2] * w), int(z[3] * h))

            # -------------------------------------------------
            # BUILD DETECTION RESULTS
            # -------------------------------------------------
            for index in range(len(boxes)):
                xyxy = boxes.xyxy[index].detach().cpu().tolist()
                x1, y1, x2, y2 = [int(value) for value in xyxy]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # ---------------------------------------------
                # Reject unusably small detections
                # ---------------------------------------------
                if x2 - x1 < 30 or y2 - y1 < 60:
                    continue
                confidence = float(boxes.conf[index].detach().cpu().item())

                # ---------------------------------------------
                # BoT-SORT LOCAL TRACK ID
                # ---------------------------------------------
                tracker_id = None
                if track_ids is not None:
                    try:
                        tracker_id = str(int(track_ids[index].detach().cpu().item()))
                    except Exception:
                        tracker_id = None

                # ---------------------------------------------
                # DETECTION ZONE
                # ---------------------------------------------
                if zone is not None:
                    (zx1, zy1, zx2, zy2) = zone
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    if not (zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2):
                        continue

                # ---------------------------------------------
                # WHOLE VISIBLE PERSON CROP
                # ---------------------------------------------
                body_crop = frame_rgb[y1:y2, x1:x2]
                if body_crop.size == 0:
                    continue
                # Preserve RGB inputs for OSNet, converting only person crops.
                if color_order == "bgr":
                    body_crop = cv2.cvtColor(body_crop, cv2.COLOR_BGR2RGB)

                # ---------------------------------------------
                # FACE REGION CANDIDATE
                # ---------------------------------------------
                face_height = int((y2 - y1) * 0.40)
                face_y2 = min(h, y1 + face_height)
                face_crop = body_crop[: face_y2 - y1]

                # ---------------------------------------------
                # RETURN FORMAT
                #
                # Keep existing fields so ArcFace / OSNet /
                # ReIDManager don't need to change.
                #
                # New field:
                # tracker_id = BoT-SORT local ID
                # ---------------------------------------------
                results.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "body_crop": body_crop,
                        "face_crop": (face_crop if face_crop.size > 0 else None),
                        "tracker_id": tracker_id,
                    }
                )
        except Exception as error:
            print(f"Body detection / " f"BoT-SORT error " f"({cam_id}): " f"{error}")
        return results
