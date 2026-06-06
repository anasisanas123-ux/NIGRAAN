import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import config

class BodyDetector:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        # run on CPU to avoid CUDA stream conflict with InsightFace
        self.model.to('cpu')
        print(f"BodyDetector ready on CPU (avoiding CUDA stream conflict)")

    def detect(self, frame_rgb, cam_id=None):
        results = []
        try:
            frame_bgr  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            detections = self.model(
                frame_bgr,
                classes=[0],
                conf=0.55,
                iou=0.45,
                verbose=False,
                device='cpu'
            )[0]

            h, w = frame_rgb.shape[:2]

            # get zone
            zone = None
            if cam_id and hasattr(config, 'DETECTION_ZONE'):
                z = config.DETECTION_ZONE.get(cam_id)
                if z:
                    zone = (
                        int(z[0]*w), int(z[1]*h),
                        int(z[2]*w), int(z[3]*h)
                    )

            for box in detections.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if (x2-x1) < 40 or (y2-y1) < 80:
                    continue

                # zone filter
                if zone:
                    zx1, zy1, zx2, zy2 = zone
                    cx = (x1+x2)//2
                    cy = (y1+y2)//2
                    if not (zx1 <= cx <= zx2 and zy1 <= cy <= zy2):
                        continue

                body_crop = frame_rgb[y1:y2, x1:x2]
                if body_crop.size == 0:
                    continue

                face_h    = int((y2-y1) * 0.40)
                face_y2   = min(h, y1 + face_h)
                face_crop = frame_rgb[y1:face_y2, x1:x2]

                results.append({
                    'bbox':       [x1, y1, x2, y2],
                    'confidence': conf,
                    'body_crop':  body_crop,
                    'face_crop':  face_crop if face_crop.size > 0 else None
                })

        except Exception as e:
            print(f"Body detection error: {e}")

        return results