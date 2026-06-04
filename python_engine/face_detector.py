import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
import config

class FaceDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=config.MIN_FACE_SIZE,
            thresholds=[0.6, 0.7, config.FACE_DETECTION_CONFIDENCE],
            post_process=False,
            select_largest=False
        )
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        print(f"FaceDetector ready on {self.device}")

    def detect(self, frame_rgb):
        results = []
        try:
            h, w    = frame_rgb.shape[:2]
            boxes, probs = self.detector.detect(frame_rgb)
            if boxes is None:
                return results

            for box, prob in zip(boxes, probs):
                if prob is None or prob < config.FACE_DETECTION_CONFIDENCE:
                    continue

                x1 = max(0, min(w-1, int(box[0])))
                y1 = max(0, min(h-1, int(box[1])))
                x2 = max(0, min(w,   int(box[2])))
                y2 = max(0, min(h,   int(box[3])))

                w_box = x2 - x1
                h_box = y2 - y1

                # size check
                if w_box < config.MIN_FACE_SIZE or \
                   h_box < config.MIN_FACE_SIZE:
                    continue

                # aspect ratio check — real faces are roughly square
                aspect = w_box / max(h_box, 1)
                if aspect < 0.4 or aspect > 2.2:
                    continue

                # position check — ignore bottom 20% of frame (floor)
                if y1 > h * 0.80:
                    continue

                # ignore top 5% of frame (ceiling artifacts)
                if y2 < h * 0.05:
                    continue

                # brightness check — floor tiles often very bright or dark
                face_crop = frame_rgb[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                mean_brightness = np.mean(face_crop)
                if mean_brightness < 20 or mean_brightness > 240:
                    continue

                # variance check — real faces have texture variance
                # floor tiles are often uniform
                gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
                variance  = np.var(gray_crop)
                if variance < 100:
                    continue

                results.append({
                    'bbox':       [x1, y1, x2, y2],
                    'confidence': float(prob),
                    'face_crop':  face_crop
                })

        except Exception as e:
            print(f"Detection error: {e}")
        return results