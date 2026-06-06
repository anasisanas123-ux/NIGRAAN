import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as T

class BodyReID:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        try:
            import torchreid
            self.model = torchreid.models.build_model(
                name='osnet_x1_0',
                num_classes=1000,
                pretrained=True
            )
            self.model = self.model.to(self.device).eval()
            print(f"BodyReID (OSNet x1_0) ready on {self.device}")
        except Exception as e:
            print(f"OSNet failed: {e} — using color histogram")
            self.model = None

    def get_embedding(self, body_crop_rgb):
        try:
            if body_crop_rgb is None or body_crop_rgb.size == 0:
                return None

            if self.model is not None:
                img    = Image.fromarray(body_crop_rgb).resize(
                    (128, 256), Image.BILINEAR
                )
                tensor = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = self.model(tensor).cpu().numpy().flatten()
                norm = np.linalg.norm(emb)
                return emb / norm if norm > 0 else emb
            else:
                return self._color_histogram(body_crop_rgb)

        except Exception:
            return self._color_histogram(body_crop_rgb)

    def _color_histogram(self, img_rgb):
        img  = cv2.resize(img_rgb, (64, 128))
        hist = []
        for ch in range(3):
            h = cv2.calcHist([img], [ch], None, [32], [0, 256])
            hist.extend(h.flatten())
        hist = np.array(hist, dtype=np.float32)
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    def similarity(self, a, b):
        if a is None or b is None:
            return 0.0
        if a.shape != b.shape:
            return 0.0
        return float(np.dot(a, b) /
                     (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))