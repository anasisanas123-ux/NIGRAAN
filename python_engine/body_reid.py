import os
import sys

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

import torchreid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class BodyReID:

    def __init__(self):

        # =====================================================
        # DEVICE
        # =====================================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nBodyReID device: {self.device}")

        # =====================================================
        # PROJECT ROOT
        # =====================================================
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # =====================================================
        # RE-ID CHECKPOINT
        # =====================================================
        self.model_path = os.path.join(
            project_root,
            "models",
            "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("\nOSNet Re-ID checkpoint not found:\n" f"{self.model_path}")

        # =====================================================
        # IMAGE PREPROCESSING
        # =====================================================
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # OSNet is trained with horizontal-flip augmentation. Averaging the
        # original and flipped descriptors makes inference less sensitive to
        # which side of a person faces the camera. Keep the switch configurable
        # because it trades additional inference work for view robustness.
        self.flip_test = bool(getattr(config, "REID_FLIP_TEST", True))
        self.min_crop_blur = float(getattr(config, "REID_MIN_CROP_BLUR", 2.0))

        # =====================================================
        # BUILD OSNET
        # =====================================================
        print("Loading OSNet x1.0 architecture...")
        self.model = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=1000, loss="softmax", pretrained=False
        )

        # =====================================================
        # LOAD PERSON RE-ID WEIGHTS
        # =====================================================
        print("Loading person-ReID checkpoint:\n" f"{self.model_path}")
        torchreid.utils.load_pretrained_weights(self.model, self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("\n========================================")
        print("BodyReID ready")
        print("Model      : OSNet x1.0")
        print("Dataset    : MSMT17 combineall")
        print("Checkpoint : " + os.path.basename(self.model_path))
        print(f"Device     : {self.device}")
        print("========================================\n")

    # =========================================================
    # NORMALIZE EMBEDDING
    # =========================================================

    @staticmethod
    def _normalize_embedding(embedding):
        if embedding is None:
            return None
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        if not embedding.size or not np.isfinite(embedding).all():
            return None
        norm = np.linalg.norm(embedding)
        if not np.isfinite(norm) or norm <= 1e-10:
            return None
        return embedding / norm

    # =========================================================
    # PREPARE BODY CROP
    # =========================================================

    def _prepare_crop(self, crop):
        if crop is None:
            return None
        if not isinstance(crop, np.ndarray):
            return None
        if crop.size == 0:
            return None
        height, width = crop.shape[:2]
        if width < 30:
            return None
        if height < 60:
            return None

        # Reject only severely blurred crops. These otherwise contaminate the
        # track/gallery prototype and can make later cross-camera matches less
        # stable. The low default is deliberately conservative.
        try:
            if crop.ndim == 3 and crop.shape[2] >= 3:
                gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_RGB2GRAY)
            elif crop.ndim == 2:
                gray = crop
            else:
                return None
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if not np.isfinite(blur_score) or blur_score < self.min_crop_blur:
                return None
        except (cv2.error, TypeError, ValueError):
            return None
        image = Image.fromarray(crop).convert("RGB")
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    # =========================================================
    # SINGLE EMBEDDING
    # =========================================================

    def get_embedding(self, crop):
        tensor = self._prepare_crop(crop)
        if tensor is None:
            return None
        try:
            model_input = tensor
            if self.flip_test:
                model_input = torch.cat(
                    [tensor, torch.flip(tensor, dims=[3])],
                    dim=0,
                )
            with torch.inference_mode():
                features = self.model(model_input)
            if self.flip_test:
                features = (features[0] + features[1]) * 0.5
            features = features.detach().cpu().numpy().flatten()
            embedding = self._normalize_embedding(features)
            return embedding
        except Exception as error:
            print(f"[BODY REID ERROR] " f"{error}")
            return None

    # =========================================================
    # BATCH EMBEDDINGS
    # =========================================================

    def get_embeddings(self, crops):
        if not crops:
            return []
        tensors = []
        valid_indices = []
        for index, crop in enumerate(crops):
            tensor = self._prepare_crop(crop)
            if tensor is None:
                continue
            tensors.append(tensor.squeeze(0))
            valid_indices.append(index)
        results = [None for _ in crops]
        if not tensors:
            return results
        batch = torch.stack(tensors).to(self.device)
        try:
            model_input = batch
            if self.flip_test:
                model_input = torch.cat(
                    [batch, torch.flip(batch, dims=[3])],
                    dim=0,
                )
            with torch.inference_mode():
                features = self.model(model_input)
            if self.flip_test:
                batch_size = len(batch)
                features = (features[:batch_size] + features[batch_size:]) * 0.5
            features = features.detach().cpu().numpy()
            for original_index, feature in zip(valid_indices, features):
                results[original_index] = self._normalize_embedding(feature)
        except Exception as error:
            print(f"[BODY REID BATCH ERROR] " f"{error}")
        return results

    # =========================================================
    # COSINE SIMILARITY
    # =========================================================

    @staticmethod
    def similarity(embedding_a, embedding_b):
        if embedding_a is None or embedding_b is None:
            return 0.0
        embedding_a = np.asarray(embedding_a, dtype=np.float32).flatten()
        embedding_b = np.asarray(embedding_b, dtype=np.float32).flatten()
        if (
            (embedding_a.shape != embedding_b.shape)
            or not np.isfinite(embedding_a).all()
            or not np.isfinite(embedding_b).all()
        ):
            return 0.0
        return float(
            np.dot(embedding_a, embedding_b)
            / (np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b) + 1e-10)
        )
