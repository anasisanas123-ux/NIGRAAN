import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import torchreid

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import config


class BodyReID:

    def __init__(self):

        # =====================================================
        # DEVICE
        # =====================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"\nBodyReID device: {self.device}")

        # =====================================================
        # PROJECT ROOT
        # =====================================================

        project_root = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        # =====================================================
        # RE-ID CHECKPOINT
        # =====================================================

        self.model_path = os.path.join(
            project_root,
            "models",
            "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
        )

        if not os.path.exists(self.model_path):

            raise FileNotFoundError(
                "\nOSNet Re-ID checkpoint not found:\n"
                f"{self.model_path}"
            )

        # =====================================================
        # IMAGE PREPROCESSING
        # =====================================================

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        # =====================================================
        # BUILD OSNET
        # =====================================================

        print("Loading OSNet x1.0 architecture...")

        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,
            loss="softmax",
            pretrained=False
        )

        # =====================================================
        # LOAD PERSON RE-ID WEIGHTS
        # =====================================================

        print(
            "Loading person-ReID checkpoint:\n"
            f"{self.model_path}"
        )

        torchreid.utils.load_pretrained_weights(
            self.model,
            self.model_path
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        print("\n========================================")
        print("BodyReID ready")
        print("Model      : OSNet x1.0")
        print("Dataset    : MSMT17 combineall")
        print(
            "Checkpoint : "
            + os.path.basename(
                self.model_path
            )
        )
        print(f"Device     : {self.device}")
        print("========================================\n")


    # =========================================================
    # NORMALIZE EMBEDDING
    # =========================================================

    @staticmethod
    def _normalize_embedding(
        embedding
    ):

        if embedding is None:
            return None

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).flatten()

        norm = np.linalg.norm(
            embedding
        )

        if norm <= 1e-10:
            return None

        return embedding / norm


    # =========================================================
    # PREPARE BODY CROP
    # =========================================================

    def _prepare_crop(
        self,
        crop
    ):

        if crop is None:
            return None

        if not isinstance(
            crop,
            np.ndarray
        ):
            return None

        if crop.size == 0:
            return None

        height, width = crop.shape[:2]

        if width < 30:
            return None

        if height < 60:
            return None

        image = Image.fromarray(
            crop
        )

        tensor = self.transform(
            image
        )

        tensor = tensor.unsqueeze(
            0
        )

        return tensor.to(
            self.device
        )


    # =========================================================
    # SINGLE EMBEDDING
    # =========================================================

    def get_embedding(
        self,
        crop
    ):

        tensor = self._prepare_crop(
            crop
        )

        if tensor is None:
            return None

        try:

            with torch.no_grad():

                features = self.model(
                    tensor
                )

            features = (
                features
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )

            embedding = (
                self._normalize_embedding(
                    features
                )
            )

            return embedding

        except Exception as error:

            print(
                f"[BODY REID ERROR] "
                f"{error}"
            )

            return None


    # =========================================================
    # BATCH EMBEDDINGS
    # =========================================================

    def get_embeddings(
        self,
        crops
    ):

        if not crops:
            return []

        tensors = []
        valid_indices = []

        for index, crop in enumerate(
            crops
        ):

            tensor = self._prepare_crop(
                crop
            )

            if tensor is None:
                continue

            tensors.append(
                tensor.squeeze(0)
            )

            valid_indices.append(
                index
            )

        results = [
            None
            for _ in crops
        ]

        if not tensors:
            return results

        batch = torch.stack(
            tensors
        ).to(
            self.device
        )

        try:

            with torch.no_grad():

                features = self.model(
                    batch
                )

            features = (
                features
                .detach()
                .cpu()
                .numpy()
            )

            for (
                original_index,
                feature
            ) in zip(
                valid_indices,
                features
            ):

                results[
                    original_index
                ] = (
                    self._normalize_embedding(
                        feature
                    )
                )

        except Exception as error:

            print(
                f"[BODY REID BATCH ERROR] "
                f"{error}"
            )

        return results


    # =========================================================
    # COSINE SIMILARITY
    # =========================================================

    @staticmethod
    def similarity(
        embedding_a,
        embedding_b
    ):

        if (
            embedding_a is None
            or
            embedding_b is None
        ):
            return 0.0

        embedding_a = np.asarray(
            embedding_a,
            dtype=np.float32
        ).flatten()

        embedding_b = np.asarray(
            embedding_b,
            dtype=np.float32
        ).flatten()

        if (
            embedding_a.shape
            !=
            embedding_b.shape
        ):
            return 0.0

        return float(
            np.dot(
                embedding_a,
                embedding_b
            )
            /
            (
                np.linalg.norm(
                    embedding_a
                )
                *
                np.linalg.norm(
                    embedding_b
                )
                +
                1e-10
            )
        )