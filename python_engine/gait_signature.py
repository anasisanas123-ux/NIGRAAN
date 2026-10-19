from pathlib import Path
import os
import threading
import uuid

import numpy as np


_GALLERY_LOCK = threading.RLock()


class GaitSignatureEngine:

    LOWER_BODY = [
        11, 12,     # hips
        13, 14,     # knees
        15, 16      # ankles
    ]

    def __init__(
        self,
        gallery_dir="data/gait_gallery_observation",
        sequence_length=32
    ):

        self.gallery_dir = Path(
            gallery_dir
        )

        self.gallery_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.sequence_length = int(
            sequence_length
        )

        print(
            f"[GAIT SIGNATURE] Ready | "
            f"sequence={self.sequence_length}"
        )


    @staticmethod
    def _angle(a, b, c):

        ba = a - b
        bc = c - b

        denom = (
            np.linalg.norm(ba)
            *
            np.linalg.norm(bc)
            +
            1e-8
        )

        cosine = np.dot(
            ba,
            bc
        ) / denom

        cosine = np.clip(
            cosine,
            -1.0,
            1.0
        )

        angle = np.arccos(
            cosine
        )

        return float(
            angle / np.pi
        )


    def frame_feature(
        self,
        keypoints,
        bbox
    ):

        if keypoints is None:
            return None

        if len(keypoints) < 17:
            return None

        kp = np.asarray(
            keypoints,
            dtype=np.float32
        )

        if kp.ndim != 2 or kp.shape[0] < 17 or kp.shape[1] < 3:
            return None

        if not np.isfinite(kp[:17, :3]).all():
            return None

        # Require all lower-body joints
        # with modest confidence.
        for idx in self.LOWER_BODY:

            if kp[idx, 2] < 0.25:
                return None


        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None

        if not np.isfinite([x1, y1, x2, y2]).all() or y2 <= y1:
            return None

        body_height = max(
            y2 - y1,
            1.0
        )


        left_hip = kp[11, :2]
        right_hip = kp[12, :2]

        left_knee = kp[13, :2]
        right_knee = kp[14, :2]

        left_ankle = kp[15, :2]
        right_ankle = kp[16, :2]


        hip_center = (
            left_hip
            +
            right_hip
        ) / 2.0


        coords = []

        for idx in self.LOWER_BODY:

            point = kp[idx, :2]

            normalized = (
                point
                -
                hip_center
            ) / body_height

            coords.extend(
                normalized.tolist()
            )


        left_knee_angle = self._angle(
            left_hip,
            left_knee,
            left_ankle
        )

        right_knee_angle = self._angle(
            right_hip,
            right_knee,
            right_ankle
        )


        ankle_distance = (
            abs(
                left_ankle[0]
                -
                right_ankle[0]
            )
            /
            body_height
        )


        ankle_height_difference = (
            abs(
                left_ankle[1]
                -
                right_ankle[1]
            )
            /
            body_height
        )


        hip_width = (
            np.linalg.norm(
                left_hip
                -
                right_hip
            )
            /
            body_height
        )


        feature = np.asarray(
            coords
            +
            [
                left_knee_angle,
                right_knee_angle,
                ankle_distance,
                ankle_height_difference,
                hip_width
            ],
            dtype=np.float32
        )

        return feature


    def build_signature(
        self,
        frame_features,
        timestamps=None
    ):

        if len(frame_features) < 15:
            return None


        data = np.asarray(
            frame_features,
            dtype=np.float32
        )


        if data.ndim != 2 or not np.isfinite(data).all():
            return None

        if timestamps is not None:
            times = np.asarray(timestamps, dtype=np.float64).flatten()
            if times.size != len(data) or not np.isfinite(times).all():
                return None
            order = np.argsort(times, kind="stable")
            times = times[order]
            data = data[order]
            keep = np.concatenate(([True], np.diff(times) > 1e-6))
            times = times[keep]
            data = data[keep]
            if len(data) < 15:
                return None
            old_x = (times - times[0]) / max(times[-1] - times[0], 1e-6)
        else:
            old_x = np.linspace(0.0, 1.0, len(data))

        new_x = np.linspace(
            0.0,
            1.0,
            self.sequence_length
        )


        resampled = np.zeros(
            (
                self.sequence_length,
                data.shape[1]
            ),
            dtype=np.float32
        )


        for col in range(
            data.shape[1]
        ):

            resampled[:, col] = np.interp(
                new_x,
                old_x,
                data[:, col]
            )


        # Dynamic movement information.
        velocity = np.diff(
            resampled,
            axis=0,
            prepend=resampled[0:1]
        )


        signature = np.concatenate(
            [
                resampled.flatten(),
                velocity.flatten()
            ]
        ).astype(
            np.float32
        )


        norm = np.linalg.norm(
            signature
        )

        if norm < 1e-8:
            return None


        return (
            signature
            /
            norm
        )


    @staticmethod
    def similarity(
        a,
        b
    ):

        return float(
            np.dot(a, b)
            /
            (
                np.linalg.norm(a)
                *
                np.linalg.norm(b)
                +
                1e-8
            )
        )


    def save(
        self,
        label,
        signature
    ):

        path = (
            self.gallery_dir
            /
            f"{label}.npy"
        )


        signature = np.asarray(
            signature,
            dtype=np.float32
        )


        if signature.ndim != 1 or not signature.size or not np.isfinite(signature).all():
            return

        with _GALLERY_LOCK:
            if path.exists():
                try:
                    old = np.load(path, allow_pickle=False)
                    if old.ndim == 1:
                        old = old.reshape(1, -1)
                    if old.ndim != 2 or old.shape[1] != signature.size or not np.isfinite(old).all():
                        old = np.empty((0, signature.size), dtype=np.float32)
                except (OSError, ValueError):
                    old = np.empty((0, signature.size), dtype=np.float32)
            else:
                old = np.empty((0, signature.size), dtype=np.float32)

            data = np.vstack([old, signature.reshape(1, -1)])[-100:]
            temp_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
            try:
                with temp_path.open("wb") as stream:
                    np.save(stream, data, allow_pickle=False)
                os.replace(temp_path, path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()


        print(
            f"[GAIT SAVE] "
            f"{label} "
            f"sequences={len(data)}"
        )


    def match(
        self,
        signature
    ):

        signature = np.asarray(signature, dtype=np.float32).flatten()
        if not signature.size or not np.isfinite(signature).all():
            return None, 0.0, 0.0

        rankings = []


        for path in sorted(
            self.gallery_dir.glob(
                "*.npy"
            )
        ):

            label = path.stem

            try:

                with _GALLERY_LOCK:
                    stored = np.load(path, allow_pickle=False)

            except Exception:

                continue


            if stored.ndim == 1:

                stored = stored.reshape(
                    1,
                    -1
                )


            scores = []


            for sample in stored:

                if (
                    sample.shape
                    !=
                    signature.shape
                ):

                    continue


                scores.append(
                    self.similarity(
                        signature,
                        sample
                    )
                )


            if not scores:
                continue


            scores.sort(
                reverse=True
            )


            # Use strongest few sequences.
            score = float(
                np.mean(
                    scores[:3]
                )
            )


            rankings.append(
                (
                    label,
                    score
                )
            )


        rankings.sort(
            key=lambda item:
                item[1],
            reverse=True
        )


        if not rankings:

            return (
                None,
                0.0,
                0.0
            )


        best_label = rankings[0][0]
        best_score = rankings[0][1]


        second_score = (
            rankings[1][1]
            if len(rankings) > 1
            else 0.0
        )


        margin = (
            best_score
            -
            second_score
        )


        return (
            best_label,
            best_score,
            margin
        )
