from pathlib import Path
import numpy as np


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

        # Require all lower-body joints
        # with modest confidence.
        for idx in self.LOWER_BODY:

            if kp[idx, 2] < 0.25:
                return None


        x1, y1, x2, y2 = [
            float(v)
            for v in bbox
        ]

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
        frame_features
    ):

        if len(frame_features) < 15:
            return None


        data = np.asarray(
            frame_features,
            dtype=np.float32
        )


        old_x = np.linspace(
            0.0,
            1.0,
            len(data)
        )

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


        if path.exists():

            old = np.load(
                path
            )

            if old.ndim == 1:

                old = old.reshape(
                    1,
                    -1
                )


            data = np.vstack(
                [
                    old,
                    signature.reshape(
                        1,
                        -1
                    )
                ]
            )

        else:

            data = signature.reshape(
                1,
                -1
            )


        np.save(
            path,
            data
        )


        print(
            f"[GAIT SAVE] "
            f"{label} "
            f"sequences={len(data)}"
        )


    def match(
        self,
        signature
    ):

        rankings = []


        for path in sorted(
            self.gallery_dir.glob(
                "*.npy"
            )
        ):

            label = path.stem

            try:

                stored = np.load(
                    path
                )

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
