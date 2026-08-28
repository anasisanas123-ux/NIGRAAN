import cv2
import numpy as np
from ultralytics import YOLO


class GaitPoseExtractor:
    """
    Observation-only gait pose extractor.

    Does NOT assign identities.
    Does NOT modify ArcFace / OSNet / ReID.
    Only extracts human pose keypoints.
    """

    def __init__(
        self,
        model_path="yolo11n-pose.pt",
        conf=0.35
    ):
        print("[GAIT] Loading pose model...")

        self.model = YOLO(model_path)
        self.conf = float(conf)

        print("[GAIT] Pose extractor ready")


    def extract(self, frame):
        """
        Returns a list of people.

        Each person:
        {
            'bbox': [x1,y1,x2,y2],
            'confidence': float,
            'keypoints': np.ndarray (17,3)
        }

        keypoints columns:
        x, y, confidence
        """

        if frame is None:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            classes=[0],
            verbose=False
        )

        people = []

        if not results:
            return people

        result = results[0]

        if result.boxes is None:
            return people

        if result.keypoints is None:
            return people

        boxes = result.boxes.xyxy.cpu().numpy()

        box_conf = (
            result.boxes.conf.cpu().numpy()
            if result.boxes.conf is not None
            else np.ones(len(boxes))
        )

        xy = result.keypoints.xy.cpu().numpy()

        if result.keypoints.conf is not None:
            kp_conf = (
                result.keypoints.conf
                .cpu()
                .numpy()
            )
        else:
            kp_conf = np.ones(
                (
                    xy.shape[0],
                    xy.shape[1]
                ),
                dtype=np.float32
            )

        count = min(
            len(boxes),
            len(xy)
        )

        for i in range(count):

            points = np.concatenate(
                [
                    xy[i],
                    kp_conf[i][:, None]
                ],
                axis=1
            )

            people.append({
                "bbox":
                    boxes[i]
                    .astype(int)
                    .tolist(),

                "confidence":
                    float(box_conf[i]),

                "keypoints":
                    points.astype(
                        np.float32
                    )
            })

        return people


    @staticmethod
    def usable_for_gait(
        keypoints,
        threshold=0.35
    ):
        """
        For gait we mainly need:

        hips
        knees
        ankles

        COCO:
        11 left hip
        12 right hip
        13 left knee
        14 right knee
        15 left ankle
        16 right ankle
        """

        if keypoints is None:
            return False

        if len(keypoints) < 17:
            return False

        required = [
            11, 12,
            13, 14,
            15, 16
        ]

        good = 0

        for idx in required:

            if keypoints[idx][2] >= threshold:
                good += 1

        # Need at least 5 of the 6
        # important lower-body joints.
        return good >= 5
