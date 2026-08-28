import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from deep_sort_realtime.deepsort_tracker import DeepSort

import config


class Tracker:

    def __init__(self):

        # =====================================================
        # DEEPSORT
        # =====================================================
        #
        # Important:
        #
        # multi_camera.py sends frame_rgb to this class.
        # Therefore bgr MUST be False.
        #
        # We also give DeepSORT slightly more tolerance so a
        # person does not constantly receive new local IDs
        # because of small detector changes / temporary misses.
        # =====================================================

        self.tracker = DeepSort(

            # -------------------------------------------------
            # TRACK LIFETIME
            # -------------------------------------------------
            # Keep an existing track alive longer during short
            # detector misses / occlusions.
            max_age=60,

            # Two detections are enough to confirm a track.
            n_init=2,

            # -------------------------------------------------
            # SPATIAL ASSOCIATION
            # -------------------------------------------------
            # Slightly more tolerant than 0.7.
            max_iou_distance=0.8,

            # -------------------------------------------------
            # APPEARANCE ASSOCIATION
            # -------------------------------------------------
            # Default is quite strict. A little more tolerance
            # helps when pose / lighting changes.
            max_cosine_distance=0.35,

            # Store a reasonable history of appearance vectors.
            nn_budget=100,

            # -------------------------------------------------
            # KALMAN GATING
            # -------------------------------------------------
            # Position-based gating is more tolerant when body
            # box height / aspect ratio changes while walking.
            gating_only_position=True,

            # -------------------------------------------------
            # EMBEDDER
            # -------------------------------------------------
            embedder="mobilenet",

            half=True,

            # CRITICAL:
            # We pass frame_rgb, not OpenCV BGR.
            bgr=False,

            embedder_gpu=True,

            # Disable additional NMS here because YOLO already
            # produced the person detections.
            nms_max_overlap=1.0
        )

        print("\n========================================")
        print("DeepSORT Tracker ready")
        print("max_age             : 60")
        print("n_init              : 2")
        print("max_iou_distance    : 0.80")
        print("max_cosine_distance : 0.35")
        print("nn_budget           : 100")
        print("frame format        : RGB")
        print("========================================\n")


    def update(
        self,
        detections,
        frame_rgb
    ):

        # =====================================================
        # BUILD DEEPSORT INPUT
        # =====================================================

        ds_input = []

        for det in detections:

            bbox = det.get(
                'bbox'
            )

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox

            width = max(
                1,
                int(x2 - x1)
            )

            height = max(
                1,
                int(y2 - y1)
            )

            confidence = float(
                det.get(
                    'confidence',
                    0.5
                )
            )

            # Ignore completely artificial / invalid boxes.
            if (
                width < 5
                or
                height < 10
            ):
                continue

            ds_input.append(
                (
                    [
                        int(x1),
                        int(y1),
                        width,
                        height
                    ],
                    confidence,
                    'person'
                )
            )

        # =====================================================
        # UPDATE DEEPSORT
        # =====================================================

        tracks = (
            self.tracker.update_tracks(
                ds_input,
                frame=frame_rgb
            )
        )

        results = []

        # =====================================================
        # RETURN ACTIVE TRACKS
        # =====================================================

        for track in tracks:

            if not track.is_confirmed():
                continue

            # -------------------------------------------------
            # Do not return stale predictions for body/ReID
            # matching.
            #
            # Internally DeepSORT can retain the track for up to
            # max_age, but multi_camera should only associate a
            # current YOLO body with tracks updated very recently.
            # -------------------------------------------------

            if (
                track.time_since_update
                >
                1
            ):
                continue

            ltrb = (
                track.to_ltrb()
            )

            x1, y1, x2, y2 = [
                int(value)
                for value
                in ltrb
            ]

            if (
                x2 <= x1
                or
                y2 <= y1
            ):
                continue

            results.append({

                'bbox': [
                    x1,
                    y1,
                    x2,
                    y2
                ],

                'tracker_id':
                    str(
                        track.track_id
                    )
            })

        return results