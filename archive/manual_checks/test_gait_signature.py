import cv2
import argparse
import time

from python_engine.gait_pose import GaitPoseExtractor
from python_engine.gait_signature import GaitSignatureEngine


TARGET_FRAMES = 32


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="0"
    )

    parser.add_argument(
        "--label",
        default=None
    )

    args = parser.parse_args()


    source = args.source

    if str(source).isdigit():
        source = int(source)


    pose = GaitPoseExtractor()

    gait = GaitSignatureEngine(
        sequence_length=TARGET_FRAMES
    )


    cap = cv2.VideoCapture(
        source
    )


    if not cap.isOpened():

        print(
            "[GAIT ERROR] "
            "Could not open camera"
        )

        return


    sequence = []

    last_added = 0.0


    print()
    print(
        "[GAIT TEST] Walk normally across the camera."
    )

    print(
        f"[GAIT TEST] Collecting "
        f"{TARGET_FRAMES} usable frames."
    )

    print()


    while True:

        ok, frame = cap.read()

        if not ok:
            break


        people = pose.extract(
            frame
        )


        # For this test only:
        # use largest detected person.
        if people:

            person = max(
                people,
                key=lambda p:
                    (
                        p["bbox"][2]
                        -
                        p["bbox"][0]
                    )
                    *
                    (
                        p["bbox"][3]
                        -
                        p["bbox"][1]
                    )
            )


            feature = gait.frame_feature(
                person["keypoints"],
                person["bbox"]
            )


            if (
                feature is not None

                and

                time.time()
                -
                last_added
                >=
                0.08
            ):

                sequence.append(
                    feature
                )

                last_added = time.time()


            x1, y1, x2, y2 = (
                person["bbox"]
            )


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )


        progress = min(
            len(sequence),
            TARGET_FRAMES
        )


        cv2.putText(
            frame,
            f"Gait frames: "
            f"{progress}/{TARGET_FRAMES}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "NIGRAAN - Gait Signature Test",
            frame
        )


        if progress >= TARGET_FRAMES:
            break


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()


    signature = gait.build_signature(
        sequence
    )


    if signature is None:

        print(
            "[GAIT TEST] "
            "Not enough usable gait data."
        )

        return


    print(
        f"[GAIT SIGNATURE] "
        f"created | dim={len(signature)}"
    )


    (
        best_label,
        best_score,
        margin
    ) = gait.match(
        signature
    )


    if best_label is None:

        print(
            "[GAIT MATCH] "
            "Gallery empty"
        )

    else:

        print(
            f"[GAIT MATCH] "
            f"best={best_label} "
            f"score={best_score:.3f} "
            f"margin={margin:.3f}"
        )


    if args.label:

        gait.save(
            args.label,
            signature
        )


if __name__ == "__main__":
    main()
