import cv2
import argparse
import time

from python_engine.gait_pose import GaitPoseExtractor


SKELETON = [
    (5, 6),

    (5, 11),
    (6, 12),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16)
]


def draw_pose(
    frame,
    person
):

    kp = person["keypoints"]

    usable = (
        GaitPoseExtractor.usable_for_gait(
            kp
        )
    )

    x1, y1, x2, y2 = (
        person["bbox"]
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0)
        if usable
        else
        (0, 165, 255),
        2
    )


    for i, point in enumerate(kp):

        x, y, conf = point

        if conf < 0.35:
            continue

        cv2.circle(
            frame,
            (
                int(x),
                int(y)
            ),
            4,
            (255, 255, 255),
            -1
        )


    for a, b in SKELETON:

        if (
            kp[a][2] >= 0.35
            and
            kp[b][2] >= 0.35
        ):

            cv2.line(
                frame,
                (
                    int(kp[a][0]),
                    int(kp[a][1])
                ),
                (
                    int(kp[b][0]),
                    int(kp[b][1])
                ),
                (255, 255, 255),
                2
            )


    label = (
        "GAIT READY"
        if usable
        else
        "POSE ONLY"
    )

    cv2.putText(
        frame,
        label,
        (
            x1,
            max(
                20,
                y1 - 10
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (
            0, 255, 0
            if usable
            else 255
        ),
        2
    )

    return usable


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="0"
    )

    args = parser.parse_args()

    source = args.source

    if source.isdigit():
        source = int(source)


    gait = GaitPoseExtractor()

    cap = cv2.VideoCapture(
        source
    )

    if not cap.isOpened():

        print(
            "[GAIT ERROR] "
            "Could not open camera"
        )

        return


    frame_count = 0
    usable_count = 0
    last_log = time.time()


    while True:

        ok, frame = cap.read()

        if not ok:
            print(
                "[GAIT] Frame read failed"
            )
            break


        people = gait.extract(
            frame
        )

        frame_count += 1

        frame_usable = 0


        for person in people:

            if draw_pose(
                frame,
                person
            ):

                frame_usable += 1


        usable_count += (
            frame_usable
        )


        if (
            time.time()
            -
            last_log
            >= 1.0
        ):

            print(
                f"[GAIT OBSERVE] "
                f"persons={len(people)} "
                f"gait_ready={frame_usable}"
            )

            last_log = (
                time.time()
            )


        cv2.putText(
            frame,
            f"Persons: {len(people)} | "
            f"Gait ready: {frame_usable}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "NIGRAAN - Gait Observation",
            frame
        )


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key == ord("q"):
            break


    cap.release()

    cv2.destroyAllWindows()


    print(
        f"[GAIT] Frames={frame_count} "
        f"usable observations={usable_count}"
    )


if __name__ == "__main__":
    main()
