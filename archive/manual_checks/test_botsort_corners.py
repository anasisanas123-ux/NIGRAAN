import cv2

from python_engine.body_detector import BodyDetector


detector = BodyDetector()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open cam_a webcam")


while True:

    ok, frame = cap.read()

    if not ok:
        continue

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    detections = detector.detect(
        frame_rgb,
        cam_id="cam_a"
    )

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]

        tracker_id = det.get(
            "tracker_id"
        )

        confidence = det.get(
            "confidence",
            0.0
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"ID {tracker_id}  {confidence:.2f}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Detections: {len(detections)}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "BoT-SORT Corner Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()