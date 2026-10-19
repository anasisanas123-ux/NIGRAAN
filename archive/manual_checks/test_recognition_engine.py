"""Smoke-test the same ArcFace recognizer used by the live application.

Offline mode reads images from a directory. Live mode samples a configured
camera for a bounded duration. The script prints aggregate metrics only and
does not save frames, embeddings, or visitor records.
"""

import argparse
from contextlib import redirect_stdout
import io
import os
import time

import cv2

import config
from python_engine.arcface_recognizer import ArcFaceRecognizer


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--images",
        metavar="DIR",
        help="recursively evaluate image files in DIR (for example test_images)",
    )
    mode.add_argument(
        "--camera",
        choices=sorted(config.CAMERA_SOURCES),
        help="sample one configured camera without saving its video",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=15,
        help="live-camera test duration in seconds (default: 15)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="minimum time between camera samples (default: 0.5 seconds)",
    )
    return parser.parse_args()


def test_frame(recognizer, frame):
    # The production method logs recognized names. Keep this test's output
    # limited to aggregate metrics and never print biometric identities.
    captured = io.StringIO()
    with redirect_stdout(captured):
        detections = recognizer.detect_and_recognize(frame)
    had_error = "ArcFace error:" in captured.getvalue()
    valid = [item for item in detections if "is_known" in item]
    known = sum(bool(item["is_known"]) for item in valid)
    recognized_ids = {
        item.get("student_id")
        for item in valid
        if item.get("is_known") and item.get("student_id")
    }
    return len(valid), known, had_error, recognized_ids


def run_images(recognizer, root):
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise SystemExit("Image directory does not exist.")

    paths = []
    for current, _, files in os.walk(root):
        paths.extend(
            os.path.join(current, filename)
            for filename in files
            if filename.lower().endswith(IMAGE_EXTENSIONS)
        )
    paths.sort()
    if not paths:
        raise SystemExit("No supported images were found in that directory.")

    frames = faces = known = no_face = unreadable = errors = 0
    labeled = correct = rejected = incorrect = 0
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            unreadable += 1
            continue
        frames += 1
        face_count, known_count, had_error, recognized_ids = test_frame(recognizer, frame)
        errors += int(had_error)
        faces += face_count
        known += known_count
        no_face += int(face_count == 0 and not had_error)

        # Optional accuracy evaluation: arrange files in folders named with
        # the enrolled ID. Flat directories remain unlabeled smoke tests.
        relative_path = os.path.relpath(path, root)
        first_folder = relative_path.split(os.sep, 1)[0]
        expected_id = first_folder.split("_", 1)[0]
        if relative_path != os.path.basename(path) and expected_id in recognizer.known_embeddings:
            labeled += 1
            if face_count == 0:
                rejected += 1
            elif expected_id in recognized_ids:
                correct += 1
            else:
                incorrect += 1

    print("Recognition engine image test")
    print(f"Images found: {len(paths)}")
    print(f"Images read: {frames}")
    print(f"Images unreadable: {unreadable}")
    print(f"Faces detected: {faces}")
    print(f"Known matches: {known}")
    print(f"Unknown faces: {faces - known}")
    print(f"Images with no usable face: {no_face}")
    print(f"Recognizer errors: {errors}")
    if labeled:
        print(f"Labeled images: {labeled}")
        print(f"Correctly recognized: {correct}")
        print(f"Rejected: {rejected}")
        print(f"Incorrect identity: {incorrect}")
    else:
        print("Accuracy: unavailable (samples have no enrolled-ID folder labels)")


def run_camera(recognizer, camera_name, seconds, interval):
    source = config.CAMERA_SOURCES.get(camera_name)
    if source is None or source == "":
        raise SystemExit(f"{camera_name} has no source configured in .env/config.py.")
    if seconds <= 0 or interval <= 0:
        raise SystemExit("--seconds and --interval must be greater than zero.")

    if isinstance(source, str):
        # OpenCV's timeout properties must be passed at construction time for
        # FFmpeg-backed streams; setting them later does not bound open().
        capture = cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                5000,
            ],
        )
    else:
        capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise SystemExit(f"Could not open {camera_name}; source is not printed for privacy.")

        deadline = time.monotonic() + seconds
        next_sample = 0.0
        frames = faces = known = errors = read_failures = 0
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok or frame is None:
                read_failures += 1
                time.sleep(min(interval, 0.2))
                continue
            now = time.monotonic()
            if now < next_sample:
                continue
            next_sample = now + interval
            frames += 1
            face_count, known_count, had_error, _ = test_frame(recognizer, frame)
            faces += face_count
            known += known_count
            errors += int(had_error)

        print(f"Recognition engine camera test ({camera_name})")
        print(f"Duration: {seconds:g} seconds")
        print(f"Frames evaluated: {frames}")
        print(f"Frame read failures: {read_failures}")
        print(f"Faces detected: {faces}")
        print(f"Known matches: {known}")
        print(f"Unknown faces: {faces - known}")
        print(f"Recognizer errors: {errors}")
    finally:
        capture.release()


def main():
    args = parse_args()
    if args.camera and (args.seconds <= 0 or args.interval <= 0):
        raise SystemExit("--seconds and --interval must be greater than zero.")

    print("Loading recognition model and local embedding gallery...")
    recognizer = ArcFaceRecognizer()
    if args.images:
        run_images(recognizer, args.images)
    else:
        run_camera(recognizer, args.camera, args.seconds, args.interval)


if __name__ == "__main__":
    main()
