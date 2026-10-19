"""Calibrate OSNet pair-similarity on labeled, held-out body crops.

Arrange a validation set as DATASET/<person_id>/*.jpg. Do not use the same
frames for enrollment and validation. This evaluates the body descriptor and
threshold behavior; it does not measure tracker ID switches or camera handoff.
"""

import argparse
import os

import cv2
import numpy as np

import config
from python_engine.body_reid import BodyReID


EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def load_video_embeddings(path, detector, reid, sample_seconds, max_samples):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        return [], 1

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_step = max(1, round((fps if fps > 0 else 25.0) * sample_seconds))
    embeddings = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_step == 0:
            result = detector.predict(
                frame, classes=[0], conf=0.35, imgsz=640, verbose=False
            )[0]
            boxes = result.boxes
            if boxes is not None and len(boxes):
                candidates = []
                for box in boxes:
                    x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].cpu().tolist()]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    if x2 > x1 and y2 > y1:
                        candidates.append((
                            (x2 - x1) * (y2 - y1),
                            frame[y1:y2, x1:x2],
                        ))
                if candidates:
                    # Labeled clips should feature one target; selecting the
                    # largest person avoids letting a distant bystander win.
                    crop = max(candidates, key=lambda item: item[0])[1]
                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    embedding = reid.get_embedding(rgb_crop)
                    if embedding is not None:
                        embeddings.append(embedding)
                        if len(embeddings) >= max_samples:
                            break
        frame_index += 1
    capture.release()
    return embeddings, 0


def load_labeled_embeddings(root, reid, video_model, sample_seconds, max_samples):
    grouped = {}
    unreadable = 0
    detector = None
    for person_id in sorted(os.listdir(root)):
        person_dir = os.path.join(root, person_id)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []
        for current, _, filenames in os.walk(person_dir):
            for filename in filenames:
                extension = os.path.splitext(filename)[1].lower()
                if extension not in EXTENSIONS + VIDEO_EXTENSIONS:
                    continue
                media_path = os.path.join(current, filename)
                if extension in VIDEO_EXTENSIONS:
                    if detector is None:
                        from ultralytics import YOLO
                        detector = YOLO(video_model)
                    video_embeddings, failed = load_video_embeddings(
                        media_path, detector, reid, sample_seconds, max_samples
                    )
                    unreadable += failed
                    embeddings.extend(video_embeddings)
                else:
                    image = cv2.imread(media_path)
                    if image is None:
                        unreadable += 1
                        continue
                    # Image files should be person-body crops. Full recordings
                    # are supported through the video path above.
                    rgb_crop = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    embedding = reid.get_embedding(rgb_crop)
                    if embedding is not None:
                        embeddings.append(embedding)
        if embeddings:
            grouped[person_id] = embeddings
    return grouped, unreadable


def pair_scores(grouped):
    same = []
    different = []
    identities = sorted(grouped)
    for i, identity_a in enumerate(identities):
        for index_a, embedding_a in enumerate(grouped[identity_a]):
            for index_b in range(index_a + 1, len(grouped[identity_a])):
                same.append(BodyReID.similarity(embedding_a, grouped[identity_a][index_b]))
            for identity_b in identities[i + 1:]:
                different.extend(
                    BodyReID.similarity(embedding_a, embedding_b)
                    for embedding_b in grouped[identity_b]
                )
    return np.asarray(same), np.asarray(different)


def metrics(same, different, threshold):
    tp = int(np.count_nonzero(same >= threshold))
    fn = len(same) - tp
    fp = int(np.count_nonzero(different >= threshold))
    tn = len(different) - fp
    frr = fn / len(same) if len(same) else float("nan")
    far = fp / len(different) if len(different) else float("nan")
    balanced = ((1 - frr) + (1 - far)) / 2 if np.isfinite(frr) and np.isfinite(far) else float("nan")
    return {"threshold": threshold, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "frr": frr, "far": far, "balanced_accuracy": balanced}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="directory containing one subfolder per identity")
    parser.add_argument("--step", type=float, default=0.01, help="threshold sweep step")
    parser.add_argument("--sample-seconds", type=float, default=1.0,
                        help="seconds between sampled video frames (default: 1)")
    parser.add_argument("--max-samples-per-video", type=int, default=60,
                        help="maximum body crops extracted from each clip")
    parser.add_argument("--yolo-model", default="yolov8s.pt",
                        help="local Ultralytics person-detector weights")
    args = parser.parse_args()

    root = os.path.abspath(args.dataset)
    if not os.path.isdir(root):
        raise SystemExit("Dataset directory does not exist.")
    if args.step <= 0 or args.step > 0.1:
        raise SystemExit("--step must be greater than 0 and at most 0.1.")

    print("Loading OSNet and held-out crops…")
    reid = BodyReID()
    if args.sample_seconds <= 0:
        raise SystemExit("--sample-seconds must be greater than zero.")
    if args.max_samples_per_video < 2:
        raise SystemExit("--max-samples-per-video must be at least 2.")
    grouped, unreadable = load_labeled_embeddings(
        root, reid, args.yolo_model, args.sample_seconds,
        args.max_samples_per_video,
    )
    same, different = pair_scores(grouped)
    crop_count = sum(map(len, grouped.values()))
    print("OSNet pair evaluation (held-out crops)")
    print(f"Identities with usable crops: {len(grouped)}")
    print(f"Usable crops: {crop_count}")
    print(f"Unreadable files: {unreadable}")
    print(f"Same-identity pairs: {len(same)}")
    print(f"Different-identity pairs: {len(different)}")
    if len(grouped) < 2 or not len(same) or not len(different):
        raise SystemExit("Need at least two identities and two usable crops per identity.")

    same_threshold = float(getattr(config, "REID_SAME_CAMERA_THRESHOLD", 0.85))
    cross_threshold = float(getattr(
        config, "REID_CROSS_CAMERA_THRESHOLD", getattr(config, "REID_THRESHOLD", 0.88)
    ))
    for label, threshold in (("Current same-camera threshold", same_threshold),
                             ("Current cross-camera threshold", cross_threshold)):
        result = metrics(same, different, threshold)
        print(f"{label} ({threshold:.2f}): FAR={result['far']:.1%}, "
              f"FRR={result['frr']:.1%}, balanced accuracy={result['balanced_accuracy']:.1%}")

    thresholds = np.arange(0.0, 1.0001, args.step)
    best = max((metrics(same, different, float(value)) for value in thresholds),
               key=lambda item: item["balanced_accuracy"])
    print(f"Validation-only candidate threshold: {best['threshold']:.2f} "
          f"(FAR={best['far']:.1%}, FRR={best['frr']:.1%}, "
          f"balanced accuracy={best['balanced_accuracy']:.1%})")
    print("Use this candidate only after checking it on a separate held-out test set.")
    print("This is pair verification; it does not represent full tracking accuracy.")


if __name__ == "__main__":
    main()
