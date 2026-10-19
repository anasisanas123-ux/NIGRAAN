"""Verify the local gun/knife model, optionally against a supplied image."""
import argparse
import json

import cv2
import numpy as np

import config
from python_engine.weapon_detector import create_default_weapon_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', help='Optional local test image; no files are modified.')
    args = parser.parse_args()
    detector = create_default_weapon_detector(config)
    detector._session = detector._load_session()
    frame = cv2.imread(args.image) if args.image else np.zeros((720, 1280, 3), np.uint8)
    if frame is None:
        raise ValueError('The supplied test image could not be opened.')
    detections = detector._infer(frame)
    print('PASS: gun/knife ONNX model loaded and completed inference on CPU.')
    print(json.dumps({'detections': detections, 'thresholds': detector.thresholds}))
    if not args.image:
        print('Synthetic-frame smoke test only; validate detections on representative images and footage.')


if __name__ == '__main__':
    main()
