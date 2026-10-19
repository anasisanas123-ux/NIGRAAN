"""Compose unchanged models and run the multi-camera application."""

import sys
import os
import threading
from pathlib import Path
import torch
import config
from python_engine.arcface_recognizer import ArcFaceRecognizer
from python_engine.body_detector import BodyDetector
from python_engine.body_reid import BodyReID
from python_engine.annotator import Annotator
from python_engine.reid_manager import ReIDManager
from python_engine.anonymous_face_reid import AnonymousFaceReID
from python_engine.engine_api import start_engine, set_identity_recognizer
from python_engine.runtime_context import RuntimeContext
from python_engine.camera_worker import camera_worker
from python_engine.frame_buffer import LatestCameraResults
from python_engine.display import run_display


def main():
    # Resolve model/config paths consistently when launched from another folder.
    os.chdir(Path(__file__).resolve().parents[1])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    print("Loading models...")
    context = RuntimeContext(
        recognizer=ArcFaceRecognizer(),
        body_detector=BodyDetector(),
        body_reid=BodyReID(),
        reid=ReIDManager(),
        anonymous_face_reid=AnonymousFaceReID(),
        annotator=Annotator(),
    )
    set_identity_recognizer(context.recognizer)
    threading.Thread(target=start_engine, daemon=True, name="engine-api").start()
    results = LatestCameraResults()
    started = 0
    try:
        for camera, source in config.CAMERA_SOURCES.items():
            if source is None or source == "":
                continue
            threading.Thread(
                target=camera_worker,
                args=(camera, source, results, context),
                daemon=True,
                name=f"inference-{camera}",
            ).start()
            started += 1
        if not started:
            raise RuntimeError("No camera sources configured.")
        print(f"{started} camera workers started. Press Q in a camera window to quit.")
        run_display(results)
    finally:
        context.stop()


if __name__ == "__main__":
    main()
