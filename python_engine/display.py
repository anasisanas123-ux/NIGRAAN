"""Display already rendered preview frames, without repeating annotation."""

import queue
import time
import cv2
import config


def run_display(results):
    windows = set()
    if not getattr(config, "SHOW_DESKTOP_WINDOWS", True):
        while True:
            time.sleep(1)
    try:
        while True:
            try:
                camera, frame, _, _ = results.get(timeout=0.03)
                name = f"Campus Surveillance - {camera}"
                if name not in windows:
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(name, *getattr(config, "PREVIEW_SIZE", (640, 480)))
                    windows.add(name)
                cv2.imshow(name, frame)
            except queue.Empty:
                pass
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
