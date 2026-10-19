"""Detection logging and shared numeric helpers."""

import requests
import numpy as np
import config


def post_detection(data):

    try:

        response = requests.post(config.NODE_DETECTION_URL, json=data, timeout=1.0)

        if response.status_code not in (200, 201):

            print(
                "[LOG ERROR] "
                f"Node returned "
                f"{response.status_code}: "
                f"{response.text[:200]}"
            )

    except Exception as e:

        print(f"[LOG ERROR] " f"{e}")


def emb_valid(emb):

    return emb is not None and isinstance(emb, np.ndarray) and emb.size > 0


def iou(a, b):

    ax1, ay1, ax2, ay2 = a

    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)

    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)

    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    if inter == 0:

        return 0.0

    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter

    return inter / max(union, 1)
