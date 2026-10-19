from pathlib import Path

path = Path("multi_camera.py")
text = path.read_text(encoding="utf-8")


# ------------------------------------------------------------
# 1. Add configurable confidence threshold globally
# ------------------------------------------------------------

marker_global = """cross_cam_fired = set()
cross_cam_lock = threading.Lock()
"""

replacement_global = """cross_cam_fired = set()
cross_cam_lock = threading.Lock()

# YOLO may use a low threshold for tracking, but a weak
# body-only detection must not create a global UNK identity.
NEW_UNK_MIN_BODY_CONF = 0.35
"""

if "NEW_UNK_MIN_BODY_CONF" not in text:

    if marker_global not in text:
        raise SystemExit(
            "ERROR: global threshold insertion point not found"
        )

    text = text.replace(
        marker_global,
        replacement_global,
        1
    )


# ------------------------------------------------------------
# 2. Preserve YOLO BODY confidence when face is matched
# ------------------------------------------------------------

old_face = """                    'confidence':
                        fd[
                            'confidence'
                        ],

                    'face_crop':
"""

new_face = """                    'confidence':
                        fd[
                            'confidence'
                        ],

                    'body_confidence':
                        bd.get(
                            'confidence',
                            0.0
                        ),

                    'face_crop':
"""

if "'body_confidence':" not in text:

    if old_face not in text:
        raise SystemExit(
            "ERROR: face-matched confidence block not found"
        )

    text = text.replace(
        old_face,
        new_face,
        1
    )


# ------------------------------------------------------------
# 3. Preserve BODY confidence when no face is matched
# ------------------------------------------------------------

old_body = """                    'confidence':
                        prev.get(
                            'confidence',
                            0.0
                        ),

                    'face_crop':
"""

new_body = """                    'confidence':
                        prev.get(
                            'confidence',
                            0.0
                        ),

                    'body_confidence':
                        bd.get(
                            'confidence',
                            0.0
                        ),

                    'face_crop':
"""

if text.count("'body_confidence':") < 2:

    if old_body not in text:
        raise SystemExit(
            "ERROR: body-only confidence block not found"
        )

    text = text.replace(
        old_body,
        new_body,
        1
    )


# ------------------------------------------------------------
# 4. Read body confidence inside UNKNOWN PERSON flow
# ------------------------------------------------------------

old_unknown = """                face_embedding = det.get(
                    'face_embedding'
                )


                body_emb = None
"""

new_unknown = """                face_embedding = det.get(
                    'face_embedding'
                )


                body_confidence = float(
                    det.get(
                        'body_confidence',
                        0.0
                    )
                    or
                    0.0
                )


                body_emb = None
"""

if "body_confidence = float(" not in text:

    if old_unknown not in text:
        raise SystemExit(
            "ERROR: unknown flow insertion point not found"
        )

    text = text.replace(
        old_unknown,
        new_unknown,
        1
    )


# ------------------------------------------------------------
# 5. Prevent weak BODY-ONLY detections from entering OSNet ReID
# ------------------------------------------------------------

old_osnet = """                    if (
                        body_crop
                        is not None
                    ):

                        try:

                            body_emb = (
"""

new_osnet = """                    allow_body_reid = (

                        face_embedding
                        is not None

                        or

                        body_confidence
                        >=
                        NEW_UNK_MIN_BODY_CONF
                    )


                    if (
                        not allow_body_reid
                    ):

                        print(
                            f"[REID PERSON GATE] "
                            f"{tracker_key} "
                            f"blocked weak body-only detection "
                            f"conf={body_confidence:.3f}"
                        )


                    if (
                        body_crop
                        is not None

                        and

                        allow_body_reid
                    ):

                        try:

                            body_emb = (
"""

if "[REID PERSON GATE]" not in text:

    if old_osnet not in text:
        raise SystemExit(
            "ERROR: OSNet gate insertion point not found"
        )

    text = text.replace(
        old_osnet,
        new_osnet,
        1
    )


path.write_text(
    text,
    encoding="utf-8"
)

print("FALSE PERSON GATE INSTALLED")
