from pathlib import Path

path = Path("multi_camera.py")
text = path.read_text(encoding="utf-8")


# ============================================================
# 1. SETTINGS
# ============================================================

marker = "ANON_FACE_HOLD_STALE_SECONDS = 1.0"

block = r'''

# ============================================================
# FAR PERSON SAFETY
#
# A small/far person is still detected and tracked, but is
# NOT allowed to create a new persistent UNK identity.
#
# Existing hard-locked identities and strong ArcFace matches
# still work normally.
# ============================================================

FAR_PERSON_MIN_HEIGHT_RATIO = 0.30

'''

if "FAR_PERSON_MIN_HEIGHT_RATIO" not in text:

    if marker not in text:
        raise SystemExit(
            "ERROR: near-face settings marker not found"
        )

    text = text.replace(
        marker,
        marker + block,
        1
    )


# ============================================================
# 2. PER-CAMERA LOG STATE
# ============================================================

marker = "    near_face_holds = {}"

block = r'''

    far_person_last_log = {}

'''

if "far_person_last_log = {}" not in text:

    if marker not in text:
        raise SystemExit(
            "ERROR: near_face_holds marker not found"
        )

    text = text.replace(
        marker,
        marker + block,
        1
    )


# ============================================================
# 3. CALCULATE WHETHER CURRENT PERSON IS TOO FAR
# ============================================================

marker = """                    near_face_hold_active = False

                    near_face_hold_candidate = None
"""

block = r'''                    # =================================
                    # FAR PERSON QUALITY CHECK
                    #
                    # Use bbox height relative to the complete
                    # frame rather than an absolute pixel size.
                    # =================================

                    far_person = False
                    body_height_ratio = 1.0


                    try:

                        if current_body_bbox is not None:

                            (
                                far_x1,
                                far_y1,
                                far_x2,
                                far_y2
                            ) = [
                                float(v)
                                for v
                                in current_body_bbox
                            ]


                            frame_height = float(
                                frame_bgr.shape[0]
                            )


                            body_height = max(
                                0.0,
                                far_y2 - far_y1
                            )


                            if frame_height > 0:

                                body_height_ratio = (
                                    body_height
                                    /
                                    frame_height
                                )


                                far_person = (
                                    body_height_ratio
                                    <
                                    FAR_PERSON_MIN_HEIGHT_RATIO
                                )


                    except Exception:

                        far_person = False
                        body_height_ratio = 1.0


                    near_face_hold_active = False

                    near_face_hold_candidate = None
'''

if "body_height_ratio = 1.0" not in text:

    if marker not in text:
        raise SystemExit(
            "ERROR: near-face initialization marker not found"
        )

    text = text.replace(
        marker,
        block,
        1
    )


# ============================================================
# 4. BLOCK BODY find_or_create WHEN PERSON IS FAR
# ============================================================

old = """                        not near_face_hold_active

                        and

                        emb_valid(
                            body_emb
                        )
"""

new = """                        not near_face_hold_active

                        and

                        not far_person

                        and

                        emb_valid(
                            body_emb
                        )
"""

if "not far_person" not in text:

    if old not in text:
        raise SystemExit(
            "ERROR: body fallback condition not found"
        )

    text = text.replace(
        old,
        new,
        1
    )


# ============================================================
# 5. DIAGNOSTIC LOG
# ============================================================

marker = """                    # =========================================
                    # FALL BACK TO OSNET / BODY RE-ID
"""

block = r'''                    if (
                        global_id is None
                        and
                        far_person
                    ):

                        previous_far_log = float(
                            far_person_last_log.get(
                                tracker_key,
                                0.0
                            )
                        )


                        if (
                            now
                            -
                            previous_far_log
                            >=
                            1.0
                        ):

                            far_person_last_log[
                                tracker_key
                            ] = now


                            print(
                                f"[REID FAR HOLD] "
                                f"{tracker_key} "
                                f"height_ratio="
                                f"{body_height_ratio:.3f} "
                                f"conf="
                                f"{body_confidence:.3f} "
                                f"no_new_id"
                            )


'''

if "[REID FAR HOLD]" not in text:

    if marker not in text:
        raise SystemExit(
            "ERROR: body fallback comment not found"
        )

    text = text.replace(
        marker,
        block + marker,
        1
    )


path.write_text(
    text,
    encoding="utf-8"
)

print("FAR PERSON HOLD INSTALLED")
