from pathlib import Path

path = Path("multi_camera.py")
text = path.read_text(encoding="utf-8")

if "[ANON FACE MATCH]" in text:
    print("Anonymous face integration already installed.")
    raise SystemExit(0)


# ============================================================
# 1. Carry face embedding into UNKNOWN PERSON processing
# ============================================================

old_1 = """                body_crop = det.get(
                    'body_crop'
                )


                body_emb = None
"""

new_1 = """                body_crop = det.get(
                    'body_crop'
                )


                face_embedding = det.get(
                    'face_embedding'
                )


                body_emb = None
"""

if old_1 not in text:
    raise SystemExit(
        "ERROR: Could not find body_crop insertion point"
    )

text = text.replace(
    old_1,
    new_1,
    1
)


# ============================================================
# 2. Try anonymous ArcFace BEFORE body-only ReID
# ============================================================

old_2 = """                    if emb_valid(
                        body_emb
                    ):

                        try:
"""

new_2 = """                    # =========================================
                    # ANONYMOUS FACE RE-ID
                    #
                    # Face is the strongest signal for an
                    # already-known UNKNOWN identity.
                    # =========================================

                    current_body_bbox = det.get(
                        'body_bbox'
                    )

                    if current_body_bbox is None:

                        current_body_bbox = det.get(
                            'bbox'
                        )


                    if face_embedding is not None:

                        try:

                            (
                                face_global_id,
                                face_reid_score,
                                face_reid_margin
                            ) = anonymous_face_reid.match(
                                face_embedding
                            )


                            if face_global_id is not None:

                                face_bound = (
                                    reid.bind_track_identity(

                                        track_key=
                                            tracker_key,

                                        global_id=
                                            face_global_id,

                                        camera_id=
                                            cam_id,

                                        body_embedding=(
                                            body_emb
                                            if emb_valid(
                                                body_emb
                                            )
                                            else None
                                        ),

                                        bbox=
                                            current_body_bbox
                                    )
                                )


                                if face_bound:

                                    global_id = (
                                        face_global_id
                                    )

                                    reid_score = (
                                        face_reid_score
                                    )


                                    cams = (
                                        reid
                                        .get_cameras_for(
                                            global_id
                                        )
                                    )


                                    reid_cameras = (

                                        list(cams)

                                        if cams

                                        else [cam_id]
                                    )


                                    margin_text = (

                                        "n/a"

                                        if face_reid_margin
                                        is None

                                        else
                                        f"{face_reid_margin:.3f}"
                                    )


                                    print(
                                        f"[ANON FACE MATCH] "
                                        f"{tracker_key} "
                                        f"-> {global_id} "
                                        f"score="
                                        f"{face_reid_score:.3f} "
                                        f"margin="
                                        f"{margin_text}"
                                    )


                            elif face_reid_score > 0.0:

                                margin_text = (

                                    "n/a"

                                    if face_reid_margin
                                    is None

                                    else
                                    f"{face_reid_margin:.3f}"
                                )


                                print(
                                    f"[ANON FACE NO MATCH] "
                                    f"{tracker_key} "
                                    f"best="
                                    f"{face_reid_score:.3f} "
                                    f"margin="
                                    f"{margin_text}"
                                )


                        except Exception as e:

                            print(
                                f"[ANON FACE ERROR] "
                                f"{tracker_key}: "
                                f"{e}"
                            )


                    # =========================================
                    # FALL BACK TO OSNET / BODY RE-ID
                    #
                    # Only run body identity creation if
                    # anonymous face did NOT already restore
                    # a global UNK identity.
                    # =========================================

                    if (
                        global_id
                        is None

                        and

                        emb_valid(
                            body_emb
                        )
                    ):

                        try:
"""

if old_2 not in text:
    raise SystemExit(
        "ERROR: Could not find OSNet insertion point"
    )

text = text.replace(
    old_2,
    new_2,
    1
)


# ============================================================
# 3. Save face samples once a global UNK is confirmed
# ============================================================

marker_3 = """                # =============================================
                # DISPLAY UNKNOWN UNTIL GLOBAL ID CONFIRMED
                # =============================================
"""

new_3 = """                # =============================================
                # PERSIST ANONYMOUS FACE FOR CONFIRMED UNK
                # =============================================

                if (
                    global_id
                    is not None

                    and

                    face_embedding
                    is not None
                ):

                    try:

                        # If this track is using the short-term
                        # unknown cache, only add the new face
                        # sample when ArcFace itself agrees with
                        # the cached global identity.
                        allow_face_add = True


                        if (
                            skip_reid

                            and

                            anonymous_face_reid
                            .get_sample_count(
                                global_id
                            )
                            > 0
                        ):

                            (
                                cached_face_id,
                                cached_face_score,
                                cached_face_margin
                            ) = anonymous_face_reid.match(
                                face_embedding
                            )


                            allow_face_add = (
                                cached_face_id
                                ==
                                global_id
                            )


                        if allow_face_add:

                            anonymous_face_reid.add(
                                global_id,
                                face_embedding
                            )


                    except Exception as e:

                        print(
                            f"[ANON FACE SAVE ERROR] "
                            f"{tracker_key}: "
                            f"{e}"
                        )


                # =============================================
                # DISPLAY UNKNOWN UNTIL GLOBAL ID CONFIRMED
                # =============================================
"""

if marker_3 not in text:
    raise SystemExit(
        "ERROR: Could not find display-name insertion point"
    )

text = text.replace(
    marker_3,
    new_3,
    1
)


path.write_text(
    text,
    encoding="utf-8"
)

print("ANONYMOUS FACE INTEGRATION INSTALLED")
