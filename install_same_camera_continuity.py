from pathlib import Path

path = Path("multi_camera.py")

text = path.read_text(
    encoding="utf-8"
)


# ============================================================
# 1. SETTINGS
# ============================================================

marker = "FAR_PERSON_MIN_HEIGHT_RATIO = 0.30"

settings = r'''

# ============================================================
# SAME-CAMERA TRACK CONTINUITY
#
# Purpose:
#
# A confirmed UNK may receive a NEW BoT-SORT local ID when:
#
# - person becomes far
# - partial occlusion happens
# - tracker temporarily drops
#
# This bridge restores the RECENT SAME-CAMERA identity only
# when body appearance strongly supports the SAME global ID.
#
# This is NOT a general OSNet threshold reduction.
# ============================================================

SAME_CAM_CONTINUITY_TTL = 3.0

# Strong body evidence required.
SAME_CAM_CONTINUITY_MIN_BODY_SCORE = 0.62

# Best identity must clearly beat second place.
SAME_CAM_CONTINUITY_MIN_BODY_MARGIN = 0.18

# New tracker must still be reasonably near the last
# position of that SAME identity.
SAME_CAM_CONTINUITY_MAX_CENTER = 0.38

# Allows scale change when person walks toward/away camera.
SAME_CAM_CONTINUITY_MIN_AREA_RATIO = 0.25

'''

if "SAME_CAM_CONTINUITY_TTL" not in text:

    if marker not in text:

        raise SystemExit(
            "ERROR: FAR_PERSON marker not found"
        )

    text = text.replace(
        marker,
        marker + settings,
        1
    )


# ============================================================
# 2. PER-CAMERA MEMORY
# ============================================================

marker = "    far_person_last_log = {}"

state = r'''

    # global_id -> most recent confirmed observation
    #
    # {
    #     'bbox': ...,
    #     'tracker_key': ...,
    #     'time': ...
    # }
    recent_same_camera_identities = {}

'''

if "recent_same_camera_identities = {}" not in text:

    if marker not in text:

        raise SystemExit(
            "ERROR: far_person_last_log marker not found"
        )

    text = text.replace(
        marker,
        marker + state,
        1
    )


# ============================================================
# 3. CONTINUITY GEOMETRY HELPER
# ============================================================

marker = """    def gait_bbox_iou(
        box_a,
        box_b
    ):
"""

helper = r'''    def same_camera_continuity_geometry(
        previous_bbox,
        current_bbox
    ):
        """
        Returns:
            center_distance,
            area_ratio

        center_distance is normalized by the average person
        box diagonal rather than full-frame size.

        This lets a far person move somewhat while still
        preventing arbitrary identity jumps across the frame.
        """

        try:

            (
                px1,
                py1,
                px2,
                py2
            ) = [
                float(v)
                for v
                in previous_bbox
            ]


            (
                cx1,
                cy1,
                cx2,
                cy2
            ) = [
                float(v)
                for v
                in current_bbox
            ]


            prev_w = max(
                px2 - px1,
                1.0
            )

            prev_h = max(
                py2 - py1,
                1.0
            )


            curr_w = max(
                cx2 - cx1,
                1.0
            )

            curr_h = max(
                cy2 - cy1,
                1.0
            )


            prev_center_x = (
                px1 + px2
            ) / 2.0

            prev_center_y = (
                py1 + py2
            ) / 2.0


            curr_center_x = (
                cx1 + cx2
            ) / 2.0

            curr_center_y = (
                cy1 + cy2
            ) / 2.0


            dx = (
                curr_center_x
                -
                prev_center_x
            )

            dy = (
                curr_center_y
                -
                prev_center_y
            )


            center_distance_pixels = (
                dx * dx
                +
                dy * dy
            ) ** 0.5


            prev_diag = (
                prev_w * prev_w
                +
                prev_h * prev_h
            ) ** 0.5


            curr_diag = (
                curr_w * curr_w
                +
                curr_h * curr_h
            ) ** 0.5


            normalizer = max(
                (
                    prev_diag
                    +
                    curr_diag
                )
                / 2.0,
                1.0
            )


            center_distance = (
                center_distance_pixels
                /
                normalizer
            )


            prev_area = (
                prev_w
                *
                prev_h
            )

            curr_area = (
                curr_w
                *
                curr_h
            )


            area_ratio = (
                min(
                    prev_area,
                    curr_area
                )
                /
                max(
                    prev_area,
                    curr_area,
                    1.0
                )
            )


            return (
                float(
                    center_distance
                ),
                float(
                    area_ratio
                )
            )


        except Exception:

            return (
                999.0,
                0.0
            )


'''

if "def same_camera_continuity_geometry(" not in text:

    if marker not in text:

        raise SystemExit(
            "ERROR: gait_bbox_iou marker not found"
        )

    text = text.replace(
        marker,
        helper + marker,
        1
    )


# ============================================================
# 4. ADD SAME-CAMERA CONTINUITY BEFORE ANON FACE MATCH
# ============================================================

marker = """                    if face_embedding is not None:

                        try:
"""

continuity = r'''                    # =================================
                    # SAME-CAMERA TRACK CONTINUITY BRIDGE
                    #
                    # Runs only when:
                    #
                    # 1. current local tracker has NO identity
                    # 2. no usable face is available
                    # 3. no near-face hold is active
                    # 4. OSNet strongly prefers one identity
                    # 5. that SAME identity was just seen on
                    #    THIS camera
                    # 6. position/scale remain plausible
                    #
                    # Multiple-person safe:
                    #
                    # we do NOT simply inherit the closest
                    # global ID.
                    # =================================

                    if (
                        global_id is None

                        and

                        face_embedding is None

                        and

                        not near_face_hold_active

                        and

                        emb_valid(
                            body_emb
                        )

                        and

                        current_body_bbox
                        is not None
                    ):

                        try:

                            continuity_rankings = (
                                reid._rank_gallery(
                                    body_emb
                                )
                            )


                        except Exception:

                            continuity_rankings = []


                        if continuity_rankings:

                            try:

                                continuity_best_id = str(
                                    continuity_rankings[
                                        0
                                    ][0]
                                )

                                continuity_best_score = float(
                                    continuity_rankings[
                                        0
                                    ][1]
                                )


                                continuity_second_score = (

                                    float(
                                        continuity_rankings[
                                            1
                                        ][1]
                                    )

                                    if
                                    len(
                                        continuity_rankings
                                    )
                                    >
                                    1

                                    else
                                    0.0
                                )


                                continuity_margin = (

                                    continuity_best_score

                                    -

                                    continuity_second_score
                                )


                            except Exception:

                                continuity_best_id = None
                                continuity_best_score = 0.0
                                continuity_margin = 0.0


                            recent_identity = (

                                recent_same_camera_identities
                                .get(
                                    continuity_best_id
                                )

                                if
                                continuity_best_id
                                is not None

                                else
                                None
                            )


                            if recent_identity is not None:

                                continuity_age = (

                                    now

                                    -

                                    float(
                                        recent_identity.get(
                                            'time',
                                            0.0
                                        )
                                    )
                                )


                                previous_tracker_key = (

                                    recent_identity.get(
                                        'tracker_key'
                                    )
                                )


                                previous_bbox = (

                                    recent_identity.get(
                                        'bbox'
                                    )
                                )


                                (
                                    continuity_center,
                                    continuity_area
                                ) = (
                                    same_camera_continuity_geometry(
                                        previous_bbox,
                                        current_body_bbox
                                    )

                                    if
                                    previous_bbox
                                    is not None

                                    else
                                    (
                                        999.0,
                                        0.0
                                    )
                                )


                                continuity_pass = (

                                    continuity_age
                                    <=
                                    SAME_CAM_CONTINUITY_TTL

                                    and

                                    previous_tracker_key
                                    !=
                                    tracker_key

                                    and

                                    continuity_best_score
                                    >=
                                    SAME_CAM_CONTINUITY_MIN_BODY_SCORE

                                    and

                                    continuity_margin
                                    >=
                                    SAME_CAM_CONTINUITY_MIN_BODY_MARGIN

                                    and

                                    continuity_center
                                    <=
                                    SAME_CAM_CONTINUITY_MAX_CENTER

                                    and

                                    continuity_area
                                    >=
                                    SAME_CAM_CONTINUITY_MIN_AREA_RATIO
                                )


                                if continuity_pass:

                                    try:

                                        continuity_bound = (
                                            reid.bind_track_identity(
                                                track_key=
                                                    tracker_key,

                                                global_id=
                                                    continuity_best_id,

                                                camera_id=
                                                    cam_id,

                                                body_embedding=
                                                    body_emb,

                                                bbox=
                                                    current_body_bbox
                                            )
                                        )


                                    except Exception as e:

                                        continuity_bound = False

                                        print(
                                            f"[REID CONTINUITY ERROR] "
                                            f"{tracker_key}: {e}"
                                        )


                                    if continuity_bound:

                                        global_id = (
                                            continuity_best_id
                                        )


                                        reid_score = (
                                            continuity_best_score
                                        )


                                        cams = (
                                            reid.get_cameras_for(
                                                global_id
                                            )
                                        )


                                        reid_cameras = (

                                            list(cams)

                                            if cams

                                            else
                                            [cam_id]
                                        )


                                        print(
                                            f"[REID CONTINUITY BIND] "
                                            f"{tracker_key} "
                                            f"-> {global_id} "
                                            f"body="
                                            f"{continuity_best_score:.3f} "
                                            f"margin="
                                            f"{continuity_margin:.3f} "
                                            f"age="
                                            f"{continuity_age:.2f}s "
                                            f"center="
                                            f"{continuity_center:.3f} "
                                            f"area="
                                            f"{continuity_area:.3f}"
                                        )


                    if face_embedding is not None:

                        try:
'''

if "[REID CONTINUITY BIND]" not in text:

    if marker not in text:

        raise SystemExit(
            "ERROR: anonymous face block marker not found"
        )

    text = text.replace(
        marker,
        continuity,
        1
    )


# ============================================================
# 5. REFRESH RECENT SAME-CAMERA IDENTITY MEMORY
#
# Do this AFTER final result dedup.
# ============================================================

marker = """            results = final_results
"""

refresh = r'''

            # =================================================
            # REFRESH SAME-CAMERA CONTINUITY MEMORY
            #
            # Only confirmed anonymous identities enter this
            # memory.
            #
            # Known students are not handled here.
            # =================================================

            for continuity_item in results:

                if continuity_item.get(
                    'is_known',
                    False
                ):

                    continue


                continuity_gid = (
                    continuity_item.get(
                        'global_id'
                    )
                )


                if continuity_gid is None:

                    continue


                continuity_gid = str(
                    continuity_gid
                )


                if not continuity_gid.startswith(
                    'UNK-'
                ):

                    continue


                continuity_tid = (
                    continuity_item.get(
                        'tracker_id'
                    )
                )


                if continuity_tid is None:

                    continue


                continuity_bbox = (

                    continuity_item.get(
                        'body_bbox'
                    )

                    or

                    continuity_item.get(
                        'bbox'
                    )
                )


                if continuity_bbox is None:

                    continue


                recent_same_camera_identities[
                    continuity_gid
                ] = {

                    'bbox':
                        list(
                            continuity_bbox
                        ),

                    'tracker_key':
                        f"{cam_id}_{continuity_tid}",

                    'time':
                        now
                }


            # Remove stale continuity memories.

            stale_continuity_ids = [

                gid

                for gid, value
                in
                recent_same_camera_identities.items()

                if (
                    now
                    -
                    float(
                        value.get(
                            'time',
                            0.0
                        )
                    )
                    >
                    (
                        SAME_CAM_CONTINUITY_TTL
                        +
                        1.0
                    )
                )
            ]


            for stale_gid in stale_continuity_ids:

                recent_same_camera_identities.pop(
                    stale_gid,
                    None
                )

'''

if "REFRESH SAME-CAMERA CONTINUITY MEMORY" not in text:

    if marker not in text:

        raise SystemExit(
            "ERROR: final results marker not found"
        )

    text = text.replace(
        marker,
        marker + refresh,
        1
    )


path.write_text(
    text,
    encoding="utf-8"
)

print(
    "SAME-CAMERA CONTINUITY BRIDGE INSTALLED"
)
