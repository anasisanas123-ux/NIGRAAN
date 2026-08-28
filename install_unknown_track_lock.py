from pathlib import Path

path = Path("multi_camera.py")
text = path.read_text(encoding="utf-8")


# ============================================================
# DISABLE OLD SPATIAL CAMERA LOCK IF IT WAS INSTALLED
# ============================================================

text = text.replace(
    "CAM_ID_LOCK_TTL = 2.5",
    "CAM_ID_LOCK_TTL = 0.0"
)

text = text.replace(
    "CAM_ID_LOCK_MIN_BODY_CONF = 0.55",
    "CAM_ID_LOCK_MIN_BODY_CONF = 2.0"
)


# ============================================================
# INSERT UNKNOWN TRACK LOCK BEFORE ARCFACE
# ============================================================

marker = """        # ====================================================
        # RUN ARCFACE
"""

block = r'''        # ====================================================
        # CONFIRMED UNKNOWN TRACK LOCK
        #
        # Same idea as confirmed known students:
        #
        # Once:
        #     cam_a_2 -> UNK-001
        #
        # has been confirmed, keep UNK-001 attached to
        # cam_a_2 while BoT-SORT continues returning that
        # tracker ID.
        #
        # Therefore:
        #
        # face visible        -> identify UNK-001
        # face turns away     -> still UNK-001
        # face hidden         -> still UNK-001
        # body remains tracked-> still UNK-001
        #
        # ArcFace/OSNet only run again if the local tracker
        # actually changes or disappears.
        # ====================================================

        if need_arcface_regions:

            remaining_arcface_regions = []


            for (
                tid,
                tracker_key,
                bd
            ) in need_arcface_regions:

                cached_unknown = (
                    confirmed_unknown_ids.get(
                        tracker_key
                    )
                )


                if (
                    cached_unknown is None

                    or

                    cached_unknown.get(
                        'global_id'
                    )
                    is None
                ):

                    remaining_arcface_regions.append(
                        (
                            tid,
                            tracker_key,
                            bd
                        )
                    )

                    continue


                # --------------------------------------------
                # SAME BOT-SORT TRACK IS STILL ALIVE
                # --------------------------------------------

                global_id = cached_unknown.get(
                    'global_id'
                )


                cached_unknown[
                    'time'
                ] = now


                # Keep local display/identity lock alive too.
                if tracker_key in locked_ids:

                    locked_ids[
                        tracker_key
                    ][
                        'time'
                    ] = now

                    locked_ids[
                        tracker_key
                    ][
                        'name'
                    ] = global_id


                else:

                    locked_ids[
                        tracker_key
                    ] = {

                        'is_known':
                            False,

                        'student_id':
                            None,

                        'name':
                            global_id,

                        'confidence':
                            0.0,

                        'time':
                            now
                    }


                cameras = cached_unknown.get(
                    'cameras',
                    [cam_id]
                )


                if not cameras:

                    cameras = [
                        cam_id
                    ]


                try:

                    cached_score = float(
                        cached_unknown.get(
                            'score',
                            0.0
                        )
                        or
                        0.0
                    )

                except Exception:

                    cached_score = 0.0


                # --------------------------------------------
                # BUILD RESULT DIRECTLY
                #
                # No ArcFace required here.
                # No OSNet global identity decision required.
                #
                # BoT-SORT is maintaining the person.
                # --------------------------------------------

                confirmed_results.append({

                    'face_bbox':
                        get_cached_face_bbox(
                            tracker_key,
                            bd['bbox'],
                            now
                        ),

                    'body_bbox':
                        bd['bbox'],

                    'bbox':
                        bd['bbox'],

                    'is_known':
                        False,

                    'student_id':
                        None,

                    'name':
                        global_id,

                    'confidence':
                        0.0,

                    'reid_confidence':
                        cached_score,

                    'body_confidence':
                        bd.get(
                            'confidence',
                            0.0
                        ),

                    'tracker_id':
                        tid,

                    'local_tracker_id':
                        str(
                            tid
                        ),

                    'global_id':
                        global_id,

                    'cameras':
                        cameras,

                    # Existing downstream cached-result path
                    # will append this directly.
                    '_cached':
                        True
                })


            # Only tracks which DO NOT already have a locked
            # unknown identity continue to ArcFace.
            need_arcface_regions = (
                remaining_arcface_regions
            )


'''

if "CONFIRMED UNKNOWN TRACK LOCK" not in text:

    if marker not in text:

        raise SystemExit(
            "ERROR: RUN ARCFACE marker not found"
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

print(
    "UNKNOWN TRACK LOCK INSTALLED"
)
