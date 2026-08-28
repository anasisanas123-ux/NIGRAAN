from pathlib import Path
import re

path = Path("multi_camera.py")
text = path.read_text(encoding="utf-8")


# ============================================================
# 1. CAMERA LOCK MUST HAVE A REAL BODY DETECTION
# ============================================================

text = text.replace(
    "CAM_ID_LOCK_MIN_BODY_CONF = 0.30",
    "CAM_ID_LOCK_MIN_BODY_CONF = 0.50"
)


# ============================================================
# 2. ADD BODY-APPEARANCE REQUIREMENTS
# ============================================================

marker = """    CAM_ID_LOCK_MIN_CONTAINMENT = 0.75
"""

block = r'''

    # Camera-lock transfer is NOT allowed from geometry alone.
    #
    # The new local track must also look like the SAME global
    # identity according to OSNet.
    CAM_ID_LOCK_MIN_BODY_SCORE = 0.52
    CAM_ID_LOCK_MIN_BODY_MARGIN = 0.08

'''

if "CAM_ID_LOCK_MIN_BODY_SCORE" not in text:

    if marker not in text:

        raise SystemExit(
            "ERROR: camera lock settings marker not found"
        )

    text = text.replace(
        marker,
        marker + block,
        1
    )


# ============================================================
# 3. REPLACE CAMERA-LOCK MATCHER
# ============================================================

start_marker = """    def match_camera_identity_lock(
"""

end_marker = """    def refresh_camera_identity_lock(
"""

start = text.find(start_marker)
end = text.find(end_marker)

if start == -1 or end == -1 or end <= start:

    raise SystemExit(
        "ERROR: camera lock function boundaries not found"
    )


new_function = r'''    def match_camera_identity_lock(
        tracker_key,
        current_bbox,
        current_time,
        body_embedding=None
    ):

        prune_camera_identity_locks(
            current_time
        )


        # ====================================================
        # APPEARANCE VERIFICATION
        #
        # Never transfer a camera lock using geometry alone.
        # ====================================================

        if body_embedding is None:

            return (
                None,
                None
            )


        try:

            rankings = reid._rank_gallery(
                body_embedding
            )

        except Exception:

            # Fallback in case internal helper changes.
            rankings = []

            try:

                for gid, identity in (
                    reid.unknown_gallery.items()
                ):

                    score = reid._identity_score(
                        body_embedding,
                        identity
                    )

                    rankings.append(
                        (
                            gid,
                            float(score)
                        )
                    )


                rankings.sort(
                    key=lambda item: item[1],
                    reverse=True
                )

            except Exception:

                rankings = []


        if not rankings:

            return (
                None,
                None
            )


        best_body_id = str(
            rankings[0][0]
        )

        best_body_score = float(
            rankings[0][1]
        )


        if len(rankings) > 1:

            second_body_score = float(
                rankings[1][1]
            )

            body_margin = (
                best_body_score
                -
                second_body_score
            )

        else:

            second_body_score = None
            body_margin = 1.0


        # Weak/ambiguous OSNet evidence:
        # do NOT inherit any camera lock.
        if (
            best_body_score
            <
            CAM_ID_LOCK_MIN_BODY_SCORE
        ):

            print(
                f"[CAM LOCK BODY REJECT] "
                f"{tracker_key} "
                f"best={best_body_id}:"
                f"{best_body_score:.3f} "
                f"reason=weak"
            )

            return (
                None,
                None
            )


        if (
            len(rankings) > 1

            and

            body_margin
            <
            CAM_ID_LOCK_MIN_BODY_MARGIN
        ):

            print(
                f"[CAM LOCK BODY REJECT] "
                f"{tracker_key} "
                f"best={best_body_id}:"
                f"{best_body_score:.3f} "
                f"margin={body_margin:.3f} "
                f"reason=ambiguous"
            )

            return (
                None,
                None
            )


        candidates = []


        for (
            global_id,
            info
        ) in camera_identity_locks.items():

            global_id = str(
                global_id
            )


            # =================================================
            # CRITICAL RULE:
            #
            # Spatial lock can ONLY transfer the identity which
            # OSNet independently ranked #1.
            # =================================================

            if global_id != best_body_id:

                continue


            # The lock exists to bridge a LOCAL TRACK CHANGE.
            if (
                str(
                    info.get(
                        'tracker_key'
                    )
                )
                ==
                str(
                    tracker_key
                )
            ):

                continue


            previous_bbox = info.get(
                'bbox'
            )


            metrics = camera_lock_metrics(
                current_bbox,
                previous_bbox
            )


            if metrics is None:

                continue


            if (
                metrics[
                    'area_ratio'
                ]
                <
                CAM_ID_LOCK_MIN_AREA_RATIO
            ):

                continue


            geometry_ok = (

                metrics[
                    'iou'
                ]
                >=
                CAM_ID_LOCK_MIN_IOU

                or

                (
                    metrics[
                        'center'
                    ]
                    <=
                    CAM_ID_LOCK_MAX_CENTER

                    and

                    metrics[
                        'containment'
                    ]
                    >=
                    CAM_ID_LOCK_MIN_CONTAINMENT
                )
            )


            if not geometry_ok:

                continue


            metrics = dict(
                metrics
            )

            metrics[
                'body_score'
            ] = best_body_score

            metrics[
                'body_margin'
            ] = body_margin


            candidates.append(
                (
                    global_id,
                    metrics
                )
            )


        if not candidates:

            return (
                None,
                None
            )


        candidates.sort(
            key=lambda item:
                item[1]['score'],
            reverse=True
        )


        best_gid = candidates[0][0]
        best_metrics = candidates[0][1]


        return (
            best_gid,
            best_metrics
        )


'''

text = (
    text[:start]
    +
    new_function
    +
    text[end:]
)


# ============================================================
# 4. PASS BODY EMBEDDING INTO CAMERA-LOCK MATCH
# ============================================================

old_call = """                        ) = match_camera_identity_lock(

                            tracker_key,

                            current_body_bbox,

                            now
                        )
"""

new_call = """                        ) = match_camera_identity_lock(

                            tracker_key,

                            current_body_bbox,

                            now,

                            body_emb
                        )
"""

if old_call in text:

    text = text.replace(
        old_call,
        new_call,
        1
    )

elif "body_emb\n                        )" not in text:

    raise SystemExit(
        "ERROR: camera lock call not found"
    )


# ============================================================
# 5. IMPROVE LOCK LOG
# ============================================================

old_log = """                                    f"area="
                                    f"{lock_metrics['area_ratio']:.3f}"
"""

new_log = """                                    f"area="
                                    f"{lock_metrics['area_ratio']:.3f} "
                                    f"body="
                                    f"{lock_metrics.get('body_score', 0.0):.3f} "
                                    f"margin="
                                    f"{lock_metrics.get('body_margin', 0.0):.3f}"
"""

if old_log in text:

    text = text.replace(
        old_log,
        new_log,
        1
    )


path.write_text(
    text,
    encoding="utf-8"
)

print(
    "APPEARANCE-VERIFIED CAMERA LOCK INSTALLED"
)
