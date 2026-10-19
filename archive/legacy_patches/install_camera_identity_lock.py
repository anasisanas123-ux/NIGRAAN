from pathlib import Path

path = Path("multi_camera.py")
text = path.read_text(encoding="utf-8")


# ============================================================
# PART 1
# CAMERA-LOCAL IDENTITY LOCK STATE + GEOMETRY HELPERS
# ============================================================

state_marker = """    UNKNOWN_RECHECK = 0.8
"""

state_block = r'''

    # ========================================================
    # CAMERA-LOCAL UNKNOWN IDENTITY LOCK
    #
    # Purpose:
    #
    # BoT-SORT local ID may change while the same physical
    # person is still inside this camera.
    #
    # Example:
    #
    # cam_b_13 -> UNK-001
    # cam_b_16 -> ?
    #
    # If the new track strongly overlaps the last position of
    # UNK-001, temporarily inherit UNK-001.
    #
    # IMPORTANT:
    # This dictionary lives inside camera_worker(), therefore
    # each camera has its OWN lock state.
    # ========================================================

    CAM_ID_LOCK_TTL = 2.5

    # Body-only inheritance must still be reasonably strong.
    # A visible face does NOT use this shortcut; ArcFace gets
    # priority instead.
    CAM_ID_LOCK_MIN_BODY_CONF = 0.55

    CAM_ID_LOCK_MIN_IOU = 0.25
    CAM_ID_LOCK_MAX_CENTER = 0.30
    CAM_ID_LOCK_MIN_AREA_RATIO = 0.50
    CAM_ID_LOCK_MIN_CONTAINMENT = 0.55

    # If two existing people are nearly equally plausible,
    # don't guess.
    CAM_ID_LOCK_AMBIGUITY_MARGIN = 0.10

    # global_id -> {
    #     bbox,
    #     tracker_key,
    #     time
    # }
    camera_identity_locks = {}


    def prune_camera_identity_locks(
        current_time
    ):

        expired = []

        for gid, info in list(
            camera_identity_locks.items()
        ):

            age = (
                current_time
                -
                float(
                    info.get(
                        'time',
                        0.0
                    )
                )
            )

            if age > CAM_ID_LOCK_TTL:

                expired.append(
                    gid
                )


        for gid in expired:

            camera_identity_locks.pop(
                gid,
                None
            )


    def camera_lock_metrics(
        box_a,
        box_b
    ):

        if (
            box_a is None
            or
            box_b is None
        ):

            return None


        try:

            ax1, ay1, ax2, ay2 = [
                float(v)
                for v in box_a
            ]

            bx1, by1, bx2, by2 = [
                float(v)
                for v in box_b
            ]


            aw = max(
                1.0,
                ax2 - ax1
            )

            ah = max(
                1.0,
                ay2 - ay1
            )

            bw = max(
                1.0,
                bx2 - bx1
            )

            bh = max(
                1.0,
                by2 - by1
            )


            area_a = (
                aw
                *
                ah
            )

            area_b = (
                bw
                *
                bh
            )


            ix1 = max(
                ax1,
                bx1
            )

            iy1 = max(
                ay1,
                by1
            )

            ix2 = min(
                ax2,
                bx2
            )

            iy2 = min(
                ay2,
                by2
            )


            iw = max(
                0.0,
                ix2 - ix1
            )

            ih = max(
                0.0,
                iy2 - iy1
            )


            intersection = (
                iw
                *
                ih
            )


            union = (
                area_a
                +
                area_b
                -
                intersection
            )


            if union > 0:

                iou = (
                    intersection
                    /
                    union
                )

            else:

                iou = 0.0


            smaller_area = min(
                area_a,
                area_b
            )

            larger_area = max(
                area_a,
                area_b
            )


            if smaller_area > 0:

                containment = (
                    intersection
                    /
                    smaller_area
                )

            else:

                containment = 0.0


            if larger_area > 0:

                area_ratio = (
                    smaller_area
                    /
                    larger_area
                )

            else:

                area_ratio = 0.0


            acx = (
                ax1 + ax2
            ) / 2.0

            acy = (
                ay1 + ay2
            ) / 2.0

            bcx = (
                bx1 + bx2
            ) / 2.0

            bcy = (
                by1 + by2
            ) / 2.0


            center_distance = (

                (
                    acx - bcx
                ) ** 2

                +

                (
                    acy - bcy
                ) ** 2

            ) ** 0.5


            diag_a = (
                aw ** 2
                +
                ah ** 2
            ) ** 0.5

            diag_b = (
                bw ** 2
                +
                bh ** 2
            ) ** 0.5


            normalization = max(

                1.0,

                (
                    diag_a
                    +
                    diag_b
                ) / 2.0
            )


            normalized_center = (
                center_distance
                /
                normalization
            )


            center_quality = max(

                0.0,

                1.0
                -
                min(
                    normalized_center,
                    1.0
                )
            )


            geometry_score = (

                0.40
                *
                iou

                +

                0.25
                *
                containment

                +

                0.20
                *
                area_ratio

                +

                0.15
                *
                center_quality
            )


            return {

                'iou':
                    float(
                        iou
                    ),

                'containment':
                    float(
                        containment
                    ),

                'area_ratio':
                    float(
                        area_ratio
                    ),

                'center':
                    float(
                        normalized_center
                    ),

                'score':
                    float(
                        geometry_score
                    )
            }


        except Exception:

            return None


    def match_camera_identity_lock(
        tracker_key,
        current_bbox,
        current_time
    ):

        prune_camera_identity_locks(
            current_time
        )


        candidates = []


        for (
            global_id,
            info
        ) in camera_identity_locks.items():

            # The lock exists to bridge a LOCAL TRACK CHANGE.
            # If BoT-SORT ID did not change, normal cached ReID
            # already handles the person.
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


            # Prevent a tiny false detection inside a real
            # person's box from inheriting the person's ID.
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
                item[
                    1
                ][
                    'score'
                ],

            reverse=True
        )


        best_gid = candidates[
            0
        ][
            0
        ]

        best_metrics = candidates[
            0
        ][
            1
        ]


        # Multiple people may be standing close together.
        # Don't guess if two locks are almost equally good.
        if len(
            candidates
        ) > 1:

            second_metrics = candidates[
                1
            ][
                1
            ]


            difference = (

                best_metrics[
                    'score'
                ]

                -

                second_metrics[
                    'score'
                ]
            )


            if (
                difference
                <
                CAM_ID_LOCK_AMBIGUITY_MARGIN
            ):

                print(
                    f"[CAM LOCK AMBIGUOUS] "
                    f"{cam_id} "
                    f"{tracker_key} "
                    f"best={best_gid}:"
                    f"{best_metrics['score']:.3f} "
                    f"margin={difference:.3f}"
                )

                return (
                    None,
                    None
                )


        return (
            best_gid,
            best_metrics
        )


    def refresh_camera_identity_lock(
        global_id,
        bbox,
        tracker_key,
        current_time
    ):

        if (
            global_id is None
            or
            bbox is None
        ):

            return


        global_id = str(
            global_id
        )


        # This mechanism is only for anonymous global IDs.
        if not global_id.startswith(
            'UNK-'
        ):

            return


        prune_camera_identity_locks(
            current_time
        )


        camera_identity_locks[
            global_id
        ] = {

            'bbox':
                list(
                    bbox
                ),

            'tracker_key':
                str(
                    tracker_key
                ),

            'time':
                float(
                    current_time
                )
        }

'''

if "CAMERA-LOCAL UNKNOWN IDENTITY LOCK" not in text:

    if state_marker not in text:

        raise SystemExit(
            "ERROR: UNKNOWN_RECHECK marker not found"
        )

    text = text.replace(

        state_marker,

        state_marker
        +
        state_block,

        1
    )


# ============================================================
# PART 2
# TRY CAMERA LOCK BEFORE ANONYMOUS FACE / BODY GLOBAL SEARCH
# ============================================================

anon_marker = """                    # =========================================
                    # ANONYMOUS FACE RE-ID
"""

inherit_block = r'''                    # =========================================
                    # CAMERA-LOCAL TRACK -> GLOBAL ID INHERIT
                    #
                    # Only used when:
                    #
                    # 1. no face embedding is available
                    # 2. body detection is reasonably strong
                    # 3. a previous confirmed UNK was at almost
                    #    the same physical location
                    #
                    # ArcFace always remains higher priority.
                    # =========================================

                    current_body_bbox = det.get(
                        'body_bbox'
                    )

                    if current_body_bbox is None:

                        current_body_bbox = det.get(
                            'bbox'
                        )


                    if (
                        global_id is None

                        and

                        face_embedding is None

                        and

                        current_body_bbox is not None

                        and

                        body_confidence
                        >=
                        CAM_ID_LOCK_MIN_BODY_CONF
                    ):

                        (
                            lock_global_id,
                            lock_metrics
                        ) = match_camera_identity_lock(

                            tracker_key,

                            current_body_bbox,

                            now
                        )


                        if lock_global_id is not None:

                            try:

                                lock_bound = (
                                    reid.bind_track_identity(

                                        track_key=
                                            tracker_key,

                                        global_id=
                                            lock_global_id,

                                        camera_id=
                                            cam_id,

                                        body_embedding=
                                            body_emb,

                                        bbox=
                                            current_body_bbox
                                    )
                                )


                            except Exception as e:

                                print(
                                    f"[CAM LOCK ERROR] "
                                    f"{tracker_key}: "
                                    f"{e}"
                                )

                                lock_bound = False


                            if lock_bound:

                                global_id = (
                                    lock_global_id
                                )

                                reid_score = float(
                                    lock_metrics[
                                        'score'
                                    ]
                                )


                                print(
                                    f"[CAM LOCK BIND] "
                                    f"{tracker_key} "
                                    f"-> {global_id} "
                                    f"geom="
                                    f"{lock_metrics['score']:.3f} "
                                    f"iou="
                                    f"{lock_metrics['iou']:.3f} "
                                    f"center="
                                    f"{lock_metrics['center']:.3f} "
                                    f"area="
                                    f"{lock_metrics['area_ratio']:.3f}"
                                )


'''

if "CAMERA-LOCAL TRACK -> GLOBAL ID INHERIT" not in text:

    if anon_marker not in text:

        raise SystemExit(
            "ERROR: anonymous face ReID marker not found"
        )

    text = text.replace(

        anon_marker,

        inherit_block
        +
        anon_marker,

        1
    )


# ============================================================
# PART 3
# REFRESH LOCK USING THE FINAL DEDUPLICATED RESULT
# ============================================================

smooth_marker = """        # ====================================================
        # SMOOTH BOUNDING BOXES
        # ====================================================
"""

refresh_block = r'''        # ====================================================
        # REFRESH CAMERA IDENTITY LOCKS
        #
        # Run AFTER final result dedup so a duplicated local
        # track cannot overwrite the lock with a second box.
        # ====================================================

        for lock_item in results:

            if lock_item.get(
                'is_known',
                False
            ):

                continue


            lock_global_id = lock_item.get(
                'global_id'
            )


            if lock_global_id is None:

                continue


            lock_bbox = lock_item.get(
                'body_bbox'
            )


            if lock_bbox is None:

                lock_bbox = lock_item.get(
                    'bbox'
                )


            if lock_bbox is None:

                continue


            lock_tid = lock_item.get(
                'tracker_id'
            )


            lock_tracker_key = (
                f"{cam_id}_{lock_tid}"
            )


            refresh_camera_identity_lock(

                lock_global_id,

                lock_bbox,

                lock_tracker_key,

                now
            )


'''

if "REFRESH CAMERA IDENTITY LOCKS" not in text:

    if smooth_marker not in text:

        raise SystemExit(
            "ERROR: smoothing marker not found"
        )

    text = text.replace(

        smooth_marker,

        refresh_block
        +
        smooth_marker,

        1
    )


path.write_text(
    text,
    encoding="utf-8"
)

print(
    "CAMERA IDENTITY LOCK INSTALLED"
)
