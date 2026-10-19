from pathlib import Path

path = Path("multi_camera.py")
text = path.read_text(encoding="utf-8")


# ============================================================
# 1. IMPORT GAIT MODULES
# ============================================================

import_marker = (
    "from python_engine.anonymous_face_reid "
    "import AnonymousFaceReID"
)

import_block = """
from python_engine.gait_pose import GaitPoseExtractor
from python_engine.gait_signature import GaitSignatureEngine
"""

if "from python_engine.gait_pose import GaitPoseExtractor" not in text:

    if import_marker not in text:

        raise SystemExit(
            "ERROR: anonymous face import marker not found"
        )

    text = text.replace(
        import_marker,
        import_marker + import_block,
        1
    )


# ============================================================
# 2. ADD PER-CAMERA PASSIVE GAIT STATE
# ============================================================

stream_marker = """    stream = CameraStream(
        source
    )
"""

state_block = r'''

    # ========================================================
    # PASSIVE GAIT OBSERVATION
    #
    # IMPORTANT:
    #
    # Gait is NOT allowed to assign/change an identity here.
    #
    # Current ArcFace + anonymous face + OSNet + camera locks
    # remain the authority.
    #
    # Gait only observes their final UNK identity and reports
    # whether the gait signature agrees with previous samples.
    # ========================================================

    GAIT_OBSERVATION_ENABLED = True

    GAIT_SEQUENCE_FRAMES = 32

    # About 5-6 gait samples / second.
    # Pose does NOT run on every video frame.
    GAIT_POSE_INTERVAL = 0.18

    # Do not collect multiple copies of nearly identical
    # frames too quickly.
    GAIT_SAMPLE_INTERVAL = 0.15

    # Current global identity must remain stable before its
    # sequence may be written to the observation gallery.
    GAIT_ID_STABLE_TIME = 3.0

    # Drop abandoned local-track buffers.
    GAIT_TRACK_TTL = 5.0

    # Conservative body <-> pose association.
    GAIT_MIN_ASSOC_IOU = 0.45


    # Lazy initialization:
    #
    # cam_b being offline will NOT load another pose model.
    # The model loads only after a confirmed unknown actually
    # appears on this camera.
    gait_pose = None
    gait_signature = None
    gait_init_attempted = False

    gait_last_pose_run = 0.0

    gait_track_state = {}


    def gait_bbox_iou(
        box_a,
        box_b
    ):

        try:

            ax1, ay1, ax2, ay2 = [
                float(v)
                for v in box_a
            ]

            bx1, by1, bx2, by2 = [
                float(v)
                for v in box_b
            ]


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


            area_a = max(
                0.0,
                ax2 - ax1
            ) * max(
                0.0,
                ay2 - ay1
            )


            area_b = max(
                0.0,
                bx2 - bx1
            ) * max(
                0.0,
                by2 - by1
            )


            union = (
                area_a
                +
                area_b
                -
                intersection
            )


            if union <= 0:

                return 0.0


            return float(
                intersection
                /
                union
            )


        except Exception:

            return 0.0

'''

if "PASSIVE GAIT OBSERVATION" not in text:

    if stream_marker not in text:

        raise SystemExit(
            "ERROR: CameraStream marker not found"
        )

    text = text.replace(
        stream_marker,
        stream_marker + state_block,
        1
    )


# ============================================================
# 3. ADD OBSERVATION AFTER FINAL RESULT DEDUP
# ============================================================

result_marker = """            results = final_results
"""

observation_block = r'''

            # =================================================
            # PASSIVE GAIT OBSERVATION
            #
            # This executes AFTER final identity dedup.
            #
            # It NEVER writes to:
            #
            #     global_id
            #     locked_ids
            #     confirmed_unknown_ids
            #     ReIDManager
            #     anonymous_face_reid
            #
            # It is diagnostic only.
            # =================================================

            if GAIT_OBSERVATION_ENABLED:

                gait_candidates = []


                # ---------------------------------------------
                # USE CONFIRMED UNKNOWN PEOPLE ONLY
                # ---------------------------------------------

                for gait_item in results:

                    if gait_item.get(
                        'is_known',
                        False
                    ):

                        continue


                    gait_gid = gait_item.get(
                        'global_id'
                    )


                    if gait_gid is None:

                        continue


                    gait_gid = str(
                        gait_gid
                    )


                    if not gait_gid.startswith(
                        'UNK-'
                    ):

                        continue


                    gait_tid = gait_item.get(
                        'tracker_id'
                    )


                    if gait_tid is None:

                        continue


                    gait_bbox = (
                        gait_item.get(
                            'body_bbox'
                        )
                        or
                        gait_item.get(
                            'bbox'
                        )
                    )


                    if gait_bbox is None:

                        continue


                    gait_candidates.append({
                        'item':
                            gait_item,

                        'global_id':
                            gait_gid,

                        'tracker_id':
                            gait_tid,

                        'tracker_key':
                            f"{cam_id}_{gait_tid}",

                        'bbox':
                            gait_bbox
                    })


                # ---------------------------------------------
                # CLEAN ABANDONED TRACK BUFFERS
                # ---------------------------------------------

                stale_gait_keys = [

                    key

                    for key, value
                    in gait_track_state.items()

                    if (
                        now
                        -
                        float(
                            value.get(
                                'last_seen',
                                0.0
                            )
                        )
                        >
                        GAIT_TRACK_TTL
                    )
                ]


                for stale_key in stale_gait_keys:

                    gait_track_state.pop(
                        stale_key,
                        None
                    )


                # ---------------------------------------------
                # THROTTLED POSE INFERENCE
                # ---------------------------------------------

                should_run_gait_pose = (

                    bool(
                        gait_candidates
                    )

                    and

                    (
                        now
                        -
                        gait_last_pose_run
                    )
                    >=
                    GAIT_POSE_INTERVAL
                )


                if should_run_gait_pose:

                    gait_last_pose_run = now


                    # -----------------------------------------
                    # LAZY MODEL INITIALIZATION
                    # -----------------------------------------

                    if (
                        not gait_init_attempted
                    ):

                        gait_init_attempted = True


                        try:

                            gait_pose = (
                                GaitPoseExtractor()
                            )


                            gait_signature = (
                                GaitSignatureEngine(
                                    gallery_dir=(
                                        "data/"
                                        "gait_gallery_"
                                        "runtime_observation"
                                    ),
                                    sequence_length=(
                                        GAIT_SEQUENCE_FRAMES
                                    )
                                )
                            )


                            print(
                                f"[GAIT OBS MODE] "
                                f"{cam_id} ready"
                            )


                        except Exception as e:

                            gait_pose = None
                            gait_signature = None


                            print(
                                f"[GAIT OBS INIT ERROR] "
                                f"{cam_id}: {e}"
                            )


                    # -----------------------------------------
                    # POSE EXTRACTION
                    # -----------------------------------------

                    gait_people = []


                    if (
                        gait_pose is not None
                        and
                        gait_signature is not None
                    ):

                        try:

                            gait_people = (
                                gait_pose.extract(
                                    frame_bgr
                                )
                            )


                        except Exception as e:

                            print(
                                f"[GAIT OBS ERROR] "
                                f"{cam_id}: {e}"
                            )

                            gait_people = []


                    # -----------------------------------------
                    # ONE-TO-ONE BODY <-> POSE ASSOCIATION
                    #
                    # Important when multiple people exist.
                    #
                    # One pose cannot be assigned to two
                    # BoT-SORT results.
                    # -----------------------------------------

                    gait_pairs = []


                    for candidate_index, candidate \
                    in enumerate(
                        gait_candidates
                    ):

                        for pose_index, pose_person \
                        in enumerate(
                            gait_people
                        ):

                            pose_bbox = (
                                pose_person.get(
                                    'bbox'
                                )
                            )


                            if pose_bbox is None:

                                continue


                            overlap = gait_bbox_iou(
                                candidate[
                                    'bbox'
                                ],
                                pose_bbox
                            )


                            if (
                                overlap
                                >=
                                GAIT_MIN_ASSOC_IOU
                            ):

                                gait_pairs.append(
                                    (
                                        overlap,
                                        candidate_index,
                                        pose_index
                                    )
                                )


                    gait_pairs.sort(
                        key=lambda value:
                            value[0],
                        reverse=True
                    )


                    used_gait_candidates = set()
                    used_gait_poses = set()


                    for (
                        association_iou,
                        candidate_index,
                        pose_index
                    ) in gait_pairs:

                        if (
                            candidate_index
                            in
                            used_gait_candidates
                        ):

                            continue


                        if (
                            pose_index
                            in
                            used_gait_poses
                        ):

                            continue


                        used_gait_candidates.add(
                            candidate_index
                        )

                        used_gait_poses.add(
                            pose_index
                        )


                        candidate = (
                            gait_candidates[
                                candidate_index
                            ]
                        )

                        pose_person = (
                            gait_people[
                                pose_index
                            ]
                        )


                        tracker_key = candidate[
                            'tracker_key'
                        ]

                        global_id = candidate[
                            'global_id'
                        ]


                        # -------------------------------------
                        # TRACK STATE
                        # -------------------------------------

                        state = (
                            gait_track_state.get(
                                tracker_key
                            )
                        )


                        if state is None:

                            state = {

                                'global_id':
                                    global_id,

                                'gid_since':
                                    now,

                                'features':
                                    [],

                                'last_sample':
                                    0.0,

                                'last_seen':
                                    now
                            }


                            gait_track_state[
                                tracker_key
                            ] = state


                        # -------------------------------------
                        # IF CURRENT SYSTEM CHANGES THE ID,
                        # DISCARD THE PARTIAL GAIT SEQUENCE.
                        #
                        # Prevents:
                        #
                        # UNK-001 frames
                        # +
                        # temporary UNK-002 frames
                        #
                        # being mixed together.
                        # -------------------------------------

                        if (
                            state.get(
                                'global_id'
                            )
                            !=
                            global_id
                        ):

                            print(
                                f"[GAIT OBS RESET] "
                                f"{tracker_key} "
                                f"{state.get('global_id')} "
                                f"-> {global_id}"
                            )


                            state[
                                'global_id'
                            ] = global_id

                            state[
                                'gid_since'
                            ] = now

                            state[
                                'features'
                            ] = []

                            state[
                                'last_sample'
                            ] = 0.0


                        state[
                            'last_seen'
                        ] = now


                        # -------------------------------------
                        # CREATE ONE FRAME GAIT FEATURE
                        # -------------------------------------

                        try:

                            gait_feature = (
                                gait_signature
                                .frame_feature(
                                    pose_person[
                                        'keypoints'
                                    ],
                                    pose_person[
                                        'bbox'
                                    ]
                                )
                            )


                        except Exception:

                            gait_feature = None


                        if gait_feature is None:

                            continue


                        if (
                            now
                            -
                            float(
                                state.get(
                                    'last_sample',
                                    0.0
                                )
                            )
                            <
                            GAIT_SAMPLE_INTERVAL
                        ):

                            continue


                        state[
                            'last_sample'
                        ] = now


                        state[
                            'features'
                        ].append(
                            gait_feature
                        )


                        feature_count = len(
                            state[
                                'features'
                            ]
                        )


                        # -------------------------------------
                        # SMALL PROGRESS LOG
                        # -------------------------------------

                        if (
                            feature_count
                            in
                            (
                                8,
                                16,
                                24
                            )
                        ):

                            print(
                                f"[GAIT COLLECT] "
                                f"{tracker_key} "
                                f"{global_id} "
                                f"{feature_count}/"
                                f"{GAIT_SEQUENCE_FRAMES}"
                            )


                        if (
                            feature_count
                            <
                            GAIT_SEQUENCE_FRAMES
                        ):

                            continue


                        # -------------------------------------
                        # CURRENT GLOBAL ID MUST HAVE BEEN
                        # STABLE LONG ENOUGH
                        # -------------------------------------

                        identity_age = (

                            now

                            -

                            float(
                                state.get(
                                    'gid_since',
                                    now
                                )
                            )
                        )


                        if (
                            identity_age
                            <
                            GAIT_ID_STABLE_TIME
                        ):

                            # Keep latest frames, but don't
                            # write a signature yet.

                            state[
                                'features'
                            ] = state[
                                'features'
                            ][
                                -GAIT_SEQUENCE_FRAMES:
                            ]

                            continue


                        # -------------------------------------
                        # BUILD GAIT SIGNATURE
                        # -------------------------------------

                        try:

                            signature = (
                                gait_signature
                                .build_signature(
                                    state[
                                        'features'
                                    ][
                                        -GAIT_SEQUENCE_FRAMES:
                                    ]
                                )
                            )


                        except Exception as e:

                            print(
                                f"[GAIT SIGNATURE ERROR] "
                                f"{tracker_key}: {e}"
                            )

                            signature = None


                        # Start a fresh non-overlapping
                        # walking sequence.

                        state[
                            'features'
                        ] = []


                        if signature is None:

                            continue


                        # -------------------------------------
                        # MATCH EXISTING GAIT OBSERVATIONS
                        # BEFORE SAVING CURRENT SAMPLE
                        # -------------------------------------

                        try:

                            (
                                gait_best,
                                gait_score,
                                gait_margin
                            ) = gait_signature.match(
                                signature
                            )


                        except Exception as e:

                            print(
                                f"[GAIT MATCH ERROR] "
                                f"{tracker_key}: {e}"
                            )

                            gait_best = None
                            gait_score = 0.0
                            gait_margin = 0.0


                        if gait_best is None:

                            agreement = "N/A"

                            best_display = "none"


                        else:

                            agreement = (
                                "YES"
                                if
                                str(
                                    gait_best
                                )
                                ==
                                str(
                                    global_id
                                )
                                else
                                "NO"
                            )

                            best_display = str(
                                gait_best
                            )


                        print(
                            f"[GAIT OBSERVE] "
                            f"{tracker_key} "
                            f"assigned={global_id} "
                            f"best={best_display} "
                            f"score={gait_score:.3f} "
                            f"margin={gait_margin:.3f} "
                            f"agree={agreement} "
                            f"iou={association_iou:.3f}"
                        )


                        # -------------------------------------
                        # OBSERVATION GALLERY ONLY
                        #
                        # This is intentionally separate from:
                        #
                        # data/reid_gallery
                        # data/anonymous_face_gallery
                        #
                        # Deleting this gait folder cannot
                        # affect your current recognition.
                        # -------------------------------------

                        try:

                            gait_signature.save(
                                global_id,
                                signature
                            )


                        except Exception as e:

                            print(
                                f"[GAIT SAVE ERROR] "
                                f"{tracker_key}: {e}"
                            )

'''

if "PASSIVE GAIT OBSERVATION\n            #" not in text:

    if result_marker not in text:

        raise SystemExit(
            "ERROR: final result marker not found"
        )

    text = text.replace(
        result_marker,
        result_marker + observation_block,
        1
    )


path.write_text(
    text,
    encoding="utf-8"
)

print(
    "PASSIVE GAIT OBSERVATION INSTALLED"
)
