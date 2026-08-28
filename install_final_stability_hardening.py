from pathlib import Path

path = Path("multi_camera.py")
text = path.read_text(encoding="utf-8")


# ============================================================
# 1. REQUIRE MORE EVIDENCE BEFORE CREATING A BRAND NEW UNK
# ============================================================

reid_marker = """reid = ReIDManager()
"""

reid_block = r'''
# ============================================================
# FINAL IDENTITY STABILITY HARDENING
#
# A temporary BoT-SORT break must not create a persistent
# UNK after only a few weak body observations.
#
# Empty-gallery first identity still uses ReIDManager's
# special quick-start handling.
# ============================================================

if hasattr(reid, "new_id_frames"):

    try:

        reid.new_id_frames = max(
            int(reid.new_id_frames),
            12
        )

        print(
            f"[STABILITY] "
            f"new UNK confirmation="
            f"{reid.new_id_frames} samples"
        )

    except Exception:

        pass

'''

if "FINAL IDENTITY STABILITY HARDENING" not in text:

    if reid_marker not in text:

        raise SystemExit(
            "ERROR: ReIDManager initialization not found"
        )

    text = text.replace(
        reid_marker,
        reid_marker + reid_block,
        1
    )


# ============================================================
# 2. PER-CAMERA DISPLAY STABILITY STATE
# ============================================================

state_marker = """    confirmed_unknown_ids = {}
"""

state_block = r'''

    # ========================================================
    # DISPLAY IDENTITY STABILITY
    #
    # Internal detector/ReID can keep calculating, but the UI
    # only shows identities after they are stable.
    #
    # This prevents:
    #
    # hand -> temporary red "person" box
    #
    # and:
    #
    # UNK-001 -> UNK-003 -> UNK-001
    #
    # display flicker.
    # ========================================================

    display_identity_state = {}

    DISPLAY_EXISTING_CONFIRM_FRAMES = 2
    DISPLAY_NEW_CONFIRM_FRAMES = 4

    # Switching away from an already displayed identity must
    # be sustained longer.
    DISPLAY_SWITCH_NEW_FRAMES = 5

    # If evidence returns to an OLDER established identity,
    # allow the correction quickly.
    DISPLAY_SWITCH_OLDER_FRAMES = 2

    # Brief moment where ReID returns no ID but the same
    # BoT-SORT track is still alive.
    DISPLAY_ID_GRACE = 1.5

    DISPLAY_STATE_TTL = 8.0


    def display_identity_first_seen(
        global_id,
        current_time
    ):

        if global_id is None:

            return current_time


        try:

            identity = reid.unknown_gallery.get(
                str(global_id),
                {}
            )

            return float(
                identity.get(
                    'first_seen',
                    current_time
                )
            )

        except Exception:

            return current_time


'''

if "DISPLAY IDENTITY STABILITY" not in text:

    if state_marker not in text:

        raise SystemExit(
            "ERROR: confirmed_unknown_ids state not found"
        )

    text = text.replace(
        state_marker,
        state_marker + state_block,
        1
    )


# ============================================================
# 3. FILTER / STABILIZE RESULTS BEFORE DISPLAY
# ============================================================

smooth_marker = """        # ====================================================
        # SMOOTH BOUNDING BOXES
        # ====================================================
"""

display_block = r'''        # ====================================================
        # FINAL DISPLAY CONFIRMATION + ID HYSTERESIS
        #
        # Recognition/tracking has already happened above.
        # This block controls only what becomes a visible
        # final result.
        # ====================================================

        # Remove old display states.
        expired_display_keys = [

            key

            for key, value
            in display_identity_state.items()

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
                DISPLAY_STATE_TTL
            )
        ]


        for key in expired_display_keys:

            display_identity_state.pop(
                key,
                None
            )


        stable_results = []


        for display_item in results:

            # ------------------------------------------------
            # KNOWN STUDENTS
            #
            # Already positively recognized and tracked.
            # Show immediately.
            # ------------------------------------------------

            if (
                display_item.get(
                    'is_known',
                    False
                )

                or

                display_item.get(
                    'student_id'
                )
                is not None
            ):

                stable_results.append(
                    display_item
                )

                continue


            display_tid = display_item.get(
                'tracker_id'
            )


            display_tracker_key = (
                f"{cam_id}_{display_tid}"
            )


            display_gid = display_item.get(
                'global_id'
            )


            state = display_identity_state.get(
                display_tracker_key
            )


            # ------------------------------------------------
            # NO CONFIRMED GLOBAL ID
            #
            # Example:
            # YOLO thinks a hand / clothes / partial object
            # may be a person.
            #
            # Never draw a fresh red square for it.
            # ------------------------------------------------

            if display_gid is None:

                # If this exact BoT-SORT track already had a
                # stable identity, allow a tiny grace window
                # rather than flickering the label off.
                if (
                    state is not None

                    and

                    state.get(
                        'shown_id'
                    )
                    is not None

                    and

                    (
                        now
                        -
                        float(
                            state.get(
                                'last_seen',
                                0.0
                            )
                        )
                    )
                    <=
                    DISPLAY_ID_GRACE
                ):

                    held_id = state[
                        'shown_id'
                    ]


                    held_item = dict(
                        display_item
                    )

                    held_item[
                        'global_id'
                    ] = held_id

                    held_item[
                        'name'
                    ] = held_id


                    state[
                        'last_seen'
                    ] = now


                    stable_results.append(
                        held_item
                    )


                # New unconfirmed detection:
                # hide it completely.
                continue


            display_gid = str(
                display_gid
            )


            # ------------------------------------------------
            # FIRST OBSERVATION FOR THIS LOCAL TRACK
            # ------------------------------------------------

            if state is None:

                state = {

                    'shown_id':
                        None,

                    'candidate_id':
                        display_gid,

                    'candidate_count':
                        1,

                    'last_seen':
                        now
                }


                display_identity_state[
                    display_tracker_key
                ] = state


            else:

                state[
                    'last_seen'
                ] = now


                # --------------------------------------------
                # ID ALREADY SHOWN AND STILL AGREES
                # --------------------------------------------

                if (
                    state.get(
                        'shown_id'
                    )
                    ==
                    display_gid
                ):

                    state[
                        'candidate_id'
                    ] = None

                    state[
                        'candidate_count'
                    ] = 0


                # --------------------------------------------
                # NEW / DIFFERENT CANDIDATE
                # --------------------------------------------

                else:

                    if (
                        state.get(
                            'candidate_id'
                        )
                        ==
                        display_gid
                    ):

                        state[
                            'candidate_count'
                        ] = (
                            int(
                                state.get(
                                    'candidate_count',
                                    0
                                )
                            )
                            +
                            1
                        )


                    else:

                        state[
                            'candidate_id'
                        ] = display_gid

                        state[
                            'candidate_count'
                        ] = 1


            shown_id = state.get(
                'shown_id'
            )


            candidate_count = int(
                state.get(
                    'candidate_count',
                    0
                )
            )


            # ------------------------------------------------
            # NOTHING SHOWN YET
            # ------------------------------------------------

            if shown_id is None:

                identity_age = (

                    now
                    -
                    display_identity_first_seen(
                        display_gid,
                        now
                    )
                )


                # Existing persistent IDs can appear quickly.
                # Brand-new IDs must remain stable longer.
                if identity_age >= 2.0:

                    required = (
                        DISPLAY_EXISTING_CONFIRM_FRAMES
                    )

                else:

                    required = (
                        DISPLAY_NEW_CONFIRM_FRAMES
                    )


                if candidate_count >= required:

                    state[
                        'shown_id'
                    ] = display_gid

                    state[
                        'candidate_id'
                    ] = None

                    state[
                        'candidate_count'
                    ] = 0

                    shown_id = display_gid


                else:

                    # Still calculating.
                    # Do not draw a misleading temporary ID.
                    continue


            # ------------------------------------------------
            # SYSTEM WANTS TO CHANGE AN ALREADY DISPLAYED ID
            # ------------------------------------------------

            elif shown_id != display_gid:

                shown_first = (
                    display_identity_first_seen(
                        shown_id,
                        now
                    )
                )

                candidate_first = (
                    display_identity_first_seen(
                        display_gid,
                        now
                    )
                )


                # Example:
                #
                # temporary UNK-003
                #       ↓
                # strong evidence returns to old UNK-001
                #
                # Older established identity wins quickly.
                if candidate_first < shown_first:

                    required_switch = (
                        DISPLAY_SWITCH_OLDER_FRAMES
                    )

                else:

                    required_switch = (
                        DISPLAY_SWITCH_NEW_FRAMES
                    )


                if candidate_count >= required_switch:

                    print(
                        f"[DISPLAY ID SWITCH] "
                        f"{display_tracker_key} "
                        f"{shown_id} -> {display_gid}"
                    )


                    state[
                        'shown_id'
                    ] = display_gid

                    state[
                        'candidate_id'
                    ] = None

                    state[
                        'candidate_count'
                    ] = 0

                    shown_id = display_gid


            # ------------------------------------------------
            # DISPLAY ONLY THE STABLE ID
            # ------------------------------------------------

            stable_item = dict(
                display_item
            )

            stable_item[
                'global_id'
            ] = shown_id

            stable_item[
                'name'
            ] = shown_id


            stable_results.append(
                stable_item
            )


        if len(stable_results) != len(results):

            print(
                f"[DISPLAY FILTER] "
                f"{cam_id} "
                f"{len(results)} -> "
                f"{len(stable_results)}"
            )


        results = stable_results


'''

if "FINAL DISPLAY CONFIRMATION + ID HYSTERESIS" not in text:

    if smooth_marker not in text:

        raise SystemExit(
            "ERROR: smoothing marker not found"
        )

    text = text.replace(
        smooth_marker,
        display_block + smooth_marker,
        1
    )


path.write_text(
    text,
    encoding="utf-8"
)

print(
    "FINAL STABILITY HARDENING INSTALLED"
)
