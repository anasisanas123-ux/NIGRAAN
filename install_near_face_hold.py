from pathlib import Path


# ============================================================
# PART 1 — anonymous_face_reid.py
# Add a NON-DECISION best_candidate() method.
#
# match() keeps its current strict behavior.
# ============================================================

anon_path = Path(
    "python_engine/anonymous_face_reid.py"
)

anon = anon_path.read_text(
    encoding="utf-8"
)


if "def best_candidate(" not in anon:

    match_marker = """    def match(
        self,
        embedding
    ):
"""

    if match_marker not in anon:

        raise SystemExit(
            "ERROR: AnonymousFaceReID match marker not found"
        )


    best_candidate_code = r'''    def best_candidate(
        self,
        embedding
    ):
        """
        Observation only.

        Returns the highest-ranked anonymous identity even
        when it does NOT satisfy the final match threshold.

        This method NEVER assigns or creates an identity.

        Returns:
            best_id, best_score, margin
        """

        emb = self._normalize(
            embedding
        )

        if emb is None:

            return (
                None,
                0.0,
                None
            )


        with self.lock:

            if not self.gallery:

                return (
                    None,
                    0.0,
                    None
                )


            rankings = []


            for (
                global_id,
                samples
            ) in self.gallery.items():

                score = self._identity_score(
                    emb,
                    samples
                )


                rankings.append(
                    (
                        global_id,
                        float(score)
                    )
                )


            rankings.sort(
                key=lambda item:
                    item[1],
                reverse=True
            )


            if not rankings:

                return (
                    None,
                    0.0,
                    None
                )


            best_id = rankings[0][0]

            best_score = float(
                rankings[0][1]
            )


            if len(rankings) > 1:

                second_score = float(
                    rankings[1][1]
                )

                margin = (
                    best_score
                    -
                    second_score
                )

            else:

                margin = None


            return (
                best_id,
                best_score,
                margin
            )


'''

    anon = anon.replace(
        match_marker,
        best_candidate_code
        +
        match_marker,
        1
    )


anon_path.write_text(
    anon,
    encoding="utf-8"
)


# ============================================================
# PART 2 — multi_camera.py
# ============================================================

multi_path = Path(
    "multi_camera.py"
)

text = multi_path.read_text(
    encoding="utf-8"
)


# ============================================================
# 1. SETTINGS
# ============================================================

settings_marker = (
    "NEW_UNK_MIN_BODY_CONF = 0.50"
)


settings_block = r'''

# ============================================================
# NEAR-FACE HOLD
#
# Do NOT lower Anonymous ArcFace's real 0.60 threshold.
#
# This only delays creation of a NEW UNK when the current
# face already looks promisingly similar to an existing UNK.
# ============================================================

ANON_FACE_HOLD_MIN_SCORE = 0.52

ANON_FACE_HOLD_MIN_MARGIN = 0.08

# A score that is already at/above the normal face threshold
# can still trigger a temporary HOLD if only the margin is
# currently failing because two gallery IDs are very similar.
ANON_FACE_HOLD_STRONG_SCORE = 0.60

# Maximum delay before normal body-based new-person logic
# is allowed to continue.
ANON_FACE_HOLD_SECONDS = 4.0

# Keep the hold briefly if the face disappears for a moment.
ANON_FACE_HOLD_STALE_SECONDS = 1.0

'''

if "ANON_FACE_HOLD_MIN_SCORE" not in text:

    if settings_marker not in text:

        raise SystemExit(
            "ERROR: NEW_UNK_MIN_BODY_CONF marker not found"
        )


    text = text.replace(
        settings_marker,
        settings_marker
        +
        settings_block,
        1
    )


# ============================================================
# 2. PER-CAMERA HOLD STATE
# ============================================================

state_marker = (
    "    confirmed_unknown_ids = {}"
)


state_block = r'''

    # Temporary possible-existing-face states.
    #
    # tracker_key -> {
    #     candidate_id,
    #     start_time,
    #     last_seen,
    #     max_score,
    #     last_log
    # }
    near_face_holds = {}

'''

if "near_face_holds = {}" not in text:

    if state_marker not in text:

        raise SystemExit(
            "ERROR: confirmed_unknown_ids marker not found"
        )


    text = text.replace(
        state_marker,
        state_marker
        +
        state_block,
        1
    )


# ============================================================
# 3. INITIALIZE HOLD VARIABLES BEFORE ANON FACE MATCH
# ============================================================

face_marker = """                    if face_embedding is not None:

                        try:
"""


face_init = r'''                    near_face_hold_active = False

                    near_face_hold_candidate = None


                    # Existing hold can survive a brief
                    # missed-face frame.
                    previous_face_hold = (
                        near_face_holds.get(
                            tracker_key
                        )
                    )


                    if previous_face_hold is not None:

                        hold_age = (
                            now
                            -
                            float(
                                previous_face_hold.get(
                                    'start_time',
                                    now
                                )
                            )
                        )

                        hold_stale = (
                            now
                            -
                            float(
                                previous_face_hold.get(
                                    'last_seen',
                                    0.0
                                )
                            )
                        )


                        if (
                            hold_age
                            <=
                            ANON_FACE_HOLD_SECONDS

                            and

                            hold_stale
                            <=
                            ANON_FACE_HOLD_STALE_SECONDS
                        ):

                            near_face_hold_active = True

                            near_face_hold_candidate = (
                                previous_face_hold.get(
                                    'candidate_id'
                                )
                            )

                        else:

                            near_face_holds.pop(
                                tracker_key,
                                None
                            )


                    if face_embedding is not None:

                        try:
'''

if "previous_face_hold = (" not in text:

    if face_marker not in text:

        raise SystemExit(
            "ERROR: face embedding block marker not found"
        )


    text = text.replace(
        face_marker,
        face_init,
        1
    )


# ============================================================
# 4. AFTER STRICT MATCH(), ALSO READ RAW BEST CANDIDATE
# ============================================================

match_call = """                            ) = anonymous_face_reid.match(
                                face_embedding
                            )


                            if face_global_id is not None:
"""


replacement = r'''                            ) = anonymous_face_reid.match(
                                face_embedding
                            )


                            (
                                raw_face_id,
                                raw_face_score,
                                raw_face_margin
                            ) = anonymous_face_reid.best_candidate(
                                face_embedding
                            )


                            # =================================
                            # STRICT MATCH WON
                            # =================================

                            if face_global_id is not None:

                                near_face_holds.pop(
                                    tracker_key,
                                    None
                                )

                                near_face_hold_active = False

                                near_face_hold_candidate = None
'''

if "raw_face_id," not in text:

    if match_call not in text:

        raise SystemExit(
            "ERROR: anonymous face match call marker not found"
        )


    text = text.replace(
        match_call,
        replacement,
        1
    )


# ============================================================
# 5. ADD NEAR-FACE HOLD BEFORE THE EXISTING NO-MATCH LOG
# ============================================================

no_match_marker = """                            elif face_reid_score > 0.0:

                                margin_text = (
"""


hold_code = r'''                            else:

                                # =================================
                                # NEAR-FACE HOLD
                                #
                                # This does NOT assign raw_face_id.
                                #
                                # It only prevents ReIDManager from
                                # creating another global UNK for a
                                # few seconds while we wait for a
                                # clearer face frame.
                                # =================================

                                margin_ok = (

                                    raw_face_margin
                                    is None

                                    or

                                    raw_face_margin
                                    >=
                                    ANON_FACE_HOLD_MIN_MARGIN
                                )


                                strong_score = (

                                    raw_face_score
                                    >=
                                    ANON_FACE_HOLD_STRONG_SCORE
                                )


                                near_score = (

                                    raw_face_score
                                    >=
                                    ANON_FACE_HOLD_MIN_SCORE
                                )


                                qualifies_for_hold = (

                                    raw_face_id
                                    is not None

                                    and

                                    near_score

                                    and

                                    (
                                        margin_ok
                                        or
                                        strong_score
                                    )
                                )


                                if qualifies_for_hold:

                                    hold = (
                                        near_face_holds.get(
                                            tracker_key
                                        )
                                    )


                                    # New candidate or previous
                                    # state became stale -> restart
                                    # the bounded timer.
                                    if (
                                        hold is None

                                        or

                                        hold.get(
                                            'candidate_id'
                                        )
                                        !=
                                        raw_face_id

                                        or

                                        (
                                            now
                                            -
                                            float(
                                                hold.get(
                                                    'last_seen',
                                                    0.0
                                                )
                                            )
                                            >
                                            ANON_FACE_HOLD_STALE_SECONDS
                                        )
                                    ):

                                        hold = {
                                            'candidate_id':
                                                raw_face_id,

                                            'start_time':
                                                now,

                                            'last_seen':
                                                now,

                                            'max_score':
                                                raw_face_score,

                                            'last_log':
                                                0.0
                                        }


                                        near_face_holds[
                                            tracker_key
                                        ] = hold


                                    else:

                                        hold[
                                            'last_seen'
                                        ] = now

                                        hold[
                                            'max_score'
                                        ] = max(
                                            float(
                                                hold.get(
                                                    'max_score',
                                                    0.0
                                                )
                                            ),
                                            raw_face_score
                                        )


                                    hold_age = (

                                        now

                                        -

                                        float(
                                            hold[
                                                'start_time'
                                            ]
                                        )
                                    )


                                    if (
                                        hold_age
                                        <=
                                        ANON_FACE_HOLD_SECONDS
                                    ):

                                        near_face_hold_active = True

                                        near_face_hold_candidate = (
                                            raw_face_id
                                        )


                                        if (
                                            now
                                            -
                                            float(
                                                hold.get(
                                                    'last_log',
                                                    0.0
                                                )
                                            )
                                            >=
                                            0.75
                                        ):

                                            hold[
                                                'last_log'
                                            ] = now


                                            hold_margin_text = (

                                                "n/a"

                                                if
                                                raw_face_margin
                                                is None

                                                else
                                                f"{raw_face_margin:.3f}"
                                            )


                                            print(
                                                f"[ANON FACE HOLD] "
                                                f"{tracker_key} "
                                                f"candidate="
                                                f"{raw_face_id} "
                                                f"score="
                                                f"{raw_face_score:.3f} "
                                                f"margin="
                                                f"{hold_margin_text} "
                                                f"age="
                                                f"{hold_age:.1f}/"
                                                f"{ANON_FACE_HOLD_SECONDS:.1f}s"
                                            )


                                    else:

                                        near_face_hold_active = False

                                        near_face_hold_candidate = None


                                        print(
                                            f"[ANON FACE HOLD EXPIRED] "
                                            f"{tracker_key} "
                                            f"candidate="
                                            f"{raw_face_id} "
                                            f"max_score="
                                            f"{float(hold.get('max_score', 0.0)):.3f}"
                                        )


                                        near_face_holds.pop(
                                            tracker_key,
                                            None
                                        )


                                # Existing original diagnostic
                                # output remains below.

                                if face_reid_score > 0.0:

                                    margin_text = (
'''

if "qualifies_for_hold = (" not in text:

    if no_match_marker not in text:

        raise SystemExit(
            "ERROR: ANON FACE NO MATCH block marker not found"
        )


    text = text.replace(
        no_match_marker,
        hold_code,
        1
    )


# ============================================================
# 6. BLOCK BODY GLOBAL-ID CREATION DURING HOLD
# ============================================================

body_fallback = """                    if (
                        global_id
                        is None

                        and

                        emb_valid(
                            body_emb
                        )
                    ):
"""


body_fallback_safe = """                    if (
                        global_id
                        is None

                        and

                        not near_face_hold_active

                        and

                        emb_valid(
                            body_emb
                        )
                    ):
"""


if "not near_face_hold_active" not in text:

    if body_fallback not in text:

        raise SystemExit(
            "ERROR: body fallback marker not found"
        )


    text = text.replace(
        body_fallback,
        body_fallback_safe,
        1
    )


# ============================================================
# 7. EXPLICIT HOLD LOG WHERE BODY FALLBACK WOULD HAVE RUN
# ============================================================

fallback_comment = """                    # =========================================
                    # FALL BACK TO OSNET / BODY RE-ID
"""


hold_status = r'''                    if (
                        global_id is None
                        and
                        near_face_hold_active
                    ):

                        # No assignment here.
                        #
                        # Keep the current person unresolved
                        # rather than creating a duplicate UNK.
                        pass


'''

if "# No assignment here." not in text:

    if fallback_comment not in text:

        raise SystemExit(
            "ERROR: body fallback comment not found"
        )


    text = text.replace(
        fallback_comment,
        hold_status
        +
        fallback_comment,
        1
    )


multi_path.write_text(
    text,
    encoding="utf-8"
)


print(
    "NEAR-FACE HOLD INSTALLED"
)
