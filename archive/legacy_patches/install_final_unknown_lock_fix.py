from pathlib import Path
import re

path = Path("multi_camera.py")
text = path.read_text(encoding="utf-8")


# ============================================================
# 1. RE-ENABLE SHORT TRACK-BREAK BRIDGE
# ============================================================

replacements = {
    "CAM_ID_LOCK_TTL = 0.0":
        "CAM_ID_LOCK_TTL = 4.0",

    "CAM_ID_LOCK_MIN_BODY_CONF = 2.0":
        "CAM_ID_LOCK_MIN_BODY_CONF = 0.30",

    "CAM_ID_LOCK_MIN_IOU = 0.25":
        "CAM_ID_LOCK_MIN_IOU = 0.60",

    "CAM_ID_LOCK_MAX_CENTER = 0.30":
        "CAM_ID_LOCK_MAX_CENTER = 0.18",

    "CAM_ID_LOCK_MIN_AREA_RATIO = 0.50":
        "CAM_ID_LOCK_MIN_AREA_RATIO = 0.65",

    "CAM_ID_LOCK_MIN_CONTAINMENT = 0.55":
        "CAM_ID_LOCK_MIN_CONTAINMENT = 0.75",

    "CAM_ID_LOCK_AMBIGUITY_MARGIN = 0.10":
        "CAM_ID_LOCK_AMBIGUITY_MARGIN = 0.12",
}


for old, new in replacements.items():

    if old in text:

        text = text.replace(
            old,
            new,
            1
        )

        print(
            "UPDATED:",
            old,
            "->",
            new
        )

    elif new in text:

        print(
            "ALREADY SET:",
            new
        )

    else:

        print(
            "WARNING: setting not found:",
            old
        )


# ============================================================
# 2. MAKE CONFIRMED UNKNOWN A HARD TRACK LOCK
#
# OLD:
# same tracker was only cached for 0.8 seconds.
#
# NEW:
# if tracker_key still exists and already owns a confirmed
# global_id, keep that identity indefinitely while BoT-SORT
# continues returning that tracker.
# ============================================================

pattern = re.compile(
    r"""                skip_reid\s*=\s*\(\s*
                    cached_unk\s*
                    is\s+not\s+None\s*
                    and\s*
                    \(\s*
                        now\s*
                        -\s*
                        cached_unk\[\s*
                            'time'\s*
                        \]\s*
                    \)\s*
                    <\s*
                    UNKNOWN_RECHECK\s*
                \)""",
    re.VERBOSE
)


replacement = """                # =============================================
                # HARD UNKNOWN TRACK LOCK
                #
                # Same behavior as known students:
                # while the SAME BoT-SORT tracker is alive,
                # do not recalculate its identity just because
                # the face disappears.
                # =============================================

                skip_reid = (

                    cached_unk
                    is not None

                    and

                    cached_unk.get(
                        'global_id'
                    )
                    is not None
                )


                if skip_reid:

                    # Refresh continuously while the person's
                    # body remains tracked.
                    cached_unk[
                        'time'
                    ] = now

"""


text, count = pattern.subn(
    replacement,
    text,
    count=1
)


if count == 1:

    print(
        "HARD SAME-TRACK UNKNOWN LOCK INSTALLED"
    )

else:

    if "HARD UNKNOWN TRACK LOCK" in text:

        print(
            "HARD SAME-TRACK UNKNOWN LOCK ALREADY INSTALLED"
        )

    else:

        raise SystemExit(
            "ERROR: could not find old skip_reid block"
        )


path.write_text(
    text,
    encoding="utf-8"
)

print(
    "FINAL UNKNOWN LOCK FIX INSTALLED"
)
