from pathlib import Path
import re
import shutil

path = Path("multi_camera.py")

if not path.exists():
    raise SystemExit("multi_camera.py NOT FOUND")

text = path.read_text(encoding="utf-8")

# ------------------------------------------------------------
# BACKUP
# ------------------------------------------------------------

backup = Path("multi_camera_before_botsort.py")

shutil.copy2(
    path,
    backup
)

print(f"Backup created: {backup}")


# ------------------------------------------------------------
# 1. REMOVE DEEPSORT IMPORT
# ------------------------------------------------------------

text, import_count = re.subn(
    r'^[ \t]*from python_engine\.tracker import Tracker[ \t]*\r?\n',
    '',
    text,
    count=1,
    flags=re.MULTILINE
)

if import_count != 1:
    raise SystemExit(
        f"ERROR: expected 1 Tracker import, found {import_count}"
    )


# ------------------------------------------------------------
# 2. REMOVE GLOBAL DEEPSORT TRACKER OBJECTS
# ------------------------------------------------------------

tracker_pattern = r'''
\n
trackers\s*=\s*\{
\s*
cam_id:\s*Tracker\(\)
\s*
for\s+cam_id
\s*
in\s+config\.CAMERA_SOURCES\.keys\(\)
\s*
\}
'''

text, tracker_count = re.subn(
    tracker_pattern,
    '\n',
    text,
    count=1,
    flags=re.VERBOSE
)

if tracker_count != 1:
    raise SystemExit(
        f"ERROR: expected 1 trackers block, found {tracker_count}"
    )


# ------------------------------------------------------------
# 3. GIVE SYNTHETIC FALLBACK BOX ITS EXISTING TRACK ID
# ------------------------------------------------------------
#
# The old fallback creates a body detection using last_known_boxes.
# Since BoT-SORT isn't producing that synthetic detection, we keep
# the previous tracker ID ourselves.
# ------------------------------------------------------------

fallback_start = text.find(
    "        if not body_dets:"
)

deep_marker = text.find(
    "# PREPARE DEEPSORT INPUT",
    fallback_start
)

if fallback_start == -1 or deep_marker == -1:
    raise SystemExit(
        "ERROR: could not find detection fallback / DeepSORT section"
    )

fallback_block = text[
    fallback_start:
    deep_marker
]

if "'tracker_id'" not in fallback_block:

    fallback_block, fallback_count = re.subn(
        r"('face_crop':\s*\n\s*None)(\s*\n\s*\}\))",
        r"""\1,

                        'tracker_id':
                            tracker_key.rsplit('_', 1)[-1]\2""",
        fallback_block,
        count=1
    )

    if fallback_count != 1:
        raise SystemExit(
            "ERROR: could not add tracker_id to fallback detection"
        )

    text = (
        text[:fallback_start]
        +
        fallback_block
        +
        text[deep_marker:]
    )


# ------------------------------------------------------------
# 4. REMOVE:
#
#    PREPARE DEEPSORT INPUT
#    temp_dets
#    trackers[cam_id].update(...)
#    get_tracker_id(...)
#
# ------------------------------------------------------------

lines = text.splitlines(
    keepends=True
)

start_index = None

for i, line in enumerate(lines):

    if "PREPARE DEEPSORT INPUT" in line:

        start_index = i

        if (
            i > 0
            and
            "====" in lines[i - 1]
        ):
            start_index = i - 1

        break


end_heading = None

for i in range(
    start_index if start_index is not None else 0,
    len(lines)
):

    if (
        "SPLIT CONFIRMED / UNCONFIRMED PEOPLE"
        in
        lines[i]
    ):

        end_heading = i
        break


if (
    start_index is None
    or
    end_heading is None
):
    raise SystemExit(
        "ERROR: could not locate complete DeepSORT section"
    )


end_remove = end_heading

if (
    end_heading > 0
    and
    "====" in lines[end_heading - 1]
):

    end_remove = (
        end_heading - 1
    )


replacement = [
    "        # ====================================================\n",
    "        # BOT-SORT TRACK IDS COME DIRECTLY FROM BODY DETECTOR\n",
    "        # ====================================================\n",
    "\n",
]


lines[
    start_index:
    end_remove
] = replacement


text = ''.join(
    lines
)


# ------------------------------------------------------------
# 5. USE BOT-SORT ID DIRECTLY FOR EACH BODY
# ------------------------------------------------------------

body_tid_pattern = r'''
(?m)
^([ \t]*)
tid
\s*=\s*
get_tracker_id
\(
\s*
bd\['bbox'\]
\s*
\)
'''

text, body_tid_count = re.subn(
    body_tid_pattern,
    r"\1tid = bd.get('tracker_id')",
    text,
    count=1,
    flags=re.VERBOSE
)

if body_tid_count != 1:
    raise SystemExit(
        f"ERROR: body tracker replacement count = {body_tid_count}"
    )


# ------------------------------------------------------------
# 6. REMOVE SECOND MANUAL TRACK LOOKUP
#
# OLD:
#
# tid = (
#     det.get('tracker_id')
#     or
#     get_tracker_id(det['bbox'])
# )
#
# NEW:
#
# tid = det.get('tracker_id')
# ------------------------------------------------------------

det_tid_pattern = r'''
(?m)
^([ \t]*)
tid
\s*=\s*
\(
\s*
det\.get
\(
\s*
'tracker_id'
\s*
\)
\s*
or
\s*
get_tracker_id
\(
\s*
det\['bbox'\]
\s*
\)
\s*
\)
'''

text, det_tid_count = re.subn(
    det_tid_pattern,
    r"\1tid = det.get('tracker_id')",
    text,
    count=1,
    flags=re.VERBOSE
)

if det_tid_count != 1:
    raise SystemExit(
        f"ERROR: final tracker replacement count = {det_tid_count}"
    )


# ------------------------------------------------------------
# 7. UPDATE COMMENTS
# ------------------------------------------------------------

text = text.replace(
    "DeepSORT's `tid`",
    "BoT-SORT's `tid`"
)

text = text.replace(
    "DeepSORT tracker ID",
    "BoT-SORT tracker ID"
)


# ------------------------------------------------------------
# 8. FINAL SAFETY CHECK
# ------------------------------------------------------------

problems = []

if "from python_engine.tracker import Tracker" in text:
    problems.append(
        "DeepSORT Tracker import still exists"
    )

if "get_tracker_id(" in text:
    problems.append(
        "get_tracker_id() still exists"
    )

if "trackers[" in text:
    problems.append(
        "old trackers[...] call still exists"
    )

if problems:

    print("MIGRATION NOT WRITTEN")

    for problem in problems:
        print(" -", problem)

    raise SystemExit(1)


# ------------------------------------------------------------
# WRITE FILE
# ------------------------------------------------------------

path.write_text(
    text,
    encoding="utf-8"
)

print()
print("BOT-SORT MIGRATION SUCCESSFUL")
print(f"Tracker import removed : {import_count}")
print(f"Tracker block removed  : {tracker_count}")
print(f"Body ID changed        : {body_tid_count}")
print(f"Final ID changed       : {det_tid_count}")
print()
print("DeepSORT is no longer used by multi_camera.py")
print("Backup:", backup)
