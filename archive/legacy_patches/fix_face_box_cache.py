from pathlib import Path
import re
import shutil
import py_compile

path = Path("multi_camera.py")
backup = Path("multi_camera_before_face_cache.py")

if not path.exists():
    raise SystemExit("multi_camera.py NOT FOUND")

text = path.read_text(encoding="utf-8")

if "last_face_boxes = {}" in text:
    raise SystemExit("Face-box cache already appears to be installed")

shutil.copy2(path, backup)
print(f"Backup created: {backup}")


# ============================================================
# 1. UPDATE OLD COMMENT
# ============================================================

text = text.replace(
    "Known identities cached by DeepSORT track",
    "Known identities cached by BoT-SORT track"
)


# ============================================================
# 2. ADD FACE CACHE STATE
# ============================================================

pattern = r'(?m)^([ \t]*)last_known_boxes\s*=\s*\{\}[ \t]*$'
match = re.search(pattern, text)

if not match:
    raise SystemExit("ERROR: last_known_boxes not found")

indent = match.group(1)

replacement = (
    match.group(0)
    + "\n\n"
    + indent + "# Last detected face position for each BoT-SORT track\n"
    + indent + "last_face_boxes = {}\n\n"
    + indent + "# Keep face box briefly when SCRFD misses during movement\n"
    + indent + "FACE_BOX_TTL = 0.8"
)

text = (
    text[:match.start()]
    + replacement
    + text[match.end():]
)


# ============================================================
# 3. REMOVE FACE CACHE WHEN TRACK STATE EXPIRES
# ============================================================

pattern = r'(?m)^([ \t]*)for key in expired:[ \t]*$'
match = re.search(pattern, text)

if not match:
    raise SystemExit("ERROR: expired cleanup loop not found")

loop_indent = match.group(1)

replacement = (
    match.group(0)
    + "\n\n"
    + loop_indent
    + "    last_face_boxes.pop(key, None)"
)

text = (
    text[:match.start()]
    + replacement
    + text[match.end():]
)


# ============================================================
# 4. ADD FUNCTION THAT MOVES CACHED FACE WITH BODY TRACK
# ============================================================

pattern = r'(?m)^([ \t]*)UNKNOWN_RECHECK\s*=\s*[0-9]+(?:\.[0-9]+)?[ \t]*$'
match = re.search(pattern, text)

if not match:
    raise SystemExit("ERROR: UNKNOWN_RECHECK not found")

indent = match.group(1)

helper = f"""

{indent}def get_cached_face_bbox(
{indent}    tracker_key,
{indent}    current_body_bbox,
{indent}    current_time
{indent}):
{indent}    cached = last_face_boxes.get(
{indent}        tracker_key
{indent}    )

{indent}    if cached is None:
{indent}        return None

{indent}    if (
{indent}        current_time
{indent}        -
{indent}        cached.get('time', 0)
{indent}        >
{indent}        FACE_BOX_TTL
{indent}    ):
{indent}        return None

{indent}    old_face = cached.get(
{indent}        'face_bbox'
{indent}    )

{indent}    old_body = cached.get(
{indent}        'body_bbox'
{indent}    )

{indent}    if (
{indent}        old_face is None
{indent}        or
{indent}        old_body is None
{indent}        or
{indent}        current_body_bbox is None
{indent}    ):
{indent}        return None

{indent}    obx1, oby1, obx2, oby2 = [
{indent}        float(v)
{indent}        for v in old_body
{indent}    ]

{indent}    fx1, fy1, fx2, fy2 = [
{indent}        float(v)
{indent}        for v in old_face
{indent}    ]

{indent}    nbx1, nby1, nbx2, nby2 = [
{indent}        float(v)
{indent}        for v in current_body_bbox
{indent}    ]

{indent}    old_w = max(
{indent}        obx2 - obx1,
{indent}        1.0
{indent}    )

{indent}    old_h = max(
{indent}        oby2 - oby1,
{indent}        1.0
{indent}    )

{indent}    new_w = max(
{indent}        nbx2 - nbx1,
{indent}        1.0
{indent}    )

{indent}    new_h = max(
{indent}        nby2 - nby1,
{indent}        1.0
{indent}    )

{indent}    # Face coordinates relative to old body box
{indent}    rx1 = (fx1 - obx1) / old_w
{indent}    ry1 = (fy1 - oby1) / old_h
{indent}    rx2 = (fx2 - obx1) / old_w
{indent}    ry2 = (fy2 - oby1) / old_h

{indent}    # Apply them to the current BoT-SORT body box
{indent}    new_fx1 = int(
{indent}        nbx1 + rx1 * new_w
{indent}    )

{indent}    new_fy1 = int(
{indent}        nby1 + ry1 * new_h
{indent}    )

{indent}    new_fx2 = int(
{indent}        nbx1 + rx2 * new_w
{indent}    )

{indent}    new_fy2 = int(
{indent}        nby1 + ry2 * new_h
{indent}    )

{indent}    if (
{indent}        new_fx2 <= new_fx1
{indent}        or
{indent}        new_fy2 <= new_fy1
{indent}    ):
{indent}        return None

{indent}    return [
{indent}        new_fx1,
{indent}        new_fy1,
{indent}        new_fx2,
{indent}        new_fy2
{indent}    ]
"""

text = (
    text[:match.end()]
    + helper
    + text[match.end():]
)


# ============================================================
# 5. CACHED KNOWN PERSON:
#    USE LAST FACE BOX INSTEAD OF None
# ============================================================

start = text.find(
    "# Already confirmed known student"
)

end = text.find(
    "# RUN ARCFACE",
    start
)

if start == -1 or end == -1:
    raise SystemExit(
        "ERROR: cached-known section not found"
    )

block = text[start:end]

block, count = re.subn(
    r"('face_bbox'\s*:\s*\n\s*)None",
    r"\1get_cached_face_bbox(tracker_key, bd['bbox'], now)",
    block,
    count=1
)

if count != 1:
    raise SystemExit(
        "ERROR: cached known face_bbox not replaced"
    )

text = (
    text[:start]
    + block
    + text[end:]
)


# ============================================================
# 6. SAVE EVERY REAL SCRFD FACE POSITION
# ============================================================

start = text.find(
    "# FACE MATCHED TO BODY"
)

end = text.find(
    "# BODY EXISTS BUT FACE WAS NOT MATCHED",
    start
)

if start == -1 or end == -1:
    raise SystemExit(
        "ERROR: matched-face section not found"
    )

block = text[start:end]

append_pos = block.find(
    "detections.append({"
)

if append_pos == -1:
    raise SystemExit(
        "ERROR: matched-face detections.append not found"
    )

line_start = block.rfind(
    "\n",
    0,
    append_pos
) + 1

append_indent = re.match(
    r'[ \t]*',
    block[line_start:]
).group(0)

cache_code = (
    append_indent
    + "last_face_boxes[tracker_key] = {\n"
    + append_indent
    + "    'face_bbox': list(fd['bbox']),\n"
    + append_indent
    + "    'body_bbox': list(bd['bbox']),\n"
    + append_indent
    + "    'time': now\n"
    + append_indent
    + "}\n\n"
)

block = (
    block[:line_start]
    + cache_code
    + block[line_start:]
)

text = (
    text[:start]
    + block
    + text[end:]
)


# ============================================================
# 7. BODY TRACK EXISTS BUT SCRFD MISSED:
#    REUSE RECENT FACE BOX
# ============================================================

start = text.find(
    "# BODY EXISTS BUT FACE WAS NOT MATCHED"
)

end = text.find(
    "# PROCESS IDENTITIES",
    start
)

if start == -1 or end == -1:
    raise SystemExit(
        "ERROR: body-without-face section not found"
    )

block = text[start:end]

block, count = re.subn(
    r"('face_bbox'\s*:\s*\n\s*)None",
    r"\1get_cached_face_bbox(tracker_key, bd['bbox'], now)",
    block,
    count=1
)

if count != 1:
    raise SystemExit(
        "ERROR: unmatched face_bbox not replaced"
    )

text = (
    text[:start]
    + block
    + text[end:]
)


# ============================================================
# WRITE + VERIFY
# ============================================================

path.write_text(
    text,
    encoding="utf-8"
)

try:
    py_compile.compile(
        str(path),
        doraise=True
    )

except Exception as error:

    shutil.copy2(
        backup,
        path
    )

    raise SystemExit(
        f"SYNTAX ERROR - original restored: {error}"
    )


print()
print("FACE BOX CACHE INSTALLED")
print("TTL: 0.8 seconds")
print("Cached face follows BoT-SORT body movement")
print("Syntax check: OK")
print("Backup:", backup)

