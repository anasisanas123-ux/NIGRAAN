from pathlib import Path

path = Path("multi_camera.py")
text = path.read_text(encoding="utf-8")

# ------------------------------------------------------------
# 1. Add shared cross-camera state globally
# ------------------------------------------------------------

old_global = """reid = ReIDManager()
anonymous_face_reid = AnonymousFaceReID()
"""

new_global = """reid = ReIDManager()
anonymous_face_reid = AnonymousFaceReID()

# Shared across ALL camera threads
cross_cam_fired = set()
cross_cam_lock = threading.Lock()
"""

if "cross_cam_lock = threading.Lock()" not in text:

    if old_global not in text:
        raise SystemExit("ERROR: global insertion point not found")

    text = text.replace(
        old_global,
        new_global,
        1
    )


# ------------------------------------------------------------
# 2. Remove per-camera cross_cam_fired set
# ------------------------------------------------------------

old_local = """    pending_unknown_time = {}

    cross_cam_fired = set()
"""

new_local = """    pending_unknown_time = {}
"""

if old_local in text:
    text = text.replace(
        old_local,
        new_local,
        1
    )


# ------------------------------------------------------------
# 3. Make check/add thread-safe
# ------------------------------------------------------------

old_block = """                    if (
                        cross_key
                        not in
                        cross_cam_fired
                    ):

                        cross_cam_fired.add(
                            cross_key
                        )


                        def send_cross_camera(
"""

new_block = """                    with cross_cam_lock:

                        should_fire_cross = (
                            cross_key
                            not in
                            cross_cam_fired
                        )

                        if should_fire_cross:

                            cross_cam_fired.add(
                                cross_key
                            )


                    if should_fire_cross:

                        def send_cross_camera(
"""

if old_block not in text:
    raise SystemExit("ERROR: cross-camera block not found")

text = text.replace(
    old_block,
    new_block,
    1
)

path.write_text(
    text,
    encoding="utf-8"
)

print("SHARED CROSS-CAMERA DEDUP INSTALLED")
