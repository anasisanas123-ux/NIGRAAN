from pathlib import Path
import re

path = Path("multi_camera.py")
text = path.read_text(encoding="utf-8")

replacement = '''                        existing_face_samples = (
                            anonymous_face_reid
                            .get_sample_count(
                                global_id
                            )
                        )


                        # A brand-new global UNK has no face
                        # gallery yet, so its first sample
                        # is allowed.
                        if existing_face_samples == 0:

                            allow_face_add = True


                        # Existing UNK: only save the face if
                        # ArcFace independently agrees that it
                        # belongs to this global identity.
                        else:

                            (
                                verified_face_id,
                                verified_face_score,
                                verified_face_margin
                            ) = anonymous_face_reid.match(
                                face_embedding
                            )


                            allow_face_add = (
                                verified_face_id
                                ==
                                global_id
                            )

'''

pattern = r'''                        allow_face_add = True.*?(?=                        if allow_face_add:)'''

new_text, count = re.subn(
    pattern,
    replacement,
    text,
    count=1,
    flags=re.DOTALL
)

if count != 1:
    raise SystemExit(
        f"ERROR: expected 1 block, replaced {count}"
    )

path.write_text(new_text, encoding="utf-8")

print("FACE GALLERY SAFETY BLOCK FIXED")
