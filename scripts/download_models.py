"""Download the person Re-ID checkpoint the engine needs, and verify it.

Run from the project root:
    .\\venv\\Scripts\\python.exe scripts\\download_models.py

YOLO and InsightFace weights download automatically on first start; the
weapon and violence models are optional (see docs/).
"""

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    {
        "path": ROOT / "models" / (
            "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_"
            "lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
        ),
        # Torchreid model zoo: OSNet x1.0, MSMT17 combineall=True, color jitter.
        "google_drive_id": "1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x",
        "sha256": "48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727",
    },
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    import gdown

    failed = False
    for model in MODELS:
        path = model["path"]
        if path.is_file() and sha256(path) == model["sha256"]:
            print(f"OK (already present): {path.name}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        print(f"Downloading {path.name} ...")
        try:
            gdown.download(id=model["google_drive_id"], output=str(temporary), quiet=False)
        except Exception as error:
            print(f"FAILED to download {path.name}: {error}", file=sys.stderr)
            failed = True
            continue
        if not temporary.is_file() or sha256(temporary) != model["sha256"]:
            print(f"FAILED: {path.name} did not match the expected checksum.", file=sys.stderr)
            if temporary.exists():
                temporary.unlink()
            failed = True
            continue
        os.replace(temporary, path)
        print(f"OK (downloaded and verified): {path.name}")
    if failed:
        print(
            "\nIf Google Drive refused the download, open the Torchreid model zoo "
            "(https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html), download "
            "osnet_x1_0 'MSMT17 (combineall=True)' and save it in models/ under the name above.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
