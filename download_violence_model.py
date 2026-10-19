"""Download the optional RWF-2000 X3D-M checkpoint into the ignored data folder."""

import os
import sys

import requests

import config


MODEL_URL = (
    "https://huggingface.co/visionlab-ai/school-violence-detection-models/"
    "resolve/main/final/final_x3d_realtime.pt"
)
MAX_BYTES = 100 * 1024 * 1024


def main():
    destination = config.VIOLENCE_MODEL_PATH
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + ".download"
    try:
        with requests.get(
            MODEL_URL,
            stream=True,
            timeout=(15, 120),
            headers={"User-Agent": "campus-surveillance-model-setup/1.0"},
        ) as response:
            response.raise_for_status()
            content_length = int(response.headers.get("Content-Length", "0") or 0)
            if content_length > MAX_BYTES:
                raise ValueError("Checkpoint is larger than the 100 MiB safety limit.")

            written = 0
            with open(temporary, "wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > MAX_BYTES:
                        raise ValueError("Checkpoint exceeded the 100 MiB safety limit.")
                    output.write(chunk)
        if written < 1024 * 1024:
            raise ValueError("Download is too small to be a model checkpoint.")
        os.replace(temporary, destination)
        print(f"Violence model checkpoint saved to: {destination}")
        print("Restart the Python engine to load it. The model only creates review cues.")
    except Exception as error:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass
        print(f"Could not download the violence checkpoint: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
