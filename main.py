"""Default launcher for the multi-camera application."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("multi_camera.py")),
        run_name="__main__",
    )
