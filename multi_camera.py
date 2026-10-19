"""Compatibility launcher; the implementation lives in python_engine/."""

import sys

if sys.platform == "win32":
    import ctypes

    ctypes.windll.kernel32.SetDllDirectoryW(
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
    )
from python_engine.runtime import main

if __name__ == "__main__":
    main()
