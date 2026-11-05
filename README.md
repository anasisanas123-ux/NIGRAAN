# NIGRAAN campus surveillance

Run commands from this project folder. Existing `main.py` and `multi_camera.py`
launch commands are preserved. Model weights, enrollment data and thresholds are
unchanged by the performance refactor.

## First-time setup

Requires Python 3.11, Node.js, MongoDB running locally, and an NVIDIA GPU for
real-time face recognition (it falls back to a much slower CPU path).

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
# insightface also installs the CPU-only onnxruntime, which overwrites the GPU
# build in the same folder. Keep only the GPU build:
.\venv\Scripts\python.exe -m pip uninstall -y onnxruntime onnxruntime-gpu
.\venv\Scripts\python.exe -m pip install --no-deps onnxruntime-gpu==1.20.2
cd dashboard; npm ci; cd ..
```

`pip check` then reports that insightface requires `onnxruntime`; the GPU build
provides the same module, so this message is expected.

Not in the repository (git-ignored) and needed before starting:

- `models/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth`
  from the [Torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html)
  (MSMT17 `combineall=True`, osnet_x1_0).
- `dashboard/.env` with `MONGODB_URI=mongodb://127.0.0.1:27017/nigraan`.
- `.env` (optional) with `CAM_B_RTSP=` for a second camera and `SHOW_DESKTOP_WINDOWS=0`.

YOLO and InsightFace weights download automatically on first start. Weapon and
violence models are optional; see `docs/`.

## Start

```powershell
.\venv\Scripts\python.exe main.py
```

In a second terminal:

```powershell
cd dashboard
npm start
```

Open http://localhost:3000 and refresh with Ctrl+F5 after a dashboard update.
Use **Set zones** under a live camera to draw recognition and restricted areas.
To restart both already-running project services:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/restart_project_engines.ps1
```

Set `SHOW_DESKTOP_WINDOWS=0` in `.env` to run only the dashboard preview, without
OpenCV desktop windows. Startup logs from the restart script are in `data/logs/`.

## Layout

| Location | Purpose |
|---|---|
| `main.py`, `multi_camera.py` | Compatibility launchers |
| `config.py` | Camera, recognition and processing settings |
| `python_engine/runtime.py` | Model composition and application lifecycle |
| `python_engine/camera_stream.py`, `frame_buffer.py` | Unique captured frames, reconnects, bounded transport |
| `python_engine/camera_worker.py` | Existing tracking and identity decisions |
| `python_engine/preview.py`, `display.py` | Independent previews and desktop display |
| `python_engine/engine_api.py` | Dashboard APIs, shared JPEG encoding and health |
| `dashboard/` | Express app and browser UI |
| `tests/` | Automated offline regressions |
| `scripts/` | Setup, restart and organization utilities |
| `docs/` | Setup, recognition, behavior and evaluation guides |
| `archive/` | Historical patches, manual checks and pre-refactor source snapshots |
| `data/` | Persistent galleries, enrollment images, local models and logs |

The remaining root enrollment/evaluation/download commands are retained for
compatibility. Historical patch programs are archived rather than deleted;
they target old source layouts and should not be run against the current code.
Unrelated folders have not been removed.

## Camera performance

- Each inference worker processes a capture sequence at most once.
- Face checks are limited to once every `FACE_INFERENCE_INTERVAL=0.2` seconds
  per camera; person tracking continues between those checks. Confirmation still
  requires three matching observations using distinct source frames.
- USB cameras request 25 FPS at the configured resolution. `USB_CAMERA_MJPEG`
  is enabled because uncompressed YUY2 limits many USB webcams to 5-10 FPS at
  720p/1080p; it falls back to the default format if MJPEG capture fails.
- Preview delivery runs independently, up to `PREVIEW_FPS=20`, preserving aspect
  ratio. It does not lower the resolution used by the recognition models.
- Stale person overlays disappear after `OVERLAY_MAX_AGE=1.0` seconds, or on a
  reconnect/deleted identity. Fresh video may continue while AI catches up.
- Browser viewers share JPEGs; desktop windows reuse the same rendered previews.
- One latest-result slot per camera prevents a backlog and camera starvation.
- The dashboard shows preview and AI FPS separately. Hover for capture FPS,
  inference time and result age. `/api/cameras/health` also exposes person/face
  inference milliseconds and skipped source-frame counts.
- Skipping source frames under load is intentional: latency takes precedence
  over analyzing old frames. This does not establish crowded-campus throughput.

Keep `FRAME_SKIP=1` for normal operation. Lowering person-detector resolution or
identity thresholds is not part of this optimization because it can reduce
distant-person detection or increase false matches.

## Validation

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
node --test tests/capture-camera.test.cjs tests/student-deletion.test.cjs tests/zone-editor.test.cjs
```

Use [the CCTV guide](docs/CCTV_RECOGNITION_GUIDE.md) and `evaluate_faces.py` for
local recognition evaluation. Other guides are in [docs](docs/).
See [performance notes](docs/PERFORMANCE_REFACTOR.md) for the live measurements,
refactor details and validation limits.

The completed 2.5 GiB PyTorch installation wheel can be removed once the installed
environment is verified. It is not needed to run the application. The setup
script can download pinned versions again if reinstalling becomes necessary.
Keep `venv/`, model weights and biometric galleries locally; they are not cleanup
targets. Optional formatting tools are listed in `requirements-dev.txt`.
