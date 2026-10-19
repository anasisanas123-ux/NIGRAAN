# Gun and knife review cues

This add-on uses a separate local YOLO11n model exported to ONNX. Existing face,
person, Re-ID, gait, and violence checkpoints are not modified. Source:
https://huggingface.co/cosgun99/gun-knife-yolo11n, revision `572728c`.
The publisher labels classes `0: gun`, `1: knife`, and describes the model as an
academic demonstration. Its reported test metrics do not establish accuracy on
these campus cameras. Export provenance and hashes are saved beside the ONNX file.

The detector runs asynchronously on CPU with two inference threads, sampling each
camera at most twice per second. Latest pending frames replace older frames;
camera publishing does not wait for inference. The dashboard and native camera
windows draw orange boxes around possible guns/knives. Results expire after two
seconds. Two consecutive observations are required for a review alert, and each
camera/weapon class has a 20-second cooldown. The cue does not identify a person
or establish possession, intent, or whether an object is real.

Controls in `config.py`:

```python
WEAPON_MODEL_ENABLED = True
WEAPON_MODEL_THRESHOLDS = {'gun': 0.55, 'knife': 0.60}
WEAPON_MODEL_INTERVAL = 0.5
WEAPON_MODEL_CONFIRM_FRAMES = 2
WEAPON_MODEL_COOLDOWN = 20.0
```

Restart the Python engine after changing these settings. Refresh Activity Review
to inspect readiness, per-camera detections, and recent review cues.

Offline checks:

```powershell
.\venv\Scripts\python.exe verify_weapon_model.py
.\venv\Scripts\python.exe verify_weapon_model.py --image "C:\path\to\test-image.jpg"
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_weapon_detector.py -v
```

The default smoke test uses a blank synthetic image and only checks execution.
Evaluate representative weapon-positive and ordinary negative images/clips before
tuning thresholds. A displayed photograph can also trigger an object cue.

To recreate the model file, download `best.pt` from the pinned source into
`data/setup_cache/weapon_gun_knife_yolo11n.pt`, then run
`.\venv\Scripts\python.exe prepare_weapon_model.py`. The preparation command uses
restricted checkpoint loading and exports `data/models/weapon_gun_knife_yolo11n.onnx`.
The running engine only loads that ONNX file and never downloads models at startup.
