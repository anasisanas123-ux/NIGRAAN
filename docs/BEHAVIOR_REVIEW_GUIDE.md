# Activity review cues

The dashboard's **Activity Review** page reports a small set of rule-based,
non-identifying cues for an operator to check against the video. The current
signals are configured restricted-zone entry, extended time inside that zone,
and rapid movement by a tracked person. The system does not label a person as
a thief or aggressor, infer intent, or take action based on a cue.

An optional scene-level violence cue model is also integrated. It uses the
RWF-2000 X3D-M checkpoint listed below and requires two successive model
windows above the configured threshold before raising a review cue.

## Install and enable the violence cue model

Install project dependencies, including PyTorchVideo, then download the
checkpoint explicitly:

```powershell
pip install -r requirements.txt
python download_violence_model.py
```

The checkpoint is saved under `data/models/` (ignored by Git). Restart the
Python engine. The Activity Review page displays whether weights are missing,
loading, ready, or failed to load. The classifier samples the full scene and
does not identify or associate a person with the cue. It runs asynchronously
so video-frame publishing does not wait for model inference.

To check the installed dependencies and the existing checkpoint without opening
cameras or starting the dashboard, run from the project directory:

```powershell
.\venv\Scripts\python.exe verify_behavior_model.py
```

This runs one synthetic clip on CPU and checks that the model produces valid
class probabilities. It does not measure action-recognition accuracy. Pass
`--device cuda` to check GPU execution. The checkpoint is never rewritten.

Activity Review shows collected frames (16 are required for a clip), the latest
scene score, and consecutive positive windows. Scores come directly from the
existing X3D probability head. Two positive windows are still required by the
default configuration. A camera outage resets clip collection so a clip does
not combine frames from unrelated moments.

If the model reports `pytorchvideo` missing, install it into this project's
virtual environment, rather than a global Python installation. The restricted
checkpoint loader requires PyTorch 2.4 or newer. Restart the Python engine after
installing dependencies; a running process keeps its previously loaded packages.

This is a research proof-of-concept checkpoint trained on RWF-2000, a binary
violent/non-violent surveillance clip dataset. Its model card reports false
alerts for sports and ordinary contact and possible domain mismatch on campus
camera views. Treat scores only as a prompt to review footage; do not take
disciplinary or legal action from the model output alone. Review the
[checkpoint model card](https://huggingface.co/visionlab-ai/school-violence-detection-models)
and its license/intended-use terms before use.

## Configure restricted areas

In `config.py`, define camera-specific normalized rectangles (left, top, right,
bottom coordinates between 0 and 1):

```python
BEHAVIOR_RESTRICTED_ZONES = {
    "cam_a": [
        {
            "id": "staff_only_door",
            "name": "Staff-only door",
            "box": [0.72, 0.15, 0.98, 0.92],
            "dwell_seconds": 10,
        }
    ]
}
```

Choose the rectangle from the camera's own view. Configure each camera
separately. Entry and dwell cues only appear when person detections have a
local tracker ID. Alerts are held in memory (up to 200) and reset when the
Python engine restarts.

## Theft and aggression model requirements

The current person detector and gait pose extractor are not action classifiers.
Theft requires contextual evidence that a particular object was moved or taken;
a person or object detection by itself cannot establish that. Aggression needs
temporal action examples and camera-specific validation because ordinary
greetings, sports, crowding, and occlusion can look similar in a single frame.

To add those as narrowly defined review cues, collect representative clips from
the actual cameras, label the visible action and time interval (including
ordinary non-event examples), split training and evaluation by person and
recording session, and measure false alerts on held-out clips before enabling
them. Do not use these cues alone to identify a person, accuse someone, or
trigger discipline or law-enforcement action.
