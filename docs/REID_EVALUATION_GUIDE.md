# Re-ID evaluation and dashboard features

## Calibrate OSNet on held-out body crops

`evaluate_reid.py` measures same-person and different-person OSNet cosine scores.
It does not change thresholds automatically because a threshold selected from
the same data can overfit.

Prepare a validation set of body crops that are not present in the Re-ID gallery:

```text
reid-validation/
  person_001/
    front_angle.mp4
    side_angle.mp4
  person_002/
    front_angle.mp4
    side_angle.mp4
  unknown_001/
    front_angle.mp4
    side_angle.mp4
```

For a final estimate, reserve a separate test set from different recording
sessions. Include unenrolled people to measure false accepts. Run:

```powershell
venv\Scripts\python.exe evaluate_reid.py reid-validation
```

The report compares the configured same-camera and cross-camera thresholds with
a validation-only candidate. Review FAR (different people merged) and FRR (same
person rejected) together before changing `REID_SAME_CAMERA_THRESHOLD` or
`REID_CROSS_CAMERA_THRESHOLD` in `config.py`. The manager now reads these values
directly; its current defaults remain 0.85 and 0.88.

The evaluator samples each video once per second and uses the local YOLO person
detector to crop the largest person in each sampled frame. It also accepts
pre-cropped body images. Keep each clip focused on one labeled person. Do not
put frames from the same moments in both the enrollment gallery and validation
set. If `yolov8s.pt` is stored elsewhere, pass its path with `--yolo-model`.

## Re-ID inference changes

OSNet inference now averages original and horizontally flipped descriptors. A
conservative Laplacian blur filter drops near-unusable crops before inference so
they cannot contaminate the track/gallery sample set. The defaults are
`REID_FLIP_TEST = True` and `REID_MIN_CROP_BLUR = 2.0`; tune them only with
held-out camera footage.

## Live camera status and zones

The Python engine publishes processed-frame time, estimated FPS, active track
count, and configured zone occupancy/dwell through `/health`. The dashboard
proxies that data through `/api/cameras/health`, polls every five seconds, and
raises a camera-offline alert after a previously live feed goes stale. Camera
URLs and credentials are not included in health responses.

To enable one rectangular normalized zone per camera, define an ROI in
`config.py`:

```python
DETECTION_ZONE = {
    "cam_a": (0.10, 0.15, 0.90, 0.95),
    "cam_b": (0.20, 0.10, 0.80, 0.90),
}
DETECTION_ZONE_NAMES = {
    "cam_a": "Main entrance",
    "cam_b": "Corridor",
}
```

Coordinates are `(left, top, right, bottom)` fractions of frame width/height.
The Zones dashboard shows current occupancy and an estimated average dwell
duration for tracks inside the ROI. Dwell is in-memory and resets when the
Python engine restarts.

## Event investigation

The History dashboard searches durable detection events by identity/name/track,
camera, event type, and time range. Each event can open that identity's
chronological cross-camera timeline. Search results are bounded to 150 events;
the API supports up to 300.
