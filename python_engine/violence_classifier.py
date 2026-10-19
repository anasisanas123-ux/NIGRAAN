"""Optional RWF-2000 X3D-M violence cue model.

The classifier produces scene-level review cues only; it never assigns a
person identity or makes a finding about an individual.
"""

from collections import deque
import os
import queue
import threading
import time

import cv2
import numpy as np


def _numpy2_scalar(dtype, data):
    """Read NumPy 2 scalar metadata using the project's pinned NumPy 1.26."""
    from numpy.core.multiarray import scalar
    return scalar(dtype, data)


# PyTorch 2.4 keys its restricted allowlist by a callable's module and name.
# The checkpoint references the NumPy 2 path; do not change NumPy's own globals.
_numpy2_scalar.__module__ = 'numpy._core.multiarray'
_numpy2_scalar.__name__ = 'scalar'
_numpy2_scalar.__qualname__ = 'scalar'


class ViolenceClassifier:
    def __init__(
        self,
        checkpoint_path,
        enabled=True,
        threshold=0.55,
        confirm_windows=2,
        sample_interval=0.25,
        inference_interval=1.0,
        cooldown=20.0,
    ):
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.enabled = bool(enabled)
        self.threshold = float(threshold)
        self.confirm_windows = max(1, int(confirm_windows))
        self.sample_interval = max(0.05, float(sample_interval))
        self.inference_interval = max(0.25, float(inference_interval))
        self.cooldown = max(1.0, float(cooldown))
        self._lock = threading.Lock()
        self._frames = {}
        self._last_sample = {}
        self._last_enqueue = {}
        self._streaks = {}
        self._last_alert = {}
        self._alerts = deque(maxlen=200)
        self._queue = queue.Queue(maxsize=2)
        self._worker_started = False
        self._state = "disabled" if not self.enabled else "weights_missing"
        self._error = None
        self._model = None
        self._torch = None
        self._device = "cpu"
        self._camera_status = {}

    def status(self):
        with self._lock:
            cameras = {}
            for camera_id, frames in self._frames.items():
                cameras[camera_id] = {
                    **self._camera_status.get(camera_id, {}),
                    "sampled_frames": len(frames),
                    "required_frames": 16,
                }
            return {"state": self._state, "error": self._error,
                    "device": str(self._device), "cameras": cameras}

    def snapshot(self):
        with self._lock:
            return list(self._alerts)[::-1]

    def update(self, cam_id, frame, now=None):
        """Sample raw BGR frames and enqueue a recent 16-frame scene clip."""
        if not self.enabled or frame is None:
            return
        if not os.path.isfile(self.checkpoint_path):
            with self._lock:
                if self._state != "load_error":
                    self._state = "weights_missing"
            return

        now = time.monotonic() if now is None else float(now)
        camera_id = str(cam_id)
        with self._lock:
            if self._state == "load_error":
                return
            if not self._worker_started:
                self._worker_started = True
                threading.Thread(target=self._worker, name="violence-model", daemon=True).start()
                self._state = "loading"
            if now - self._last_sample.get(camera_id, -float("inf")) < self.sample_interval:
                return
            if now - self._last_sample.get(camera_id, now) > max(2.0, self.sample_interval * 4):
                # Never classify a clip stitched together across a camera outage.
                self._frames.pop(camera_id, None)
                self._streaks[camera_id] = 0
                self._last_enqueue.pop(camera_id, None)
                self._camera_status.pop(camera_id, None)
            self._last_sample[camera_id] = now
            self._frames.setdefault(camera_id, deque(maxlen=16)).append(frame.copy())
            frames = self._frames[camera_id]
            if len(frames) < 16 or now - self._last_enqueue.get(camera_id, -float("inf")) < self.inference_interval:
                return
            self._last_enqueue[camera_id] = now
            clip = list(frames)

        try:
            self._queue.put_nowait((camera_id, clip))
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((camera_id, clip))
            except queue.Full:
                pass

    def _load_model(self, device_override=None):
        import torch
        from torch import nn
        from pytorchvideo.models.hub import x3d_m

        model = x3d_m(pretrained=False)
        # NumPy 1.26 can expose an incomplete `_core` namespace. Import the
        # scalar constructor directly rather than treating that as NumPy 2.
        from numpy.core.multiarray import scalar as numpy_scalar
        from _codecs import encode as bytes_encode
        # Protocol-2 NumPy scalar metadata also reconstructs its raw bytes via
        # _codecs.encode, which PyTorch 2.4 does not include by default.
        safe_types = [numpy_scalar, _numpy2_scalar, np.dtype,
                      type(np.dtype(np.float64)), bytes_encode]
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is not None:
            with safe_globals(safe_types):
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
        elif hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals(safe_types)
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        else:
            raise RuntimeError(
                "This checkpoint needs PyTorch 2.4 or newer for restricted safe loading. "
                "Update torch and torchvision from the matching PyTorch index."
            )
        state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
        if not isinstance(state, dict):
            raise ValueError("Checkpoint does not contain a model state dictionary.")
        clean_state = {}
        for key, value in state.items():
            key = key.removeprefix("module.")
            key = key.removeprefix("backbone.")
            clean_state[key] = value
        # The selected final checkpoint was trained with dropout before its
        # binary projection, while baseline X3D checkpoints use a plain Linear.
        head_key = "blocks.5.proj.1.weight"
        if head_key in clean_state:
            dropout = float(checkpoint.get("hyper_params", {}).get("dropout", 0.3))
            model.blocks[5].proj = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(2048, 2),
            )
        else:
            model.blocks[5].proj = nn.Linear(2048, 2)
        model.load_state_dict(clean_state, strict=True)
        device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device).eval()
        return torch, model, device

    @staticmethod
    def _preprocess(frames, torch, device):
        clip = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            clip.append(rgb.astype(np.float32) / 255.0)
        array = np.stack(clip, axis=0)  # T,H,W,C
        tensor = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
        mean = torch.tensor([0.45, 0.45, 0.45], dtype=tensor.dtype).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225], dtype=tensor.dtype).view(3, 1, 1, 1)
        return ((tensor - mean) / std).unsqueeze(0).to(device)

    def _worker(self):
        try:
            torch, self._model, self._device = self._load_model()
            self._torch = torch
            with self._lock:
                self._state = "ready"
                self._error = None
        except Exception as error:
            with self._lock:
                self._state = "load_error"
                self._error = f"{type(error).__name__}: {error}"
            return

        while True:
            camera_id, frames = self._queue.get()
            try:
                inputs = self._preprocess(frames, self._torch, self._device)
                with self._torch.inference_mode():
                    probabilities = self._model(inputs)
                    score = self._violence_probability(probabilities)
                with self._lock:
                    self._state = "ready"
                    self._error = None
                self._record_score(camera_id, score)
            except Exception as error:
                with self._lock:
                    self._state = "inference_error"
                    self._error = f"{type(error).__name__}: {error}"
                    self._streaks[camera_id] = 0
            finally:
                self._queue.task_done()

    @staticmethod
    def _violence_probability(prediction):
        # PyTorchVideo's X3D head already applies Softmax. Applying it again
        # compresses the scores and makes the configured threshold misleading.
        values = prediction.detach().float().cpu().numpy()
        if (values.shape != (1, 2) or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values > 1)
                or not np.isclose(values.sum(), 1.0, atol=1e-4)):
            raise ValueError('Expected two finite X3D class probabilities summing to one.')
        return float(values[0, 1])

    def _record_score(self, camera_id, score):
        now = time.monotonic()
        with self._lock:
            if score >= self.threshold:
                self._streaks[camera_id] = self._streaks.get(camera_id, 0) + 1
            else:
                self._streaks[camera_id] = 0
            self._camera_status[camera_id] = {
                "score": round(float(score), 4),
                "threshold": self.threshold,
                "positive_windows": self._streaks[camera_id],
                "required_positive_windows": self.confirm_windows,
                "last_inference_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if self._streaks[camera_id] < self.confirm_windows:
                return
            if now - self._last_alert.get(camera_id, -float("inf")) < self.cooldown:
                return
            self._last_alert[camera_id] = now
            self._alerts.append({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
                "camera_id": camera_id,
                "event_type": "possible_violent_activity",
                "description": (
                    f"Video model raised a possible violence cue (score {score:.2f}). "
                    "Review the footage; this is not a confirmed event."
                ),
                "score": round(score, 4),
                "review_only": True,
                "model": "RWF-2000 X3D-M",
            })


def create_default_classifier(config):
    return ViolenceClassifier(
        checkpoint_path=getattr(
            config,
            "VIOLENCE_MODEL_PATH",
            os.path.join(config.DATA_DIR, "models", "violence_rwf2000_x3d_m.pt"),
        ),
        enabled=getattr(config, "VIOLENCE_MODEL_ENABLED", True),
        threshold=getattr(config, "VIOLENCE_MODEL_THRESHOLD", 0.55),
        confirm_windows=getattr(config, "VIOLENCE_MODEL_CONFIRM_WINDOWS", 2),
        sample_interval=getattr(config, "VIOLENCE_MODEL_SAMPLE_INTERVAL", 0.25),
        inference_interval=getattr(config, "VIOLENCE_MODEL_INFERENCE_INTERVAL", 1.0),
        cooldown=getattr(config, "VIOLENCE_MODEL_ALERT_COOLDOWN", 20.0),
    )
