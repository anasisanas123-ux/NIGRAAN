# Dashboard bug fixes: live verification

Restart both the Python engine and Node dashboard after applying these changes.
Open `http://localhost:3000` and hard-refresh the page to load the updated capture code.

1. **Student removal:** recognize an enrolled student, then delete them while they
   remain in view. Their known label must disappear on subsequent processed frames.
   Restart the engine and confirm the deleted enrollment does not return.
2. **Student and visitor capture:** choose **Camera A (running engine)** or Camera B,
   then Start Camera. Capture uses the camera already open in Python, including when
   the dashboard is accessed over HTTP on the local network. The optional **Browser
   webcam** source requires localhost/HTTPS, permission, and a camera available to the
   browser. Uploading photos remains available.
3. **Visitor recognition:** register a visitor from a clear photo, including while
   they already have an unknown tracker ID. Keep their face visible. Unknown tracks
   retry face matching every 0.8 seconds and immediately after gallery changes.
   A successful match must show `VISITOR: <name>`; checkout or expiry must remove it.
4. **Activity:** the dashboard polls cues every five seconds and adds cues to the
   alert panel. Frames reach behavior processing once per processed camera frame,
   rather than repeatedly resubmitting the display's last frame. Check the Activity
   Review model status for readiness or a specific dependency/loading error.
   Motion cues work without restricted zones. Entry/dwell cues require real camera
   regions in `BEHAVIOR_RESTRICTED_ZONES` in `config.py`; no arbitrary region is
   enabled by this fix. The existing violence model still requires enough sampled
   frames and consecutive positive windows; a cue is a prompt to review footage.

Offline checks (no cameras or project servers):

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -p test_dashboard_regressions.py -v
node --test tests/capture-camera.test.cjs
```

The Python tests use temporary galleries and a fake face extractor; they verify
state changes and API behavior, not real recognition accuracy. Live camera,
browser-permission, MongoDB, and model accuracy checks still need the running app.
