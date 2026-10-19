# CCTV recognition and camera zones

Existing face, body Re-ID, behavior and weapon weights are unchanged.

## Camera zones

In the dashboard Live view, click **Set zones** on Camera A or B. Drag a rectangle
on the source camera snapshot, choose a type and click **Add drawn zone**, then
**Save zones**. Refresh camera image gets another snapshot without discarding
drawn zones. Remove and redraw a zone to change its position. Closing prompts
before discarding unsaved changes. An offline camera cannot supply an image.

- Recognition / occupancy: one rectangle per camera; restricts new face matches
  and supplies the existing occupancy/dwell summary. Already tracked identities
  may continue outside this rectangle. Removing it restores full-frame matching.
- Restricted: up to 20 named rectangles per camera, each with a dwell time.
  Entry and dwell produce review alerts. Existing rules use body-box centers,
  so draw areas around where a person's center is expected, rather than just feet.
- Boxes appear on dashboard feeds. Coordinates are normalized to the source
  image, independent of dashboard size. Reconfigure after moving a camera.
- Saved to `data/camera_zones.json` with atomic replacement, applied immediately,
  loaded on restart. Saving resets old occupancy and restricted-zone timers.

## Recognition changes

Face matching now requires at least 60 pixels in both face dimensions, detector
confidence 0.65, brightness 25–235, and pitch/yaw at most
45 degrees when pose is available. Live sharpness must be at least 8; frames
between 8 and 35 additionally require a face-match score of 0.70, the existing
identity margin, and three consistent observations. Weak matches on such frames
are excluded from anonymous face galleries. Enrollment still requires sharpness
35, 80 pixels and exactly
one detected face per photo. These are initial configurable gates, not validated
campus operating thresholds. Settings are `FACE_*` in `config.py`.

Student enrollment already accepts multiple photos and retains multiple gallery
embeddings. Capture sharp front and slight side views, including normal glasses.
Poor images are skipped; an entirely unusable set returns an actionable error.
Existing enrollments are not automatically rebuilt or removed.

Before a live camera assigns a new name, three spatially associated observations
must agree. Camera histories are separate; gallery changes, mismatches and gaps
reset confirmation. Existing person tracking then maintains continuity. A track
label is not fresh face confirmation in every frame. Existing Re-ID safeguards
remain in place; no claim is made that body appearance proves identity.

Full-frame recognition uses the original camera frame. Every 0.8 seconds, up to
four unresolved person crops are also checked using the same face models. Crop
selection rotates to avoid starving later tracks; duplicate faces are merged.
Inference is serialized with enrollment to protect shared model sessions. This
adds compute cost, which must be profiled with the actual camera count. It cannot
recover facial detail absent from the stream. The displayed 640x480 preview does
not control recognition resolution.

## Measure before tuning

Use separate, consented test photos not copied from enrollment, covering different
distances, illumination, head angles and unregistered people. Each image should
contain one person. Example CSV `camera_test.csv`:

```csv
image,expected_id,group
test_images/near.jpg,S001,near_day
test_images/far.jpg,S001,far_day
test_images/unknown.jpg,,unknown_day
```

Run from the project root (preferably with the live engine stopped to avoid GPU
contention):

```powershell
.\venv\Scripts\python.exe evaluate_faces.py camera_test.csv --output face_evaluation.json
```

The report separates correct known matches, missed known identities, wrong names
and correctly rejected unknowns by group. It evaluates image matching, not video
tracking. Evaluate video separately for identity switches and confirmation delay.
Do not lower the 0.55 identity threshold just to hide missed recognition: measure
incorrect matches against the whole gallery, including unknown people. No campus
accuracy or recognition distance is established by the synthetic regression tests.
