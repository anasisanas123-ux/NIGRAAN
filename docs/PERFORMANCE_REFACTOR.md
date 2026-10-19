# Performance and project cleanup

Existing model weights, identity thresholds, detection resolution and galleries
are preserved. Start with `python main.py` or `python multi_camera.py` as before.

## Changes

The old monolithic entry script was split into capture, bounded frame transport,
identity worker, model composition, preview and display modules. The identity
worker receives shared models/state explicitly through `RuntimeContext`.
Thirteen nested identity helpers were checked against their original Python
syntax trees and are unchanged. Re-ID, body extraction and annotation formatting
were compacted with syntax-equivalence checks; comments remain.

Captured frames carry a sequence and camera generation. Independent preview and
inference consumers keep their own sequence cursor. Each processes new frames
only; neither accumulates an unbounded backlog. Reconnects reset local tracking
using the packet's generation. An offline camera continues attempting reconnects.

Fresh previews no longer wait for recognition. Existing annotations are scaled
to a preview that preserves the source aspect ratio, and expire after one second
or a reconnect. Deleted/expired enrollment labels are filtered between inference
runs. Browser viewers share lazily encoded JPEGs per frame, and desktop windows
reuse the rendered preview. A per-camera result slot replaces the shared FIFO.
Browser proxy connections are released when viewers disconnect.

The BGR camera frame goes directly to YOLO. RGB person crops are converted only
after detection, preserving the original OSNet/snapshot color contract. Full-frame
BGR-to-RGB-to-BGR conversions are removed from the main pipeline.

Face checks are scheduled at most five times per second per camera; body tracking
continues between checks. Face quality gates, three-observation confirmation and
identity match thresholds are unchanged. This scheduling changes the timing of
face checks and should be evaluated on representative moving crowds. The unknown
retry timestamp advances only when a face pass is actually scheduled.

FPS now counts actual events over a rolling window instead of averaging inverse
frame intervals, which could inflate results on bursty capture. Capture, preview
and AI processing rates are separate. Health also exposes source resolution,
person/face timings, skipped inference frames and result age. Preview is capped
at 20 FPS, while the current laptop camera supplies approximately 10 unique FPS.

## Live verification

On 19 October 2026, a 20-second Camera A sample with at most one visible person
recorded approximately:

| Stage | Mean FPS |
|---|---:|
| Camera capture | 9.99 |
| Preview | 9.99 |
| AI processing | 7.82 |

This is a limited live sanity check, not a campus load benchmark or an accuracy
evaluation. Two simultaneous dashboard proxy viewers received valid 640x360
frames, while model inputs retained the original 1280x720 source. Camera B was
offline and was not benchmarked. Violence and weapon models reported ready.
Trying MJPEG/30 FPS did not increase the laptop's actual capture rate, so original
25 FPS/default format requests were restored; MJPEG remains an optional setting.

Reproduce measurements without opening extra cameras/models:

```powershell
.\venv\Scripts\python.exe scripts/measure_camera_performance.py --seconds 20 --output data/logs/performance.json
.\venv\Scripts\python.exe scripts/verify_camera_stream.py
```

Health is sampled once per second. Latest inference times in these samples are
not a complete per-frame latency distribution. Source delivery and scene load
can change results; do not compare old inflated counters as a speedup percentage.

## Cleanup

- 33 historical patch/backup files moved under `archive/legacy_patches/`.
- 12 manual diagnostics moved under `archive/manual_checks/`.
- Other backup/reference material retained under `archive/`.
- Guides moved to `docs/`; root CLI tools remain for compatibility.
- Completed PyTorch installer removed, reclaiming 2.51 GiB; installed CUDA-enabled
  PyTorch/torchvision verified first. Setup still supports reinstalling pinned
  packages without this cached wheel.
- Active logs go under `data/logs/`. Persistent data, original weights and `venv/`
  were retained. VS Code excludes archives/environments from searches and bulky
  generated directories from file watching.

Validation: 37 offline Python regressions, 11 Node/dashboard tests, preserved
identity helper syntax, both live model statuses, zone API and two live viewers.
