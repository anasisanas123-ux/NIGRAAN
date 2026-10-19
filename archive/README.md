# Historical material

This folder preserves old source patches, manual camera checks, reference files
and the source immediately before the performance refactor. Active code is in
`python_engine/`; automated regressions are in `tests/`.

The old patch scripts expect the previous monolithic `multi_camera.py`. Do not
execute them against the refactored project. Manual checks may also rely on their
original working directory. They are retained for reference, not imported by the
running application.
