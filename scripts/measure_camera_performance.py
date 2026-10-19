"""Sample existing camera health; no models or cameras are opened by this tool."""

import argparse
import json
import statistics
import time
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--url", default="http://127.0.0.1:5001/health")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    samples = {}
    for _ in range(max(1, args.seconds)):
        with urlopen(args.url, timeout=3) as response:
            health = json.load(response)
        for camera, value in health["cameras"].items():
            if value["status"] == "online":
                samples.setdefault(camera, []).append(value)
        time.sleep(1)
    summary = {}
    fields = (
        "fps",
        "capture_fps",
        "processed_fps",
        "inference_ms",
        "person_ms",
        "face_ms",
        "inference_age_seconds",
    )
    for camera, values in samples.items():
        summary[camera] = {
            "samples": len(values),
            "max_tracks": max(v.get("active_tracks", 0) for v in values),
        }
        for field in fields:
            measured = [v[field] for v in values if type(v.get(field)) in (int, float)]
            if measured:
                summary[camera][field] = {
                    "mean": round(statistics.mean(measured), 2),
                    "min": round(min(measured), 2),
                    "max": round(max(measured), 2),
                }
    report = {
        "summary": summary,
        "samples": samples,
        "note": "FPS values are stage-specific. Preview FPS is not recognition throughput. Scene load determines applicability.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "note": report["note"]}, indent=2))


if __name__ == "__main__":
    main()
