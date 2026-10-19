"""Evaluate a saved gait gallery with leave-one-out identity matching."""

import argparse
import json
from pathlib import Path

from python_engine.gait_evaluator import evaluate_gallery


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Measure gait-signature top-1 accuracy by holding out each saved "
            "signature in turn. Gallery filenames are identity labels."
        )
    )
    parser.add_argument(
        "--gallery",
        default="data/gait_gallery_observation",
        help="Folder containing one <identity>.npy file per identity",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum cosine similarity to accept a match",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=None,
        help="Optional minimum best-versus-second score margin to accept a match",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path for the full metrics and per-sample predictions JSON",
    )
    args = parser.parse_args()

    result = evaluate_gallery(args.gallery, args.min_score, args.min_margin)
    print(f"Gait gallery: {result['gallery_dir']}")
    print(
        f"Identities: {result['identities_in_gallery']} total, "
        f"{result['evaluable_identities']} evaluable"
    )

    if result["evaluated_samples"]:
        print(
            f"Leave-one-out: {result['correct_samples']}/{result['evaluated_samples']} "
            f"correct; top-1 accuracy={result['top1_accuracy']:.1%}; "
            f"macro F1={result['macro_f1']:.1%}"
        )
        print(
            f"Accepted: {result['accepted_samples']}/{result['evaluated_samples']} "
            f"({result['coverage']:.1%} coverage); "
            f"accepted accuracy="
            f"{result['accepted_accuracy']:.1%}"
            if result["accepted_accuracy"] is not None
            else f"Accepted: 0/{result['evaluated_samples']} (0.0% coverage)"
        )
    else:
        print(
            "No evaluable samples. Each identity needs at least two saved "
            "signatures, and at least two identities must meet that minimum."
        )

    for label, metrics in result["per_identity"].items():
        print(
            f"  {label}: {metrics['correct']}/{metrics['samples']} correct "
            f"(F1={metrics['f1']:.1%})"
        )
    for label, count in result["excluded_identities"].items():
        print(f"  {label}: excluded ({count} saved signature; need at least 2)")
    for item in result["skipped_files"]:
        print(f"  skipped {item['file']}: {item['reason']}")

    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Full report: {output}")


if __name__ == "__main__":
    main()
