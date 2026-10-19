"""Offline leave-one-out evaluation for saved gait signature galleries."""

from pathlib import Path

import numpy as np

from .gait_signature import GaitSignatureEngine


def load_gallery(gallery_dir):
    """Load valid signature rows, using each .npy filename as its identity label."""
    gallery = {}
    skipped = []

    for path in sorted(Path(gallery_dir).glob("*.npy")):
        try:
            data = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            skipped.append({"file": path.name, "reason": f"could not load: {exc}"})
            continue

        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[1] == 0:
            skipped.append({"file": path.name, "reason": "expected a 1D or 2D array of signatures"})
            continue

        rows = data[np.isfinite(data).all(axis=1)]
        if len(rows) != len(data):
            skipped.append({"file": path.name, "reason": f"ignored {len(data) - len(rows)} non-finite signature(s)"})
        if len(rows):
            gallery[path.stem] = rows.astype(np.float32, copy=False)

    return gallery, skipped


def _rank(query, gallery, excluded_label, excluded_index):
    rankings = []
    for label, samples in gallery.items():
        candidates = samples
        if label == excluded_label:
            candidates = np.delete(samples, excluded_index, axis=0)
        if not len(candidates):
            continue

        scores = [GaitSignatureEngine.similarity(query, sample) for sample in candidates]
        scores.sort(reverse=True)
        rankings.append((label, float(np.mean(scores[:3]))))

    rankings.sort(key=lambda item: item[1], reverse=True)
    if not rankings:
        return None, 0.0, 0.0
    second = rankings[1][1] if len(rankings) > 1 else 0.0
    return rankings[0][0], rankings[0][1], rankings[0][1] - second


def evaluate_gallery(gallery_dir, min_score=None, min_margin=None):
    """Return leave-one-out predictions and summary metrics for a labeled gallery."""
    gallery, skipped = load_gallery(gallery_dir)
    dimensions = {}
    for rows in gallery.values():
        dimensions[rows.shape[1]] = dimensions.get(rows.shape[1], 0) + 1
    if dimensions:
        expected_dimension = max(dimensions, key=dimensions.get)
        for label in list(gallery):
            if gallery[label].shape[1] != expected_dimension:
                skipped.append({
                    "file": f"{label}.npy",
                    "reason": f"signature length {gallery[label].shape[1]} does not match common length {expected_dimension}",
                })
                del gallery[label]

    usable = {label: rows for label, rows in gallery.items() if len(rows) >= 2}
    excluded = {label: int(len(rows)) for label, rows in gallery.items() if len(rows) < 2}
    labels = sorted(usable)
    predictions = []

    # A single identity forms a trivial classifier, so require at least two
    # identities with multiple samples before reporting recognition metrics.
    if len(labels) >= 2:
        for truth in labels:
            for index, query in enumerate(usable[truth]):
                predicted, score, margin = _rank(query, usable, truth, index)
                accepted = predicted is not None
                if min_score is not None and score < min_score:
                    accepted = False
                if min_margin is not None and margin < min_margin:
                    accepted = False
                predictions.append({
                    "truth": truth,
                    "prediction": predicted,
                    "score": score,
                    "margin": margin,
                    "accepted": accepted,
                    "correct": predicted == truth,
                })

    count = len(predictions)
    correct = sum(item["correct"] for item in predictions)
    accepted_items = [item for item in predictions if item["accepted"]]
    accepted_correct = sum(item["correct"] for item in accepted_items)

    per_identity = {}
    for label in labels:
        rows = [item for item in predictions if item["truth"] == label]
        true_positive = sum(item["prediction"] == label for item in rows)
        false_positive = sum(item["prediction"] == label and item["truth"] != label for item in predictions)
        false_negative = len(rows) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_identity[label] = {
            "samples": len(rows),
            "correct": true_positive,
            "accuracy": true_positive / len(rows),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "gallery_dir": str(Path(gallery_dir)),
        "identities_in_gallery": len(gallery),
        "evaluable_identities": len(labels),
        "excluded_identities": excluded,
        "evaluated_samples": count,
        "correct_samples": correct,
        "top1_accuracy": correct / count if count else None,
        "macro_f1": float(np.mean([item["f1"] for item in per_identity.values()])) if per_identity else None,
        "accepted_samples": len(accepted_items),
        "coverage": len(accepted_items) / count if count else None,
        "accepted_accuracy": accepted_correct / len(accepted_items) if accepted_items else None,
        "min_score": min_score,
        "min_margin": min_margin,
        "per_identity": per_identity,
        "predictions": predictions,
        "skipped_files": skipped,
    }
