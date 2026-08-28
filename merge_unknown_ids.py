from pathlib import Path
from datetime import datetime
import argparse
import shutil
import numpy as np


GALLERIES = [
    {
        "name": "anonymous_face",
        "dir": Path("data/anonymous_face_gallery"),
        "max_samples": 20,
        "normalize": True,
    },
    {
        "name": "body_reid",
        "dir": Path("data/reid_gallery"),
        "max_samples": 20,
        "normalize": True,
    },
    {
        "name": "gait",
        "dir": Path("data/gait_gallery_runtime_observation"),
        "max_samples": None,
        "normalize": False,
    },
]


def load_array(path):
    arr = np.load(
        path,
        allow_pickle=False
    )

    arr = np.asarray(
        arr,
        dtype=np.float32
    )

    if arr.ndim == 1:
        arr = arr.reshape(
            1,
            -1
        )

    return arr


def normalize_rows(arr):
    arr = np.asarray(
        arr,
        dtype=np.float32
    )

    norms = np.linalg.norm(
        arr,
        axis=1,
        keepdims=True
    )

    valid = (
        norms[:, 0]
        >
        1e-12
    )

    arr = arr[valid]

    if len(arr) == 0:
        return arr

    norms = np.linalg.norm(
        arr,
        axis=1,
        keepdims=True
    )

    return (
        arr
        /
        np.maximum(
            norms,
            1e-12
        )
    ).astype(
        np.float32
    )


def deduplicate_embeddings(
    arr,
    similarity_threshold=0.995
):
    if len(arr) <= 1:
        return arr

    kept = []

    for sample in arr:

        sample = np.asarray(
            sample,
            dtype=np.float32
        )

        keep = True

        for existing in kept:

            denom = (
                np.linalg.norm(sample)
                *
                np.linalg.norm(existing)
                +
                1e-12
            )

            similarity = float(
                np.dot(
                    sample,
                    existing
                )
                /
                denom
            )

            if (
                similarity
                >=
                similarity_threshold
            ):
                keep = False
                break

        if keep:
            kept.append(
                sample.copy()
            )

    if not kept:
        return np.empty(
            (
                0,
                arr.shape[1]
            ),
            dtype=np.float32
        )

    return np.stack(
        kept,
        axis=0
    ).astype(
        np.float32
    )


def choose_diverse(
    arr,
    max_samples
):
    if (
        max_samples is None
        or
        len(arr) <= max_samples
    ):
        return arr

    # Start with first sample, then repeatedly
    # pick the sample least similar to what
    # is already selected.
    selected = [0]

    while (
        len(selected)
        <
        max_samples
    ):

        best_index = None
        best_diversity = None

        for idx in range(
            len(arr)
        ):

            if idx in selected:
                continue

            similarities = []

            for chosen_idx in selected:

                a = arr[idx]
                b = arr[chosen_idx]

                denom = (
                    np.linalg.norm(a)
                    *
                    np.linalg.norm(b)
                    +
                    1e-12
                )

                similarities.append(
                    float(
                        np.dot(
                            a,
                            b
                        )
                        /
                        denom
                    )
                )

            # Lower maximum similarity means
            # this sample adds a more different view.
            score = max(
                similarities
            )

            if (
                best_diversity is None
                or
                score
                <
                best_diversity
            ):

                best_diversity = score
                best_index = idx

        if best_index is None:
            break

        selected.append(
            best_index
        )

    return arr[
        selected
    ]


def find_identity_files(
    directory,
    identity
):
    if not directory.exists():
        return []

    files = []

    for path in directory.iterdir():

        if not path.is_file():
            continue

        # Accept:
        #
        # UNK-005.npy
        # UNK-005.npy.disabled
        # UNK-005.npy.disabled_duplicate
        #
        if (
            path.name
            ==
            f"{identity}.npy"
            or
            path.name.startswith(
                f"{identity}.npy."
            )
        ):

            files.append(
                path
            )

    return sorted(
        files
    )


def active_target_path(
    directory,
    identity
):
    return (
        directory
        /
        f"{identity}.npy"
    )


def merge_gallery(
    gallery,
    keep_id,
    merge_id,
    backup_root
):
    name = gallery[
        "name"
    ]

    directory = gallery[
        "dir"
    ]

    max_samples = gallery[
        "max_samples"
    ]

    should_normalize = gallery[
        "normalize"
    ]


    print()
    print(
        "=" * 60
    )

    print(
        f"[MERGE] Gallery: {name}"
    )

    print(
        "=" * 60
    )


    directory.mkdir(
        parents=True,
        exist_ok=True
    )


    keep_files = (
        find_identity_files(
            directory,
            keep_id
        )
    )

    merge_files = (
        find_identity_files(
            directory,
            merge_id
        )
    )


    print(
        f"Keep files : "
        f"{[p.name for p in keep_files]}"
    )

    print(
        f"Merge files: "
        f"{[p.name for p in merge_files]}"
    )


    if (
        not keep_files
        and
        not merge_files
    ):

        print(
            "[MERGE] Nothing found."
        )

        return


    gallery_backup = (
        backup_root
        /
        name
    )

    gallery_backup.mkdir(
        parents=True,
        exist_ok=True
    )


    # Backup every related file.
    for path in (
        keep_files
        +
        merge_files
    ):

        shutil.copy2(
            path,
            gallery_backup
            /
            path.name
        )


    arrays = []


    for path in keep_files:

        try:

            arr = load_array(
                path
            )

            arrays.append(
                arr
            )

            print(
                f"[LOAD] "
                f"{path.name}: "
                f"{arr.shape}"
            )

        except Exception as e:

            print(
                f"[WARNING] Could not load "
                f"{path}: {e}"
            )


    for path in merge_files:

        try:

            arr = load_array(
                path
            )

            arrays.append(
                arr
            )

            print(
                f"[LOAD] "
                f"{path.name}: "
                f"{arr.shape}"
            )

        except Exception as e:

            print(
                f"[WARNING] Could not load "
                f"{path}: {e}"
            )


    if not arrays:

        print(
            "[MERGE] No valid arrays."
        )

        return


    dimensions = {
        arr.shape[1]
        for arr
        in arrays
    }


    if len(
        dimensions
    ) != 1:

        print(
            "[ERROR] Different embedding "
            f"dimensions found: {dimensions}"
        )

        print(
            "Skipping this gallery."
        )

        return


    combined = np.vstack(
        arrays
    ).astype(
        np.float32
    )


    before = len(
        combined
    )


    if should_normalize:

        combined = normalize_rows(
            combined
        )


    combined = (
        deduplicate_embeddings(
            combined
        )
    )


    after_dedup = len(
        combined
    )


    combined = choose_diverse(
        combined,
        max_samples
    )


    after_cap = len(
        combined
    )


    target = (
        active_target_path(
            directory,
            keep_id
        )
    )


    np.save(
        target,
        combined.astype(
            np.float32
        )
    )


    print(
        f"[SAVE] {target}"
    )

    print(
        f"[STATS] "
        f"combined={before} "
        f"dedup={after_dedup} "
        f"final={after_cap}"
    )


    # Disable all duplicate identity files.
    for source in merge_files:

        disabled = (
            source.parent
            /
            (
                source.name
                +
                f".merged_into_{keep_id}"
            )
        )

        if disabled.exists():

            disabled.unlink()


        source.rename(
            disabled
        )


        print(
            f"[DISABLE] "
            f"{source.name} "
            f"-> "
            f"{disabled.name}"
        )


    # There may also be disabled copies of the KEEP identity.
    # Keep only the newly-generated active .npy.
    for keep_file in keep_files:

        if (
            keep_file
            ==
            target
        ):
            continue

        if not keep_file.exists():
            continue

        archived = (
            keep_file.parent
            /
            (
                keep_file.name
                +
                ".old_after_merge"
            )
        )

        if archived.exists():

            archived.unlink()


        keep_file.rename(
            archived
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Safely merge one NIGRAAN "
            "unknown identity into another."
        )
    )


    parser.add_argument(
        "--keep",
        required=True,
        help="Identity to keep, e.g. UNK-001"
    )


    parser.add_argument(
        "--merge",
        required=True,
        help="Duplicate identity, e.g. UNK-005"
    )


    args = parser.parse_args()


    keep_id = args.keep.strip()
    merge_id = args.merge.strip()


    if not (
        keep_id.startswith(
            "UNK-"
        )
        and
        merge_id.startswith(
            "UNK-"
        )
    ):

        raise SystemExit(
            "Only UNK-* identities are allowed."
        )


    if (
        keep_id
        ==
        merge_id
    ):

        raise SystemExit(
            "Keep and merge IDs cannot be the same."
        )


    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )


    backup_root = (
        Path(
            "data/"
            "unknown_merge_backups"
        )
        /
        (
            f"{timestamp}_"
            f"{merge_id}_into_"
            f"{keep_id}"
        )
    )


    backup_root.mkdir(
        parents=True,
        exist_ok=True
    )


    print()
    print(
        "=========================================="
    )

    print(
        "NIGRAAN UNKNOWN ID MERGE"
    )

    print(
        "=========================================="
    )

    print(
        f"KEEP : {keep_id}"
    )

    print(
        f"MERGE: {merge_id}"
    )

    print(
        f"BACKUP: {backup_root}"
    )


    for gallery in GALLERIES:

        merge_gallery(
            gallery,
            keep_id,
            merge_id,
            backup_root
        )


    print()
    print(
        "=========================================="
    )

    print(
        "MERGE COMPLETE"
    )

    print(
        "=========================================="
    )

    print(
        f"{merge_id} has been merged "
        f"into {keep_id}."
    )

    print(
        "Restart multi_camera.py so "
        "the galleries reload."
    )


if __name__ == "__main__":
    main()
