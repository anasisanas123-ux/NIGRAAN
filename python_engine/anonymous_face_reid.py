from pathlib import Path
import threading

import numpy as np


class AnonymousFaceReID:
    """
    Persistent ArcFace gallery for UNKNOWN global identities.

    Example:
        UNK-001 -> several normalized 512-D ArcFace embeddings
        UNK-002 -> several normalized 512-D ArcFace embeddings

    This module does not create UNK IDs.
    ReIDManager remains responsible for global ID creation.
    """

    def __init__(
        self,
        gallery_dir=None,
        match_threshold=0.60,
        margin_threshold=0.08,
        max_embeddings=20
    ):

        if gallery_dir is None:
            project_root = (
                Path(__file__)
                .resolve()
                .parent
                .parent
            )

            gallery_dir = (
                project_root
                / "data"
                / "anonymous_face_gallery"
            )

        self.gallery_dir = Path(
            gallery_dir
        )

        self.gallery_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.match_threshold = float(
            match_threshold
        )

        self.margin_threshold = float(
            margin_threshold
        )

        self.max_embeddings = int(
            max_embeddings
        )

        self.lock = threading.RLock()

        self.gallery = {}

        self._load_gallery()

        print(
            "[ANON FACE] Ready | "
            f"identities={len(self.gallery)} | "
            f"threshold={self.match_threshold:.2f} | "
            f"margin={self.margin_threshold:.2f}"
        )


    # =========================================================
    # NORMALIZATION
    # =========================================================

    def _normalize(
        self,
        embedding
    ):

        if embedding is None:

            return None

        emb = np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(-1)

        if emb.size == 0:

            return None

        norm = float(
            np.linalg.norm(
                emb
            )
        )

        if (
            not np.isfinite(norm)
            or
            norm <= 1e-12
        ):

            return None

        emb = emb / norm

        if not np.all(
            np.isfinite(
                emb
            )
        ):

            return None

        return emb.astype(
            np.float32
        )


    # =========================================================
    # FILE PATH
    # =========================================================

    def _identity_path(
        self,
        global_id
    ):

        safe_id = str(
            global_id
        ).replace(
            "/",
            "_"
        ).replace(
            "\\",
            "_"
        )

        return (
            self.gallery_dir
            / f"{safe_id}.npy"
        )


    # =========================================================
    # LOAD
    # =========================================================

    def _load_gallery(
        self
    ):

        with self.lock:

            self.gallery = {}

            for path in (
                self.gallery_dir
                .glob("UNK-*.npy")
            ):

                try:

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

                    valid = []

                    for emb in arr:

                        emb = self._normalize(
                            emb
                        )

                        if emb is not None:

                            valid.append(
                                emb
                            )

                    if not valid:

                        continue

                    global_id = path.stem

                    self.gallery[
                        global_id
                    ] = valid[
                        -self.max_embeddings:
                    ]

                except Exception as e:

                    print(
                        f"[ANON FACE LOAD ERROR] "
                        f"{path.name}: {e}"
                    )


    # =========================================================
    # SAVE
    # =========================================================

    def _save_identity(
        self,
        global_id
    ):

        embeddings = self.gallery.get(
            global_id,
            []
        )

        if not embeddings:

            return

        arr = np.stack(
            embeddings,
            axis=0
        ).astype(
            np.float32
        )

        np.save(
            self._identity_path(
                global_id
            ),
            arr
        )


    # =========================================================
    # ADD FACE TO GLOBAL UNK ID
    # =========================================================

    def add(
        self,
        global_id,
        embedding
    ):

        if not global_id:

            return False

        if not str(
            global_id
        ).startswith(
            "UNK-"
        ):

            return False

        emb = self._normalize(
            embedding
        )

        if emb is None:

            return False

        with self.lock:

            samples = self.gallery.setdefault(
                global_id,
                []
            )

            # Avoid saving nearly identical frames repeatedly.
            if samples:

                best_existing = max(
                    float(
                        np.dot(
                            emb,
                            existing
                        )
                    )
                    for existing in samples
                )

                if best_existing >= 0.995:

                    return True

            samples.append(
                emb.copy()
            )

            self.gallery[
                global_id
            ] = samples[
                -self.max_embeddings:
            ]

            self._save_identity(
                global_id
            )

            print(
                f"[ANON FACE ADD] "
                f"{global_id} "
                f"samples="
                f"{len(self.gallery[global_id])}"
            )

            return True


    # =========================================================
    # SCORE ONE GLOBAL ID
    # =========================================================

    def _identity_score(
        self,
        embedding,
        samples
    ):

        if not samples:

            return -1.0

        scores = sorted(
            (
                float(
                    np.dot(
                        embedding,
                        sample
                    )
                )
                for sample in samples
            ),
            reverse=True
        )

        # Use several good views rather than trusting only
        # one accidental high similarity.
        top = scores[
            :min(
                3,
                len(scores)
            )
        ]

        return float(
            np.mean(
                top
            )
        )


    # =========================================================
    # MATCH
    # =========================================================

    def best_candidate(
        self,
        embedding
    ):
        """
        Observation only.

        Returns the highest-ranked anonymous identity even
        when it does NOT satisfy the final match threshold.

        This method NEVER assigns or creates an identity.

        Returns:
            best_id, best_score, margin
        """

        emb = self._normalize(
            embedding
        )

        if emb is None:

            return (
                None,
                0.0,
                None
            )


        with self.lock:

            if not self.gallery:

                return (
                    None,
                    0.0,
                    None
                )


            rankings = []


            for (
                global_id,
                samples
            ) in self.gallery.items():

                score = self._identity_score(
                    emb,
                    samples
                )


                rankings.append(
                    (
                        global_id,
                        float(score)
                    )
                )


            rankings.sort(
                key=lambda item:
                    item[1],
                reverse=True
            )


            if not rankings:

                return (
                    None,
                    0.0,
                    None
                )


            best_id = rankings[0][0]

            best_score = float(
                rankings[0][1]
            )


            if len(rankings) > 1:

                second_score = float(
                    rankings[1][1]
                )

                margin = (
                    best_score
                    -
                    second_score
                )

            else:

                margin = None


            return (
                best_id,
                best_score,
                margin
            )


    def match(
        self,
        embedding
    ):

        """
        Returns:
            global_id, best_score, margin

        global_id is None unless both:
          - score >= match_threshold
          - separation from second identity >= margin_threshold

        For a gallery containing only one identity,
        only the score threshold is required.
        """

        emb = self._normalize(
            embedding
        )

        if emb is None:

            return (
                None,
                0.0,
                None
            )

        with self.lock:

            if not self.gallery:

                return (
                    None,
                    0.0,
                    None
                )

            rankings = []

            for (
                global_id,
                samples
            ) in self.gallery.items():

                score = self._identity_score(
                    emb,
                    samples
                )

                rankings.append(
                    (
                        global_id,
                        score
                    )
                )

            rankings.sort(
                key=lambda item: item[1],
                reverse=True
            )

            best_id = rankings[0][0]
            best_score = float(
                rankings[0][1]
            )

            if len(rankings) == 1:

                margin = None

                if (
                    best_score
                    >=
                    self.match_threshold
                ):

                    return (
                        best_id,
                        best_score,
                        margin
                    )

                return (
                    None,
                    best_score,
                    margin
                )

            second_score = float(
                rankings[1][1]
            )

            margin = (
                best_score
                -
                second_score
            )

            if (
                best_score
                >=
                self.match_threshold

                and

                margin
                >=
                self.margin_threshold
            ):

                return (
                    best_id,
                    best_score,
                    margin
                )

            return (
                None,
                best_score,
                margin
            )


    # =========================================================
    # INFO
    # =========================================================

    def get_sample_count(
        self,
        global_id
    ):

        with self.lock:

            return len(
                self.gallery.get(
                    global_id,
                    []
                )
            )


    def identities(
        self
    ):

        with self.lock:

            return list(
                self.gallery.keys()
            )
