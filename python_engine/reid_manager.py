import os
import re
import json
import time
import cv2
import threading
import numpy as np

import config


class ReIDManager:

    def __init__(self):
        self.lock = threading.RLock()

        # =====================================================
        # GLOBAL GALLERY
        # =====================================================
        self.unknown_gallery = {}

        # =====================================================
        # LOCAL DEEPSORT TRACK STATE
        # =====================================================
        self.track_states = {}
        self.next_global_id = 1

        # =====================================================
        # NORMAL MATCH SETTINGS
        # =====================================================

        # Keep the conservative current operating points explicit in config
        # so a held-out evaluation can tune them without editing this class.
        self.same_camera_threshold = float(getattr(config, "REID_SAME_CAMERA_THRESHOLD", 0.85))
        self.cross_camera_threshold = float(
            getattr(
                config,
                "REID_CROSS_CAMERA_THRESHOLD",
                getattr(config, "REID_THRESHOLD", 0.88),
            )
        )
        self.same_camera_threshold = min(1.0, max(0.0, self.same_camera_threshold))
        self.cross_camera_threshold = min(1.0, max(0.0, self.cross_camera_threshold))
        self.min_margin = getattr(config, "REID_MIN_MARGIN", 0.12)
        self.confirm_frames = max(getattr(config, "REID_CONFIRM_FRAMES", 3), 3)

        # =====================================================
        # NEW PERSON SETTINGS
        # =====================================================
        self.new_id_frames = max(getattr(config, "REID_NEW_ID_FRAMES", 8), 8)
        self.new_person_threshold = getattr(config, "REID_NEW_PERSON_THRESHOLD", 0.50)

        # =====================================================
        # SINGLE-GALLERY CASE
        # =====================================================
        #
        # Earlier:
        #
        # <= 0.80 = new
        #
        # was too aggressive.
        #
        # The same person produced ~0.62-0.78 when DeepSORT
        # fragmented.
        #
        # Spatial handoff now handles those cases, but we still
        # make the normal "new person" rule safer.
        # =====================================================
        self.single_match_threshold = getattr(config, "REID_SINGLE_MATCH_THRESHOLD", 0.88)
        configured_single_new = getattr(config, "REID_SINGLE_NEW_THRESHOLD", 0.74)
        self.single_new_threshold = min(configured_single_new, 0.74)

        # =====================================================
        # SPATIAL TRACK HANDOFF SETTINGS
        # =====================================================
        #
        # This handles:
        #
        # cam_a_4 -> UNK-001
        #
        # DeepSORT suddenly creates:
        #
        # cam_a_7
        #
        # while it is still the same physical person.
        #
        # We compare:
        #
        # 1. same camera
        # 2. very recent previous confirmed track
        # 3. bounding-box overlap / position
        # 4. OSNet body similarity
        #
        # This allows UNK-001 to transfer to cam_a_7.
        # =====================================================
        self.handoff_max_gap = getattr(config, "REID_HANDOFF_MAX_GAP", 2.5)

        # Bounding boxes overlap enough.
        self.handoff_iou_threshold = getattr(config, "REID_HANDOFF_IOU_THRESHOLD", 0.25)

        # Or their centers are very close relative to body size.
        self.handoff_center_threshold = getattr(config, "REID_HANDOFF_CENTER_THRESHOLD", 0.30)

        # Prevent very differently-sized boxes from being
        # considered the same physical person.
        self.handoff_area_ratio = getattr(config, "REID_HANDOFF_AREA_RATIO", 0.45)

        # Even noisy OSNet samples around 0.50 should prevent a
        # new UNK ID from being created when spatial continuity
        # is very strong.
        self.handoff_guard_threshold = getattr(config, "REID_HANDOFF_GUARD_THRESHOLD", 0.48)

        # We don't require a single embedding to suddenly reach
        # 0.85. We accumulate several spatially-consistent body
        # scores.
        self.handoff_average_threshold = getattr(config, "REID_HANDOFF_AVERAGE_THRESHOLD", 0.58)
        self.handoff_peak_threshold = getattr(config, "REID_HANDOFF_PEAK_THRESHOLD", 0.60)
        self.handoff_confirm_frames = max(getattr(config, "REID_HANDOFF_CONFIRM_FRAMES", 3), 3)
        self.handoff_score_window = 5

        # =====================================================
        # GALLERY SETTINGS
        # =====================================================
        self.max_gallery_embeddings = getattr(config, "REID_MAX_GALLERY_EMBEDDINGS", 20)
        self.gallery_update_threshold = max(
            getattr(config, "REID_GALLERY_UPDATE_THRESHOLD", 0.88), 0.88
        )
        self.sample_interval = getattr(config, "REID_SAMPLE_INTERVAL", 0.30)
        self.track_timeout = max(getattr(config, "REID_TRACK_TIMEOUT", 8.0), 8.0)

        # =====================================================
        # CONFIRMED TRACK REVALIDATION
        # =====================================================
        #
        # Do not release a confirmed UNK identity because of one
        # bad body crop.
        # =====================================================
        self.revalidate_threshold = getattr(config, "REID_REVALIDATE_THRESHOLD", 0.55)
        self.revalidate_frames = max(getattr(config, "REID_REVALIDATE_FRAMES", 3), 5)

        # =====================================================
        # STORAGE
        # =====================================================
        self.snapshot_dir = os.path.join(config.DATA_DIR, "snapshots")
        self.gallery_dir = os.path.join(config.DATA_DIR, "reid_gallery")
        self.metadata_path = os.path.join(self.gallery_dir, "gallery.json")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.gallery_dir, exist_ok=True)
        self._load_gallery()
        print("\n========================================")
        print("ReIDManager ready")
        print(f"Loaded identities       : " f"{len(self.unknown_gallery)}")
        print(f"Same-camera strong match: " f"{self.same_camera_threshold:.2f}")
        print(f"Cross-camera strong     : " f"{self.cross_camera_threshold:.2f}")
        print(f"Single-ID new threshold : " f"{self.single_new_threshold:.2f}")
        print(f"Handoff guard           : " f"{self.handoff_guard_threshold:.2f}")
        print(f"Handoff avg match       : " f"{self.handoff_average_threshold:.2f}")
        print(f"Handoff max gap         : " f"{self.handoff_max_gap:.1f}s")
        print(f"Gallery update threshold: " f"{self.gallery_update_threshold:.2f}")
        print(
            f"Revalidation threshold  : "
            f"{self.revalidate_threshold:.2f} "
            f"x {self.revalidate_frames}"
        )
        print("========================================\n")

    # =========================================================
    # NORMALIZE
    # =========================================================

    @staticmethod
    def _normalize(embedding):
        if embedding is None:
            return None
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        if embedding.size == 0 or not np.isfinite(embedding).all():
            return None
        norm = np.linalg.norm(embedding)
        if not np.isfinite(norm) or norm <= 1e-10:
            return None
        return embedding / norm

    # =========================================================
    # COSINE SIMILARITY
    # =========================================================

    @staticmethod
    def _cosine_similarity(a, b):
        if a is None or b is None:
            return 0.0
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            return 0.0
        denominator = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
        return float(np.dot(a, b) / denominator)

    # =========================================================
    # BUILD PROTOTYPE
    # =========================================================

    def _build_prototype(self, embeddings):
        if not embeddings:
            return None
        matrix = np.stack(embeddings)
        prototype = np.mean(matrix, axis=0)
        return self._normalize(prototype)

    # =========================================================
    # FILTER EMBEDDING OUTLIERS
    # =========================================================

    def _filter_embeddings(self, embeddings):
        valid = []
        for embedding in embeddings:
            normalized = self._normalize(embedding)
            if normalized is not None:
                valid.append(normalized)
        if len(valid) <= 2:
            return valid
        prototype = self._build_prototype(valid)
        scores = [self._cosine_similarity(embedding, prototype) for embedding in valid]
        median_score = float(np.median(scores))
        cutoff = max(0.55, median_score - 0.15)
        filtered = [embedding for embedding, score in zip(valid, scores) if score >= cutoff]
        if len(filtered) >= 2:
            return filtered
        return valid

    # =========================================================
    # BBOX VALIDATION
    # =========================================================

    @staticmethod
    def _clean_bbox(bbox):
        if bbox is None:
            return None
        try:
            if len(bbox) != 4:
                return None
            x1, y1, x2, y2 = [float(value) for value in bbox]
            if x2 <= x1 or y2 <= y1:
                return None
            return [x1, y1, x2, y2]
        except Exception:
            return None

    # =========================================================
    # BBOX SPATIAL METRICS
    # =========================================================

    @staticmethod
    def _bbox_metrics(bbox_a, bbox_b):
        if bbox_a is None or bbox_b is None:
            return (0.0, 999.0, 0.0)
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        # -----------------------------------------------------
        # IoU
        # -----------------------------------------------------
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        iou = intersection / union if union > 0 else 0.0

        # -----------------------------------------------------
        # Center distance normalized by average body diagonal
        # -----------------------------------------------------
        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0
        center_distance = float(np.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2))
        aw = max(1.0, ax2 - ax1)
        ah = max(1.0, ay2 - ay1)
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        diagonal_a = np.sqrt(aw**2 + ah**2)
        diagonal_b = np.sqrt(bw**2 + bh**2)
        average_diagonal = max(1.0, (diagonal_a + diagonal_b) / 2.0)
        normalized_center_distance = center_distance / average_diagonal

        # -----------------------------------------------------
        # Relative body-box size
        # -----------------------------------------------------
        if area_a <= 0 or area_b <= 0:
            area_ratio = 0.0
        else:
            area_ratio = min(area_a, area_b) / max(area_a, area_b)
        return (float(iou), float(normalized_center_distance), float(area_ratio))

    # =========================================================
    # PERSISTENCE
    # =========================================================

    def _embedding_file(self, global_id):
        return os.path.join(self.gallery_dir, f"{global_id}.npy")

    def _save_identity_embeddings(self, global_id):
        identity = self.unknown_gallery.get(global_id)
        if identity is None:
            return
        embeddings = identity.get("embeddings", [])
        if not embeddings:
            return
        np.save(self._embedding_file(global_id), np.stack(embeddings).astype(np.float32))

    def _save_metadata(self):
        metadata = {}
        for global_id, identity in self.unknown_gallery.items():
            metadata[global_id] = {
                "first_seen": identity.get("first_seen"),
                "last_seen": identity.get("last_seen"),
                "last_camera": identity.get("last_camera"),
                "cameras": sorted(list(identity.get("cameras", set()))),
                "snapshot_path": identity.get("snapshot_path"),
                "detection_count": identity.get("detection_count", 0),
            }
        temp_path = self.metadata_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        os.replace(temp_path, self.metadata_path)

    def _save_identity(self, global_id):
        try:
            self._save_identity_embeddings(global_id)
            self._save_metadata()
        except Exception as error:
            print(f"[REID SAVE ERROR] " f"{global_id}: " f"{error}")

    # =========================================================
    # LOAD SAVED GALLERY
    # =========================================================

    def _load_gallery(self):
        if not os.path.exists(self.metadata_path):
            return
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
        except Exception as error:
            print(f"[REID LOAD ERROR] " f"{error}")
            return
        highest_id = 0
        for global_id, info in metadata.items():
            embedding_path = self._embedding_file(global_id)
            if not os.path.exists(embedding_path):
                continue
            try:
                data = np.load(embedding_path)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                embeddings = self._filter_embeddings(list(data))
                if not embeddings:
                    continue
                self.unknown_gallery[global_id] = {
                    "embeddings": embeddings,
                    "prototype": self._build_prototype(embeddings),
                    "first_seen": info.get("first_seen", time.time()),
                    "last_seen": info.get("last_seen", time.time()),
                    "last_camera": info.get("last_camera"),
                    "cameras": set(info.get("cameras", [])),
                    "snapshot_path": info.get("snapshot_path"),
                    "detection_count": info.get("detection_count", len(embeddings)),
                }
                match = re.search(r"UNK-(\d+)", global_id)
                if match:
                    highest_id = max(highest_id, int(match.group(1)))
            except Exception as error:
                print(f"[REID LOAD ERROR] " f"{global_id}: " f"{error}")
        self.next_global_id = highest_id + 1

    # =========================================================
    # CLEAN EXPIRED TRACK STATES
    # =========================================================

    def _cleanup_tracks(self):
        now = time.time()
        expired = []
        for track_key, state in self.track_states.items():
            age = now - state.get("last_seen", 0)
            if age > self.track_timeout:
                expired.append(track_key)
        for track_key in expired:
            self.track_states.pop(track_key, None)

    # =========================================================
    # IDENTITY SCORE
    # =========================================================

    def _identity_score(self, query, identity):
        prototype_score = self._cosine_similarity(query, identity.get("prototype"))
        individual_scores = []
        for stored in identity.get("embeddings", []):
            if stored.shape != query.shape:
                continue
            individual_scores.append(self._cosine_similarity(query, stored))
        if individual_scores:
            individual_scores.sort(reverse=True)
            sample_score = float(np.mean(individual_scores[:3]))
        else:
            sample_score = prototype_score
        return float(0.30 * prototype_score + 0.70 * sample_score)

    # =========================================================
    # RANK GALLERY
    # =========================================================

    def _rank_gallery(self, query):
        rankings = []
        for global_id, identity in self.unknown_gallery.items():
            score = self._identity_score(query, identity)
            rankings.append((global_id, score))
        rankings.sort(key=lambda item: item[1], reverse=True)
        return rankings

    # =========================================================
    # TRACK PROTOTYPE
    # =========================================================

    def _track_embedding(self, samples):
        filtered = self._filter_embeddings(samples)
        return self._build_prototype(filtered)

    # =========================================================
    # ID THRESHOLD
    # =========================================================

    def _required_threshold(self, global_id, camera_id):
        identity = self.unknown_gallery.get(global_id)
        if identity is None:
            return self.cross_camera_threshold
        if camera_id in identity.get("cameras", set()):
            return self.same_camera_threshold
        return self.cross_camera_threshold

    # =========================================================
    # UPDATE EXISTING IDENTITY
    # =========================================================

    def _update_identity(self, global_id, embedding, camera_id):
        identity = self.unknown_gallery.get(global_id)
        if identity is None:
            return False
        identity["last_seen"] = time.time()
        identity["last_camera"] = camera_id
        identity["cameras"].add(camera_id)
        score = self._cosine_similarity(embedding, identity.get("prototype"))

        # -----------------------------------------------------
        # Weak samples update metadata only.
        #
        # They do NOT enter the saved gallery.
        # -----------------------------------------------------
        if score < self.gallery_update_threshold:
            self._save_metadata()
            return False
        identity["embeddings"].append(embedding.copy())
        identity["embeddings"] = self._filter_embeddings(identity["embeddings"])
        if len(identity["embeddings"]) > self.max_gallery_embeddings:
            identity["embeddings"] = identity["embeddings"][-self.max_gallery_embeddings :]
        identity["prototype"] = self._build_prototype(identity["embeddings"])
        identity["detection_count"] += 1
        self._save_identity(global_id)
        return True

    # =========================================================
    # CREATE NEW ID
    # =========================================================

    def _create_identity(self, embedding, camera_id, frame=None, initial_embeddings=None):
        global_id = f"UNK-" f"{self.next_global_id:03d}"
        self.next_global_id += 1
        now = time.time()
        snapshot_path = None
        if frame is not None:
            try:
                snapshot_path = os.path.join(self.snapshot_dir, f"{global_id}.jpg")
                cv2.imwrite(snapshot_path, frame)
            except Exception:
                snapshot_path = None
        seed_embeddings = []
        if initial_embeddings:
            seed_embeddings.extend(initial_embeddings)
        seed_embeddings.append(embedding.copy())
        seed_embeddings = self._filter_embeddings(seed_embeddings)
        if not seed_embeddings:
            seed_embeddings = [embedding.copy()]
        if len(seed_embeddings) > self.max_gallery_embeddings:
            seed_embeddings = seed_embeddings[-self.max_gallery_embeddings :]
        self.unknown_gallery[global_id] = {
            "embeddings": [item.copy() for item in seed_embeddings],
            "prototype": self._build_prototype(seed_embeddings),
            "first_seen": now,
            "last_seen": now,
            "last_camera": camera_id,
            "cameras": {camera_id},
            "snapshot_path": snapshot_path,
            "detection_count": len(seed_embeddings),
        }
        self._save_identity(global_id)
        print(f"[REID NEW] " f"{global_id} " f"camera={camera_id}")
        return global_id

    # =========================================================
    # RESET NORMAL CANDIDATE
    # =========================================================

    @staticmethod
    def _reset_candidate(state):
        state["candidate_id"] = None
        state["candidate_count"] = 0

    # =========================================================
    # RESET HANDOFF CANDIDATE
    # =========================================================

    @staticmethod
    def _reset_handoff(state):
        state["handoff_id"] = None
        state["handoff_source"] = None
        state["handoff_scores"] = []

    # =========================================================
    # FIND SPATIAL HANDOFF
    # =========================================================

    def _find_spatial_handoff(self, query, camera_id, track_key, bbox, now):
        if bbox is None:
            return None
        candidates = []
        for other_key, other_state in self.track_states.items():
            if other_key == track_key:
                continue
            global_id = other_state.get("global_id")
            if global_id is None:
                continue
            if other_state.get("camera_id") != camera_id:
                continue
            age = now - other_state.get("last_seen", 0)
            if age < 0 or age > self.handoff_max_gap:
                continue
            previous_bbox = other_state.get("last_bbox")
            if previous_bbox is None:
                continue
            (overlap, center_distance, area_ratio) = self._bbox_metrics(bbox, previous_bbox)

            # -------------------------------------------------
            # Spatial continuity
            # -------------------------------------------------
            spatial_ok = (overlap >= self.handoff_iou_threshold) or (
                center_distance <= self.handoff_center_threshold
                and area_ratio >= self.handoff_area_ratio
            )
            if not spatial_ok:
                continue
            identity = self.unknown_gallery.get(global_id)
            if identity is None:
                continue
            embedding_score = self._identity_score(query, identity)

            # Higher IoU and smaller center distance means
            # stronger spatial evidence.
            spatial_score = max(overlap, max(0.0, 1.0 - center_distance))
            combined_score = 0.70 * embedding_score + 0.30 * spatial_score
            candidates.append(
                {
                    "global_id": global_id,
                    "source_track": other_key,
                    "embedding_score": embedding_score,
                    "iou": overlap,
                    "center_distance": center_distance,
                    "area_ratio": area_ratio,
                    "age": age,
                    "combined": combined_score,
                }
            )
        if not candidates:
            return None
        candidates.sort(key=lambda item: item["combined"], reverse=True)
        return candidates[0]

    # =========================================================
    # RECENT IDENTITY RECOVERY GUARD
    # =========================================================

    def _find_recent_identity_guard(self, query, camera_id, track_key, bbox, now):
        """
        Safety layer for broken local tracker IDs.

        This does NOT immediately assign an identity.

        It only prevents a new UNK ID from being created when
        a recently-seen identity from the same camera is still
        reasonably plausible.

        This gives the new local track time to collect a better
        track-level embedding.
        """
        if bbox is None:
            return None
        recovery_max_gap = 5.0

        # Deliberately lower than the real match threshold.
        # This is only a HOLD / WAIT threshold, not an ID match.
        recovery_min_score = 0.55
        candidates = []
        for other_key, other_state in self.track_states.items():
            if other_key == track_key:
                continue
            global_id = other_state.get("global_id")
            if global_id is None:
                continue
            if other_state.get("camera_id") != camera_id:
                continue
            age = now - other_state.get("last_seen", 0)
            if age < 0 or age > recovery_max_gap:
                continue
            previous_bbox = other_state.get("last_bbox")
            if previous_bbox is None:
                continue
            (overlap, center_distance, area_ratio) = self._bbox_metrics(bbox, previous_bbox)

            # Softer geometry than the strict handoff.
            #
            # We are NOT declaring that these are the same
            # person. We are only deciding whether it is too
            # early to create a brand-new identity.
            soft_spatial_ok = overlap >= 0.05 or (center_distance <= 0.65 and area_ratio >= 0.25)
            if not soft_spatial_ok:
                continue
            identity = self.unknown_gallery.get(global_id)
            if identity is None:
                continue
            embedding_score = self._identity_score(query, identity)
            if embedding_score < recovery_min_score:
                continue

            # More recent observations are preferred.
            time_score = max(0.0, 1.0 - age / recovery_max_gap)
            spatial_score = max(overlap, max(0.0, 1.0 - center_distance))
            combined_score = 0.65 * embedding_score + 0.20 * spatial_score + 0.15 * time_score
            candidates.append(
                {
                    "global_id": global_id,
                    "source_track": other_key,
                    "embedding_score": embedding_score,
                    "iou": overlap,
                    "center_distance": center_distance,
                    "area_ratio": area_ratio,
                    "age": age,
                    "combined": combined_score,
                }
            )
        if not candidates:
            return None
        candidates.sort(key=lambda item: item["combined"], reverse=True)
        return candidates[0]

    # =========================================================
    # PROCESS HANDOFF
    # =========================================================

    def _process_handoff(self, state, track_key, handoff, track_embedding, camera_id):
        global_id = handoff["global_id"]
        source_track = handoff["source_track"]
        score = handoff["embedding_score"]
        strong_spatial_overlap = handoff["iou"] >= 0.45

        # -----------------------------------------------------
        # Guard against duplicate UNK creation even if OSNet is
        # temporarily noisy.
        # -----------------------------------------------------
        guarded = score >= self.handoff_guard_threshold or strong_spatial_overlap
        if not guarded:
            self._reset_handoff(state)
            return (False, None, score)

        # -----------------------------------------------------
        # Accumulate several observations.
        # -----------------------------------------------------
        if state["handoff_id"] != global_id:
            state["handoff_id"] = global_id
            state["handoff_source"] = source_track
            state["handoff_scores"] = [score]
        else:
            state["handoff_scores"].append(score)
        state["handoff_scores"] = state["handoff_scores"][-self.handoff_score_window :]
        scores = state["handoff_scores"]
        average_score = float(np.mean(scores))
        peak_score = float(np.max(scores))

        # -----------------------------------------------------
        # While spatial continuity exists:
        #
        # NEVER count this track toward a new UNK identity.
        # -----------------------------------------------------
        state["new_count"] = 0
        self._reset_candidate(state)

        # -----------------------------------------------------
        # Confirm the handoff once enough evidence accumulates.
        # -----------------------------------------------------
        if (
            len(scores) >= self.handoff_confirm_frames
            and average_score >= self.handoff_average_threshold
            and peak_score >= self.handoff_peak_threshold
        ):
            state["global_id"] = global_id
            state["weak_count"] = 0
            self._update_identity(global_id, track_embedding, camera_id)
            print(
                f"[REID HANDOFF] "
                f"{source_track} "
                f"-> {track_key} "
                f"-> {global_id} "
                f"score={score:.3f} "
                f"avg={average_score:.3f} "
                f"iou={handoff['iou']:.3f} "
                f"center="
                f"{handoff['center_distance']:.3f}"
            )
            self._reset_handoff(state)
            return (True, global_id, score)
        print(
            f"[REID HANDOFF WAIT] "
            f"{track_key} "
            f"recent={global_id} "
            f"score={score:.3f} "
            f"avg={average_score:.3f} "
            f"iou={handoff['iou']:.3f} "
            f"center="
            f"{handoff['center_distance']:.3f}"
        )
        return (True, None, score)

    # =========================================================
    # NORMAL MATCH CONFIRMATION
    # =========================================================

    def _process_candidate_match(
        self, state, track_key, global_id, score, margin, track_embedding, camera_id
    ):
        state["new_count"] = 0
        self._reset_handoff(state)
        if state["candidate_id"] == global_id:
            state["candidate_count"] += 1
        else:
            state["candidate_id"] = global_id
            state["candidate_count"] = 1
        if state["candidate_count"] < self.confirm_frames:
            return (None, score)
        state["global_id"] = global_id
        state["weak_count"] = 0
        self._update_identity(global_id, track_embedding, camera_id)
        if margin is None:
            print(f"[REID MATCH] " f"{track_key} " f"-> {global_id} " f"score={score:.3f}")
        else:
            print(
                f"[REID MATCH] "
                f"{track_key} "
                f"-> {global_id} "
                f"score={score:.3f} "
                f"margin={margin:.3f}"
            )
        return (global_id, score)

    # =========================================================
    # BIND LOCAL TRACK TO GLOBAL ID FROM FACE EVIDENCE
    # =========================================================

    def bind_track_identity(self, track_key, global_id, camera_id, body_embedding=None, bbox=None):
        """
        Bind a local BoT-SORT track to an existing unknown
        global identity when another strong modality, such as
        anonymous ArcFace recognition, confirms the identity.

        This method does NOT create a new global identity.
        """
        body_embedding = self._normalize(body_embedding)
        bbox = self._clean_bbox(bbox)
        with self.lock:
            self._cleanup_tracks()
            identity = self.unknown_gallery.get(global_id)

            # Face is only allowed to bind to an identity that
            # already exists in the global unknown gallery.
            if identity is None:
                print(f"[REID FACE BIND ERROR] " f"{global_id} does not exist")
                return False
            now = time.time()
            if track_key not in self.track_states:
                self.track_states[track_key] = {
                    "samples": [],
                    "candidate_id": None,
                    "candidate_count": 0,
                    "new_count": 0,
                    "global_id": None,
                    "weak_count": 0,
                    "camera_id": camera_id,
                    "first_seen": now,
                    "last_seen": now,
                    "last_sample_time": 0.0,
                    "last_bbox": bbox,
                    "handoff_id": None,
                    "handoff_source": None,
                    "handoff_scores": [],
                }
            state = self.track_states[track_key]

            # Face evidence overrides an uncertain body-only
            # candidate for this local track.
            state["global_id"] = global_id
            state["candidate_id"] = None
            state["candidate_count"] = 0
            state["new_count"] = 0
            state["weak_count"] = 0
            state["camera_id"] = camera_id
            state["last_seen"] = now
            if bbox is not None:
                state["last_bbox"] = bbox
            self._reset_handoff(state)

            # Keep the current body observation in the LOCAL
            # tracklet, but do not automatically contaminate
            # the persistent OSNet gallery with a weak body
            # embedding.
            if body_embedding is not None:
                state["samples"].append(body_embedding.copy())
                state["samples"] = state["samples"][-8:]
                state["last_sample_time"] = now

            # Update global camera/time metadata.
            identity["last_seen"] = now
            identity["last_camera"] = camera_id
            identity.setdefault("cameras", set()).add(camera_id)
            print(f"[REID FACE BIND] " f"{track_key} " f"-> {global_id} " f"camera={camera_id}")
            return True

    # =========================================================
    # FIND OR CREATE
    # =========================================================

    # =========================================================
    # CAMERA SESSION TRACK RESET
    # =========================================================
    def reset_camera_tracks(self, camera_id):
        """
        Clear temporary per-camera track state only.
        Persistent UNK identities and galleries remain intact.
        """
        camera_id = str(camera_id)
        prefix = f"{camera_id}_"
        with self.lock:
            track_states = getattr(self, "track_states", {})
            stale_keys = [
                key
                for key, state in list(track_states.items())
                if (
                    str(key).startswith(prefix)
                    or (isinstance(state, dict) and str(state.get("camera_id")) == camera_id)
                )
            ]
            for key in stale_keys:
                track_states.pop(key, None)
        print(f"[REID CAMERA RESET] " f"{camera_id} " f"tracks={len(stale_keys)}")
        return len(stale_keys)

    def find_or_create(self, embedding, camera_id, track_key, frame=None, bbox=None, snapshot=None):
        embedding = self._normalize(embedding)
        if embedding is None:
            return (None, 0.0)
        bbox = self._clean_bbox(bbox)
        with self.lock:
            self._cleanup_tracks()
            now = time.time()

            # =================================================
            # INITIALIZE TRACK
            # =================================================
            if track_key not in self.track_states:
                self.track_states[track_key] = {
                    "samples": [],
                    "candidate_id": None,
                    "candidate_count": 0,
                    "new_count": 0,
                    "global_id": None,
                    "weak_count": 0,
                    "camera_id": camera_id,
                    "first_seen": now,
                    "last_seen": now,
                    "last_sample_time": 0.0,
                    "last_bbox": bbox,
                    "handoff_id": None,
                    "handoff_source": None,
                    "handoff_scores": [],
                }
            state = self.track_states[track_key]
            state["camera_id"] = camera_id
            state["last_seen"] = now
            if bbox is not None:
                state["last_bbox"] = bbox

            # =================================================
            # SAMPLE EMBEDDING
            # =================================================
            sampled = False
            if now - state["last_sample_time"] >= self.sample_interval:
                sampled = True
                state["last_sample_time"] = now
                state["samples"].append(embedding.copy())
                if len(state["samples"]) > 8:
                    state["samples"] = state["samples"][-8:]
            track_embedding = self._track_embedding(state["samples"])
            if track_embedding is None:
                track_embedding = embedding

            # =================================================
            # ALREADY CONFIRMED GLOBAL ID
            # =================================================
            if state["global_id"] is not None:
                global_id = state["global_id"]
                identity = self.unknown_gallery.get(global_id)
                if identity is not None:
                    score = self._identity_score(track_embedding, identity)

                    # -----------------------------------------
                    # Weak confirmed track
                    # -----------------------------------------
                    if score < self.revalidate_threshold:

                        # NIGRAAN_REVALIDATION_SAMPLE_GUARD
                        # Count only a genuinely new OSNet sample.
                        if sampled:
                            state["weak_count"] += 1
                        if state["weak_count"] >= self.revalidate_frames:
                            print(
                                f"[REID RELEASE] "
                                f"{track_key} "
                                f"dropped "
                                f"{global_id} "
                                f"score={score:.3f}"
                            )
                            state["global_id"] = None
                            state["weak_count"] = 0
                            state["new_count"] = 0
                            state["samples"] = [embedding.copy()]
                            state["last_sample_time"] = now
                            track_embedding = embedding.copy()
                            sampled = True
                            self._reset_candidate(state)
                            self._reset_handoff(state)
                        else:
                            return (global_id, score)
                    else:
                        state["weak_count"] = 0
                        if sampled:
                            self._update_identity(global_id, track_embedding, camera_id)
                        return (global_id, score)

            # =================================================
            # DON'T COUNT SAME SAMPLE MULTIPLE TIMES
            # =================================================
            if not sampled:
                return (None, 0.0)

            # =================================================
            # EMPTY GALLERY
            # =================================================
            if not self.unknown_gallery:
                state["new_count"] += 1
                first_required = min(self.new_id_frames, 5)
                if state["new_count"] < first_required:
                    return (None, 0.0)
                new_id = self._create_identity(
                    embedding=track_embedding,
                    camera_id=camera_id,
                    frame=frame,
                    initial_embeddings=state["samples"],
                )
                state["global_id"] = new_id
                return (new_id, 0.0)

            # =================================================
            # SPATIAL DEEPSORT HANDOFF
            # =================================================
            spatial_handoff = self._find_spatial_handoff(
                query=track_embedding, camera_id=camera_id, track_key=track_key, bbox=bbox, now=now
            )
            if spatial_handoff is not None:
                (handoff_guarded, handoff_global_id, handoff_score) = self._process_handoff(
                    state=state,
                    track_key=track_key,
                    handoff=spatial_handoff,
                    track_embedding=track_embedding,
                    camera_id=camera_id,
                )
                if handoff_guarded:
                    if handoff_global_id is not None:
                        return (handoff_global_id, handoff_score)

                    # Important:
                    #
                    # Spatially this still looks like the recent
                    # person, but we do not have enough evidence
                    # to confirm yet.
                    #
                    # DO NOT create UNK-002.
                    return (None, handoff_score)
            else:
                self._reset_handoff(state)

            # =================================================
            # RECENT IDENTITY RECOVERY GUARD
            # =================================================
            #
            # Strict spatial handoff may fail when BoT-SORT
            # changes the box significantly near frame edges,
            # after partial occlusion, or after an ID switch.
            #
            # Do NOT immediately create UNK-002 if a recently
            # seen identity is still reasonably plausible.
            # =================================================
            recent_guard = self._find_recent_identity_guard(
                query=track_embedding, camera_id=camera_id, track_key=track_key, bbox=bbox, now=now
            )
            if recent_guard is not None:
                recovery_score = recent_guard["embedding_score"]

                # If appearance is already strong enough,
                # allow the normal gallery matching logic below
                # to make the real identity decision.
                #
                # Otherwise HOLD this local track temporarily
                # instead of counting it toward UNK-002.
                if recovery_score < self.same_camera_threshold:
                    state["new_count"] = 0
                    self._reset_candidate(state)
                    print(
                        f"[REID RECOVERY WAIT] "
                        f"{track_key} "
                        f"recent="
                        f"{recent_guard['global_id']} "
                        f"score="
                        f"{recovery_score:.3f} "
                        f"age="
                        f"{recent_guard['age']:.2f}s "
                        f"iou="
                        f"{recent_guard['iou']:.3f} "
                        f"center="
                        f"{recent_guard['center_distance']:.3f}"
                    )
                    return (None, recovery_score)

            # =================================================
            # NORMAL GALLERY RANKING
            # =================================================
            rankings = self._rank_gallery(track_embedding)
            best_id = rankings[0][0]
            best_score = rankings[0][1]

            # =================================================
            # ONLY ONE GLOBAL ID
            # =================================================
            if len(rankings) == 1:

                # ---------------------------------------------
                # STRONG MATCH
                # ---------------------------------------------
                if best_score >= self.single_match_threshold:
                    return self._process_candidate_match(
                        state=state,
                        track_key=track_key,
                        global_id=best_id,
                        score=best_score,
                        margin=None,
                        track_embedding=track_embedding,
                        camera_id=camera_id,
                    )

                # ---------------------------------------------
                # CLEARLY NEW
                # ---------------------------------------------
                if best_score <= self.single_new_threshold:
                    self._reset_candidate(state)
                    state["new_count"] += 1
                    print(
                        f"[REID NOVEL] "
                        f"{track_key} "
                        f"vs {best_id} "
                        f"score="
                        f"{best_score:.3f} "
                        f"{state['new_count']}"
                        f"/"
                        f"{self.new_id_frames}"
                    )
                    if state["new_count"] < self.new_id_frames:
                        return (None, best_score)
                    new_id = self._create_identity(
                        embedding=track_embedding,
                        camera_id=camera_id,
                        frame=frame,
                        initial_embeddings=state["samples"],
                    )
                    state["global_id"] = new_id
                    return (new_id, best_score)

                # ---------------------------------------------
                # UNCERTAIN
                # ---------------------------------------------
                self._reset_candidate(state)
                state["new_count"] = 0
                print(
                    f"[REID AMBIGUOUS] " f"{track_key} " f"only=" f"{best_id}:" f"{best_score:.3f}"
                )
                return (None, best_score)

            # =================================================
            # MULTIPLE GLOBAL IDS
            # =================================================
            second_id = rankings[1][0]
            second_score = rankings[1][1]
            margin = best_score - second_score
            threshold = self._required_threshold(best_id, camera_id)

            # =================================================
            # CLEAR EXISTING MATCH
            # =================================================
            if best_score >= threshold and margin >= self.min_margin:
                return self._process_candidate_match(
                    state=state,
                    track_key=track_key,
                    global_id=best_id,
                    score=best_score,
                    margin=margin,
                    track_embedding=track_embedding,
                    camera_id=camera_id,
                )

            # =================================================
            # POSSIBLE EXISTING IDENTITY
            # =================================================
            if best_score >= self.new_person_threshold:
                self._reset_candidate(state)
                state["new_count"] = 0
                print(
                    f"[REID AMBIGUOUS] "
                    f"{track_key} "
                    f"best="
                    f"{best_id}:"
                    f"{best_score:.3f} "
                    f"second="
                    f"{second_id}:"
                    f"{second_score:.3f} "
                    f"margin="
                    f"{margin:.3f}"
                )
                return (None, best_score)

            # =================================================
            # CLEARLY NEW
            # =================================================
            self._reset_candidate(state)
            state["new_count"] += 1
            print(
                f"[REID NOVEL] "
                f"{track_key} "
                f"best="
                f"{best_id}:"
                f"{best_score:.3f} "
                f"{state['new_count']}"
                f"/"
                f"{self.new_id_frames}"
            )
            if state["new_count"] < self.new_id_frames:
                return (None, best_score)
            new_id = self._create_identity(
                embedding=track_embedding,
                camera_id=camera_id,
                frame=frame,
                initial_embeddings=state["samples"],
            )
            state["global_id"] = new_id
            return (new_id, best_score)

    # =========================================================
    # CAMERAS FOR GLOBAL ID
    # =========================================================

    def get_cameras_for(self, global_id):
        with self.lock:
            identity = self.unknown_gallery.get(global_id)
            if identity is None:
                return []
            return list(identity.get("cameras", set()))

    # =========================================================
    # SNAPSHOT PATH
    # =========================================================

    def get_snapshot_path(self, global_id):
        with self.lock:
            identity = self.unknown_gallery.get(global_id)
            if identity is None:
                return None
            return identity.get("snapshot_path")

    # =========================================================
    # DEBUG / STATUS
    # =========================================================

    def get_all(self):
        with self.lock:
            result = {}
            for global_id, identity in self.unknown_gallery.items():
                result[global_id] = {
                    "first_seen": identity.get("first_seen"),
                    "last_seen": identity.get("last_seen"),
                    "last_camera": identity.get("last_camera"),
                    "cameras": list(identity.get("cameras", set())),
                    "detection_count": identity.get("detection_count", 0),
                    "embedding_count": len(identity.get("embeddings", [])),
                    "snapshot_path": identity.get("snapshot_path"),
                }
            return result
