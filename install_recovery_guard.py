from pathlib import Path

path = Path("python_engine/reid_manager.py")
text = path.read_text(encoding="utf-8")

marker = """    # =========================================================
    # PROCESS HANDOFF
    # =========================================================
"""

new_method = r'''    # =========================================================
    # RECENT IDENTITY RECOVERY GUARD
    # =========================================================

    def _find_recent_identity_guard(
        self,
        query,
        camera_id,
        track_key,
        bbox,
        now
    ):

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

            age = now - other_state.get(
                "last_seen",
                0
            )

            if age < 0 or age > recovery_max_gap:
                continue

            previous_bbox = other_state.get(
                "last_bbox"
            )

            if previous_bbox is None:
                continue

            (
                overlap,
                center_distance,
                area_ratio
            ) = self._bbox_metrics(
                bbox,
                previous_bbox
            )

            # Softer geometry than the strict handoff.
            #
            # We are NOT declaring that these are the same
            # person. We are only deciding whether it is too
            # early to create a brand-new identity.
            soft_spatial_ok = (
                overlap >= 0.05
                or
                (
                    center_distance <= 0.65
                    and
                    area_ratio >= 0.25
                )
            )

            if not soft_spatial_ok:
                continue

            identity = self.unknown_gallery.get(
                global_id
            )

            if identity is None:
                continue

            embedding_score = self._identity_score(
                query,
                identity
            )

            if embedding_score < recovery_min_score:
                continue

            # More recent observations are preferred.
            time_score = max(
                0.0,
                1.0 - age / recovery_max_gap
            )

            spatial_score = max(
                overlap,
                max(
                    0.0,
                    1.0 - center_distance
                )
            )

            combined_score = (
                0.65 * embedding_score
                +
                0.20 * spatial_score
                +
                0.15 * time_score
            )

            candidates.append({
                "global_id": global_id,
                "source_track": other_key,
                "embedding_score": embedding_score,
                "iou": overlap,
                "center_distance": center_distance,
                "area_ratio": area_ratio,
                "age": age,
                "combined": combined_score
            })

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item["combined"],
            reverse=True
        )

        return candidates[0]


'''

if marker not in text:
    raise SystemExit("ERROR: PROCESS HANDOFF marker not found")

if "def _find_recent_identity_guard(" in text:
    print("Recovery guard already exists.")
else:
    text = text.replace(
        marker,
        new_method + marker,
        1
    )

# ---------------------------------------------------------
# Insert recovery check between strict spatial handoff
# and normal gallery ranking.
# ---------------------------------------------------------

ranking_marker = """            # =================================================
            # NORMAL GALLERY RANKING
            # =================================================
"""

recovery_call = r'''            # =================================================
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
                query=track_embedding,
                camera_id=camera_id,
                track_key=track_key,
                bbox=bbox,
                now=now
            )

            if recent_guard is not None:

                recovery_score = recent_guard[
                    "embedding_score"
                ]

                # If appearance is already strong enough,
                # allow the normal gallery matching logic below
                # to make the real identity decision.
                #
                # Otherwise HOLD this local track temporarily
                # instead of counting it toward UNK-002.
                if recovery_score < self.same_camera_threshold:

                    state["new_count"] = 0

                    self._reset_candidate(
                        state
                    )

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

                    return (
                        None,
                        recovery_score
                    )

'''

if ranking_marker not in text:
    raise SystemExit("ERROR: gallery ranking marker not found")

if "[REID RECOVERY WAIT]" not in text:
    text = text.replace(
        ranking_marker,
        recovery_call + ranking_marker,
        1
    )

path.write_text(
    text,
    encoding="utf-8"
)

print("RECOVERY GUARD INSTALLED")
