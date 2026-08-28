from pathlib import Path

path = Path("python_engine/reid_manager.py")
text = path.read_text(encoding="utf-8")

marker = """    # =========================================================
    # FIND OR CREATE
    # =========================================================
"""

method = r'''    # =========================================================
    # BIND LOCAL TRACK TO GLOBAL ID FROM FACE EVIDENCE
    # =========================================================

    def bind_track_identity(
        self,
        track_key,
        global_id,
        camera_id,
        body_embedding=None,
        bbox=None
    ):

        """
        Bind a local BoT-SORT track to an existing unknown
        global identity when another strong modality, such as
        anonymous ArcFace recognition, confirms the identity.

        This method does NOT create a new global identity.
        """

        body_embedding = self._normalize(
            body_embedding
        )

        bbox = self._clean_bbox(
            bbox
        )

        with self.lock:

            self._cleanup_tracks()

            identity = self.unknown_gallery.get(
                global_id
            )

            # Face is only allowed to bind to an identity that
            # already exists in the global unknown gallery.
            if identity is None:

                print(
                    f"[REID FACE BIND ERROR] "
                    f"{global_id} does not exist"
                )

                return False

            now = time.time()

            if track_key not in self.track_states:

                self.track_states[
                    track_key
                ] = {

                    "samples": [],

                    "candidate_id":
                        None,

                    "candidate_count":
                        0,

                    "new_count":
                        0,

                    "global_id":
                        None,

                    "weak_count":
                        0,

                    "camera_id":
                        camera_id,

                    "first_seen":
                        now,

                    "last_seen":
                        now,

                    "last_sample_time":
                        0.0,

                    "last_bbox":
                        bbox,

                    "handoff_id":
                        None,

                    "handoff_source":
                        None,

                    "handoff_scores":
                        []
                }

            state = self.track_states[
                track_key
            ]

            # Face evidence overrides an uncertain body-only
            # candidate for this local track.
            state[
                "global_id"
            ] = global_id

            state[
                "candidate_id"
            ] = None

            state[
                "candidate_count"
            ] = 0

            state[
                "new_count"
            ] = 0

            state[
                "weak_count"
            ] = 0

            state[
                "camera_id"
            ] = camera_id

            state[
                "last_seen"
            ] = now

            if bbox is not None:

                state[
                    "last_bbox"
                ] = bbox

            self._reset_handoff(
                state
            )

            # Keep the current body observation in the LOCAL
            # tracklet, but do not automatically contaminate
            # the persistent OSNet gallery with a weak body
            # embedding.
            if body_embedding is not None:

                state[
                    "samples"
                ].append(
                    body_embedding.copy()
                )

                state[
                    "samples"
                ] = state[
                    "samples"
                ][-8:]

                state[
                    "last_sample_time"
                ] = now

            # Update global camera/time metadata.
            identity[
                "last_seen"
            ] = now

            identity[
                "last_camera"
            ] = camera_id

            identity.setdefault(
                "cameras",
                set()
            ).add(
                camera_id
            )

            print(
                f"[REID FACE BIND] "
                f"{track_key} "
                f"-> {global_id} "
                f"camera={camera_id}"
            )

            return True


'''

if "def bind_track_identity(" in text:
    print("bind_track_identity already exists.")
elif marker not in text:
    raise SystemExit("ERROR: FIND OR CREATE marker not found")
else:
    text = text.replace(
        marker,
        method + marker,
        1
    )

    path.write_text(
        text,
        encoding="utf-8"
    )

    print("FACE BIND API INSTALLED")
