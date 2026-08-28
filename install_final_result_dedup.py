from pathlib import Path

path = Path("multi_camera.py")
text = path.read_text(encoding="utf-8")


# ============================================================
# 1. STRONGER BODY-ONLY NEW-IDENTITY GATE
#
# YOLO still detects/tracks at its existing low threshold.
# This ONLY makes body-only global identity decisions stricter.
# Face evidence still bypasses this gate.
# ============================================================

text = text.replace(
    "NEW_UNK_MIN_BODY_CONF = 0.35",
    "NEW_UNK_MIN_BODY_CONF = 0.50"
)


# ============================================================
# 2. FINAL PER-FRAME IDENTITY DEDUP
# ============================================================

marker = """        # ====================================================
        # SMOOTH BOUNDING BOXES
        # ====================================================
"""

if "FINAL PER-FRAME IDENTITY DEDUP" not in text:

    if marker not in text:
        raise SystemExit(
            "ERROR: smoothing insertion point not found"
        )

    block = r'''        # ====================================================
        # FINAL PER-FRAME IDENTITY DEDUP
        #
        # BoT-SORT may temporarily represent one physical
        # person with several local track IDs.
        #
        # Example:
        #     cam_b_3 -> UNK-001
        #     cam_b_6 -> UNK-001
        #
        # Keep only ONE final result for the stable identity.
        # This does not modify ArcFace, OSNet or ReIDManager.
        # ====================================================

        if results:

            def result_bbox(
                item
            ):

                bbox = item.get(
                    'body_bbox'
                )

                if bbox is None:

                    bbox = item.get(
                        'bbox'
                    )

                return bbox


            def bbox_area(
                bbox
            ):

                if (
                    bbox is None
                    or
                    len(bbox) != 4
                ):

                    return 0.0

                try:

                    x1, y1, x2, y2 = [
                        float(v)
                        for v in bbox
                    ]

                    return max(
                        0.0,
                        x2 - x1
                    ) * max(
                        0.0,
                        y2 - y1
                    )

                except Exception:

                    return 0.0


            def containment_overlap(
                box_a,
                box_b
            ):

                """
                Intersection / smaller-box area.

                Better than normal IoU for detecting a small
                duplicate box sitting inside a larger body box.
                """

                if (
                    box_a is None
                    or
                    box_b is None
                ):

                    return 0.0

                try:

                    ax1, ay1, ax2, ay2 = [
                        float(v)
                        for v in box_a
                    ]

                    bx1, by1, bx2, by2 = [
                        float(v)
                        for v in box_b
                    ]


                    ix1 = max(
                        ax1,
                        bx1
                    )

                    iy1 = max(
                        ay1,
                        by1
                    )

                    ix2 = min(
                        ax2,
                        bx2
                    )

                    iy2 = min(
                        ay2,
                        by2
                    )


                    intersection = (

                        max(
                            0.0,
                            ix2 - ix1
                        )

                        *

                        max(
                            0.0,
                            iy2 - iy1
                        )
                    )


                    area_a = bbox_area(
                        box_a
                    )

                    area_b = bbox_area(
                        box_b
                    )


                    smaller = min(
                        area_a,
                        area_b
                    )


                    if smaller <= 0.0:

                        return 0.0


                    return float(
                        intersection
                        /
                        smaller
                    )

                except Exception:

                    return 0.0


            def result_quality(
                item
            ):

                score = 0.0


                # Registered student is strongest.
                if item.get(
                    'student_id'
                ) is not None:

                    score += 100.0


                # Confirmed global unknown.
                if item.get(
                    'global_id'
                ) is not None:

                    score += 50.0


                # Prefer observation with an actual face.
                if item.get(
                    'face_bbox'
                ) is not None:

                    score += 20.0


                try:

                    score += (
                        float(
                            item.get(
                                'reid_confidence',
                                0.0
                            )
                            or
                            0.0
                        )
                        *
                        10.0
                    )

                except Exception:

                    pass


                # Prefer fuller/larger body observation when
                # everything else is equal.
                area = bbox_area(
                    result_bbox(
                        item
                    )
                )

                score += min(
                    area / 100000.0,
                    5.0
                )


                return score


            # ------------------------------------------------
            # PASS 1:
            # One result per stable student/global identity
            # ------------------------------------------------

            identity_results = []

            identity_positions = {}


            for item in results:

                student_id = item.get(
                    'student_id'
                )

                global_id = item.get(
                    'global_id'
                )


                if student_id is not None:

                    stable_key = (
                        'student',
                        str(student_id)
                    )


                elif global_id is not None:

                    stable_key = (
                        'global',
                        str(global_id)
                    )


                else:

                    stable_key = None


                if stable_key is None:

                    identity_results.append(
                        item
                    )

                    continue


                existing_index = (
                    identity_positions.get(
                        stable_key
                    )
                )


                if existing_index is None:

                    identity_positions[
                        stable_key
                    ] = len(
                        identity_results
                    )

                    identity_results.append(
                        item
                    )

                    continue


                existing = identity_results[
                    existing_index
                ]


                if (
                    result_quality(
                        item
                    )
                    >
                    result_quality(
                        existing
                    )
                ):

                    identity_results[
                        existing_index
                    ] = item


            # ------------------------------------------------
            # PASS 2:
            # Suppress an unconfirmed temporary track if its
            # body box substantially sits inside/overlaps a
            # confirmed identity's body box.
            #
            # This handles:
            #
            # UNK-001 full box
            # + temporary "Unknown" partial duplicate box
            # ------------------------------------------------

            confirmed_results = [

                item

                for item
                in identity_results

                if (
                    item.get(
                        'student_id'
                    )
                    is not None

                    or

                    item.get(
                        'global_id'
                    )
                    is not None
                )
            ]


            final_results = []


            for item in identity_results:

                is_confirmed = (

                    item.get(
                        'student_id'
                    )
                    is not None

                    or

                    item.get(
                        'global_id'
                    )
                    is not None
                )


                if is_confirmed:

                    final_results.append(
                        item
                    )

                    continue


                current_bbox = result_bbox(
                    item
                )


                overlaps_confirmed = False


                for confirmed_item in confirmed_results:

                    overlap = containment_overlap(

                        current_bbox,

                        result_bbox(
                            confirmed_item
                        )
                    )


                    if overlap >= 0.70:

                        overlaps_confirmed = True

                        break


                if overlaps_confirmed:

                    continue


                final_results.append(
                    item
                )


            if len(final_results) < len(results):

                print(
                    f"[RESULT DEDUP] "
                    f"{cam_id} "
                    f"{len(results)} -> "
                    f"{len(final_results)}"
                )


            results = final_results


'''

    text = text.replace(
        marker,
        block + marker,
        1
    )


path.write_text(
    text,
    encoding="utf-8"
)

print("FINAL RESULT DEDUP INSTALLED")
