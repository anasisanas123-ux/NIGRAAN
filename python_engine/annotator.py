import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import cv2


class Annotator:

    def __init__(self):

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.55
        self.thickness = 2


    # =========================================================
    # IOU FOR DISPLAY-ONLY DUPLICATE SUPPRESSION
    # =========================================================

    def _iou(
        self,
        box_a,
        box_b
    ):

        if box_a is None or box_b is None:
            return 0.0

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

        iw = max(
            0.0,
            ix2 - ix1
        )

        ih = max(
            0.0,
            iy2 - iy1
        )

        intersection = (
            iw
            *
            ih
        )

        area_a = max(
            0.0,
            ax2 - ax1
        ) * max(
            0.0,
            ay2 - ay1
        )

        area_b = max(
            0.0,
            bx2 - bx1
        ) * max(
            0.0,
            by2 - by1
        )

        union = (
            area_a
            +
            area_b
            -
            intersection
        )

        if union <= 0:
            return 0.0

        return float(
            intersection
            /
            union
        )


    # =========================================================
    # DISPLAY-ONLY DUPLICATE FILTER
    # =========================================================

    def _deduplicate(
        self,
        detections
    ):

        kept = []

        for det in detections:

            body_bbox = det.get(
                'body_bbox'
            )

            if body_bbox is None:

                body_bbox = det.get(
                    'bbox'
                )

            if body_bbox is None:

                continue


            duplicate_index = None


            for index, existing in enumerate(
                kept
            ):

                existing_bbox = existing.get(
                    'body_bbox'
                )

                if existing_bbox is None:

                    existing_bbox = existing.get(
                        'bbox'
                    )


                overlap = self._iou(
                    body_bbox,
                    existing_bbox
                )


                # Very strong overlap generally means the same
                # person has been represented twice by the
                # display pipeline.
                if overlap >= 0.80:

                    duplicate_index = index

                    break


            if duplicate_index is None:

                kept.append(
                    det
                )

                continue


            existing = kept[
                duplicate_index
            ]


            # Prefer the richer detection:
            # global UNK/student identity > generic Unknown.
            current_name = str(
                det.get(
                    'name',
                    'Unknown'
                )
            )

            existing_name = str(
                existing.get(
                    'name',
                    'Unknown'
                )
            )


            current_score = 0
            existing_score = 0


            if current_name != 'Unknown':
                current_score += 2

            if existing_name != 'Unknown':
                existing_score += 2


            if det.get(
                'face_bbox'
            ) is not None:

                current_score += 1


            if existing.get(
                'face_bbox'
            ) is not None:

                existing_score += 1


            if len(
                det.get(
                    'cameras',
                    []
                )
            ) > 1:

                current_score += 1


            if len(
                existing.get(
                    'cameras',
                    []
                )
            ) > 1:

                existing_score += 1


            if current_score > existing_score:

                kept[
                    duplicate_index
                ] = det


        return kept


    # =========================================================
    # DRAW
    # =========================================================

    def draw(
        self,
        frame,
        detections
    ):

        if not detections:

            return frame


        h, w = frame.shape[
            :2
        ]


        # DISPLAY ONLY:
        # recognition/ReID detections are untouched.
        detections = self._deduplicate(
            detections
        )


        used_label_rects = []


        def find_label_pos(
            x1,
            y1,
            x2,
            y2,
            tw,
            th
        ):

            positions = [

                (
                    x1,
                    y1 - th - 10,
                    x1 + tw + 8,
                    y1 - 2
                ),

                (
                    x1,
                    y1 + 2,
                    x1 + tw + 8,
                    y1 + th + 10
                ),

                (
                    x2 + 2,
                    y1,
                    x2 + tw + 10,
                    y1 + th + 8
                )
            ]


            for (
                lx1,
                ly1,
                lx2,
                ly2
            ) in positions:

                if lx1 < 0:

                    lx2 -= lx1
                    lx1 = 0


                if ly1 < 0:

                    ly2 -= ly1
                    ly1 = 0


                if lx2 > w:

                    lx1 -= (
                        lx2 - w
                    )

                    lx2 = w


                if ly2 > h:

                    ly1 -= (
                        ly2 - h
                    )

                    ly2 = h


                overlap = False


                for (
                    ux1,
                    uy1,
                    ux2,
                    uy2
                ) in used_label_rects:

                    if not (
                        lx2 < ux1
                        or
                        lx1 > ux2
                        or
                        ly2 < uy1
                        or
                        ly1 > uy2
                    ):

                        overlap = True

                        break


                if not overlap:

                    used_label_rects.append(
                        (
                            lx1,
                            ly1,
                            lx2,
                            ly2
                        )
                    )

                    return (
                        lx1,
                        ly1,
                        lx2,
                        ly2
                    )


            lx1 = max(
                0,
                x1
            )

            ly1 = max(
                0,
                y1 - th - 10
            )

            lx2 = min(
                w,
                x1 + tw + 8
            )

            ly2 = max(
                0,
                y1 - 2
            )


            used_label_rects.append(
                (
                    lx1,
                    ly1,
                    lx2,
                    ly2
                )
            )


            return (
                lx1,
                ly1,
                lx2,
                ly2
            )


        # =====================================================
        # ONE BOX PER PERSON
        # =====================================================

        for det in detections:

            is_known = det.get(
                'is_known',
                False
            )

            name = det.get(
                'name',
                'Unknown'
            )

            confidence = det.get(
                'confidence',
                0.0
            )

            body_bbox = det.get(
                'body_bbox'
            )

            face_bbox = det.get(
                'face_bbox'
            )

            cameras = det.get(
                'cameras',
                []
            )


            # Prefer the person's BODY as the single display box.
            # Face bbox remains available internally for ArcFace.
            display_bbox = body_bbox

            if display_bbox is None:

                display_bbox = face_bbox


            if display_bbox is None:

                continue


            x1, y1, x2, y2 = [
                int(v)
                for v in display_bbox
            ]


            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                w,
                x2
            )

            y2 = min(
                h,
                y2
            )


            known_color = (
                0,
                220,
                80
            )

            unknown_color = (
                0,
                60,
                220
            )


            color = (
                known_color
                if is_known
                else unknown_color
            )


            # Main person box
            cv2.rectangle(
                frame,
                (
                    x1,
                    y1
                ),
                (
                    x2,
                    y2
                ),
                color,
                2
            )


            # Corner accents
            cs = 14


            for (
                cx,
                cy
            ) in [

                (
                    x1,
                    y1
                ),

                (
                    x2,
                    y1
                ),

                (
                    x1,
                    y2
                ),

                (
                    x2,
                    y2
                )
            ]:

                dx = (
                    cs
                    if cx == x1
                    else -cs
                )

                dy = (
                    cs
                    if cy == y1
                    else -cs
                )


                cv2.line(
                    frame,
                    (
                        cx,
                        cy
                    ),
                    (
                        cx + dx,
                        cy
                    ),
                    color,
                    2
                )


                cv2.line(
                    frame,
                    (
                        cx,
                        cy
                    ),
                    (
                        cx,
                        cy + dy
                    ),
                    color,
                    2
                )


            # Identity label
            if is_known:

                label = (
                    f"{name} "
                    f"({int(confidence * 100)}%)"
                )

            else:

                label = str(
                    name
                )


                if len(
                    cameras
                ) > 1:

                    label += " [x-cam]"


            (
                tw,
                th
            ), _ = cv2.getTextSize(

                label,
                self.font,
                self.font_scale,
                self.thickness
            )


            (
                lx1,
                ly1,
                lx2,
                ly2
            ) = find_label_pos(

                x1,
                y1,
                x2,
                y2,
                tw,
                th
            )


            cv2.rectangle(
                frame,
                (
                    lx1,
                    ly1
                ),
                (
                    lx2,
                    ly2
                ),
                color,
                -1
            )


            cv2.putText(
                frame,
                label,
                (
                    lx1 + 4,
                    ly2 - 3
                ),
                self.font,
                self.font_scale,
                (
                    255,
                    255,
                    255
                ),
                self.thickness,
                cv2.LINE_AA
            )


        return frame
