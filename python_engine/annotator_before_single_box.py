import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


class Annotator:
    def __init__(self):
        self.font       = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.55
        self.thickness  = 2

    def draw(self, frame, detections):
        if not detections:
            return frame

        h, w = frame.shape[:2]

        # collect all label rects to avoid overlap
        used_label_rects = []

        def find_label_pos(fx1, fy1, fx2, fy2, tw, th):
            # try above face box first
            positions = [
                (fx1, fy1 - th - 10, fx1 + tw + 8, fy1 - 2),   # above
                (fx1, fy2 + 2,       fx1 + tw + 8, fy2 + th + 10),  # below
                (fx2 + 2, fy1,       fx2 + tw + 10, fy1 + th + 8),  # right
            ]
            for (lx1, ly1, lx2, ly2) in positions:
                # keep inside frame
                if lx1 < 0:
                    lx2 -= lx1
                    lx1 = 0
                if ly1 < 0:
                    ly2 -= ly1
                    ly1 = 0
                if lx2 > w:
                    lx1 -= (lx2 - w)
                    lx2 = w
                if ly2 > h:
                    ly1 -= (ly2 - h)
                    ly2 = h

                # check overlap with existing labels
                overlap = False
                for (ux1, uy1, ux2, uy2) in used_label_rects:
                    if not (lx2 < ux1 or lx1 > ux2 or
                            ly2 < uy1 or ly1 > uy2):
                        overlap = True
                        break

                if not overlap:
                    used_label_rects.append((lx1, ly1, lx2, ly2))
                    return lx1, ly1, lx2, ly2

            # fallback — use above regardless of overlap
            lx1 = max(0, fx1)
            ly1 = max(0, fy1 - th - 10)
            lx2 = min(w, fx1 + tw + 8)
            ly2 = fy1 - 2
            used_label_rects.append((lx1, ly1, lx2, ly2))
            return lx1, ly1, lx2, ly2

        for det in detections:
            is_known   = det.get('is_known', False)
            name       = det.get('name', 'Unknown')
            confidence = det.get('confidence', 0)
            body_bbox  = det.get('body_bbox')
            face_bbox  = det.get('face_bbox')
            cameras    = det.get('cameras', [])

            known_color   = (0, 220, 80)
            unknown_color = (0, 60, 220)
            color         = known_color if is_known else unknown_color

            # body box
            if body_bbox is not None:
                bx1, by1, bx2, by2 = [int(v) for v in body_bbox]
                bx1 = max(0, bx1)
                by1 = max(0, by1)
                bx2 = min(w, bx2)
                by2 = min(h, by2)

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)

                cs = 14
                for (x, y) in [(bx1,by1),(bx2,by1),(bx1,by2),(bx2,by2)]:
                    dx = cs if x == bx1 else -cs
                    dy = cs if y == by1 else -cs
                    cv2.line(frame, (x,y), (x+dx,y), color, 2)
                    cv2.line(frame, (x,y), (x,y+dy), color, 2)

            # face box
            if face_bbox is not None:
                fx1, fy1, fx2, fy2 = [int(v) for v in face_bbox]
                fx1 = max(0, fx1)
                fy1 = max(0, fy1)
                fx2 = min(w, fx2)
                fy2 = min(h, fy2)

                face_color = (0, 255, 100) if is_known else (50, 50, 255)
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2),
                              face_color, 2)

                if is_known:
                    label = f"{name} ({int(confidence*100)}%)"
                else:
                    label = f"{name}"
                    if len(cameras) > 1:
                        label += " [x-cam]"

                (tw, th), _ = cv2.getTextSize(
                    label, self.font, self.font_scale, self.thickness
                )

                lx1, ly1, lx2, ly2 = find_label_pos(
                    fx1, fy1, fx2, fy2, tw, th
                )

                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2),
                              face_color, -1)
                cv2.putText(frame, label,
                            (lx1 + 4, ly2 - 3),
                            self.font, self.font_scale,
                            (0, 0, 0), self.thickness,
                            cv2.LINE_AA)

            elif body_bbox is not None:
                if is_known:
                    label = f"{name} ({int(confidence*100)}%)"
                else:
                    label = f"{name}"

                bx1, by1, bx2, by2 = [int(v) for v in body_bbox]
                (tw, th), _ = cv2.getTextSize(
                    label, self.font, self.font_scale, self.thickness
                )

                lx1, ly1, lx2, ly2 = find_label_pos(
                    bx1, by1, bx2, by2, tw, th
                )

                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2),
                              color, -1)
                cv2.putText(frame, label,
                            (lx1 + 4, ly2 - 3),
                            self.font, self.font_scale,
                            (255, 255, 255), self.thickness,
                            cv2.LINE_AA)

        return frame