import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import config

class Annotator:
    def __init__(self):
        self.smooth = {}

    def _smooth_bbox(self, tid, bbox, alpha=0.6):
        if tid not in self.smooth:
            self.smooth[tid] = bbox
        else:
            sx1, sy1, sx2, sy2 = self.smooth[tid]
            x1,  y1,  x2,  y2  = bbox
            self.smooth[tid] = [
                int(alpha * x1 + (1 - alpha) * sx1),
                int(alpha * y1 + (1 - alpha) * sy1),
                int(alpha * x2 + (1 - alpha) * sx2),
                int(alpha * y2 + (1 - alpha) * sy2),
            ]
        return self.smooth[tid]

    def draw(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            is_known        = det.get('is_known', False)
            name            = det.get('name', 'Unknown')
            confidence      = det.get('confidence', 0.0)
            tracker_id      = det.get('tracker_id', None)
            global_id       = det.get('global_id', None)
            cameras         = det.get('cameras', [])

            tid = tracker_id if tracker_id else f"{x1}{y1}"
            x1, y1, x2, y2 = self._smooth_bbox(tid, [x1, y1, x2, y2])

            color = config.COLOR_KNOWN if is_known else config.COLOR_UNKNOWN

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if is_known:
                label = f"{name} ({confidence:.0%})"
            else:
                if global_id:
                    cam_str = '+'.join(sorted(cameras)) \
                              if len(cameras) > 1 else \
                              (cameras[0] if cameras else '')
                    label = f"{global_id} [{cam_str}]"
                else:
                    label = f"Unknown #{tracker_id}" if tracker_id \
                            else "Unknown"

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, config.FONT_THICKNESS
            )
            cv2.rectangle(frame,
                          (x1, y1 - th - 10),
                          (x1 + tw + 6, y1),
                          color, -1)
            cv2.putText(frame, label,
                        (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        config.FONT_SCALE,
                        (255, 255, 255),
                        config.FONT_THICKNESS)

        cv2.putText(frame, f"Detections: {len(detections)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        return frame