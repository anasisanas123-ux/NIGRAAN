import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_sort_realtime.deepsort_tracker import DeepSort
import config

class Tracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=config.MAX_AGE,
            n_init=config.N_INIT,
            max_iou_distance=config.MAX_IOU_DIST
        )
        print("Tracker ready")

    def update(self, detections, frame_rgb):
        if not detections:
            self.tracker.update_tracks([], frame=frame_rgb)
            return []

        ds_input = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            ds_input.append(([x1, y1, w, h], det['confidence'], 'face'))

        tracks  = self.tracker.update_tracks(ds_input, frame=frame_rgb)
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            results.append({
                'bbox':       [x1, y1, x2, y2],
                'tracker_id': str(track.track_id)
            })
        return results