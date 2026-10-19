"""Image quality and spatial evidence checks; no model weights are changed."""
import threading
import time
import cv2
import numpy as np


def assess_quality(face, image, settings, enrollment=False):
    def rejected(reason):
        return {'accepted':False, 'reason':reason}
    box = np.asarray(face.bbox, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        return rejected('invalid_box')
    x1, y1, x2, y2 = box.astype(int)
    size = getattr(settings, 'FACE_ENROLL_MIN_SIZE', 80) if enrollment else getattr(settings, 'FACE_MATCH_MIN_SIZE', 60)
    if min(x2-x1, y2-y1) < size or float(face.det_score) < getattr(settings, 'FACE_QUALITY_CONFIDENCE', .65):
        return rejected('size_or_detector_confidence')
    crop = image[max(0,y1):min(image.shape[0],y2), max(0,x1):min(image.shape[1],x2)]
    if not crop.size:
        return rejected('empty_crop')
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    light = float(gray.mean())
    if not getattr(settings, 'FACE_MIN_BRIGHTNESS', 25) <= light <= getattr(settings, 'FACE_MAX_BRIGHTNESS', 235):
        return rejected('lighting')
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    minimum = getattr(settings, 'FACE_ENROLL_MIN_SHARPNESS', 35) if enrollment else getattr(settings, 'FACE_MIN_SHARPNESS', 8)
    if not np.isfinite(sharpness) or sharpness < minimum:
        return rejected('severe_blur')
    pose = getattr(face, 'pose', None)
    if isinstance(pose, (list, tuple, np.ndarray)) and len(pose) >= 2:
        if not np.isfinite(pose).all() or max(abs(float(pose[0])), abs(float(pose[1]))) > getattr(settings, 'FACE_MAX_POSE', 45):
            return rejected('pose')
    soft = not enrollment and sharpness < getattr(settings, 'FACE_SOFT_SHARPNESS', 35)
    threshold = getattr(settings, 'RECOGNITION_THRESHOLD', .55)
    if soft:
        threshold = max(threshold, getattr(settings, 'FACE_SOFT_MATCH_THRESHOLD', .70))
    return {'accepted':True, 'reason':'soft_requires_strong_match' if soft else 'clear',
            'sharpness':sharpness, 'match_threshold':threshold, 'soft':soft}


def quality(face, image, settings, enrollment=False):
    return assess_quality(face, image, settings, enrollment)['accepted']


class FaceEvidence:
    """Short-lived per-camera evidence, one-to-one IoU association per frame."""
    def __init__(self):
        self.lock = threading.Lock()
        self.tracks = {}

    def confirm(self, camera, observations, version, required=3, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            tracks = [t for t in self.tracks.get(camera, []) if now-t['time'] < 2 and t['version'] == version]
            used = set()
            confirmed = []
            for box, identity in observations:
                candidates = []
                for i, t in enumerate(tracks):
                    if i in used:
                        continue
                    a = t['box']
                    intersection = max(0,min(a[2],box[2])-max(a[0],box[0])) * max(0,min(a[3],box[3])-max(a[1],box[1]))
                    union = (a[2]-a[0])*(a[3]-a[1])+(box[2]-box[0])*(box[3]-box[1])-intersection
                    if union > 0 and intersection/union >= .4:
                        candidates.append((intersection/union, i))
                if candidates:
                    _, index = max(candidates)
                    t = tracks[index]
                    t['count'] = t['count']+1 if identity and t['identity'] == identity else int(bool(identity))
                else:
                    index = len(tracks)
                    t = {'count': int(bool(identity))}
                    tracks.append(t)
                used.add(index)
                t.update(box=list(box), identity=identity, time=now, version=version)
                confirmed.append(bool(identity) and t['count'] >= required)
            self.tracks[camera] = tracks[-200:]
            return confirmed
