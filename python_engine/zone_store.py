"""Validated, atomically persisted camera regions."""
import json
import math
import os
import tempfile
import threading


class ZoneStore:
    def __init__(self, settings, path=None):
        self.settings = settings
        self.path = path or os.path.join(settings.DATA_DIR, 'camera_zones.json')
        self.lock = threading.RLock()
        self.data = {}
        if os.path.isfile(self.path):
            with open(self.path, encoding='utf-8') as stream:
                saved = json.load(stream)
            for camera, value in saved.items():
                self.data[camera] = self.validate(camera, value)
            for camera, value in self.data.items():
                self._apply(camera, value)

    def validate(self, camera, value):
        if camera not in self.settings.CAMERA_SOURCES:
            raise ValueError('Unknown camera.')
        if not isinstance(value, dict):
            raise ValueError('Expected zone settings.')
        recognition = value.get('recognition')
        def box(values):
            if not isinstance(values, list) or len(values) != 4 or any(type(v) not in (int,float) for v in values):
                raise ValueError('Zone box must contain four numbers.')
            a,b,c,d = values
            if not all(math.isfinite(v) for v in values) or not (0 <= a < c <= 1 and 0 <= b < d <= 1):
                raise ValueError('Draw a nonempty zone inside the camera image.')
            return list(values)
        recognition = box(recognition) if recognition is not None else None
        restricted = value.get('restricted', [])
        if not isinstance(restricted, list) or len(restricted) > 20:
            raise ValueError('Use at most 20 restricted zones.')
        cleaned = []
        ids = set()
        for zone in restricted:
            if not isinstance(zone, dict):
                raise ValueError('Invalid restricted zone.')
            identity = str(zone.get('id', '')).strip()
            name = str(zone.get('name', '')).strip()
            dwell = zone.get('dwell_seconds', 10)
            if not identity or len(identity)>60 or identity in ids or not name or len(name)>100:
                raise ValueError('Each zone needs a unique ID and a name (100 characters maximum).')
            if type(dwell) not in (int,float) or not math.isfinite(dwell) or not 0 <= dwell <= 86400:
                raise ValueError('Dwell time must be between 0 and 86400 seconds.')
            ids.add(identity)
            cleaned.append(dict(id=identity, name=name, box=box(zone.get('box')), dwell_seconds=dwell))
        return {'recognition': recognition, 'restricted': cleaned}

    def _apply(self, camera, value):
        # Replace dictionaries, rather than mutate dictionaries being iterated.
        self.settings.DETECTION_ZONE = {**self.settings.DETECTION_ZONE, camera: value['recognition']}
        self.settings.BEHAVIOR_RESTRICTED_ZONES = {**self.settings.BEHAVIOR_RESTRICTED_ZONES, camera: value['restricted']}

    def get(self, camera):
        with self.lock:
            if camera not in self.settings.CAMERA_SOURCES:
                raise ValueError('Unknown camera.')
            return json.loads(json.dumps(self.data.get(camera, {
                'recognition': self.settings.DETECTION_ZONE.get(camera),
                'restricted': self.settings.BEHAVIOR_RESTRICTED_ZONES.get(camera, []),
            })))

    def save(self, camera, value):
        cleaned = self.validate(camera, value)
        with self.lock:
            updated = {**self.data, camera: cleaned}
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=os.path.dirname(self.path), prefix='.zones-', suffix='.json')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    json.dump(updated, stream, indent=2)
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self.data = updated
            self._apply(camera, cleaned)
            return self.get(camera)

    def annotate(self, camera, frame):
        import cv2
        value = self.get(camera)
        if not value['recognition'] and not value['restricted']:
            return frame
        output = frame.copy()
        height, width = output.shape[:2]
        boxes = [(z['box'], z['name'], (60,170,255)) for z in value['restricted']]
        if value['recognition']:
            boxes.append((value['recognition'], 'Recognition', (255,210,80)))
        for box, name, color in boxes:
            a,b,c,d = box
            x1,y1,x2,y2 = int(a*width),int(b*height),int(c*width),int(d*height)
            cv2.rectangle(output,(x1,y1),(x2,y2),color,2)
            cv2.putText(output,name[:35],(x1+3,max(15,y1+16)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
        return output
