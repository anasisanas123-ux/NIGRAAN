import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import config

class ArcFaceRecognizer:
    def __init__(self):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.known_embeddings = {}
        self.known_names      = {}
        self.emb_matrix       = None
        self.emb_ids          = []

        self.load_database()
        self._build_embedding_matrix()
        print(f"ArcFaceRecognizer ready — "
              f"{len(self.known_embeddings)} students loaded")

    def _build_embedding_matrix(self):
        if not self.known_embeddings:
            self.emb_matrix = None
            self.emb_ids    = []
            return
        self.emb_ids    = list(self.known_embeddings.keys())
        self.emb_matrix = np.array([
            self.known_embeddings[sid] for sid in self.emb_ids
        ]).astype('float32')

    def load_database(self):
        self.known_embeddings = {}
        self.known_names      = {}
        if not os.path.exists(config.EMBEDDINGS_DIR):
            return
        for filename in os.listdir(config.EMBEDDINGS_DIR):
            if not filename.endswith('.npy'):
                continue
            sid  = filename.replace('.npy', '')
            emb  = np.load(os.path.join(config.EMBEDDINGS_DIR, filename))
            norm = np.linalg.norm(emb)
            self.known_embeddings[sid] = emb / norm if norm > 0 else emb
        names_file = os.path.join(config.EMBEDDINGS_DIR, 'names.txt')
        if os.path.exists(names_file):
            with open(names_file, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    if ',' in line:
                        sid, name = line.strip().split(',', 1)
                        self.known_names[sid.strip()] = name.strip()

    def _quality_check(self, face):
        bbox = face.bbox.astype(int)
        w    = bbox[2] - bbox[0]
        h    = bbox[3] - bbox[1]
        if w < 20 or h < 20:
            return False
        if face.det_score < 0.45:
            return False
        return True

    def _in_zone(self, cx, cy, frame_bgr, cam_id):
        if not hasattr(config, 'DETECTION_ZONE'):
            return True
        z = config.DETECTION_ZONE.get(cam_id)
        if not z:
            return True
        h_f, w_f = frame_bgr.shape[:2]
        zx1 = int(z[0]*w_f)
        zy1 = int(z[1]*h_f)
        zx2 = int(z[2]*w_f)
        zy2 = int(z[3]*h_f)
        return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

    def _match(self, emb):
        if self.emb_matrix is None:
            return None, -1.0
        scores   = self.emb_matrix @ emb
        best_idx = int(np.argmax(scores))
        return self.emb_ids[best_idx], float(scores[best_idx])

    def _open_set_check(self, emb, best_id, best_score):
        # must be above threshold
        if best_score < config.RECOGNITION_THRESHOLD:
            return False

        if self.emb_matrix is None:
            return False

        # all scores sorted descending
        scores        = self.emb_matrix @ emb
        sorted_scores = sorted(scores, reverse=True)

        # best score must be significantly higher than second best
        # minimum margin of 0.10 — prevents confused identities
        if len(sorted_scores) >= 2:
            margin = sorted_scores[0] - sorted_scores[1]
            if margin < 0.10:
                print(f"Rejected — margin too small: {margin:.4f}")
                return False

        return True

    def detect_and_recognize(self, frame_bgr, cam_id=None):
        results = []
        try:
            faces = self.app.get(frame_bgr)

            for face in faces:
                if not self._quality_check(face):
                    continue

                bbox = face.bbox.astype(int).tolist()
                x1   = max(0, bbox[0])
                y1   = max(0, bbox[1])
                x2   = min(frame_bgr.shape[1], bbox[2])
                y2   = min(frame_bgr.shape[0], bbox[3])

                cx = (x1+x2)//2
                cy = (y1+y2)//2

                if cam_id and not self._in_zone(
                    cx, cy, frame_bgr, cam_id
                ):
                    continue

                emb  = face.embedding
                norm = np.linalg.norm(emb)
                emb  = emb / norm if norm > 0 else emb

                best_id, best_score = self._match(emb)
                is_known = self._open_set_check(
                    emb, best_id, best_score
                )

                name = self.known_names.get(best_id, best_id) \
                       if is_known else 'Unknown'

                if is_known:
                    print(f"Recognized: {name} | "
                          f"Score: {best_score:.4f} | "
                          f"Det: {face.det_score:.2f}")

                results.append({
                    'bbox':       [x1, y1, x2, y2],
                    'is_known':   is_known,
                    'student_id': best_id if is_known else None,
                    'name':       name,
                    'confidence': best_score,
                    'face_crop':  frame_bgr[y1:y2, x1:x2],
                    'body_crop':  None,
                    'tracker_id': None,
                    'global_id':  None,
                    'cameras':    []
                })
        except Exception as e:
            print(f"ArcFace error: {e}")
        return results

    def get_embedding(self, face_crop_rgb):
        try:
            if face_crop_rgb is None or face_crop_rgb.size == 0:
                return None
            img   = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
            img   = cv2.resize(img, (112, 112))
            faces = self.app.get(img)
            if not faces:
                return None
            emb  = faces[0].embedding
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except:
            return None

    def recognize(self, face_crop_rgb, tracker_id=None):
        return {
            'is_known': False, 'student_id': None,
            'name': 'Unknown', 'confidence': 0.0
        }