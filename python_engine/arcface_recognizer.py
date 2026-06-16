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
            name='antelopev2',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))

        self.known_embeddings = {}
        self.known_names      = {}
        self.emb_matrix       = None
        self.emb_ids          = []

        self.load_database()
        self._build_matrix()
        print(f"ArcFaceRecognizer ready (SCRFD + ResNet100) — "
              f"{len(self.known_embeddings)} students loaded")

    def _build_matrix(self):
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
            data = np.load(
                os.path.join(config.EMBEDDINGS_DIR, filename),
                allow_pickle=True
            )
            # support both single embedding and array of embeddings
            if data.ndim == 1:
                self.known_embeddings[sid] = [data]
            else:
                self.known_embeddings[sid] = list(data)

        names_file = os.path.join(config.EMBEDDINGS_DIR, 'names.txt')
        if os.path.exists(names_file):
            with open(names_file, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line = line.strip()
                    if ',' in line:
                        sid, name = line.split(',', 1)
                        self.known_names[sid.strip()] = name.strip()

    def _build_matrix(self):
        # flatten all embeddings — each student may have multiple
        self.emb_ids    = []
        all_embeddings  = []
        for sid, embs in self.known_embeddings.items():
            for emb in embs:
                self.emb_ids.append(sid)
                all_embeddings.append(emb)
        if not all_embeddings:
            self.emb_matrix = None
            return
        self.emb_matrix = np.array(all_embeddings).astype('float32')

    def _quality_check(self, face):
        bbox = face.bbox.astype(int)
        w    = bbox[2] - bbox[0]
        h    = bbox[3] - bbox[1]
        if w < 30 or h < 30:
            return False
        if face.det_score < 0.50:
            return False
        aspect = w / max(h, 1)
        if aspect < 0.5 or aspect > 2.0:
            return False
        return True

    def _in_zone(self, cx, cy, frame_bgr, cam_id):
        if not hasattr(config, 'DETECTION_ZONE'):
            return True
        z = config.DETECTION_ZONE.get(cam_id)
        if not z:
            return True
        h_f, w_f = frame_bgr.shape[:2]
        return (int(z[0]*w_f) <= cx <= int(z[2]*w_f) and
                int(z[1]*h_f) <= cy <= int(z[3]*h_f))

    def _match(self, emb):
        if self.emb_matrix is None:
            return None, -1.0
        scores   = self.emb_matrix @ emb
        best_idx = int(np.argmax(scores))
        return self.emb_ids[best_idx], float(scores[best_idx])

    def _open_set_check(self, emb, best_id, best_score):
        if best_score < config.RECOGNITION_THRESHOLD:
            return False
        if self.emb_matrix is None:
            return False
        scores        = self.emb_matrix @ emb
        sorted_scores = sorted(scores, reverse=True)
        if len(sorted_scores) >= 2:
            margin = sorted_scores[0] - sorted_scores[1]
            if margin < 0.05:
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
                cx   = (x1+x2)//2
                cy   = (y1+y2)//2

                if cam_id and not self._in_zone(cx, cy, frame_bgr, cam_id):
                    continue

                emb  = face.embedding
                norm = np.linalg.norm(emb)
                emb  = emb / norm if norm > 0 else emb

                best_id, best_score = self._match(emb)
                is_known = self._open_set_check(emb, best_id, best_score)
                name     = self.known_names.get(best_id, best_id) \
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