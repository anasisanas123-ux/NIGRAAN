import sys
import os
import json
import re
import tempfile
import threading
import time
from datetime import datetime, timezone

# Fix CUDA/cuDNN DLL loading on Windows â€” must run before onnxruntime import
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetDllDirectoryW(
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin'
    )
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import config
from python_engine.face_quality import quality, assess_quality, FaceEvidence

class ArcFaceRecognizer:
    def __init__(self):
        from insightface.app import FaceAnalysis
        # TensorRT is not listed: when its libraries are missing, ONNX Runtime
        # drops every GPU provider and silently runs these models on the CPU.
        self.app = FaceAnalysis(
            name='antelopev2',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(1280, 1280))
        self.inference_lock = threading.RLock()
        self.evidence = FaceEvidence()
        self.crop_times = {}
        self.crop_offsets = {}

        self.known_embeddings = {}
        self.known_names      = {}
        self.known_roles      = {}
        self.valid_until      = {}
        self.emb_matrix       = None
        self.emb_ids          = []
        self.gallery_lock     = threading.RLock()
        self.gallery_version  = 0

        self.load_database()
        self._build_matrix()
        print(f"ArcFaceRecognizer ready (SCRFD + ResNet100) â€” "
              f"{len(self.known_embeddings)} students loaded")

    def load_database(self):
        self.known_embeddings = {}
        self.known_names      = {}
        self.known_roles      = {}
        self.valid_until      = {}
        if not os.path.exists(config.EMBEDDINGS_DIR):
            return

        role_data = {}
        roles_path = os.path.join(config.EMBEDDINGS_DIR, 'roles.json')
        try:
            with open(roles_path, 'r', encoding='utf-8') as f:
                role_data = json.load(f)
        except (OSError, ValueError):
            pass

        for filename in os.listdir(config.EMBEDDINGS_DIR):
            if not filename.endswith('.npy'):
                continue
            sid  = filename.replace('.npy', '')
            metadata = role_data.get(sid, {})
            role = metadata.get('role', 'student')
            expires = metadata.get('valid_until')
            if role == 'visitor':
                try:
                    if not metadata.get('active', True) or (expires and datetime.fromisoformat(expires.replace('Z', '+00:00')) <= datetime.now(timezone.utc)):
                        continue
                except (TypeError, ValueError):
                    continue
            data = np.load(
                os.path.join(config.EMBEDDINGS_DIR, filename),
                allow_pickle=False
            )
            # support both single embedding and array of embeddings
            if data.ndim == 1:
                self.known_embeddings[sid] = [data]
            else:
                self.known_embeddings[sid] = list(data)
            self.known_roles[sid] = role
            self.valid_until[sid] = expires

        names_file = os.path.join(config.EMBEDDINGS_DIR, 'names.txt')
        if os.path.exists(names_file):
            with open(names_file, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line = line.strip()
                    if ',' in line:
                        sid, name = line.split(',', 1)
                        self.known_names[sid.strip()] = name.strip()

    def _build_matrix(self):
        # flatten all embeddings â€” each student may have multiple
        ids    = []
        all_embeddings  = []
        for sid, embs in self.known_embeddings.items():
            for emb in embs:
                ids.append(sid)
                all_embeddings.append(emb)
        if not all_embeddings:
            self.emb_ids = []
            self.emb_matrix = None
            return
        matrix = np.array(all_embeddings).astype('float32')
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        valid = np.isfinite(matrix).all(axis=1) & (norms[:, 0] > 0)
        matrix = matrix[valid] / norms[valid]
        ids = [identity for identity, keep in zip(ids, valid) if keep]
        self.emb_ids = ids
        self.emb_matrix = matrix if len(ids) else None

    def _write_gallery_metadata(self):
        os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
        names_path = os.path.join(config.EMBEDDINGS_DIR, 'names.txt')
        roles_path = os.path.join(config.EMBEDDINGS_DIR, 'roles.json')
        names = [f"{sid},{name}" for sid, name in self.known_names.items()]
        roles = {
            sid: {
                'role': self.known_roles.get(sid, 'student'),
                'valid_until': self.valid_until.get(sid),
                'active': True,
            }
            for sid in self.known_embeddings
        }
        for target, content in ((names_path, '\n'.join(names)), (roles_path, json.dumps(roles, indent=2))):
            fd, temp = tempfile.mkstemp(prefix='.gallery-', dir=config.EMBEDDINGS_DIR, text=True)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    stream.write(content)
                os.replace(temp, target)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)

    def add_identity_from_images(self, identity_id, name, images, role='student', valid_until=None):
        """Extract and persist face embeddings, then publish them to live recognition."""
        identity_id = str(identity_id).strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identity_id):
            raise ValueError('Identity ID must contain only letters, digits, hyphens, or underscores.')
        if role not in ('student', 'visitor'):
            raise ValueError('Role must be student or visitor.')
        if not images or len(images) > 256:
            raise ValueError('Provide between 1 and 256 enrollment photos.')

        embeddings = []
        for image in images:
            if image is None or image.size == 0:
                continue
            with self.inference_lock:
                faces = self.app.get(image)
            if len(faces) != 1:
                continue
            face = max(faces, key=lambda item: float(item.det_score))
            if not self._quality_check(face) or not quality(face, image, config, enrollment=True):
                continue
            emb = np.asarray(face.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 0 and np.isfinite(norm):
                embeddings.append(emb / norm)
        if not embeddings:
            raise ValueError('No usable enrollment photos. Use one face per photo, sharp and well lit, at least 80 pixels wide, with a front or slight side view.')

        valid_until_value = None
        if valid_until:
            parsed = datetime.fromisoformat(str(valid_until).replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            valid_until_value = parsed.astimezone(timezone.utc).isoformat()
            if role == 'visitor' and parsed <= datetime.now(timezone.utc):
                raise ValueError('Visitor enrollment has already expired.')

        arr = np.stack(embeddings).astype(np.float32)
        os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
        emb_path = os.path.join(config.EMBEDDINGS_DIR, f'{identity_id}.npy')
        fd, temp = tempfile.mkstemp(prefix='.embedding-', suffix='.npy', dir=config.EMBEDDINGS_DIR)
        os.close(fd)
        try:
            np.save(temp, arr, allow_pickle=False)
            os.replace(temp, emb_path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

        with self.gallery_lock:
            self.known_embeddings[identity_id] = list(arr)
            self.known_names[identity_id] = str(name).strip()
            self.known_roles[identity_id] = role
            self.valid_until[identity_id] = valid_until_value
            self._build_matrix()
            self._write_gallery_metadata()
            self.gallery_version += 1
        return {'identity_id': identity_id, 'name': str(name).strip(), 'role': role, 'embeddings': len(arr), 'gallery_version': self.gallery_version}

    def remove_identity(self, identity_id):
        identity_id = str(identity_id)
        with self.gallery_lock:
            self.known_embeddings.pop(identity_id, None)
            self.known_names.pop(identity_id, None)
            self.known_roles.pop(identity_id, None)
            self.valid_until.pop(identity_id, None)
            path = os.path.join(config.EMBEDDINGS_DIR, f'{identity_id}.npy')
            if os.path.isfile(path):
                os.unlink(path)
            self._build_matrix()
            self._write_gallery_metadata()
            self.gallery_version += 1
        return {'identity_id': identity_id, 'gallery_version': self.gallery_version}

    def is_identity_active(self, identity_id):
        """Check cached labels against the live gallery, including visitor expiry."""
        with self.gallery_lock:
            if identity_id not in self.known_embeddings:
                return False
            if self.known_roles.get(identity_id) != 'visitor':
                return True
            expires = self.valid_until.get(identity_id)
            try:
                return bool(expires) and datetime.fromisoformat(
                    expires.replace('Z', '+00:00')
                ) > datetime.now(timezone.utc)
            except (TypeError, ValueError):
                return False

    def rebuild_from_saved_photos(self):
        """Recompute student embeddings from data/raw_images without a separate CLI."""
        root = config.RAW_IMAGES_DIR
        if not os.path.isdir(root):
            return {'updated': 0, 'errors': ['No saved enrollment photo folders were found.']}
        updated = 0
        errors = []
        for folder in sorted(os.listdir(root)):
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue
            if '__' in folder:
                identity_id, encoded_name = folder.split('__', 1)
            else:
                identity_id, _, encoded_name = folder.partition('_')
            if not identity_id or not encoded_name:
                continue
            name = encoded_name.replace('_', ' ')
            images = []
            for filename in sorted(os.listdir(folder_path)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image = cv2.imread(os.path.join(folder_path, filename))
                    if image is not None:
                        images.append(image)
            if not images:
                continue
            try:
                self.add_identity_from_images(identity_id, name, images, role='student')
                updated += 1
            except Exception as error:
                errors.append(f'{identity_id}: {error}')
        if updated == 0 and not errors:
            errors.append('No saved student photo folders were found to rebuild.')
        return {'updated': updated, 'errors': errors, 'gallery_version': self.gallery_version}

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
        with self.gallery_lock:
            if self.emb_matrix is None:
                return None, -1.0
            scores   = self.emb_matrix @ emb
            best_idx = int(np.argmax(scores))
            return self.emb_ids[best_idx], float(scores[best_idx])

    def _open_set_check(self, emb, best_id, best_score):
        if self.emb_matrix is None:
            return False

        # Compare identities, not individual gallery samples. Multiple
        # enrollment images for one student must not count as a competing
        # second identity in the open-set margin check.
        with self.gallery_lock:
            matrix = self.emb_matrix
            ids = list(self.emb_ids)
        if matrix is None:
            return False
        scores = matrix @ emb
        best_by_identity = {}
        for student_id, score in zip(ids, scores):
            score = float(score)
            if score > best_by_identity.get(student_id, -float('inf')):
                best_by_identity[student_id] = score

        ranked = sorted(
            best_by_identity.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked or ranked[0][0] != best_id:
            return False
        if ranked[0][1] < config.RECOGNITION_THRESHOLD:
            return False

        if len(ranked) >= 2:
            margin = ranked[0][1] - ranked[1][1]
            if margin < 0.05:
                return False
        return True

    def _detect_faces(self, frame, cam_id, regions):
        # Keep coordinates in the original stream. Crops avoid shrinking small
        # faces in wide CCTV images; they do not synthesize missing detail.
        faces = list(self.app.get(frame))
        now = time.monotonic()
        if now - self.crop_times.get(cam_id, -float('inf')) < config.FACE_CROP_INTERVAL:
            return faces
        self.crop_times[cam_id] = now
        height, width = frame.shape[:2]
        regions = regions or []
        if not hasattr(self, 'crop_offsets'):
            self.crop_offsets = {}
        offset = self.crop_offsets.get(cam_id, 0) % max(len(regions), 1)
        selected = (regions[offset:] + regions[:offset])[:config.FACE_CROP_MAX_REGIONS]
        self.crop_offsets[cam_id] = offset + len(selected)
        for region in selected:
            x1, y1, x2, y2 = map(int, region)
            pad = int((x2-x1)*.15)
            x1, x2 = max(0,x1-pad), min(width,x2+pad)
            y1, y2 = max(0,y1-pad), min(height,y2)
            if x2-x1 < 30 or y2-y1 < 30 or (x2-x1)*(y2-y1) > width*height*.6:
                continue
            for face in self.app.get(frame[y1:y2,x1:x2]):
                face.bbox = face.bbox + np.array([x1,y1,x1,y1])
                duplicate = next((i for i, existing in enumerate(faces) if self._box_iou(face.bbox, existing.bbox) > .4), None)
                if duplicate is not None:
                    if float(face.det_score) > float(faces[duplicate].det_score):
                        faces[duplicate] = face
                    continue
                faces.append(face)
        return faces

    @staticmethod
    def _box_iou(a, b):
        area = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
        return area/union if union > 0 else 0

    def detect_and_recognize(self, frame_bgr, cam_id=None, regions=None):
        results = []
        try:
            with self.inference_lock:
                faces = self._detect_faces(frame_bgr, cam_id, regions)
            for face in faces:
                assessment = assess_quality(face, frame_bgr, config)
                if not self._quality_check(face) or not assessment['accepted']:
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
                if not np.isfinite(emb).all() or not np.isfinite(norm) or norm <= 0:
                    continue
                emb = emb / norm

                best_id, best_score = self._match(emb)
                is_known = (best_score >= assessment['match_threshold']
                            and self._open_set_check(emb, best_id, best_score))
                # Soft frames can confirm a strong existing identity, but must
                # not contaminate anonymous face galleries with weak embeddings.
                if assessment['soft'] and not is_known:
                    continue
                with self.gallery_lock:
                    role = self.known_roles.get(best_id, 'student') if is_known else 'unknown'
                    expires = self.valid_until.get(best_id) if is_known else None
                if is_known and role == 'visitor' and expires:
                    try:
                        if datetime.fromisoformat(expires.replace('Z', '+00:00')) <= datetime.now(timezone.utc):
                            is_known = False
                            role = 'unknown'
                    except ValueError:
                        is_known = False
                        role = 'unknown'
                name     = self.known_names.get(best_id, best_id) \
                           if is_known else 'Unknown'

                results.append({
                    'bbox':       [x1, y1, x2, y2],
                    'is_known':   is_known,
                    'student_id': best_id if is_known else None,
                    'name':       name,
                    'role':       role,
                    'confidence': best_score,
                    'embedding':  emb.astype(np.float32).copy(),
                    'face_crop':  frame_bgr[y1:y2, x1:x2],
                    'body_crop':  None,
                    'tracker_id': None,
                    'global_id':  None,
                    'cameras':    []
                })
            if cam_id:
                accepted = self.evidence.confirm(cam_id,
                    [(r['bbox'], r['student_id']) for r in results],
                    self.gallery_version, config.FACE_CONFIRM_OBSERVATIONS)
                for result, confirmed in zip(results, accepted):
                    result['identity_pending'] = result['is_known'] and not confirmed
                    if not confirmed:
                        result.update(is_known=False, student_id=None, name='Unknown', role='unknown')
        except Exception as e:
            print(f"ArcFace error: {e}")
        return results
