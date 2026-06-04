import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import cv2
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as T
from collections import deque, Counter
import config

class FaceRecognizer:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = InceptionResnetV1(
            pretrained='vggface2'
        ).eval().to(self.device)

        self.transform = T.Compose([
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.known_embeddings = {}
        self.known_names      = {}
        self.emb_matrix       = None
        self.emb_ids          = []

        # voting buffer per tracker_id
        # keeps last N predictions and returns majority
        self.vote_buffer = {}
        self.vote_size   = 7

        self.load_database()
        self._build_embedding_matrix()

        # GPU warmup
        if self.device.type == 'cuda':
            dummy = torch.zeros(1, 3, 160, 160).to(self.device)
            with torch.no_grad():
                self.model(dummy)
            torch.cuda.synchronize()

        print(f"FaceRecognizer ready on {self.device} "
              f"— {len(self.known_embeddings)} students loaded")

    def _build_embedding_matrix(self):
        if not self.known_embeddings:
            self.emb_matrix = None
            self.emb_ids    = []
            return
        self.emb_ids    = list(self.known_embeddings.keys())
        self.emb_matrix = np.array([
            self.known_embeddings[sid] for sid in self.emb_ids
        ])

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
                    line = line.strip()
                    if ',' in line:
                        sid, name = line.split(',', 1)
                        self.known_names[sid.strip()] = name.strip()

    def _enhance(self, img_np):
        try:
            clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            channels = cv2.split(img_np)
            enhanced = [clahe.apply(ch) for ch in channels]
            return cv2.merge(enhanced)
        except Exception:
            return img_np

    def get_embedding(self, face_crop_rgb):
        try:
            if face_crop_rgb is None or face_crop_rgb.size == 0:
                return None
            h, w = face_crop_rgb.shape[:2]
            if h < 20 or w < 20:
                return None

            img         = Image.fromarray(face_crop_rgb).resize(
                (160, 160), Image.BILINEAR
            )
            face_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.model(face_tensor).cpu().numpy().flatten()

            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb

        except Exception:
            return None

    def cosine_similarity_batch(self, query_emb):
        if self.emb_matrix is None:
            return None, -1.0
        scores   = self.emb_matrix @ query_emb
        best_idx = int(np.argmax(scores))
        return self.emb_ids[best_idx], float(scores[best_idx])

    def recognize(self, face_crop_rgb, tracker_id=None):
        """
        Recognize with majority voting for stability.
        tracker_id: use to maintain per-person vote buffer
        """
        unknown = {
            'is_known': False, 'student_id': None,
            'name': 'Unknown', 'confidence': 0.0
        }

        if not self.known_embeddings:
            return unknown

        query_emb = self.get_embedding(face_crop_rgb)
        if query_emb is None:
            return unknown

        best_id, best_score = self.cosine_similarity_batch(query_emb)

        # raw prediction
        if best_score >= config.RECOGNITION_THRESHOLD:
            raw_pred = best_id
        else:
            raw_pred = 'unknown'

        # ── majority voting ────────────────────────────
        key = tracker_id if tracker_id else 'default'
        if key not in self.vote_buffer:
            self.vote_buffer[key] = deque(maxlen=self.vote_size)

        self.vote_buffer[key].append(raw_pred)

        # get majority vote
        votes  = Counter(self.vote_buffer[key])
        winner, win_count = votes.most_common(1)[0]

        # only accept if majority (more than half) agree
        if winner == 'unknown' or win_count < (self.vote_size // 2 + 1):
            return unknown

        # get average confidence for winner
        winner_scores = []
        temp_emb      = query_emb
        for sid, emb in self.known_embeddings.items():
            if sid == winner:
                s = float(np.dot(temp_emb, emb) /
                         (np.linalg.norm(temp_emb) *
                          np.linalg.norm(emb) + 1e-10))
                winner_scores.append(s)

        avg_conf = float(np.mean(winner_scores)) if winner_scores else best_score
        name     = self.known_names.get(winner, winner)

        print(f"Recognized: {name} | Score: {avg_conf:.4f} | Votes: {win_count}/{self.vote_size}")

        return {
            'is_known':   True,
            'student_id': winner,
            'name':       name,
            'confidence': avg_conf
        }