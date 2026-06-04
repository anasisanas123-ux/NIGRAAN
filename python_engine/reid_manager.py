import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import threading
import time
import config

class ReIDManager:
    def __init__(self):
        self.lock            = threading.Lock()
        self.unknown_gallery = {}
        self.next_global_id  = 1
        self.threshold       = getattr(config, 'REID_THRESHOLD', 0.35)
        print(f"ReIDManager ready — threshold: {self.threshold}")

    def _cosine_similarity(self, a, b):
        return float(np.dot(a, b) /
                     (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def find_or_create(self, embedding, camera_id, snapshot=None):
        with self.lock:
            best_id    = None
            best_score = -1.0

            for gid, data in self.unknown_gallery.items():
                score = self._cosine_similarity(embedding, data['embedding'])
                # boost score for recently seen persons on same camera
                if camera_id in data.get('cameras', set()):
                    score = min(score * 1.15, 1.0)
                if score > best_score:
                    best_score = score
                    best_id    = gid

            now = time.time()

            if best_score >= self.threshold and best_id is not None:
                # matched — update existing record
                self.unknown_gallery[best_id]['last_seen'] = now
                self.unknown_gallery[best_id]['cameras'].add(camera_id)

                # update embedding with running average
                old_emb = self.unknown_gallery[best_id]['embedding']
                new_emb = 0.7 * old_emb + 0.3 * embedding
                norm    = np.linalg.norm(new_emb)
                self.unknown_gallery[best_id]['embedding'] = \
                    new_emb / norm if norm > 0 else new_emb

                cameras = self.unknown_gallery[best_id]['cameras']
                if len(cameras) > 1:
                    print(f"Cross-camera match: {best_id} "
                          f"seen on {sorted(cameras)} "
                          f"(score={best_score:.3f})")
                return best_id, best_score

            else:
                # new unknown person
                gid = f"UNK-{self.next_global_id:03d}"
                self.next_global_id += 1
                self.unknown_gallery[gid] = {
                    'embedding':  embedding,
                    'first_seen': now,
                    'last_seen':  now,
                    'cameras':    {camera_id},
                    'snapshot':   snapshot,
                    'detection_count': 1
                }
                print(f"New unknown registered: {gid} on {camera_id}")
                return gid, 0.0

    def get_all(self):
        with self.lock:
            return dict(self.unknown_gallery)

    def get_cameras_for(self, global_id):
        with self.lock:
            if global_id in self.unknown_gallery:
                return list(self.unknown_gallery[global_id]['cameras'])
            return []