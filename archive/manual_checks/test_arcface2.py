import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import config

app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider','CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

known_embeddings = {}
known_names = {}
for f in os.listdir(config.EMBEDDINGS_DIR):
    if f.endswith('.npy'):
        sid = f.replace('.npy','')
        emb = np.load(os.path.join(config.EMBEDDINGS_DIR, f))
        norm = np.linalg.norm(emb)
        known_embeddings[sid] = emb / norm if norm > 0 else emb

names_file = os.path.join(config.EMBEDDINGS_DIR, 'names.txt')
if os.path.exists(names_file):
    with open(names_file, 'r') as f:
        for line in f:
            if ',' in line:
                sid, name = line.strip().split(',', 1)
                known_names[sid.strip()] = name.strip()

source = config.CAMERA_SOURCES.get('cam_b', 0)
cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print('Connecting to cam_b...')

for _ in range(15):
    cap.read()

ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print('No frame captured')
    sys.exit(1)

print(f'Frame size: {frame.shape}')
cv2.imwrite('test_frame.jpg', frame)
print('Saved test_frame.jpg - check if you are visible in it')

faces = app.get(frame)
print(f'Faces detected: {len(faces)}')

if len(faces) == 0:
    print('No faces found - try standing closer or check test_frame.jpg')
    sys.exit(0)

for face in faces:
    emb = face.embedding
    norm = np.linalg.norm(emb)
    emb = emb / norm if norm > 0 else emb

    scores = {}
    for sid, stored in known_embeddings.items():
        scores[sid] = float(np.dot(emb, stored))

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_sid, best_score = sorted_scores[0]
    best_name = known_names.get(best_sid, best_sid)

    print(f'Best: {best_name} score={round(best_score,4)} det={round(face.det_score,2)}')
    for sid, score in sorted_scores:
        name = known_names.get(sid, sid)
        print(f'  {name}: {round(score,4)}')
