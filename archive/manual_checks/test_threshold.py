import sys, os
sys.path.insert(0, os.path.abspath('.'))
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import config

# load embeddings
embeddings = {}
names = {}
for f in os.listdir(config.EMBEDDINGS_DIR):
    if f.endswith('.npy'):
        sid = f.replace('.npy','')
        emb = np.load(os.path.join(config.EMBEDDINGS_DIR, f))
        norm = np.linalg.norm(emb)
        embeddings[sid] = emb / norm if norm > 0 else emb
with open(os.path.join(config.EMBEDDINGS_DIR, 'names.txt')) as f:
    for line in f:
        if ',' in line:
            sid, name = line.strip().split(',',1)
            names[sid.strip()] = name.strip()

app = FaceAnalysis(name='antelopev2',
    providers=['CUDAExecutionProvider','CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640,640))

# test against camera B live frame
import time
source = config.CAMERA_SOURCES.get('cam_b', 0)
cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
for _ in range(10): cap.read()

print('Stand in front of Camera B or step away to test unknown')
print('Press ENTER to capture and test, Q to quit')

while True:
    cmd = input('Press ENTER to test (or q to quit): ').strip().lower()
    if cmd == 'q':
        break

    for _ in range(5): cap.read()
    ret, frame = cap.read()
    if not ret:
        print('No frame')
        continue

    faces = app.get(frame)
    if not faces:
        print('No face detected')
        continue

    for i, face in enumerate(faces):
        emb = face.embedding
        norm = np.linalg.norm(emb)
        emb = emb / norm if norm > 0 else emb

        scores = {}
        for sid, stored in embeddings.items():
            scores[sid] = float(np.dot(emb, stored))

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_sid, best_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
        margin = best_score - second_score

        best_name = names.get(best_sid, best_sid)

        print(f'\nFace {i+1}:')
        print(f'  Best match : {best_name} = {best_score:.4f}')
        print(f'  Margin     : {margin:.4f}')
        print(f'  Det score  : {face.det_score:.2f}')
        print(f'  All scores :')
        for sid, score in sorted_scores:
            bar = '#' * int(score * 40) if score > 0 else ''
            print(f'    {names.get(sid,sid):15} {score:.4f}  {bar}')

        # threshold simulation
        for thresh in [0.35, 0.38, 0.42, 0.45, 0.50, 0.55]:
            margin_ok = margin >= 0.05
            result = best_name if best_score >= thresh and margin_ok else 'UNKNOWN'
            print(f'  Threshold {thresh}: {result}')

cap.release()
