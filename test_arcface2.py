import sys, os
sys.path.insert(0, '.')
import numpy as np
import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640,640))

stored = {}
for f in os.listdir('data/embeddings'):
    if f.endswith('.npy'):
        sid = f.replace('.npy','')
        e   = np.load('data/embeddings/' + f)
        stored[sid] = e / np.linalg.norm(e)

names = {}
with open('data/embeddings/names.txt','r') as f:
    for line in f:
        if ',' in line:
            sid, name = line.strip().split(',',1)
            names[sid.strip()] = name.strip()

print('Connecting to cam_b...')
cap = cv2.VideoCapture('RTSP_URL_REDACTED', cv2.CAP_FFMPEG)
for i in range(50): cap.read()

found = False
for attempt in range(10):
    ret, frame = cap.read()
    if not ret:
        continue
    faces = app.get(frame)
    print('Attempt ' + str(attempt+1) + ' — faces: ' + str(len(faces)))
    if len(faces) > 0:
        found = True
        for face in faces:
            emb    = face.embedding
            emb    = emb / np.linalg.norm(emb)
            scores = {sid: float(np.dot(emb, e)) for sid, e in stored.items()}
            best   = max(scores, key=scores.get)
            print('Best: ' + names.get(best,best) + ' score=' + str(round(scores[best],4)))
            for sid, score in sorted(scores.items(), key=lambda x: -x[1]):
                print('  ' + names.get(sid,sid) + ': ' + str(round(score,4)))
        break

cap.release()
if not found:
    print('No face detected in any frame — stand closer to camera')
