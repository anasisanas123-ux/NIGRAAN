import sys, os
sys.path.insert(0, '.')
import numpy as np
import torch
import cv2
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn  = MTCNN(keep_all=False, device=device, min_face_size=20, post_process=True)
model  = InceptionResnetV1(pretrained='vggface2').eval().to(device)
transform = T.Compose([
    T.Resize((160,160)), T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

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

print('Capturing from Camera B...')
cap = cv2.VideoCapture(__import__("config").CAMERA_SOURCES["cam_b"], cv2.CAP_FFMPEG)

for i in range(30):
    cap.read()

ret, frame = cap.read()
cap.release()

if not ret:
    print('Cannot read camera')
else:
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img  = Image.fromarray(rgb)
    face = mtcnn(img)
    if face is None:
        print('NO FACE DETECTED in live frame')
    else:
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(face).cpu().numpy().flatten()
        emb    = emb / np.linalg.norm(emb)
        scores = {sid: float(np.dot(emb, e)) for sid, e in stored.items()}
        for sid, score in sorted(scores.items(), key=lambda x: -x[1]):
            print(names.get(sid,sid) + ': ' + str(round(score,4)))
