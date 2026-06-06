import sys, os
sys.path.insert(0, '.')
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn  = MTCNN(keep_all=False, device=device, min_face_size=20, post_process=True)
model  = InceptionResnetV1(pretrained='vggface2').eval().to(device)

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

folder = 'data/raw_images/S001_Umair_Iqbal'
photos = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]

print('Testing ' + str(len(photos)) + ' Umair photos:')
for p in photos[:15]:
    try:
        img  = Image.open(os.path.join(folder,p)).convert('RGB')
        face = mtcnn(img)
        if face is None:
            print('  ' + p[:30] + ' --- NO FACE DETECTED')
            continue
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(face).cpu().numpy().flatten()
        emb    = emb / np.linalg.norm(emb)
        scores = {sid: float(np.dot(emb, e)) for sid, e in stored.items()}
        best   = max(scores, key=scores.get)
        s001   = scores.get('S001', 0)
        print('  ' + p[:30] + ' best=' + names.get(best,best) + ' score=' + str(round(scores[best],4)) + ' S001=' + str(round(s001,4)))
    except Exception as ex:
        print('  ' + p + ' ERROR: ' + str(ex))
