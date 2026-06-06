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

folder = 'data/raw_images/S001_Umair_Iqbal'
photos = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]

print('Extracting embeddings from ' + str(len(photos)) + ' photos...')
embeddings = []
good = []
bad  = []

for p in photos:
    try:
        img  = Image.open(os.path.join(folder,p)).convert('RGB')
        face = mtcnn(img)
        if face is None:
            bad.append(p)
            continue
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(face).cpu().numpy().flatten()
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
        good.append(p)
    except Exception as ex:
        bad.append(p)

print('Good photos (face detected): ' + str(len(good)))
print('Bad photos (no face):        ' + str(len(bad)))

if len(embeddings) < 3:
    print('Not enough good photos!')
else:
    arr      = np.array(embeddings)
    sims     = []
    for i in range(len(arr)):
        s = np.mean([float(np.dot(arr[i], arr[j])) for j in range(len(arr)) if i != j])
        sims.append(s)
    mean_sim = np.mean(sims)
    kept     = [arr[i] for i,s in enumerate(sims) if s >= mean_sim * 0.85]
    print('Kept after outlier removal: ' + str(len(kept)))
    avg  = np.mean(kept, axis=0)
    norm = np.linalg.norm(avg)
    avg  = avg / norm
    np.save('data/embeddings/S001.npy', avg)
    print('Saved new S001.npy with ' + str(len(kept)) + ' consistent embeddings')
    print('Done!')
