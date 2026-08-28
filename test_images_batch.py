import sys, os
sys.path.insert(0, os.path.abspath("."))
import ctypes
ctypes.windll.kernel32.SetDllDirectoryW(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin")

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import config

app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320,320))

# load embeddings
embeddings = {}
names = {}
for f in os.listdir(config.EMBEDDINGS_DIR):
    if f.endswith(".npy"):
        sid = f.replace(".npy","")
        emb = np.load(os.path.join(config.EMBEDDINGS_DIR, f))
        norm = np.linalg.norm(emb)
        embeddings[sid] = emb / norm if norm > 0 else emb

with open(os.path.join(config.EMBEDDINGS_DIR, "names.txt")) as f:
    for line in f:
        if "," in line:
            sid, name = line.strip().split(",", 1)
            names[sid.strip()] = name.strip()

# ----- put test images in this folder -----
TEST_FOLDER = "test_images"
os.makedirs(TEST_FOLDER, exist_ok=True)

threshold = config.RECOGNITION_THRESHOLD
print(f"Using threshold: {threshold}\n")

image_files = [f for f in os.listdir(TEST_FOLDER)
               if f.lower().endswith((".jpg",".jpeg",".png"))]

if not image_files:
    print(f"No images found. Put test photos in: {os.path.abspath(TEST_FOLDER)}")
    sys.exit(0)

for img_file in image_files:
    img_path = os.path.join(TEST_FOLDER, img_file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"{img_file}: could not read image")
        continue

    faces = app.get(img)
    if not faces:
        print(f"{img_file}: NO FACE DETECTED")
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

        is_known = best_score >= threshold and margin >= 0.08
        result = best_name if is_known else "UNKNOWN"

        tag = "[KNOWN]  " if is_known else "[UNKNOWN]"
        print(f"{tag} {img_file} (face {i+1}) -> {result}  "
              f"(best={best_score:.3f} margin={margin:.3f} det={face.det_score:.2f})")
