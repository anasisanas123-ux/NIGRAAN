import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from PIL import Image
import config

def build_arcface_database():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name='antelopev2',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)

    student_folders = [
        f for f in os.listdir(config.RAW_IMAGES_DIR)
        if os.path.isdir(os.path.join(config.RAW_IMAGES_DIR, f))
    ]

    print(f"Found {len(student_folders)} student folders")

    for folder_name in sorted(student_folders):
        folder_path = os.path.join(config.RAW_IMAGES_DIR, folder_name)
        parts       = folder_name.split('_', 1)
        student_id  = parts[0]
        name        = parts[1].replace('_', ' ') if len(parts) > 1 else folder_name

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if not image_files:
            print(f"  Skipping {folder_name} — no images")
            continue

        print(f"  Processing {name} ({student_id}) — {len(image_files)} images")

        embeddings = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            try:
                img   = cv2.imread(img_path)
                if img is None:
                    continue
                faces = app.get(img)
                if not faces:
                    continue
                emb  = faces[0].embedding
                norm = np.linalg.norm(emb)
                embeddings.append(emb / norm if norm > 0 else emb)
            except Exception as e:
                continue

        if len(embeddings) < 3:
            print(f"  Not enough embeddings for {name}")
            continue

        # remove outliers
        arr  = np.array(embeddings)
        sims = []
        for i in range(len(arr)):
            s = np.mean([float(np.dot(arr[i], arr[j]))
                         for j in range(len(arr)) if i != j])
            sims.append(s)
        mean_sim = np.mean(sims)
        kept = [arr[i] for i, s in enumerate(sims) if s >= mean_sim * 0.85]

        avg  = np.mean(kept, axis=0)
        norm = np.linalg.norm(avg)
        avg  = avg / norm

        emb_path = os.path.join(config.EMBEDDINGS_DIR, f"{student_id}.npy")
        np.save(emb_path, avg)
        print(f"  Saved {emb_path} — {len(kept)}/{len(embeddings)} embeddings kept")

    print("\nDone. ArcFace database built.")


if __name__ == "__main__":
    build_arcface_database()