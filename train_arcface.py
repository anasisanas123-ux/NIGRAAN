import sys
import os
sys.path.insert(0, '.')

import numpy as np
import cv2
from insightface.app import FaceAnalysis
import config

def build_augmented_database():
    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)

    student_folders = [
        f for f in os.listdir(config.RAW_IMAGES_DIR)
        if os.path.isdir(os.path.join(config.RAW_IMAGES_DIR, f))
    ]

    print('Found ' + str(len(student_folders)) + ' student folders')

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
            print('  Skipping ' + folder_name + ' no images')
            continue

        print('  Processing ' + name + ' (' + student_id + ') ' + str(len(image_files)) + ' images')

        embeddings = []

        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                augmented = [
                    img,
                    cv2.convertScaleAbs(img, alpha=1.3, beta=20),
                    cv2.convertScaleAbs(img, alpha=0.7, beta=-20),
                    cv2.flip(img, 1),
                    cv2.GaussianBlur(img, (3,3), 0),
                ]

                for aug in augmented:
                    try:
                        faces = app.get(aug)
                        if not faces:
                            continue
                        if faces[0].det_score < 0.6:
                            continue
                        emb  = faces[0].embedding
                        norm = np.linalg.norm(emb)
                        embeddings.append(emb / norm if norm > 0 else emb)
                    except:
                        continue

            except Exception as e:
                continue

        if len(embeddings) < 3:
            print('  Not enough embeddings for ' + name)
            continue

        arr  = np.array(embeddings)
        sims = []
        for i in range(len(arr)):
            s = np.mean([float(np.dot(arr[i], arr[j]))
                         for j in range(len(arr)) if i != j])
            sims.append(s)
        mean_sim = np.mean(sims)
        kept     = np.array([arr[i] for i, s in enumerate(sims)
                              if s >= mean_sim * 0.85])

        avg  = np.mean(kept, axis=0)
        norm = np.linalg.norm(avg)
        avg  = avg / norm

        emb_path = os.path.join(config.EMBEDDINGS_DIR, student_id + '.npy')
        np.save(emb_path, avg)
        print('  Saved ' + emb_path + ' ' + str(len(kept)) + '/' + str(len(embeddings)) + ' embeddings kept')

    print()
    print('Done. Augmented database built.')

if __name__ == '__main__':
    build_augmented_database()
