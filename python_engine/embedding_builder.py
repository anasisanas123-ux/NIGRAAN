import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import config

class EmbeddingBuilder:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.detector = MTCNN(
            image_size=160,
            keep_all=False,
            device=self.device,
            min_face_size=config.MIN_FACE_SIZE,
            post_process=True
        )

        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        print(f"EmbeddingBuilder ready on {self.device}")

    def get_embedding(self, image_input):
        try:
            import cv2 as cv
            import torchvision.transforms as T

            if isinstance(image_input, np.ndarray):
                img = Image.fromarray(image_input)
            else:
                img = image_input

            # Resize
            img = img.resize((160, 160), Image.LANCZOS)

            # CLAHE enhancement
            img_np = np.array(img)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            channels = cv.split(img_np)
            eq_ch = [clahe.apply(ch) for ch in channels]
            img_eq = cv.merge(eq_ch)
            img = Image.fromarray(img_eq)

            # Face detection
            face_tensor = self.detector(img)

            if face_tensor is None:
                transform = T.Compose([
                    T.Resize((160, 160)),
                    T.ToTensor(),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                face_tensor = transform(img)

            face_tensor = face_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(face_tensor)

            emb = embedding.cpu().numpy().flatten()

            # Normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            return emb

        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def build_database(self):
        os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
        records = []

        student_folders = [
            f for f in os.listdir(config.RAW_IMAGES_DIR)
            if os.path.isdir(os.path.join(config.RAW_IMAGES_DIR, f))
        ]

        if not student_folders:
            print("No student folders found in", config.RAW_IMAGES_DIR)
            return records

        print(f"Found {len(student_folders)} student folders")

        for folder_name in student_folders:
            folder_path = os.path.join(config.RAW_IMAGES_DIR, folder_name)

            parts = folder_name.split('_', 1)
            student_id = parts[0]
            name = parts[1].replace('_', ' ') if len(parts) > 1 else folder_name

            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            if not image_files:
                print(f"Skipping {folder_name} — no images")
                continue

            print(f"Processing {name} ({student_id}) — {len(image_files)} images")

            embeddings = []

            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    emb = self.get_embedding(img)

                    if emb is not None:
                        embeddings.append(emb)
                    else:
                        print(f"No face in {img_file}")
                except Exception as e:
                    print(f"Error {img_file}: {e}")

            if not embeddings:
                print(f"No embeddings for {name}")
                continue

           # remove outlier embeddings before averaging
            embeddings_array = np.array(embeddings)
            if len(embeddings) >= 4:
                # compute pairwise similarities
                sims = []
                for i in range(len(embeddings_array)):
                    sim = np.mean([
                        float(np.dot(embeddings_array[i], embeddings_array[j]) /
                              (np.linalg.norm(embeddings_array[i]) *
                               np.linalg.norm(embeddings_array[j]) + 1e-10))
                        for j in range(len(embeddings_array)) if i != j
                    ])
                    sims.append(sim)
                # keep only embeddings with above-average similarity
                mean_sim = np.mean(sims)
                good = [embeddings_array[i] for i, s in enumerate(sims)
                        if s >= mean_sim * 0.8]
                if len(good) >= 2:
                    embeddings_array = np.array(good)
                    print(f"    Kept {len(good)}/{len(embeddings)} "
                          f"consistent embeddings")

            avg_embedding = np.mean(embeddings_array, axis=0)
            norm = np.linalg.norm(avg_embedding)
            avg_embedding = avg_embedding / norm if norm > 0 else avg_embedding
            
            emb_path = os.path.join(config.EMBEDDINGS_DIR, f"{student_id}.npy")
            np.save(emb_path, avg_embedding)

            records.append({
                'student_id': student_id,
                'name': name,
                'embedding_path': emb_path,
                'photo_path': os.path.join(folder_path, image_files[0])
            })

            print(f"Saved: {emb_path}")

        print(f"\nDone. {len(records)} students processed.")
        return records


if __name__ == "__main__":
    builder = EmbeddingBuilder()
    records = builder.build_database()

    print("\nStudents ready:")
    for r in records:
        print(f"{r['student_id']} — {r['name']}")