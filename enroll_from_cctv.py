import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import time
import numpy as np
from facenet_pytorch import MTCNN
import torch
import config

device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(
    keep_all=False,
    device=device,
    min_face_size=60,
    thresholds=[0.6, 0.7, 0.85],
    post_process=False
)

def enroll_from_camera(student_id, student_name, camera_id, num_photos=50):
    save_dir = os.path.join(
        config.RAW_IMAGES_DIR,
        f"{student_id}_{student_name.replace(' ', '_')}"
    )
    os.makedirs(save_dir, exist_ok=True)

    existing = len([
        f for f in os.listdir(save_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    source = config.CAMERA_SOURCES[camera_id]
    print(f"\nConnecting to {camera_id}: {source}")

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Cannot connect to {camera_id}")
        return 0

    print(f"Camera connected.")
    print(f"\nEnrolling: {student_name} ({student_id})")
    print(f"Existing photos: {existing}")
    print(f"Target: {num_photos} new photos")
    print(f"\nInstructions:")
    print(f"  Stand directly under the camera")
    print(f"  Look UP at the camera")
    print(f"  Slowly turn head left and right")
    print(f"  Move closer and further")
    print(f"  Auto-captures when face detected")
    print(f"  Press Q to finish\n")

    count         = 0
    last_saved    = time.time()
    save_interval = 0.4

    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display   = frame.copy()

        # detect face
        try:
            boxes, probs = detector.detect(frame_rgb)
        except Exception:
            boxes, probs = None, None

        face_detected = False
        face_size     = 0

        if boxes is not None and len(boxes) > 0:
            box  = boxes[0]
            prob = probs[0] if probs is not None else 0

            if prob and prob > 0.85:
                x1 = max(0, int(box[0]))
                y1 = max(0, int(box[1]))
                x2 = min(frame.shape[1], int(box[2]))
                y2 = min(frame.shape[0], int(box[3]))
                face_size = min(x2-x1, y2-y1)

                if face_size >= 40:
                    face_detected = True
                    cv2.rectangle(display, (x1, y1), (x2, y2),
                                  (0, 255, 0), 3)

                    if time.time() - last_saved >= save_interval:
                        total = existing + count + 1
                        path  = os.path.join(
                            save_dir,
                            f"cctv_{camera_id}_{total:04d}.jpg"
                        )
                        cv2.imwrite(path, frame)
                        count      += 1
                        last_saved  = time.time()
                        print(f"  Captured {count}/{num_photos} "
                              f"(face: {face_size}px)")
                else:
                    cv2.rectangle(display, (x1, y1), (x2, y2),
                                  (0, 255, 255), 2)

        # progress bar
        progress = int((count / num_photos) * (frame.shape[1] - 40))
        cv2.rectangle(display,
                      (20, display.shape[0]-50),
                      (display.shape[1]-20, display.shape[0]-20),
                      (50, 50, 50), -1)
        if progress > 0:
            cv2.rectangle(display,
                          (20, display.shape[0]-50),
                          (20+progress, display.shape[0]-20),
                          (0, 255, 0) if face_detected else (100,100,100),
                          -1)

        # status
        status = f"CAPTURING {count}/{num_photos}" \
                 if face_detected \
                 else "LOOK UP AT CAMERA — FACE NOT DETECTED"
        color  = (0, 255, 0) if face_detected else (0, 0, 255)

        cv2.putText(display, f"Student: {student_name}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,255), 2)
        cv2.putText(display, f"Camera: {camera_id}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,255), 1)
        cv2.putText(display, status,
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
        cv2.putText(display, f"Face size: {face_size}px  "
                              f"(need 40px+)",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (180,180,180), 1)
        cv2.putText(display, "Q = finish early",
                    (20, display.shape[0]-60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (150,150,150), 1)

        cv2.imshow(f"CCTV Enrollment — {student_name}", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDone! {count} photos captured.")
    print(f"Total photos for {student_name}: {existing + count}")
    return count


def main():
    students = [
        ("S001", "Umair Iqbal"),
        ("S002", "Anas Ahmed"),
        ("S003", "Rahim"),
        ("S004", "Abdul Basit"),
    ]

    cameras = list(config.CAMERA_SOURCES.keys())

    print("=" * 55)
    print("  Campus Surveillance — CCTV Enrollment Station")
    print("=" * 55)

    print("\nStudents:")
    for i, (sid, name) in enumerate(students, 1):
        folder = os.path.join(
            config.RAW_IMAGES_DIR,
            f"{sid}_{name.replace(' ', '_')}"
        )
        count = len([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg','.jpeg','.png'))
        ]) if os.path.exists(folder) else 0
        status = "✓ Ready" if count >= 30 else f"Need more ({count}/30)"
        print(f"  {i}. {name:20s} — {count:3d} photos — {status}")

    print("\nCameras:")
    for i, cam in enumerate(cameras, 1):
        print(f"  {i}. {cam}")

    print("\n  0. Exit")
    print("  R. Rebuild embeddings only")

    while True:
        try:
            choice = input("\nStudent number >> ").strip()

            if choice == '0':
                print("Exiting.")
                break

            if choice.upper() == 'R':
                print("\nRebuilding all embeddings...")
                from python_engine.embedding_builder import EmbeddingBuilder
                builder = EmbeddingBuilder()
                records = builder.build_database()
                print(f"\nDone. {len(records)} students rebuilt.")
                continue

            choice = int(choice)
            if not (1 <= choice <= len(students)):
                print(f"Enter 1-{len(students)}, R, or 0")
                continue

            sid, name = students[choice - 1]

            print(f"\nSelect camera for {name}:")
            for i, cam in enumerate(cameras, 1):
                print(f"  {i}. {cam}")
            cam_choice = int(input("Camera >> ")) - 1
            if not (0 <= cam_choice < len(cameras)):
                print("Invalid camera")
                continue

            cam_id = cameras[cam_choice]

            # enroll
            captured = enroll_from_camera(
                sid, name, cam_id, num_photos=50
            )

            if captured > 0:
                print("\nRebuilding embeddings automatically...")
                from python_engine.embedding_builder import EmbeddingBuilder
                builder = EmbeddingBuilder()
                builder.build_database()
                print("Embeddings updated. Student ready for detection.\n")

        except ValueError:
            print("Enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()