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
    min_face_size=80,
    thresholds=[0.6, 0.7, 0.9],
    post_process=False
)

def enroll_student(student_id, student_name, num_photos=15):
    save_dir = os.path.join(
        config.RAW_IMAGES_DIR,
        f"{student_id}_{student_name.replace(' ', '_')}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # delete old photos and start fresh
    old_photos = [
        f for f in os.listdir(save_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    for f in old_photos:
        os.remove(os.path.join(save_dir, f))
    print(f"  Cleared {len(old_photos)} old photos")

    print(f"\nEnrolling: {student_name} ({student_id})")
    print(f"Target: {num_photos} photos")
    print(f"\nInstructions:")
    print(f"  Look straight at camera")
    print(f"  Slowly turn head left and right")
    print(f"  Move slightly closer and further")
    print(f"  Auto-captures every 0.3 seconds when face detected")
    print(f"  Press Q to finish\n")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Cannot open webcam")
        return 0

    count         = 0
    last_saved    = time.time()
    save_interval = 0.3

    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display   = frame.copy()

        try:
            boxes, probs = detector.detect(frame_rgb)
        except Exception:
            boxes, probs = None, None

        face_detected = False
        face_size     = 0

        if boxes is not None and len(boxes) > 0:
            box  = boxes[0]
            prob = probs[0] if probs is not None else 0

            if prob and prob > 0.90:
                x1 = max(0, int(box[0]))
                y1 = max(0, int(box[1]))
                x2 = min(frame.shape[1], int(box[2]))
                y2 = min(frame.shape[0], int(box[3]))
                face_size = min(x2-x1, y2-y1)

                if face_size >= 80:
                    face_detected = True
                    cv2.rectangle(display, (x1, y1), (x2, y2),
                                  (0, 255, 0), 3)

                    if time.time() - last_saved >= save_interval:
                        path = os.path.join(
                            save_dir, f"enroll_{count+1:04d}.jpg"
                        )
                        cv2.imwrite(path, frame)
                        count     += 1
                        last_saved = time.time()
                        print(f"  Captured {count}/{num_photos}")
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
                          (0, 255, 0) if face_detected else (100, 100, 100),
                          -1)

        status = f"CAPTURING {count}/{num_photos}" \
                 if face_detected \
                 else "MOVE CLOSER — FACE NOT DETECTED"
        color  = (0, 255, 0) if face_detected else (0, 0, 255)

        cv2.putText(display, f"Student: {student_name}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
        cv2.putText(display, status,
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
        cv2.putText(display, f"Face size: {face_size}px",
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
        cv2.putText(display, "Q = finish early",
                    (20, display.shape[0]-60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (150, 150, 150), 1)

        cv2.imshow(f"Enrollment — {student_name}", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDone! {count} photos captured.")
    return count


def main():
    students = [
        ("S001", "Umair Iqbal"),
        ("S002", "Anas Ahmed"),
        ("S003", "Rahim"),
        ("S004", "Abdul Basit"),
    ]

    print("=" * 50)
    print("  Campus Surveillance — Student Enrollment")
    print("=" * 50)

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
        status = "Ready" if count >= 15 else f"Need more ({count}/15)"
        print(f"  {i}. {name:20s} — {count:3d} photos — {status}")

    print("\n  0. Exit")
    print("  R. Rebuild embeddings only")
    print("\nEnter student number:")

    while True:
        try:
            choice = input(">> ").strip()

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
            captured  = enroll_student(sid, name, num_photos=15)

            if captured > 0:
                print("\nRebuilding embeddings automatically...")
                from python_engine.embedding_builder import EmbeddingBuilder
                builder = EmbeddingBuilder()
                builder.build_database()
                print("Done. Student ready for detection.\n")
                print("Enroll another? (number) or 0 to exit:")

        except ValueError:
            print("Enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()