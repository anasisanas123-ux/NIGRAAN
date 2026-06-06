import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import time
import numpy as np
import config

# ── Pose guide ────────────────────────────────────────────
POSES = [
    (3, "Look STRAIGHT at camera",        "Normal position"),
    (2, "Turn head 20° LEFT",             "Slow turn left"),
    (2, "Turn head 20° RIGHT",            "Slow turn right"),
    (2, "SMILE naturally",                "Natural smile"),
    (2, "Look slightly UP",               "Chin up slightly"),
    (2, "Look slightly DOWN",             "Chin down slightly"),
    (2, "Look STRAIGHT — move CLOSER",    "1 meter from camera"),
]

def connect_camera(source):
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = \
        'rtsp_transport;udp|fflags;nobuffer|flags;low_delay'
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(2)
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return cap


def draw_overlay(frame, pose_name, pose_hint, count,
                 total, pose_idx, total_poses,
                 countdown, face_detected):

    h, w = frame.shape[:2]
    overlay = frame.copy()

    # top bar
    cv2.rectangle(overlay, (0,0), (w,70), (10,15,30), -1)

    # pose counter
    cv2.putText(overlay,
                f"Pose {pose_idx+1}/{total_poses}",
                (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (100,180,255), 2)

    # pose name
    cv2.putText(overlay, pose_name,
                (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255,255,255), 2)

    # pose hint
    cv2.putText(overlay, pose_hint,
                (w-280, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (150,150,150), 1)

    # face guide box
    bx1 = w//2 - 130
    by1 = h//2 - 160
    bx2 = w//2 + 130
    by2 = h//2 + 160
    color = (0,255,100) if face_detected else (0,100,255)
    cv2.rectangle(overlay, (bx1,by1), (bx2,by2), color, 2)

    # corner guides
    cs = 20
    for (x,y) in [(bx1,by1),(bx2,by1),(bx1,by2),(bx2,by2)]:
        dx = cs if x == bx1 else -cs
        dy = cs if y == by1 else -cs
        cv2.line(overlay, (x,y), (x+dx,y), color, 3)
        cv2.line(overlay, (x,y), (x,y+dy), color, 3)

    # face status
    status     = "Face detected — hold position" \
                 if face_detected else "Position face inside box"
    status_col = (0,255,100) if face_detected else (0,100,255)
    cv2.putText(overlay, status,
                (w//2 - 160, by2+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                status_col, 2)

    # countdown circle
    if countdown > 0 and face_detected:
        cv2.circle(overlay, (w-70, h-70), 45, (30,40,60), -1)
        cv2.circle(overlay, (w-70, h-70), 45, (0,255,100), 2)
        cv2.putText(overlay, str(countdown),
                    (w-82 if countdown>9 else w-76, h-57),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0,255,100), 3)

    # progress bar bottom
    bar_w = w - 40
    cv2.rectangle(overlay, (20,h-20), (20+bar_w,h-8),
                  (30,40,60), -1)
    filled = int(bar_w * (count / max(total,1)))
    if filled > 0:
        cv2.rectangle(overlay,
                      (20, h-20),
                      (20+filled, h-8),
                      (0,200,80), -1)

    # photo count
    cv2.putText(overlay,
                f"Photos: {count}/{total}",
                (20, h-28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (150,150,150), 1)

    # blend overlay
    result = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    return result


def enroll_student_cctv(student_id, student_name,
                        camera_source, cam_id):

    save_dir = os.path.join(
        config.RAW_IMAGES_DIR,
        f"{student_id}_{student_name.replace(' ','_')}"
    )
    os.makedirs(save_dir, exist_ok=True)

    existing = len([
        f for f in os.listdir(save_dir)
        if f.lower().endswith(('.jpg','.jpeg','.png'))
    ])

    print(f"\nConnecting to {cam_id}...")
    cap = connect_camera(camera_source)
    if cap is None:
        print(f"Cannot connect to {cam_id}")
        return 0

    print(f"Connected. Enrolling: {student_name} ({student_id})")
    print(f"Existing photos: {existing}")
    print("\nInstructions:")
    print("  Follow each pose shown on screen")
    print("  System auto-captures when face is detected")
    print("  Press SPACE to capture manually")
    print("  Press S to skip pose")
    print("  Press Q to finish early\n")

    # try to load face detector
    try:
        from facenet_pytorch import MTCNN
        import torch
        device   = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        detector = MTCNN(
            keep_all=False, device=device,
            min_face_size=40,
            thresholds=[0.6,0.7,0.85],
            post_process=False
        )
        use_detector = True
        print("Face detector loaded")
    except Exception:
        use_detector = False
        print("Face detector not available — manual capture only")

    total_photos  = sum(p[0] for p in POSES)
    count         = 0
    pose_idx      = 0
    pose_count    = 0
    last_capture  = 0
    capture_interval = 0.8

    while pose_idx < len(POSES):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        pose_photos = POSES[pose_idx][0]
        pose_name   = POSES[pose_idx][1]
        pose_hint   = POSES[pose_idx][2]

        # detect face
        face_detected = False
        if use_detector:
            try:
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, probs = detector.detect(rgb)
                if boxes is not None and len(boxes) > 0:
                    if probs[0] and probs[0] > 0.85:
                        face_detected = True
            except:
                pass
        else:
            face_detected = True

        # auto capture
        now = time.time()
        if face_detected and \
           pose_count < pose_photos and \
           now - last_capture >= capture_interval:
            total_num = existing + count + 1
            path      = os.path.join(
                save_dir,
                f"pose{pose_idx+1:02d}_{total_num:04d}.jpg"
            )
            cv2.imwrite(path, frame)
            count      += 1
            pose_count += 1
            last_capture = now
            print(f"  Pose {pose_idx+1} photo "
                  f"{pose_count}/{pose_photos} "
                  f"(total {count}/{total_photos})")

        # move to next pose
        if pose_count >= pose_photos:
            pose_idx  += 1
            pose_count = 0
            if pose_idx < len(POSES):
                print(f"\n  Next pose: {POSES[pose_idx][1]}")
                time.sleep(1.5)
            continue

        countdown = pose_photos - pose_count
        display   = draw_overlay(
            frame.copy(), pose_name, pose_hint,
            count, total_photos, pose_idx,
            len(POSES), countdown, face_detected
        )

        cv2.imshow(
            f"Enrollment — {student_name} — {cam_id}",
            display
        )

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # manual capture
            total_num = existing + count + 1
            path      = os.path.join(
                save_dir,
                f"pose{pose_idx+1:02d}_{total_num:04d}.jpg"
            )
            cv2.imwrite(path, frame)
            count      += 1
            pose_count += 1
            last_capture = now
            print(f"  Manual: pose {pose_idx+1} "
                  f"photo {pose_count}/{pose_photos}")

        elif key == ord('s'):
            print(f"  Skipped pose {pose_idx+1}")
            pose_idx  += 1
            pose_count = 0

        elif key == ord('q'):
            print("  Enrollment stopped early")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDone! {count} new photos captured for {student_name}")
    print(f"Total photos: {existing + count}")
    return count


def main():
    students = [
        ("S001", "Umair Iqbal"),
        ("S002", "Anas Ahmed"),
        ("S003", "Rahim"),
        ("S004", "Abdul Basit"),
        ("S005", "New Student"),
    ]

    cameras = {
        "cam_b": config.CAMERA_SOURCES.get("cam_b"),
    }

    print("=" * 55)
    print("  Campus Surveillance — CCTV Enrollment System")
    print("=" * 55)

    print("\nRegistered students:")
    for i, (sid, name) in enumerate(students, 1):
        folder = os.path.join(
            config.RAW_IMAGES_DIR,
            f"{sid}_{name.replace(' ','_')}"
        )
        count = len([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg','.jpeg','.png'))
        ]) if os.path.exists(folder) else 0
        status = "✓ Ready" if count >= 10 else f"Need more ({count}/10)"
        print(f"  {i}. {name:20s} — {count:3d} photos — {status}")

    print("\n  0. Exit")
    print("\nSelect student number:")

    while True:
        try:
            choice = input(">> ").strip()
            if choice == '0':
                break

            # allow custom student
            if choice.upper() == 'N':
                new_id   = input("Enter student ID (e.g. S005): ").strip()
                new_name = input("Enter student name: ").strip()
                sid, name = new_id, new_name
            else:
                idx = int(choice) - 1
                if not (0 <= idx < len(students)):
                    print(f"Enter 1-{len(students)}, N for new, or 0")
                    continue
                sid, name = students[idx]

            print(f"\nStarting enrollment for {name}...")
            print(f"Camera: cam_b")
            print("Stand 1.5-2 meters from camera")
            print("Press ENTER when ready...")
            input()

            captured = enroll_student_cctv(
                sid, name,
                cameras["cam_b"],
                "cam_b"
            )

            if captured > 0:
                again = input(
                    "\nEnroll another student? (y/n): "
                ).strip().lower()
                if again != 'y':
                    break
                print("\nSelect student number:")

        except ValueError:
            print("Enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

    # rebuild embeddings after enrollment
    rebuild = input(
        "\nRebuild embeddings now? (y/n): "
    ).strip().lower()
    if rebuild == 'y':
        print("\nRunning augmented training...")
        os.system("python train_arcface.py")
        print("\nDone! Run python multi_camera.py to test.")


if __name__ == "__main__":
    main()