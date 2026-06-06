import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import config

def capture_cctv_photos(student_folder, camera_source, cam_name, num_photos=25):
    save_dir = os.path.join(config.RAW_IMAGES_DIR, student_folder)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nConnecting to {cam_name}...")
    cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Cannot connect to {cam_name}")
        return 0

    # count existing photos
    existing = len([f for f in os.listdir(save_dir)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))])

    count = 0
    print(f"Camera connected. Existing photos: {existing}")
    print(f"Target: {num_photos} new photos")
    print("Stand in front of camera at normal position")
    print("Press SPACE to capture | Q to finish\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        cv2.putText(display, f"Student: {student_folder}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.putText(display, f"Camera: {cam_name}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)
        cv2.putText(display, f"Captured: {count}/{num_photos}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=capture  |  Q=done",
                    (10, 135), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)

        if count >= num_photos:
            cv2.putText(display, "TARGET REACHED — press Q",
                        (10, 170), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        cv2.imshow(f"CCTV Capture — {cam_name}", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            total = existing + count + 1
            path  = os.path.join(save_dir, f"cctv_{cam_name}_{total:03d}.jpg")
            cv2.imwrite(path, frame)
            count += 1
            print(f"  Saved {count}/{num_photos}: {path}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {count} photos saved for {student_folder} on {cam_name}")
    return count


if __name__ == "__main__":
    students = [
    ("S001", "Umair_Iqbal"),
    ("S002", "Anas_Ahmed"),
    ("S003", "Rahim"),
    ("S004", "Abdul_Basit"),
]

    cameras = list(config.CAMERA_SOURCES.items())

    print("=" * 50)
    print("  CCTV Photo Collector")
    print("=" * 50)

    print("\nSelect STUDENT:")
    for i, s in enumerate(students, 1):
        print(f"  {i}. {s}")

    while True:
        try:
            s_choice = int(input("\nStudent number >> ")) - 1
            if 0 <= s_choice < len(students):
                break
            print("Invalid — enter 1-7")
        except ValueError:
            print("Enter a number")

    print("\nSelect CAMERA:")
    for i, (cam_id, source) in enumerate(cameras, 1):
        print(f"  {i}. {cam_id} — {source[:50]}...")

    while True:
        try:
            c_choice = int(input("\nCamera number >> ")) - 1
            if 0 <= c_choice < len(cameras):
                break
            print(f"Invalid — enter 1-{len(cameras)}")
        except ValueError:
            print("Enter a number")

    cam_id, source = cameras[c_choice]
    student_id, student_name = students[s_choice]
    folder_name = student_id + '_' + student_name
    capture_cctv_photos(
        folder_name,
        source,
        cam_id,
        num_photos=25
    )