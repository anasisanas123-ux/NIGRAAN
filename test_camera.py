import cv2

RTSP_URL = __import__("config").CAMERA_SOURCES["cam_b"]

print("Connecting to Dahua camera...")
print(f"URL: {RTSP_URL}")

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("\nFAILED — trying sub stream...")
    SUB_URL = __import__("config").CAMERA_SOURCES["cam_b"]
    cap = cv2.VideoCapture(SUB_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("FAILED — cannot connect to camera")
    print("\nCheck:")
    print("  1. Camera and PC are on same WiFi network")
    print("  2. Ping the camera: ping 192.168.100.2")
    print("  3. Username and password are correct")
else:
    print("SUCCESS — Dahua camera connected!")
    print(f"Resolution : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS        : {cap.get(cv2.CAP_PROP_FPS)}")
    print("\nShowing live feed — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection — retrying...")
            continue

        cv2.putText(frame, "Dahua Camera — Connected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow("Dahua Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Test complete.")