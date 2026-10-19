with open("multi_camera.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "body_dets = body_detector.detect(frame_rgb, cam_id=cam_id)" in line:
        new_lines = (
            lines[:i+1]
            + ["        if cam_id == 'cam_a' and len(body_dets) > 1:\n",
               "            for bd in body_dets:\n",
               "                print(f\"  ghost-check conf={bd['confidence']:.3f} bbox={bd['bbox']}\")\n"]
            + lines[i+1:]
        )
        with open("multi_camera.py", "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"INSERTED after line {i+1}")
        break
else:
    print("NOT FOUND")
