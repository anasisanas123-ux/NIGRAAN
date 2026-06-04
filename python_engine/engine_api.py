import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, Response
import cv2
import threading
import config

app   = Flask(__name__)
feeds = {}  # cam_id -> latest annotated frame
lock  = threading.Lock()

def update_feed(cam_id, frame):
    """Called by multi_camera.py to push latest frame."""
    with lock:
        feeds[cam_id] = frame

def generate(cam_id):
    while True:
        with lock:
            frame = feeds.get(cam_id)
        if frame is None:
            import time
            import time as t
            t.sleep(0.05)
            continue
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    return Response(
        generate(cam_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/health')
def health():
    return {'status': 'ok', 'cameras': list(feeds.keys())}

def start_engine():
    app.run(
        host=config.ENGINE_HOST,
        port=config.ENGINE_PORT,
        threaded=True,
        debug=False,
        use_reloader=False
    )