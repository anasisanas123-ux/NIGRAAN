"""One-time structural extraction, retaining camera identity logic verbatim."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root/'multi_camera.py').read_text(encoding='utf-8')
lines = source.splitlines(keepends=True)
tree = ast.parse(source)
nodes = {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
segment = lambda n: ''.join(lines[n.lineno-1:n.end_lineno])
constants = [n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id.isupper()]
(root/'python_engine/runtime_settings.py').write_text('"""Existing identity continuity settings, unchanged by the refactor."""\n'+ '\n'.join(segment(n) for n in constants),encoding='utf-8')
helpers = '\n\n'.join(segment(nodes[name]) for name in ('post_detection','emb_valid','iou'))
(root/'python_engine/runtime_helpers.py').write_text('"""Detection logging and shared numeric helpers."""\nimport requests\nimport numpy as np\nimport config\n\n'+helpers,encoding='utf-8')
worker = segment(nodes['camera_worker'])
worker = worker.replace('    result_queue\n):','    result_queue,\n    context\n):',1)
dependencies = ['recognizer','body_detector','body_reid','reid','anonymous_face_reid','annotator','global_logged_students','global_logged_unknowns','log_lock','cross_cam_fired','cross_cam_lock']
bindings = ''.join(f'    {name} = context.{name}\n' for name in dependencies)
worker = worker.replace('):\n','):\n'+bindings,1)
worker = worker.replace('    # ========================================================\n    # WAIT FOR CAMERA CONNECTION', '''    context.register_stream(cam_id, stream)
    preview = CameraPreview(cam_id, stream, result_queue, annotator, recognizer,
                            publish_preview, (violence_classifier, weapon_detector), zone_store)
    preview.thread.start()

    # ========================================================
    # WAIT FOR CAMERA CONNECTION''',1)
start = worker.index('    connected = False')
end = worker.index('    # Current physical camera session.',start)
worker = worker[:start]+'''    while stream.running:
        packet = stream.read_packet(timeout=1)
        if packet is not None:
            break
    else:
        return
    last_sequence = 0
    dropped_frames = 0

'''+worker[end:]
# Only the main inference loop reads new packets. Preview has its own cursor.
start = worker.index('    while True:\n\n        ret, frame = stream.read()')
end = worker.index('        frame_count += 1',start)
worker = worker[:start]+'''    while stream.running:
        packet = stream.read_packet(last_sequence, timeout=.5)
        if packet is None:
            continue
        if last_sequence:
            dropped_frames += max(0,packet.sequence-last_sequence-1)
        last_sequence = packet.sequence
        frame = packet.frame
        inference_started = time.perf_counter()
        face_ms = 0.0

'''+worker[end:]
start = worker.index('        if (\n            frame_count\n            %')
end = worker.index('        frame_bgr = frame',start)
worker = worker[:start]+'''        if frame_count % max(1,config.FRAME_SKIP) != 0:
            continue

'''+worker[end:]
start = worker.index('        frame_rgb = cv2.cvtColor(')
end = worker.index('        # ====================================================\n        # SHORT DETECTION',start)
worker = worker[:start]+'''        stage_started = time.perf_counter()
        body_dets = body_detector.detect(frame_bgr, cam_id=cam_id, color_order='bgr')
        person_ms = (time.perf_counter()-stage_started)*1000

'''+worker[end:]
worker = worker.replace('        if need_arcface_regions:\n\n            arc_results = (','        if need_arcface_regions:\n\n            stage_started = time.perf_counter()\n            arc_results = (',1)
start = worker.index('            stage_started = time.perf_counter()\n            arc_results = (')
end = worker.index('        # ====================================================\n        # FACE/BODY MATCHING',start)
worker = worker[:end]+ '        face_ms = (time.perf_counter()-stage_started)*1000 if need_arcface_regions else 0.0\n\n'+worker[end:]
start = worker.index('        display_frame = annotator.draw(frame.copy(), results)')
end = worker.index('    stream.stop()',start)
worker = worker[:start]+'''        telemetry = {'processed_fps':round(fps,1),
                     'inference_ms':round((time.perf_counter()-inference_started)*1000,1),
                     'person_ms':round(person_ms,1), 'face_ms':round(face_ms,1),
                     'dropped_inference_frames':dropped_frames,
                     'analysis_frame_at':packet.captured_at}
        update_analysis(cam_id, results, frame.shape[:2], telemetry)
        preview.submit(packet, results)

'''+worker[end:]
header = '''"""Per-camera tracking and identity decisions; models are injected explicitly."""
import time
import threading
import requests
import cv2
import config
from python_engine.camera_stream import CameraStream
from python_engine.identity_cache import invalidate_inactive, needs_face_check
from python_engine.gait_pose import GaitPoseExtractor
from python_engine.gait_signature import GaitSignatureEngine
from python_engine.runtime_helpers import post_detection, emb_valid
from python_engine.runtime_settings import *
from python_engine.preview import CameraPreview
from python_engine.engine_api import update_analysis, publish_preview, violence_classifier, weapon_detector, zone_store

'''
(root/'python_engine/camera_worker.py').write_text(header+worker,encoding='utf-8')
# Keep runtime composition compact; all startup commands still enter main().
runtime = '''"""Compose unchanged models and run the multi-camera application."""
import sys
import os
import threading
from pathlib import Path
import torch
import config
from python_engine.arcface_recognizer import ArcFaceRecognizer
from python_engine.body_detector import BodyDetector
from python_engine.body_reid import BodyReID
from python_engine.annotator import Annotator
from python_engine.reid_manager import ReIDManager
from python_engine.anonymous_face_reid import AnonymousFaceReID
from python_engine.engine_api import start_engine, set_identity_recognizer
from python_engine.runtime_context import RuntimeContext
from python_engine.camera_worker import camera_worker
from python_engine.frame_buffer import LatestCameraResults
from python_engine.display import run_display


def main():
    # Resolve model/config paths consistently when launched from another folder.
    os.chdir(Path(__file__).resolve().parents[1])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    print('Loading models...')
    context = RuntimeContext(
        recognizer=ArcFaceRecognizer(), body_detector=BodyDetector(),
        body_reid=BodyReID(), reid=ReIDManager(),
        anonymous_face_reid=AnonymousFaceReID(), annotator=Annotator())
    set_identity_recognizer(context.recognizer)
    threading.Thread(target=start_engine, daemon=True, name='engine-api').start()
    results = LatestCameraResults()
    started = 0
    try:
        for camera, source in config.CAMERA_SOURCES.items():
            if source is None or source == '':
                continue
            threading.Thread(target=camera_worker, args=(camera,source,results,context),
                             daemon=True, name=f'inference-{camera}').start()
            started += 1
        if not started:
            raise RuntimeError('No camera sources configured.')
        print(f'{started} camera workers started. Press Q in a camera window to quit.')
        run_display(results)
    finally:
        context.stop()


if __name__ == '__main__':
    main()
'''
(root/'python_engine/runtime.py').write_text(runtime,encoding='utf-8')
# Preserve early Windows DLL setup from the original entry point.
wrapper = '''"""Compatibility launcher; the implementation lives in python_engine/."""
import sys
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetDllDirectoryW(r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin')
from python_engine.runtime import main

if __name__ == '__main__':
    main()
'''
(root/'multi_camera.py').write_text(wrapper,encoding='utf-8')
print('Extracted camera runtime, worker, settings and helpers; launchers preserved.')
