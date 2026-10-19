"""Read-only CPU diagnosis of face quality and gallery matching on engine snapshots."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
from urllib.request import urlopen

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import config
from python_engine.arcface_recognizer import ArcFaceRecognizer
from python_engine.face_quality import quality, assess_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',default='cam_a')
    parser.add_argument('--frames',type=int,default=3)
    args = parser.parse_args()
    if not (Path.home()/'.insightface/models/antelopev2').is_dir():
        raise RuntimeError('Existing antelopev2 models must be installed; this diagnostic does not download them.')
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='antelopev2',providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1,det_size=(1280,1280))
    rec = ArcFaceRecognizer.__new__(ArcFaceRecognizer)
    rec.gallery_lock = threading.RLock(); rec.load_database(); rec._build_matrix()
    observations = []
    for _ in range(max(1,args.frames)):
        with urlopen(f'http://127.0.0.1:{config.ENGINE_PORT}/snapshot/{args.camera}',timeout=5) as response:
            frame = cv2.imdecode(np.frombuffer(response.read(),np.uint8),cv2.IMREAD_COLOR)
        reports = []
        for face in app.get(frame):
            x1,y1,x2,y2 = face.bbox.astype(int)
            crop = frame[max(0,y1):y2,max(0,x1):x2]
            gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            emb = face.embedding/np.linalg.norm(face.embedding)
            best,score = rec._match(emb)
            reports.append({'width':int(x2-x1),'height':int(y2-y1),
                'detector_confidence':round(float(face.det_score),3),
                'brightness':round(float(gray.mean()),2),
                'sharpness':round(float(cv2.Laplacian(gray,cv2.CV_64F).var()),2),
                'pose':np.asarray(face.pose).round(2).tolist() if face.pose is not None else None,
                'quality_passed':quality(face,frame,config),'best_id':best,
                'quality_assessment':assess_quality(face,frame,config),
                'score':round(score,4),'identity_margin_passed':rec._open_set_check(emb,best,score),
                'role':rec.known_roles.get(best)})
        observations.append(reports)
        time.sleep(.3)
    print('FACE_DIAGNOSIS='+json.dumps(observations))


if __name__=='__main__': main()
