"""Check two concurrent dashboard MJPEG viewers without saving camera images."""
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
import argparse
import json
import cv2
import numpy as np


def first_frame(url):
    with urlopen(url,timeout=5) as response:
        data = b''
        while len(data) < 1024*1024:
            chunk = response.read(4096)
            if not chunk:
                break
            data += chunk
            start = data.find(b'\xff\xd8')
            end = data.find(b'\xff\xd9',start+2) if start >= 0 else -1
            if end >= 0:
                image = cv2.imdecode(np.frombuffer(data[start:end+2],np.uint8),cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('Preview JPEG could not be decoded.')
                return {'status':response.status,'width':image.shape[1],'height':image.shape[0]}
    raise ValueError('No complete preview frame received.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:3000/video_feed/cam_a')
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        print(json.dumps({'viewers':list(pool.map(first_frame,[args.url,args.url]))},indent=2))
