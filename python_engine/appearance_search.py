"""Search people in a video by appearance, clothing or face.

Indexing (once per video): sample frames, detect and track people (YOLO +
ByteTrack), and for every person track keep a few of the best crops. Each track
is then described by:
  - CLIP image embeddings of the body and of the head region (for text such as
    "red hoodie", "glasses", "backpack", "long hair")
  - upper / lower clothing colour names (for the colour filters)
  - an ArcFace face embedding when a usable face was seen (for photo search)
The result is stored in a folder: tracks.json, index.npz, thumbs/*.jpg.

Search: a text query is compared with the CLIP embeddings; a photo is compared
by face (ArcFace, same model as live recognition) and by overall look (CLIP).
This module is used by search_service.py, a separate process from the live
camera engine, and does not change recognition.
"""
import json
import os
import tempfile
import time

import cv2
import numpy as np

COLOUR_NAMES = ["black", "white", "gray", "red", "orange", "yellow", "green",
                "blue", "purple", "pink", "brown"]


# ---------------------------------------------------------------- colours
def colour_name(region_bgr):
    """Dominant clothing colour of a crop region, or None if unclear."""
    if region_bgr is None or region_bgr.size == 0:
        return None
    h, w = region_bgr.shape[:2]
    # Central strip, to stay off the background at the sides.
    region = region_bgr[:, int(w * 0.2):max(int(w * 0.8), int(w * 0.2) + 1)]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    if len(hsv) == 0:
        return None
    # Ignore skin (hands, arms, face) unless almost everything looks like skin.
    skin = (hsv[:, 0] <= 22) & (hsv[:, 1] >= 45) & (hsv[:, 1] <= 175) & (hsv[:, 2] >= 90)
    if (~skin).sum() >= 0.3 * len(hsv):
        hsv = hsv[~skin]
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    names = np.empty(len(hsv), dtype=object)
    black = val < 55
    # Bright, slightly tinted pixels are white under a camera colour cast.
    white = (~black) & (((sat < 40) & (val > 185)) | ((sat < 65) & (val > 205)))
    gray = (~black) & (~white) & (sat < 45)
    chroma = ~(black | white | gray)
    names[black], names[white], names[gray] = "black", "white", "gray"
    hc = hue[chroma]
    vc = val[chroma]
    cname = np.full(len(hc), "red", dtype=object)
    cname[(hc >= 8) & (hc < 20)] = "orange"
    cname[(hc >= 8) & (hc < 20) & (vc < 150)] = "brown"
    cname[(hc >= 20) & (hc < 34)] = "yellow"
    cname[(hc >= 34) & (hc < 85)] = "green"
    cname[(hc >= 85) & (hc < 130)] = "blue"
    cname[(hc >= 130) & (hc < 150)] = "purple"
    cname[(hc >= 150) & (hc < 172)] = "pink"
    names[chroma] = cname
    values, counts = np.unique(names.astype(str), return_counts=True)
    best = int(np.argmax(counts))
    return str(values[best]) if counts[best] / len(names) >= 0.25 else None


def clothing_colours(crop):
    h = crop.shape[0]
    upper = crop[int(h * 0.20):int(h * 0.55)]
    lower = crop[int(h * 0.58):int(h * 0.92)]
    return colour_name(upper), colour_name(lower)


# ---------------------------------------------------------------- models
class Models:
    """Loads the models once; shared by indexing and search."""

    def __init__(self, data_dir, device=None):
        import torch
        import open_clip

        self.data_dir = data_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k",
            cache_dir=os.path.join(data_dir, "models", "clip"), device=self.device)
        self.clip.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self._faces = None
        self._yolo_path = self._weights("yolov8s.pt")

    def _weights(self, name):
        path = os.path.join(self.data_dir, "models", name)
        if os.path.isfile(path):
            return path
        from ultralytics.utils.downloads import attempt_download_asset
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return str(attempt_download_asset(path))

    def person_detector(self):
        from ultralytics import YOLO
        return YOLO(self._yolo_path)

    def faces(self):
        if self._faces is None:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="antelopev2", allowed_modules=["detection", "recognition"],
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
            self._faces = app
        return self._faces

    def image_embeddings(self, crops_bgr):
        from PIL import Image
        if not crops_bgr:
            return np.zeros((0, 512), np.float32)
        out = []
        for start in range(0, len(crops_bgr), 64):
            batch = [self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                     for c in crops_bgr[start:start + 64]]
            with self.torch.no_grad():
                emb = self.clip.encode_image(self.torch.stack(batch).to(self.device))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out)

    def text_embedding(self, query):
        query = query.strip()
        prompts = [f"a photo of a person wearing {query}", f"a photo of a person with {query}",
                   f"a person, {query}", query]
        with self.torch.no_grad():
            emb = self.clip.encode_text(self.tokenizer(prompts).to(self.device))
        emb = emb / emb.norm(dim=-1, keepdim=True)
        mean = emb.mean(dim=0)
        return (mean / mean.norm()).float().cpu().numpy()

    def face_embedding(self, image_bgr):
        """Best face embedding in an image, or None."""
        if image_bgr is None or image_bgr.size == 0:
            return None
        h, w = image_bgr.shape[:2]
        if max(h, w) < 400:  # small crops: upscale so the detector finds the face
            scale = 400 / max(h, w)
            image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        faces = self.faces().get(image_bgr)
        faces = [f for f in faces if float(f.det_score) >= 0.6 and (f.bbox[2] - f.bbox[0]) >= 28]
        if not faces:
            return None
        face = max(faces, key=lambda f: float(f.det_score) * (f.bbox[2] - f.bbox[0]))
        emb = np.asarray(face.normed_embedding if face.normed_embedding is not None else face.embedding, np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else None


# ---------------------------------------------------------------- indexing
def _write_json(path, data):
    fd, temp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(temp, path)


def index_video(models, video_path, out_dir, sample_fps=3.0, progress=None, should_stop=None):
    """Build the search index for one video. Returns a short summary."""
    os.makedirs(os.path.join(out_dir, "thumbs"), exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("The video could not be opened. Use MP4, AVI, MOV or MKV.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    if not np.isfinite(fps) or fps <= 1 or fps > 240:
        fps = 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / sample_fps)))
    detector = models.person_detector()
    tracks = {}
    frame_index = 0
    last_report = 0.0
    started = time.time()
    try:
        while True:
            if should_stop and should_stop():
                raise InterruptedError("Indexing was cancelled.")
            ok = capture.grab()
            if not ok:
                break
            if frame_index % step:
                frame_index += 1
                continue
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                frame_index += 1
                continue
            t = frame_index / fps
            frame_index += 1
            result = detector.track(frame, persist=True, classes=[0], conf=0.35, verbose=False,
                                    tracker="bytetrack.yaml", device=0 if models.device == "cuda" else "cpu")[0]
            if result.boxes is None or result.boxes.id is None:
                continue
            fh, fw = frame.shape[:2]
            for box, track_id, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.id.cpu().numpy(),
                                           result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(fw, x2), min(fh, y2)
                if y2 - y1 < 64 or x2 - x1 < 24:
                    continue
                track = tracks.setdefault(int(track_id), {"first": t, "last": t, "seen": 0, "crops": []})
                track["last"] = t
                track["seen"] += 1
                quality = (y2 - y1) * (x2 - x1) * float(conf)
                crops = track["crops"]
                if len(crops) < 5 or quality > crops[-1][0]:
                    crop = frame[y1:y2, x1:x2]
                    scale = min(1.0, 384 / crop.shape[0])
                    if scale < 1:
                        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), 384))
                    context = cv2.resize(frame, (480, max(1, int(fh * 480 / fw))))
                    cv2.rectangle(context, (int(x1 * 480 / fw), int(y1 * 480 / fw)),
                                  (int(x2 * 480 / fw), int(y2 * 480 / fw)), (0, 200, 255), 2)
                    crops.append((quality, crop, t, context))
                    crops.sort(key=lambda item: -item[0])
                    del crops[5:]
            now = time.time()
            if progress and total and now - last_report > 1.0:
                last_report = now
                progress(min(0.95, frame_index / total * 0.9))
    finally:
        capture.release()

    # A track seen only once is usually a false detection.
    kept = [(tid, tr) for tid, tr in sorted(tracks.items()) if tr["seen"] >= 2 or len(tracks) <= 3]
    records, body, head, face, has_face = [], [], [], [], []
    for number, (tid, track) in enumerate(kept, 1):
        crops = [c[1] for c in track["crops"]]
        heads = [c[: max(8, int(c.shape[0] * 0.3))] for c in crops]
        body_emb = models.image_embeddings(crops).mean(axis=0)
        head_emb = models.image_embeddings(heads).mean(axis=0)
        body.append(body_emb / (np.linalg.norm(body_emb) or 1))
        head.append(head_emb / (np.linalg.norm(head_emb) or 1))
        face_emb = None
        for crop in crops[:3]:
            face_emb = models.face_embedding(crop)
            if face_emb is not None:
                break
        has_face.append(face_emb is not None)
        face.append(face_emb if face_emb is not None else np.zeros(512, np.float32))
        best = track["crops"][0]
        upper, lower = clothing_colours(best[1])
        thumb = f"thumbs/{number}.jpg"
        context = f"thumbs/{number}_scene.jpg"
        cv2.imwrite(os.path.join(out_dir, thumb), best[1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(os.path.join(out_dir, context), best[3], [cv2.IMWRITE_JPEG_QUALITY, 80])
        records.append({
            "number": number, "start": round(track["first"], 2), "end": round(track["last"], 2),
            "best_time": round(best[2], 2), "upper_colour": upper, "lower_colour": lower,
            "has_face": face_emb is not None, "thumb": thumb, "scene": context,
        })
        if progress:
            progress(0.9 + 0.1 * number / max(1, len(kept)))
    np.savez_compressed(os.path.join(out_dir, "index.npz"),
                        body=np.asarray(body, np.float32).reshape(-1, 512),
                        head=np.asarray(head, np.float32).reshape(-1, 512),
                        face=np.asarray(face, np.float32).reshape(-1, 512),
                        has_face=np.asarray(has_face, bool))
    summary = {"people": len(records), "duration_seconds": round(frame_index / fps, 1),
               "fps": round(fps, 2), "seconds_taken": round(time.time() - started, 1)}
    _write_json(os.path.join(out_dir, "tracks.json"), {"tracks": records, **summary})
    return summary


# ---------------------------------------------------------------- search
def load_index(folder):
    with open(os.path.join(folder, "tracks.json"), "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    arrays = np.load(os.path.join(folder, "index.npz"), allow_pickle=False)
    return meta["tracks"], {key: arrays[key] for key in arrays.files}


def photo_query(models, image_bgr):
    """Describe a query photo: face embedding and CLIP look of the main person."""
    face = models.face_embedding(image_bgr)
    detector = models.person_detector()
    result = detector.predict(image_bgr, classes=[0], conf=0.35, verbose=False,
                              device=0 if models.device == "cuda" else "cpu")[0]
    crop = image_bgr
    if result.boxes is not None and len(result.boxes):
        boxes = result.boxes.xyxy.cpu().numpy()
        x1, y1, x2, y2 = [int(v) for v in max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))]
        if y2 - y1 > 32 and x2 - x1 > 16:
            crop = image_bgr[max(0, y1):y2, max(0, x1):x2]
    look = models.image_embeddings([crop])[0]
    return {"face": face, "look": look, "person_found": crop is not image_bgr}


# Similarity levels used to decide what counts as a match.
FACE_MATCH = 0.40        # ArcFace cosine; live recognition uses 0.55 on a curated gallery
LOOK_MATCH = 0.80        # CLIP image-to-image
TEXT_RELATIVE = 0.035    # CLIP text: keep results within this of the best score


COLOUR_WORDS = {name: name for name in COLOUR_NAMES}
COLOUR_WORDS.update({"grey": "gray", "navy": "blue", "maroon": "red", "violet": "purple",
                     "khaki": "brown", "beige": "brown", "golden": "yellow", "silver": "gray"})
UPPER_WORDS = {"shirt", "t-shirt", "tshirt", "jacket", "hoodie", "top", "sweater", "coat", "kurta",
               "blazer", "jersey", "uniform", "vest", "polo", "cardigan", "dupatta"}
LOWER_WORDS = {"pants", "jeans", "trousers", "shorts", "skirt", "shalwar", "trouser", "leggings"}


def colour_terms(query):
    """[("red", "upper"|"lower"|None), ...] for colour words in a text query."""
    words = [w.strip(".,;:!?") for w in str(query or "").lower().replace("-", " - ").split()]
    terms = []
    for index, word in enumerate(words):
        colour = COLOUR_WORDS.get(word)
        if not colour:
            continue
        part = None
        for following in words[index + 1:index + 4]:
            if following in UPPER_WORDS or following == "t":
                part = "upper"
                break
            if following in LOWER_WORDS:
                part = "lower"
                break
        terms.append((colour, part))
    return terms


def _colour_ok(track, terms):
    for colour, part in terms:
        if part == "upper" and track.get("upper_colour") != colour:
            return False
        if part == "lower" and track.get("lower_colour") != colour:
            return False
        if part is None and colour not in (track.get("upper_colour"), track.get("lower_colour")):
            return False
    return True


def search(indexes, text=None, photo=None, upper=None, lower=None, limit=60, text_query=None):
    """indexes: list of (source_id, folder, tracks, arrays). Returns ranked matches.

    text is the CLIP embedding of text_query. Colour words in text_query
    ("red jacket", "blue jeans") must agree with the detected clothing colours;
    when nobody matches them, the closest looks are returned, marked as such.
    """
    terms = colour_terms(text_query) if text is not None else []
    rows = []
    for source_id, _folder, tracks, arrays in indexes:
        if not tracks:
            continue
        scores = np.zeros(len(tracks), np.float32)
        reasons = [[] for _ in tracks]
        if text is not None:
            text_score = np.maximum(arrays["body"] @ text, arrays["head"] @ text * 0.97)
            scores += text_score
        if photo is not None:
            look = arrays["body"] @ photo["look"]
            combined = look.copy()
            if photo["face"] is not None and arrays["has_face"].any():
                face = arrays["face"] @ photo["face"]
                face[~arrays["has_face"]] = -1
                # A face match outranks any clothing similarity.
                combined = np.where(face >= FACE_MATCH, 1.0 + face, look)
                for i in np.where(face >= FACE_MATCH)[0]:
                    reasons[i].append(f"face match {face[i]:.2f}")
            for i in np.where((look >= LOOK_MATCH) & (combined < 1.0))[0]:
                reasons[i].append(f"similar look {look[i]:.2f}")
            scores += combined
        for i, track in enumerate(tracks):
            if upper and track.get("upper_colour") != upper:
                continue
            if lower and track.get("lower_colour") != lower:
                continue
            rows.append({**track, "source_id": source_id, "score": round(float(scores[i]), 4),
                         "reasons": reasons[i]})
    rows.sort(key=lambda r: -r["score"])
    if photo is not None:
        rows = [r for r in rows if r["reasons"]]
    elif text is not None and rows:
        if terms:
            coloured = [r for r in rows if _colour_ok(r, terms)]
            if coloured:
                rows = coloured
                for row in rows:
                    row["reasons"].append("clothing colour matches")
            else:
                rows = rows[:5]
                for row in rows:
                    row["reasons"].append("closest look only - no clothing in that colour was detected")
                return rows
        best = rows[0]["score"]
        rows = [r for r in rows if r["score"] >= best - TEXT_RELATIVE]
    return rows[:limit]
