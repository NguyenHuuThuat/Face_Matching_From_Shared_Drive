import cv2, json, time, io, os, sys
import numpy as np
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from insightface.app import FaceAnalysis

# ── CONFIG ──────────────────────────────────────────────────
SA_KEY   = '/workspace/ductc/face_matching/face-matching-497405-8b4a9762743f.json'
REF_DIR  = Path('/workspace/ductc/face_matching/reference_faces')
OUT_DIR  = Path('/workspace/ductc/face_matching/matched')
LOG_FILE = Path('/workspace/ductc/face_matching/run.log')

SOURCE_FOLDER_ID = '15UUq8C7VOH-q8-CvIbTl8Xm35xLlZn3Z'
THRESHOLD        = 0.40
IMG_EXTS         = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
# ────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Log ra cả terminal lẫn file
class Logger:
    def __init__(self, path):
        self.f = open(path, 'a', encoding='utf-8')
    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + '\n')
        self.f.flush()

L = Logger(LOG_FILE)

# Model
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
L.log('Loading model...')
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
L.log('Model OK')

# Reference embedding
embeddings = []
for p in REF_DIR.iterdir():
    if p.suffix.lower() not in IMG_EXTS: continue
    img = cv2.imread(str(p))
    faces = app.get(img)
    if faces:
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        embeddings.append(f.normed_embedding)
        L.log(f'  Ref OK: {p.name}')
    else:
        L.log(f'  Ref FAIL (no face): {p.name}')

ref = np.mean(embeddings, axis=0)
ref /= np.linalg.norm(ref)
L.log(f'Reference từ {len(embeddings)} ảnh')

# Drive
creds = service_account.Credentials.from_service_account_file(SA_KEY, scopes=['https://www.googleapis.com/auth/drive.readonly'])
svc   = build('drive', 'v3', credentials=creds, cache_discovery=False)

def list_images(folder_id):
    result, token = [], None
    while True:
        r = svc.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields='nextPageToken, files(id, name, mimeType)',
            pageToken=token, pageSize=1000,
            supportsAllDrives=True, includeItemsFromAllDrives=True
        ).execute()
        for f in r.get('files', []):
            if f['mimeType'] == 'application/vnd.google-apps.folder':
                result.extend(list_images(f['id']))
            elif Path(f['name']).suffix.lower() in IMG_EXTS:
                result.append((f['id'], f['name']))
        token = r.get('nextPageToken')
        if not token: break
    return result

L.log('Đang quét folder...')
all_files = list_images(SOURCE_FOLDER_ID)
L.log(f'Tổng: {len(all_files)} ảnh')

matched = 0
for i, (fid, fname) in enumerate(all_files):
    try:
        buf = io.BytesIO()
        dl  = MediaIoBaseDownload(buf, svc.files().get_media(fileId=fid, supportsAllDrives=True), chunksize=4<<20)
        done = False
        while not done: _, done = dl.next_chunk()
        img_bytes = buf.getvalue()

        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: continue

        h, w = img.shape[:2]
        if max(h,w) > 1920:
            s = 1920/max(h,w); img = cv2.resize(img, (int(w*s), int(h*s)))

        faces = app.get(img)
        best  = max((np.dot(f.normed_embedding, ref) for f in faces
                     if (f.bbox[2]-f.bbox[0])>20 and (f.bbox[3]-f.bbox[1])>20),
                    default=-1)

        if best >= THRESHOLD:
            out = OUT_DIR / fname
            out.write_bytes(img_bytes)
            matched += 1
            L.log(f'  MATCH [{i+1}/{len(all_files)}] {fname}  sim={best:.3f}  (total={matched})')
        elif (i+1) % 100 == 0:
            L.log(f'  ... [{i+1}/{len(all_files)}] matched={matched}')

    except Exception as e:
        L.log(f'  ERR {fname}: {e}')

L.log(f'DONE — {matched}/{len(all_files)} ảnh khớp → {OUT_DIR}')