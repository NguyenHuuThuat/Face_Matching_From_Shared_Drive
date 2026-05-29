import cv2, time, io, os, re, sys, shutil, threading, subprocess
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── CONFIG ──────────────────────────────────────────────────
SA_KEY   = '/workspace/ductc/face_matching/face-matching-497405-8b4a9762743f.json'
REF_DIR  = Path('/workspace/ductc/face_matching/reference_faces')
OUT_DIR  = Path('/workspace/ductc/face_matching/matched_per_person')
LOG_FILE = Path('/workspace/ductc/face_matching/run_multi.log')

SOURCE_FOLDER_ID = '15UUq8C7VOH-q8-CvIbTl8Xm35xLlZn3Z'
THRESHOLD        = 0.40
IMG_EXTS         = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
SCOPES           = ['https://www.googleapis.com/auth/drive.readonly']

# Auto-detect GPUs; override with env var GPUS="0,1,2,3"
GPUS = [int(x) for x in os.environ.get('GPUS', '').split(',') if x.strip().isdigit()]
DOWNLOAD_THREADS = 6          # download threads per GPU worker
PREFETCH         = 12         # in-flight downloads per worker
DET_SIZE         = (640, 640)
MAX_LONG_SIDE    = 1920
# ────────────────────────────────────────────────────────────


def stamp(msg):
    return f"[{time.strftime('%H:%M:%S')}] {msg}"


def person_key(stem: str) -> str:
    """'huu thuat_1' → 'huu thuat'; 'bao han' → 'bao han'."""
    m = re.match(r'^(.*?)(?:_\d+)?$', stem.strip())
    return m.group(1).strip()


def sanitize_folder(name: str) -> str:
    return re.sub(r'[^\w\s.-]', '_', name).strip() or 'unknown'


def build_drive():
    creds = service_account.Credentials.from_service_account_file(SA_KEY, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds, cache_discovery=False)


def list_images(folder_id, svc):
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
                result.extend(list_images(f['id'], svc))
            elif Path(f['name']).suffix.lower() in IMG_EXTS:
                result.append((f['id'], f['name']))
        token = r.get('nextPageToken')
        if not token:
            break
    return result


def build_reference_embeddings(log):
    """Group ref images by person, return (names, embeddings[N,D])."""
    from insightface.app import FaceAnalysis
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUS[0]) if GPUS else '0'
    app = FaceAnalysis(name='buffalo_l',
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    groups = defaultdict(list)
    for p in sorted(REF_DIR.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        groups[person_key(p.stem)].append(p)

    names, embs = [], []
    for name, paths in sorted(groups.items()):
        per_person = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                log(f'  Ref FAIL (read): {p.name}'); continue
            faces = app.get(img)
            if not faces:
                log(f'  Ref FAIL (no face): {p.name}'); continue
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            per_person.append(f.normed_embedding)
            log(f'  Ref OK: {p.name}')
        if not per_person:
            log(f'  Person SKIPPED (no embeddings): {name}'); continue
        e = np.mean(per_person, axis=0)
        e /= np.linalg.norm(e)
        names.append(name); embs.append(e)
    return names, np.stack(embs).astype(np.float32)


# ──────────── Worker (one process per GPU) ──────────────────
def worker(gpu_id, file_chunk, names, embs_bytes, embs_shape, log_q, progress_q, out_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # Load model on this GPU
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l',
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    embs = np.frombuffer(embs_bytes, dtype=np.float32).reshape(embs_shape)  # [N, D]

    # googleapiclient/httplib2 KHÔNG thread-safe → mỗi thread một service riêng.
    tls = threading.local()
    def get_svc():
        s = getattr(tls, 'svc', None)
        if s is None:
            s = build_drive()
            tls.svc = s
        return s

    def download(fid, retries=3):
        last = None
        for k in range(retries):
            try:
                svc = get_svc()
                buf = io.BytesIO()
                dl = MediaIoBaseDownload(
                    buf,
                    svc.files().get_media(fileId=fid, supportsAllDrives=True),
                    chunksize=4 << 20,
                )
                done = False
                while not done:
                    _, done = dl.next_chunk()
                return buf.getvalue()
            except Exception as e:
                last = e
                # rebuild service on next try
                tls.svc = None
                time.sleep(0.5 * (k + 1))
        raise last

    pool = ThreadPoolExecutor(DOWNLOAD_THREADS)
    futures = {}

    def submit(idx):
        fid, _ = file_chunk[idx]
        futures[idx] = pool.submit(download, fid)

    for i in range(min(PREFETCH, len(file_chunk))):
        submit(i)
    next_idx = PREFETCH

    matched_total = 0
    for i in range(len(file_chunk)):
        fid, fname = file_chunk[i]
        try:
            img_bytes = futures.pop(i).result()
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                progress_q.put((gpu_id, 1, 0)); continue

            h, w = img.shape[:2]
            if max(h, w) > MAX_LONG_SIDE:
                s = MAX_LONG_SIDE / max(h, w)
                img = cv2.resize(img, (int(w*s), int(h*s)))

            faces = app.get(img)
            # collect each face's embedding (filter too-small)
            face_embs = [f.normed_embedding for f in faces
                         if (f.bbox[2]-f.bbox[0]) > 20 and (f.bbox[3]-f.bbox[1]) > 20]

            matched_persons = []
            if face_embs:
                F = np.stack(face_embs).astype(np.float32)            # [K, D]
                sims = F @ embs.T                                     # [K, N]
                best_per_person = sims.max(axis=0)                    # [N]
                hits = np.where(best_per_person >= THRESHOLD)[0]
                for j in hits:
                    matched_persons.append((names[j], float(best_per_person[j])))

            if matched_persons:
                # Write once per file, then copy to each person's folder.
                # Suffix dup names to avoid clobber across folders.
                for pname, sim in matched_persons:
                    folder = out_dir / sanitize_folder(pname)
                    folder.mkdir(parents=True, exist_ok=True)
                    dest = folder / fname
                    if dest.exists():
                        # different file with same name → add short suffix
                        stem, ext = os.path.splitext(fname)
                        dest = folder / f"{stem}__{fid[:6]}{ext}"
                    dest.write_bytes(img_bytes)
                matched_total += 1
                tag = ', '.join(f'{n}={s:.2f}' for n, s in matched_persons)
                log_q.put(stamp(f'  [gpu{gpu_id}] MATCH {fname} → {tag}'))

            progress_q.put((gpu_id, 1, 1 if matched_persons else 0))
        except Exception as e:
            log_q.put(stamp(f'  [gpu{gpu_id}] ERR {fname}: {e}'))
            progress_q.put((gpu_id, 1, 0))
        finally:
            if next_idx < len(file_chunk):
                submit(next_idx); next_idx += 1

    pool.shutdown(wait=True)
    log_q.put(stamp(f'  [gpu{gpu_id}] worker done; matched={matched_total}'))


# ────────────────── Main ────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(LOG_FILE, 'a', encoding='utf-8')
    def log(msg):
        line = stamp(msg)
        print(line, flush=True); log_f.write(line + '\n'); log_f.flush()

    global GPUS
    if not GPUS:
        try:
            import torch  # type: ignore
            GPUS = list(range(torch.cuda.device_count())) or [0]
        except Exception:
            try:
                out = subprocess.check_output(['nvidia-smi', '-L'], text=True,
                                              stderr=subprocess.DEVNULL)
                GPUS = [int(m.group(1)) for m in re.finditer(r'GPU (\d+):', out)]
            except Exception:
                GPUS = []
            if not GPUS:
                # last-resort fallback: parse /proc
                try:
                    n = len(os.listdir('/proc/driver/nvidia/gpus'))
                    GPUS = list(range(n)) if n else [0]
                except Exception:
                    GPUS = [0]
    log(f'Using GPUs: {GPUS}')

    log('Building reference embeddings...')
    names, embs = build_reference_embeddings(log)
    log(f'{len(names)} người: {names}')
    if not names:
        log('Không có reference nào hợp lệ. Thoát.'); return

    log('Đang quét folder Drive...')
    svc = build_drive()
    all_files = list_images(SOURCE_FOLDER_ID, svc)
    log(f'Tổng: {len(all_files)} ảnh')
    if not all_files:
        return

    # Shuffle by id to balance any folder-clustered slow files across workers,
    # then round-robin split so each GPU gets an interleaved slice.
    all_files.sort(key=lambda x: x[0])
    chunks = [[] for _ in GPUS]
    for i, item in enumerate(all_files):
        chunks[i % len(GPUS)].append(item)

    ctx = mp.get_context('spawn')
    log_q = ctx.Queue()
    progress_q = ctx.Queue()
    procs = []
    embs_bytes = embs.tobytes()
    for w, gid in enumerate(GPUS):
        p = ctx.Process(
            target=worker,
            args=(gid, chunks[w], names, embs_bytes, embs.shape,
                  log_q, progress_q, OUT_DIR),
        )
        p.start(); procs.append(p)
        log(f'  spawned worker gpu{gid}  files={len(chunks[w])}')

    total = len(all_files); done = 0; matched = 0; t0 = time.time()
    alive = lambda: any(p.is_alive() for p in procs)
    while alive() or not log_q.empty() or not progress_q.empty():
        # drain logs
        try:
            while True:
                line = log_q.get_nowait()
                print(line, flush=True); log_f.write(line + '\n')
        except Exception:
            pass
        # drain progress
        try:
            while True:
                _, d, m = progress_q.get_nowait()
                done += d; matched += m
                if done % 200 == 0:
                    rate = done / max(time.time() - t0, 1e-6)
                    log(f'  progress {done}/{total}  matched={matched}  '
                        f'rate={rate:.1f} img/s')
        except Exception:
            pass
        log_f.flush()
        time.sleep(0.2)

    for p in procs:
        p.join()
    # final drain
    try:
        while True:
            line = log_q.get_nowait()
            print(line, flush=True); log_f.write(line + '\n')
    except Exception:
        pass

    log(f'DONE — {matched}/{total} ảnh có match → {OUT_DIR}')
    log_f.close()


if __name__ == '__main__':
    main()
