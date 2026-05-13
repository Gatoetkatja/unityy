"""
Fase 0 (v2): Student Tracking dengan ByteTrack + Bbox Ground Truth
===================================================================
Pendekatan baru:
  - Bbox berasal dari LABEL GROUND TRUTH (dataset/labels/) bukan YOLO inference
  - Tracking pakai ByteTrack untuk konsistensi student_id
  - ByteTrack dipilih karena robust terhadap occlusion & kamera bergerak pelan

Input:
  dataset/<split>/videos/<video_id>/*.jpg
  dataset/<split>/labels/<video_id>/*.txt   (YOLO format)

Output:
  crop/<split>/<video_id>/student_XXX/student_XXX_NNNN.jpg

Kenapa ByteTrack?
  - Kalman filter di dalamnya bisa memprediksi posisi siswa di frame berikutnya
    sehingga tetap akurat meski kamera bergerak pelan ke kanan/kiri
  - Two-stage matching: high-conf dulu, lalu low-conf → bagus untuk
    occlusion ringan (siswa menunduk)
  - Track buffer: ID lama tetap "hidup" beberapa frame setelah hilang

Dependency:
  pip install ultralytics opencv-python numpy
  (ByteTrack tersedia langsung di ultralytics, tidak perlu install terpisah)
"""

import os
import re
import logging
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────────────────────────
class Config:
    # ByteTrack parameters
    TRACK_HIGH_THRESH: float = 0.5   # Threshold deteksi high-confidence
    TRACK_LOW_THRESH:  float = 0.1   # Threshold deteksi low-confidence
    NEW_TRACK_THRESH:  float = 0.6   # Threshold membuat track baru
    TRACK_BUFFER:      int   = 60    # Frame tunggu sebelum track dihapus (kameragerakan pelan → buffer besar)
    MATCH_THRESH:      float = 0.8   # IoU threshold untuk matching

    # Untuk bbox dari label (ground truth), kita set confidence = 1.0
    # agar selalu masuk kategori high-confidence di ByteTrack
    GT_CONFIDENCE: float = 0.99

    # Filter bbox terlalu kecil (mungkin noise di label)
    MIN_BBOX_SIZE: int = 10  # pixel


# ─────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────
def natural_sort_key(path: Path) -> list:
    """Sort alami berdasarkan angka dalam nama file."""
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalized → pixel (x1,y1,x2,y2)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
    return x1, y1, x2, y2


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """
    Baca file label YOLO, return array bbox dengan dummy confidence.

    Format YOLO: <class> <cx> <cy> <w> <h>  (normalized)

    Return: np.ndarray shape (N, 6) = [x1, y1, x2, y2, conf, class]
            class disetel 0 untuk semua (kita track "orang", tidak peduli
            cheating/not_cheating — itu urusan Fase 2)
    """
    if not label_path.exists():
        return np.zeros((0, 6), dtype=np.float32)

    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_orig = int(parts[0])     # 0=cheating, 1=not_cheating
                cx, cy, w, h = map(float, parts[1:5])
            except ValueError:
                continue

            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)

            # Filter bbox terlalu kecil
            if (x2 - x1) < Config.MIN_BBOX_SIZE or (y2 - y1) < Config.MIN_BBOX_SIZE:
                continue

            # Format: x1, y1, x2, y2, confidence, class_for_tracker
            # Kita pakai class 0 (semuanya "person") agar tracker tidak
            # memisahkan track berdasarkan cheating/not_cheating.
            # Class asli akan dipulihkan saat Fase 2 lewat anotasi.
            boxes.append([
                x1, y1, x2, y2,
                Config.GT_CONFIDENCE,
                0  # class untuk tracker (bukan cheating class)
            ])

    if not boxes:
        return np.zeros((0, 6), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# ByteTrack Wrapper
# ─────────────────────────────────────────────────────────────────
def build_bytetracker():
    """
    Buat instance ByteTrack dari ultralytics.

    Catatan: ByteTracker di ultralytics mengharapkan input berupa
    objek 'Results' dari YOLO. Kita akan bypass dengan langsung
    memanggil tracker.update() dengan format detection custom.
    """
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
    except ImportError:
        raise ImportError(
            "ultralytics tidak terinstall atau versi lama.\n"
            "Install: pip install --upgrade ultralytics"
        )

    # ByteTracker butuh objek args (namespace)
    class TrackerArgs:
        track_high_thresh = Config.TRACK_HIGH_THRESH
        track_low_thresh  = Config.TRACK_LOW_THRESH
        new_track_thresh  = Config.NEW_TRACK_THRESH
        track_buffer      = Config.TRACK_BUFFER
        match_thresh      = Config.MATCH_THRESH
        fuse_score        = True

    args = TrackerArgs()
    tracker = BYTETracker(args)
    return tracker


# Adapter agar bbox dari label bisa diumpankan ke BYTETracker
# (BYTETracker biasanya menerima objek Results dari YOLO inference)
class DetectionResults:
    """
    Wrapper minimal yang meniru API ultralytics.engine.results.Boxes.
    Diperlukan karena BYTETracker mengakses .conf, .xywh, .xyxy, .cls,
    dan juga melakukan boolean indexing: results[mask].
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        # Pastikan array 2D / 1D yang benar
        if len(xyxy) == 0:
            self.xyxy = np.zeros((0, 4), dtype=np.float32)
            self.conf = np.zeros(0, dtype=np.float32)
            self.cls  = np.zeros(0, dtype=np.float32)
            self.xywh = np.zeros((0, 4), dtype=np.float32)
        else:
            self.xyxy = xyxy.astype(np.float32)
            self.conf = conf.astype(np.float32)
            self.cls  = cls.astype(np.float32)
            # Convert xyxy → xywh (center format)
            x1, y1, x2, y2 = self.xyxy[:, 0], self.xyxy[:, 1], self.xyxy[:, 2], self.xyxy[:, 3]
            self.xywh = np.stack([
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                x2 - x1,
                y2 - y1
            ], axis=1).astype(np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        """Support boolean mask indexing yang dipakai BYTETracker internal."""
        return DetectionResults(
            xyxy=self.xyxy[idx],
            conf=self.conf[idx],
            cls=self.cls[idx]
        )


def run_tracker_on_detections(
    tracker,
    detections: np.ndarray,
    img_shape: tuple
):
    """
    Jalankan ByteTracker pada satu frame.

    Args:
        tracker    : instance BYTETracker
        detections : np.ndarray (N, 6) [x1,y1,x2,y2,conf,cls]
        img_shape  : (H, W) dari frame

    Return: list of (track_id, x1, y1, x2, y2)
    """
    if len(detections) == 0:
        fake = DetectionResults(
            xyxy=np.zeros((0, 4)),
            conf=np.zeros(0),
            cls=np.zeros(0)
        )
    else:
        fake = DetectionResults(
            xyxy=detections[:, :4],
            conf=detections[:, 4],
            cls=detections[:, 5]
        )

    tracks = tracker.update(fake, img=None)

    # tracks shape: (M, 7) = [x1, y1, x2, y2, track_id, conf, cls]
    output = []
    if len(tracks) > 0:
        for t in tracks:
            x1, y1, x2, y2 = t[:4]
            tid = int(t[4])
            output.append((tid, x1, y1, x2, y2))

    return output


# ─────────────────────────────────────────────────────────────────
# Pipeline Per Video
# ─────────────────────────────────────────────────────────────────
def process_video(
    frames_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    split_name: str,
    video_id: str
) -> dict:
    """
    Proses satu video: load label → ByteTrack → crop per student_id.

    Return: dict {student_id: jumlah_crop}
    """
    log.info(f"Memproses [{split_name}] video {video_id} ...")

    # Frame & label
    frame_files = sorted(
        [f for f in frames_dir.iterdir()
         if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=natural_sort_key
    )
    if not frame_files:
        log.warning(f"  Tidak ada frame di {frames_dir}")
        return {}

    # Map stem → label path (jaga-jaga jika jumlah tidak sama)
    label_files = list(labels_dir.glob("*.txt"))
    label_map = {lf.stem: lf for lf in label_files}

    # Initialize tracker per-video (tracker reset per video)
    tracker = build_bytetracker()

    # ID remapping: ByteTrack pakai ID global yang bisa jadi besar
    # Kita remap ke student_001, student_002, ... secara urut kemunculan
    id_remap: dict = {}        # bytetrack_id → student_id (int 1,2,3,...)
    next_student_id: int = 1

    crops_saved: dict = defaultdict(int)

    for frame_idx, frame_file in enumerate(frame_files):
        img = cv2.imread(str(frame_file))
        if img is None:
            log.warning(f"  Frame rusak: {frame_file}")
            continue
        img_h, img_w = img.shape[:2]

        # Load bbox dari label ground truth
        label_file = label_map.get(frame_file.stem)
        if label_file is None:
            # Tidak ada label untuk frame ini — beri tracker frame kosong
            # agar dia tetap update internal state (Kalman predict)
            detections = np.zeros((0, 6), dtype=np.float32)
        else:
            detections = load_yolo_labels(label_file, img_w, img_h)

        # Update ByteTracker
        try:
            tracks = run_tracker_on_detections(tracker, detections, (img_h, img_w))
        except Exception as e:
            log.error(f"  Tracker error di frame {frame_idx}: {e}")
            continue

        # Simpan crop per track
        for (bt_id, x1, y1, x2, y2) in tracks:
            # Remap ID ke student_XXX yang berurutan
            if bt_id not in id_remap:
                id_remap[bt_id] = next_student_id
                next_student_id += 1
            student_int = id_remap[bt_id]
            student_str = f"student_{student_int:03d}"

            # Clip bbox ke batas gambar
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(img_w - 1, int(x2)), min(img_h - 1, int(y2))

            if (x2 - x1) < Config.MIN_BBOX_SIZE or (y2 - y1) < Config.MIN_BBOX_SIZE:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Simpan
            student_dir = output_dir / student_str
            student_dir.mkdir(parents=True, exist_ok=True)
            crops_saved[student_int] += 1
            out_name = f"{student_str}_{crops_saved[student_int]:04d}.jpg"
            cv2.imwrite(str(student_dir / out_name), crop)

    total_students = len(crops_saved)
    total_crops    = sum(crops_saved.values())
    log.info(
        f"  Selesai: {total_students} siswa terdeteksi, "
        f"{total_crops} crops tersimpan"
    )
    return dict(crops_saved)


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
def run_phase0_v2(
    dataset_root: str,
    output_root:  str,
    splits: Optional[list] = None
):
    """
    Jalankan Fase 0 v2 untuk semua split.
    """
    if splits is None:
        splits = ["train", "valid", "test"]

    dataset_path = Path(dataset_root)
    output_path  = Path(output_root)

    overall_stats = {}

    for split in splits:
        videos_root = dataset_path / split / "videos"
        labels_root = dataset_path / split / "labels"

        if not videos_root.exists():
            log.warning(f"Split '{split}' tidak ditemukan, dilewati.")
            continue

        video_ids = sorted([d.name for d in videos_root.iterdir() if d.is_dir()])
        log.info(f"\n{'='*55}")
        log.info(f"SPLIT: {split.upper()} | {len(video_ids)} video")
        log.info(f"{'='*55}")

        for video_id in video_ids:
            frames_dir = videos_root / video_id
            labels_dir = labels_root / video_id
            out_dir    = output_path / split / video_id
            out_dir.mkdir(parents=True, exist_ok=True)

            if not labels_dir.exists():
                log.warning(f"  Label tidak ditemukan untuk video {video_id}, dilewati.")
                continue

            stats = process_video(
                frames_dir=frames_dir,
                labels_dir=labels_dir,
                output_dir=out_dir,
                split_name=split,
                video_id=video_id
            )
            overall_stats[f"{split}/{video_id}"] = stats

    # Ringkasan
    log.info(f"\n{'='*55}")
    log.info("RINGKASAN FASE 0 v2 (ByteTrack + GT Labels)")
    log.info(f"{'='*55}")
    total_st, total_cr = 0, 0
    for key, stats in overall_stats.items():
        if stats:
            n_st = len(stats)
            n_cr = sum(stats.values())
            total_st += n_st
            total_cr += n_cr
            log.info(f"  {key:<20s}: {n_st:3d} siswa | {n_cr:5d} crops")
    log.info(f"\n  TOTAL: {total_st} track siswa, {total_cr} crops")
    log.info(f"  Output: {output_path.resolve()}\n")

    return overall_stats


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 0 v2: ByteTrack + Bbox Ground Truth"
    )
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--output",  type=str, default="crop")
    parser.add_argument("--splits",  type=str, nargs="+",
                        default=["train", "valid", "test"])
    parser.add_argument("--track-buffer", type=int, default=Config.TRACK_BUFFER,
                        help="Buffer frame untuk track yang hilang sementara")
    parser.add_argument("--match-thresh", type=float, default=Config.MATCH_THRESH,
                        help="IoU threshold untuk matching")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    Config.TRACK_BUFFER = args.track_buffer
    Config.MATCH_THRESH = args.match_thresh

    run_phase0_v2(
        dataset_root=args.dataset,
        output_root=args.output,
        splits=args.splits
    )
