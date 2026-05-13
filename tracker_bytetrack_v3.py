"""
Fase 0 v3 (FIXED): ByteTrack + Per-Student Label Assignment
=============================================================
Perbaikan dari v2: Selain crop, sekarang juga simpan label per-frame
                   untuk SETIAP siswa (bukan agregat per video).

Cara kerja label matching:
  1. Untuk setiap frame, baca semua bbox label (dengan class)
  2. ByteTrack hasilkan bbox track per student_id
  3. Untuk setiap track bbox, cari bbox label dengan IoU tertinggi
  4. Ambil class dari bbox label tersebut → ini label siswa di frame ini
  5. Simpan ke labels_per_student.json

Output tambahan:
  crop/<split>/<video_id>/labels_per_student.json
    {
      "student_001": [0, 0, 1, 1, 0, ...],   // class per frame
      "student_002": [1, 1, 1, 1, 1, ...],
      ...
    }
"""

import os
import re
import json
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
# Konfigurasi (sama seperti v2)
# ─────────────────────────────────────────────────────────────────
class Config:
    TRACK_HIGH_THRESH: float = 0.5
    TRACK_LOW_THRESH:  float = 0.1
    NEW_TRACK_THRESH:  float = 0.6
    TRACK_BUFFER:      int   = 90
    MATCH_THRESH:      float = 0.7
    GT_CONFIDENCE:     float = 0.99
    MIN_BBOX_SIZE:     int   = 10

    # IoU threshold untuk mencocokkan track bbox ke label bbox
    # Untuk dapat label yang benar
    LABEL_MATCH_IOU: float = 0.3


# ─────────────────────────────────────────────────────────────────
# Helper functions (sama seperti v2)
# ─────────────────────────────────────────────────────────────────
def natural_sort_key(path: Path) -> list:
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return max(0, x1), max(0, y1), min(img_w-1, x2), min(img_h-1, y2)


def compute_iou(boxA, boxB):
    """IoU antara dua box (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)


def load_yolo_labels_with_class(label_path: Path, img_w: int, img_h: int):
    """
    Baca file label YOLO. Return list [(x1,y1,x2,y2, original_class), ...]
    """
    if not label_path.exists():
        return []

    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
            if (x2-x1) < Config.MIN_BBOX_SIZE or (y2-y1) < Config.MIN_BBOX_SIZE:
                continue
            boxes.append((x1, y1, x2, y2, cls))
    return boxes


def to_detections_array(labels_with_class):
    """Convert [(x1,y1,x2,y2,cls), ...] ke np.array (N, 6) untuk ByteTrack."""
    if not labels_with_class:
        return np.zeros((0, 6), dtype=np.float32)
    arr = np.zeros((len(labels_with_class), 6), dtype=np.float32)
    for i, (x1, y1, x2, y2, _) in enumerate(labels_with_class):
        arr[i] = [x1, y1, x2, y2, Config.GT_CONFIDENCE, 0]
    return arr


# ─────────────────────────────────────────────────────────────────
# ByteTracker wrapper
# ─────────────────────────────────────────────────────────────────
def build_bytetracker():
    from ultralytics.trackers.byte_tracker import BYTETracker
    class TrackerArgs:
        track_high_thresh = Config.TRACK_HIGH_THRESH
        track_low_thresh  = Config.TRACK_LOW_THRESH
        new_track_thresh  = Config.NEW_TRACK_THRESH
        track_buffer      = Config.TRACK_BUFFER
        match_thresh      = Config.MATCH_THRESH
        fuse_score        = True
    return BYTETracker(TrackerArgs())


class DetectionResults:
    def __init__(self, xyxy, conf, cls):
        if len(xyxy) == 0:
            self.xyxy = np.zeros((0, 4), dtype=np.float32)
            self.conf = np.zeros(0, dtype=np.float32)
            self.cls  = np.zeros(0, dtype=np.float32)
            self.xywh = np.zeros((0, 4), dtype=np.float32)
        else:
            self.xyxy = xyxy.astype(np.float32)
            self.conf = conf.astype(np.float32)
            self.cls  = cls.astype(np.float32)
            x1, y1, x2, y2 = self.xyxy[:,0], self.xyxy[:,1], self.xyxy[:,2], self.xyxy[:,3]
            self.xywh = np.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], axis=1).astype(np.float32)

    def __len__(self): return len(self.conf)
    def __getitem__(self, idx):
        return DetectionResults(self.xyxy[idx], self.conf[idx], self.cls[idx])


def run_tracker(tracker, detections):
    if len(detections) == 0:
        fake = DetectionResults(np.zeros((0,4)), np.zeros(0), np.zeros(0))
    else:
        fake = DetectionResults(detections[:, :4], detections[:, 4], detections[:, 5])
    tracks = tracker.update(fake, img=None)
    output = []
    if len(tracks) > 0:
        for t in tracks:
            output.append((int(t[4]), t[0], t[1], t[2], t[3]))
    return output


# ─────────────────────────────────────────────────────────────────
# PROCESS VIDEO — dengan label per-siswa
# ─────────────────────────────────────────────────────────────────
def process_video(frames_dir, labels_dir, output_dir, split_name, video_id):
    log.info(f"Memproses [{split_name}] video {video_id}...")

    frame_files = sorted(
        [f for f in frames_dir.iterdir()
         if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=natural_sort_key
    )
    if not frame_files:
        log.warning(f"  Tidak ada frame")
        return {}

    label_files = list(labels_dir.glob("*.txt"))
    label_map = {lf.stem: lf for lf in label_files}

    tracker = build_bytetracker()
    id_remap = {}
    next_student_id = 1
    crops_saved = defaultdict(int)

    # ⭐ Label per siswa per frame (mapping student_id → list of class per crop)
    student_labels_per_frame = defaultdict(list)

    for frame_idx, frame_file in enumerate(frame_files):
        img = cv2.imread(str(frame_file))
        if img is None:
            log.warning(f"  Frame rusak: {frame_file}")
            continue
        img_h, img_w = img.shape[:2]

        # Baca labels dengan class info
        label_file = label_map.get(frame_file.stem)
        if label_file is None:
            labels_with_class = []
        else:
            labels_with_class = load_yolo_labels_with_class(label_file, img_w, img_h)

        # Konversi ke array untuk tracker
        detections = to_detections_array(labels_with_class)

        try:
            tracks = run_tracker(tracker, detections)
        except Exception as e:
            log.error(f"  Tracker error frame {frame_idx}: {e}")
            continue

        for (bt_id, x1, y1, x2, y2) in tracks:
            # Remap ID
            if bt_id not in id_remap:
                id_remap[bt_id] = next_student_id
                next_student_id += 1
            student_int = id_remap[bt_id]
            student_str = f"student_{student_int:03d}"

            # Clip bbox
            x1c, y1c = max(0, int(x1)), max(0, int(y1))
            x2c, y2c = min(img_w-1, int(x2)), min(img_h-1, int(y2))
            if (x2c-x1c) < Config.MIN_BBOX_SIZE or (y2c-y1c) < Config.MIN_BBOX_SIZE:
                continue
            crop = img[y1c:y2c, x1c:x2c]
            if crop.size == 0: continue

            # ⭐ CARI LABEL CLASS untuk track bbox ini
            # Match track bbox ke label bbox dengan IoU tertinggi
            track_bbox = (x1, y1, x2, y2)
            best_iou = 0.0
            best_class = -1  # -1 = tidak ada label match
            for lbl_x1, lbl_y1, lbl_x2, lbl_y2, lbl_cls in labels_with_class:
                iou = compute_iou(track_bbox, (lbl_x1, lbl_y1, lbl_x2, lbl_y2))
                if iou > best_iou:
                    best_iou = iou
                    best_class = lbl_cls

            # Hanya assign label kalau IoU cukup tinggi
            if best_iou >= Config.LABEL_MATCH_IOU:
                assigned_class = best_class  # 0=cheating, 1=not_cheating
            else:
                assigned_class = -1  # tidak ada label yang cocok

            # Simpan crop
            student_dir = output_dir / student_str
            student_dir.mkdir(parents=True, exist_ok=True)
            crops_saved[student_int] += 1
            out_name = f"{student_str}_{crops_saved[student_int]:04d}.jpg"
            cv2.imwrite(str(student_dir / out_name), crop)

            # ⭐ Simpan label class untuk crop ini
            student_labels_per_frame[student_str].append(int(assigned_class))

    # ⭐ Simpan label per-siswa per-frame ke JSON
    labels_json = {
        student_str: classes
        for student_str, classes in student_labels_per_frame.items()
    }
    labels_json_path = output_dir / "labels_per_student.json"
    with open(labels_json_path, "w") as f:
        json.dump(labels_json, f, indent=2)

    # Stats
    total_st = len(crops_saved)
    total_cr = sum(crops_saved.values())

    # Hitung distribusi label
    class_counts = {-1: 0, 0: 0, 1: 0}
    for classes in student_labels_per_frame.values():
        for c in classes:
            class_counts[c] = class_counts.get(c, 0) + 1

    log.info(
        f"  Selesai: {total_st} siswa, {total_cr} crops | "
        f"label dist: cheat(0)={class_counts.get(0,0)}, "
        f"not_cheat(1)={class_counts.get(1,0)}, "
        f"unmatched(-1)={class_counts.get(-1,0)}"
    )
    return dict(crops_saved)


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
def run_phase0_v3(dataset_root, output_root, splits=None):
    if splits is None:
        splits = ["train", "valid", "test"]

    dataset_path = Path(dataset_root)
    output_path  = Path(output_root)
    overall_stats = {}

    for split in splits:
        videos_root = dataset_path / split / "videos"
        labels_root = dataset_path / split / "labels"
        if not videos_root.exists():
            log.warning(f"Split '{split}' tidak ada, dilewati.")
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
                log.warning(f"  Label tidak ada untuk video {video_id}")
                continue

            stats = process_video(frames_dir, labels_dir, out_dir, split, video_id)
            overall_stats[f"{split}/{video_id}"] = stats

    log.info(f"\n{'='*55}")
    log.info("RINGKASAN FASE 0 v3")
    log.info(f"{'='*55}")
    log.info(f"  Output: {output_path.resolve()}")
    log.info(f"  Tiap folder video punya labels_per_student.json")
    log.info(f"  Pakai dataset.py v4 di Fase 1/2 untuk membacanya")
    log.info(f"{'='*55}\n")
    return overall_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output",  default="crop")
    parser.add_argument("--splits",  nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--track-buffer", type=int, default=Config.TRACK_BUFFER)
    parser.add_argument("--match-thresh", type=float, default=Config.MATCH_THRESH)
    parser.add_argument("--label-iou", type=float, default=Config.LABEL_MATCH_IOU)
    args = parser.parse_args()

    Config.TRACK_BUFFER    = args.track_buffer
    Config.MATCH_THRESH    = args.match_thresh
    Config.LABEL_MATCH_IOU = args.label_iou

    run_phase0_v3(args.dataset, args.output, args.splits)
