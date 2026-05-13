"""
Fase 1 v3: Head Feature Extraction — YOLO Pose + Geometric + Temporal
======================================================================
Dirancang khusus untuk:
  - CCTV resolusi rendah (640x640 full frame, crop kepala ~60-120 px)
  - Siswi berkerudung yang menunduk (mata/dahi/telinga tertutup)
  - Siswa menghadap belakang (semua head keypoint hilang)

Strategi:
  1. YOLO11-pose deteksi 17 keypoint COCO; kita pakai 5 head + 2 shoulder
  2. Hitung head pose secara GEOMETRIS (tidak butuh 3D model wajah)
     → robust ke wajah parsial / tertutup kerudung
  3. Pakai BAHU sebagai reference frame yang stabil (jarang tertutup saat duduk)
  4. Tambahkan turunan temporal (velocity) → sinyal kuat untuk GRU

Input  : crop/<split>/<video_id>/student_XXX/*.jpg
Output : features/<split>/<video_id>/student_XXX.npy
         shape (60, 38)

Layout 38 fitur per frame:
  [0:21]   Raw keypoints       : 7 keypoint × 3 (x_norm, y_norm, conf)
                                  - 5 head (nose, eyes, ears) + 2 shoulders
  [21:24]  Geometric head pose : yaw_proxy, pitch_proxy, roll_proxy
  [24:26]  Head-body relation  : head_y_relative, head_size_ratio
  [26:28]  Visibility          : n_head_visible, head_facing_back_flag
  [28:38]  Temporal velocity   : Δ(x,y) untuk 5 head keypoints
"""

import os
import re
import json
import math
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import cv2

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    log.error("ultralytics belum terinstall. pip install ultralytics")


# ─────────────────────────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────────────────────────
class Config:
    SEQ_LEN: int = 8
    FEATURE_DIM: int = 38

    # Model YOLO pose
    YOLO_MODEL: str = "yolo11n-pose.pt"

    # Confidence threshold untuk dianggap "terlihat"
    KP_VISIBILITY_THRESH: float = 0.3

    # Confidence threshold deteksi pose secara keseluruhan
    POSE_CONF: float = 0.25

    # Indeks keypoint COCO yang dipakai
    # Format COCO 17-kp: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
    # 5=left_shoulder, 6=right_shoulder, ...
    KP_NOSE     = 0
    KP_LEYE     = 1
    KP_REYE     = 2
    KP_LEAR     = 3
    KP_REAR     = 4
    KP_LSHOULD  = 5
    KP_RSHOULD  = 6

    # Index urutan kita di vektor fitur
    HEAD_KP_INDICES     = [KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR]
    SHOULDER_KP_INDICES = [KP_LSHOULD, KP_RSHOULD]


# ─────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────
def natural_sort_key(path: Path) -> list:
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# ─────────────────────────────────────────────────────────────────
# Geometric Head Pose dari Keypoints
# ─────────────────────────────────────────────────────────────────
def compute_geometric_pose(
    keypoints: np.ndarray,  # shape (17, 3): x_norm, y_norm, conf
    visibility_thresh: float = Config.KP_VISIBILITY_THRESH
) -> Tuple[float, float, float, float]:
    """
    Hitung head pose proxy dari relasi geometris keypoint.
    TIDAK butuh model 3D wajah, jadi robust ke wajah parsial/tertutup.

    Return: (yaw, pitch, roll, n_head_visible)
        yaw   ∈ [-1, 1] : negatif=hadap kanan, positif=hadap kiri, 0=frontal
        pitch ∈ [-1, 1] : negatif=menunduk, positif=mendongak
        roll  ∈ [-1, 1] : kemiringan kepala
        n_head_visible: jumlah head keypoint yang confident (0-5)

    Strategi:
      - Yaw   : dari asimetri jarak nose ke eye/ear kiri vs kanan
                (jika hidung dekat ke mata kiri → menghadap kiri)
      - Pitch : dari rasio (Y_nose) vs (Y_shoulder_mid)
                (semakin dekat ke bahu = makin menunduk)
                FALLBACK: jika nose hilang tapi shoulder ada,
                pitch tetap bisa dihitung dari Y_head_proxy
      - Roll  : dari kemiringan garis mata atau bahu
    """
    kp = keypoints
    yaw = pitch = roll = 0.0

    nose = kp[Config.KP_NOSE]
    leye = kp[Config.KP_LEYE]
    reye = kp[Config.KP_REYE]
    lear = kp[Config.KP_LEAR]
    rear = kp[Config.KP_REAR]
    lsh  = kp[Config.KP_LSHOULD]
    rsh  = kp[Config.KP_RSHOULD]

    # Hitung head keypoints yang visible
    head_visibility = np.array([
        nose[2] >= visibility_thresh,
        leye[2] >= visibility_thresh,
        reye[2] >= visibility_thresh,
        lear[2] >= visibility_thresh,
        rear[2] >= visibility_thresh,
    ], dtype=bool)
    n_head_visible = int(head_visibility.sum())

    # Bahu sebagai reference yang stabil
    sh_l_visible = lsh[2] >= visibility_thresh
    sh_r_visible = rsh[2] >= visibility_thresh
    both_shoulders = sh_l_visible and sh_r_visible

    # ─── YAW ──────────────────────────────────────────────────────
    # Pakai asimetri eye-to-nose atau ear-to-nose
    if nose[2] >= visibility_thresh:
        if leye[2] >= visibility_thresh and reye[2] >= visibility_thresh:
            # Jarak hidung ke setiap mata di sumbu X
            d_left  = abs(nose[0] - leye[0])
            d_right = abs(nose[0] - reye[0])
            denom = d_left + d_right + 1e-6
            # Positif jika hidung lebih dekat ke mata kanan (= hadap kanan dari POV kamera)
            yaw = (d_left - d_right) / denom
        elif lear[2] >= visibility_thresh and rear[2] >= visibility_thresh:
            # Fallback: pakai telinga (kalau pakai kerudung biasanya keduanya hilang)
            d_left  = abs(nose[0] - lear[0])
            d_right = abs(nose[0] - rear[0])
            denom = d_left + d_right + 1e-6
            yaw = (d_left - d_right) / denom
        # else: yaw tetap 0 (tidak cukup info)

    # ─── PITCH ────────────────────────────────────────────────────
    # Strategi 1: nose & shoulder visible → rasio Y_nose vs Y_shoulder
    # Kalibrasi: normal head-frontal, nose berada ~1.0 × shoulder_width di atas bahu.
    # Saat menunduk, nose mendekati bahu → ratio mendekati 0.
    # Saat mendongak, nose menjauh ke atas → ratio > 1.5.
    if nose[2] >= visibility_thresh and both_shoulders:
        y_sh_mid = (lsh[1] + rsh[1]) / 2
        sh_width = abs(lsh[0] - rsh[0]) + 1e-6
        nose_above_sh = (y_sh_mid - nose[1]) / sh_width
        # Map: ratio 1.0 → pitch 0 (normal), 0 → -1 (menunduk), 2 → +1 (mendongak)
        pitch = np.clip(nose_above_sh - 1.0, -1.0, 1.0)
    elif both_shoulders and n_head_visible > 0:
        # Fallback: pakai pusat head keypoints yang visible
        visible_y = []
        for kp_visible, kp_data in zip(head_visibility,
                                        [nose, leye, reye, lear, rear]):
            if kp_visible:
                visible_y.append(kp_data[1])
        y_head_avg = np.mean(visible_y)
        y_sh_mid = (lsh[1] + rsh[1]) / 2
        sh_width = abs(lsh[0] - rsh[0]) + 1e-6
        head_above_sh = (y_sh_mid - y_head_avg) / sh_width
        pitch = np.clip(head_above_sh - 1.0, -1.0, 1.0)

    # ─── ROLL ─────────────────────────────────────────────────────
    # Prioritas: eyes line, lalu ears line, lalu shoulders line
    # PENTING: pakai abs(dx) agar atan2 menghitung sudut kemiringan thd horizontal,
    # bukan sudut absolut (yang bisa = π saat dx negatif).
    if leye[2] >= visibility_thresh and reye[2] >= visibility_thresh:
        dy = leye[1] - reye[1]
        dx = abs(leye[0] - reye[0]) + 1e-6
        roll_angle = math.atan2(dy, dx)
        roll = np.clip(roll_angle / (math.pi / 4), -1.0, 1.0)
    elif lear[2] >= visibility_thresh and rear[2] >= visibility_thresh:
        dy = lear[1] - rear[1]
        dx = abs(lear[0] - rear[0]) + 1e-6
        roll_angle = math.atan2(dy, dx)
        roll = np.clip(roll_angle / (math.pi / 4), -1.0, 1.0)
    elif both_shoulders:
        dy = lsh[1] - rsh[1]
        dx = abs(lsh[0] - rsh[0]) + 1e-6
        roll_angle = math.atan2(dy, dx)
        roll = np.clip(roll_angle / (math.pi / 4), -1.0, 1.0) * 0.5

    return yaw, pitch, roll, n_head_visible


# ─────────────────────────────────────────────────────────────────
# Head-Body Relation
# ─────────────────────────────────────────────────────────────────
def compute_head_body_relation(
    keypoints: np.ndarray,
    visibility_thresh: float = Config.KP_VISIBILITY_THRESH
) -> Tuple[float, float]:
    """
    Hitung:
      head_y_relative : posisi vertikal kepala relatif ke bahu
                        (1.0 = jauh di atas, 0.0 = sejajar bahu, -1.0 = di bawah)
      head_size_ratio : ukuran "spread" keypoint kepala relatif ke jarak bahu
                        (kepala yang menunduk biasanya tampak lebih kecil)
    """
    kp = keypoints

    # Kumpulkan head keypoints yang visible
    head_kp = []
    for idx in Config.HEAD_KP_INDICES:
        if kp[idx, 2] >= visibility_thresh:
            head_kp.append(kp[idx, :2])

    lsh = kp[Config.KP_LSHOULD]
    rsh = kp[Config.KP_RSHOULD]
    both_sh = (lsh[2] >= visibility_thresh and rsh[2] >= visibility_thresh)

    head_y_rel = 0.0
    head_size = 0.0

    if both_sh:
        sh_width = max(abs(lsh[0] - rsh[0]), 1e-3)
        y_sh = (lsh[1] + rsh[1]) / 2

        if head_kp:
            head_arr = np.array(head_kp)
            y_head = float(head_arr[:, 1].mean())
            # head_y_relative: positif jika di atas bahu, dinormalisasi shoulder width
            head_y_rel = np.clip((y_sh - y_head) / sh_width, -2.0, 2.0) / 2.0  # → [-1,1]

            # Spread = jangkauan keypoint head
            if len(head_kp) >= 2:
                xs = head_arr[:, 0]
                ys = head_arr[:, 1]
                spread = math.sqrt((xs.max() - xs.min())**2 + (ys.max() - ys.min())**2)
                head_size = np.clip(spread / sh_width, 0.0, 2.0) / 2.0  # → [0,1]

    return float(head_y_rel), float(head_size)


# ─────────────────────────────────────────────────────────────────
# Extract Pose dari satu crop
# ─────────────────────────────────────────────────────────────────
def extract_pose_keypoints(
    model: "YOLO",
    img: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Jalankan YOLO Pose pada satu crop.
    Return: array (17, 3) [x_norm, y_norm, conf] atau None jika tidak terdeteksi.
    """
    if img is None or img.size == 0:
        return None

    results = model(img, conf=Config.POSE_CONF, verbose=False, half=False)
    if not results:
        return None

    result = results[0]
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None

    kp_data = result.keypoints.data
    boxes = result.boxes

    if len(kp_data) == 1:
        kp = kp_data[0]
    else:
        # Ambil orang dengan box confidence tertinggi (di dalam crop seharusnya 1 orang)
        best_idx = int(boxes.conf.argmax())
        kp = kp_data[best_idx]

    kp_np = kp.cpu().numpy().astype(np.float32)  # (17, 3)

    # Normalisasi koordinat terhadap ukuran crop
    img_h, img_w = result.orig_shape
    kp_np[:, 0] /= max(img_w, 1)
    kp_np[:, 1] /= max(img_h, 1)
    kp_np[:, :2] = np.clip(kp_np[:, :2], 0.0, 1.0)

    return kp_np  # (17, 3)


# ─────────────────────────────────────────────────────────────────
# Bangun feature vector 38-dim dari keypoints
# ─────────────────────────────────────────────────────────────────
def build_feature_vector(
    keypoints: Optional[np.ndarray],
    prev_head_kp: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Bangun feature vector 38-dim dari (17, 3) keypoint YOLO.

    Args:
        keypoints     : (17, 3) atau None
        prev_head_kp  : (5, 2) head kp dari frame sebelumnya, untuk velocity

    Return:
        feat              : np.ndarray (38,)
        current_head_kp   : (5, 2) untuk dipakai sebagai prev di frame berikutnya
        is_valid          : True jika YOLO mendeteksi pose
    """
    feat = np.zeros(Config.FEATURE_DIM, dtype=np.float32)

    if keypoints is None:
        # YOLO tidak deteksi sama sekali → "facing back" / occlusion total
        feat[27] = 1.0  # head_facing_back_flag
        return feat, np.zeros((5, 2), dtype=np.float32), False

    # ── [0:21] Raw keypoints (7 keypoint × 3) ──────────────────
    selected_kp_idx = Config.HEAD_KP_INDICES + Config.SHOULDER_KP_INDICES
    for i, kp_idx in enumerate(selected_kp_idx):
        feat[i*3 + 0] = keypoints[kp_idx, 0]
        feat[i*3 + 1] = keypoints[kp_idx, 1]
        feat[i*3 + 2] = keypoints[kp_idx, 2]

    # ── [21:24] Geometric head pose ────────────────────────────
    yaw, pitch, roll, n_visible = compute_geometric_pose(keypoints)
    feat[21] = yaw
    feat[22] = pitch
    feat[23] = roll

    # ── [24:26] Head-body relation ─────────────────────────────
    h_y_rel, h_size = compute_head_body_relation(keypoints)
    feat[24] = h_y_rel
    feat[25] = h_size

    # ── [26:28] Visibility flags ───────────────────────────────
    feat[26] = n_visible / 5.0   # normalize ke [0,1]
    # head_facing_back = 1 jika 0 atau 1 head kp visible (kepala menghadap belakang)
    feat[27] = 1.0 if n_visible <= 1 else 0.0

    # ── [28:38] Temporal velocity (5 head kp × Δx, Δy) ─────────
    current_head_kp = np.zeros((5, 2), dtype=np.float32)
    for i, kp_idx in enumerate(Config.HEAD_KP_INDICES):
        current_head_kp[i, 0] = keypoints[kp_idx, 0]
        current_head_kp[i, 1] = keypoints[kp_idx, 1]

    if prev_head_kp is not None:
        delta = current_head_kp - prev_head_kp  # (5, 2)
        # Clamp velocity ke range wajar (frame-to-frame gerakan tidak ekstrem)
        delta = np.clip(delta, -0.5, 0.5)
        feat[28:38] = delta.flatten()
    # Else: velocity = 0 (frame pertama)

    return feat, current_head_kp, True


# ─────────────────────────────────────────────────────────────────
# Interpolate gaps (jaga visibility flags tetap apa adanya)
# ─────────────────────────────────────────────────────────────────
def interpolate_gaps(
    sequence: np.ndarray,
    valid_mask: np.ndarray,
    preserve_indices: Optional[list] = None
) -> np.ndarray:
    """
    Interpolasi linier untuk frame invalid.
    Kolom di preserve_indices TIDAK diinterpolasi (tetap nilai aslinya).
    """
    if preserve_indices is None:
        preserve_indices = [26, 27]  # visibility flags

    T, D = sequence.shape
    filled = sequence.copy()

    invalid_idx = np.where(~valid_mask)[0]
    valid_idx   = np.where(valid_mask)[0]

    if len(invalid_idx) == 0:
        return filled
    if len(valid_idx) == 0:
        return np.zeros_like(sequence)

    for idx in invalid_idx:
        before = valid_idx[valid_idx < idx]
        after  = valid_idx[valid_idx > idx]

        if len(before) == 0:
            interp = filled[after[0]].copy()
        elif len(after) == 0:
            interp = filled[before[-1]].copy()
        else:
            l, r = before[-1], after[0]
            alpha = (idx - l) / (r - l)
            interp = ((1 - alpha) * filled[l] + alpha * filled[r]).astype(np.float32)

        # Jaga visibility flags supaya GRU tahu frame ini hasil interpolasi
        original_row = sequence[idx]
        for col in preserve_indices:
            interp[col] = original_row[col]

        # Velocity (28:38) di-set 0 untuk frame interpolasi
        # karena velocity dari frame interpolasi tidak bermakna
        interp[28:38] = 0.0

        filled[idx] = interp

    return filled


def pad_or_truncate(seq: np.ndarray, seq_len: int) -> np.ndarray:
    T, D = seq.shape
    if T >= seq_len:
        return seq[:seq_len]
    pad = np.zeros((seq_len - T, D), dtype=np.float32)
    return np.concatenate([seq, pad], axis=0)


# ─────────────────────────────────────────────────────────────────
# Process satu siswa
# ─────────────────────────────────────────────────────────────────
def process_student(
    model: "YOLO",
    student_dir: Path,
    seq_len: int = Config.SEQ_LEN
) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Proses semua crop di folder student.
    Return (feature_array (seq_len, 38), stats).
    """
    crop_files = sorted(
        [f for f in student_dir.iterdir()
         if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=natural_sort_key
    )

    if not crop_files:
        return None

    raw_seq = []
    valid_mask = []
    prev_head_kp = None

    for crop_path in crop_files:
        img = cv2.imread(str(crop_path))
        kp = extract_pose_keypoints(model, img)
        feat, cur_head, is_valid = build_feature_vector(kp, prev_head_kp)

        raw_seq.append(feat)
        valid_mask.append(is_valid)

        # Update prev_head_kp hanya jika frame valid
        if is_valid:
            prev_head_kp = cur_head
        # else: pertahankan prev_head_kp lama agar velocity tetap masuk akal saat resume

    raw_seq    = np.stack(raw_seq, axis=0)
    valid_mask = np.array(valid_mask, dtype=bool)

    interp_seq = interpolate_gaps(raw_seq, valid_mask)
    final_seq  = pad_or_truncate(interp_seq, seq_len)

    stats = {
        "n_frames":          int(len(crop_files)),
        "n_valid":           int(valid_mask.sum()),
        "valid_ratio":       float(valid_mask.mean()),
        "avg_head_visible":  float(raw_seq[valid_mask, 26].mean() * 5)
                              if valid_mask.any() else 0.0,
        "facing_back_ratio": float(raw_seq[:, 27].mean()),
    }
    return final_seq.astype(np.float32), stats


# ─────────────────────────────────────────────────────────────────
# Pipeline per Split
# ─────────────────────────────────────────────────────────────────
def process_split(
    model: "YOLO",
    crop_split_dir: Path,
    feature_split_dir: Path,
    seq_len: int,
    overwrite: bool
) -> dict:
    stats_all = {}

    video_dirs = sorted([d for d in crop_split_dir.iterdir() if d.is_dir()])
    log.info(f"  {len(video_dirs)} video ditemukan di {crop_split_dir.name}/")

    for video_dir in video_dirs:
        video_id = video_dir.name
        out_video_dir = feature_split_dir / video_id
        out_video_dir.mkdir(parents=True, exist_ok=True)

        student_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])
        log.info(f"  Video {video_id}: {len(student_dirs)} siswa")
        video_stats = {}

        for student_dir in student_dirs:
            student_id = student_dir.name
            out_path = out_video_dir / f"{student_id}.npy"

            if out_path.exists() and not overwrite:
                log.debug(f"    Skip (sudah ada): {out_path.name}")
                continue

            result = process_student(model, student_dir, seq_len)
            if result is None:
                log.warning(f"    Gagal: {student_id} (tidak ada crop)")
                continue

            feat_array, stats = result
            np.save(str(out_path), feat_array)

            video_stats[student_id] = stats
            log.info(
                f"    ✓ {student_id}: {stats['n_frames']:>3d} crops "
                f"| pose detected: {stats['valid_ratio']:.0%} "
                f"| avg head kp: {stats['avg_head_visible']:.1f}/5 "
                f"| facing back: {stats['facing_back_ratio']:.0%}"
            )

        stats_all[video_id] = video_stats

    return stats_all


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
def run_phase1_v3(
    crop_root: str,
    feature_root: str,
    model_path: str = Config.YOLO_MODEL,
    seq_len: int = Config.SEQ_LEN,
    splits: list = None,
    overwrite: bool = False
):
    if not _YOLO_AVAILABLE:
        raise RuntimeError("ultralytics tidak terinstall. pip install ultralytics")

    if splits is None:
        splits = ["train", "valid", "test"]

    crop_path    = Path(crop_root)
    feature_path = Path(feature_root)

    log.info(f"Memuat model YOLO Pose: {model_path}")
    model = YOLO(model_path)

    all_stats = {}
    for split in splits:
        crop_split    = crop_path / split
        feature_split = feature_path / split

        if not crop_split.exists():
            log.warning(f"Split '{split}' tidak ditemukan, dilewati.")
            continue

        log.info(f"\n{'='*55}")
        log.info(f"SPLIT: {split.upper()}")
        log.info(f"{'='*55}")

        feature_split.mkdir(parents=True, exist_ok=True)
        all_stats[split] = process_split(
            model, crop_split, feature_split, seq_len, overwrite
        )

    # Simpan stats
    stats_path = feature_path / "extraction_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    # Ringkasan
    log.info(f"\n{'='*55}")
    log.info("RINGKASAN FASE 1 v3 — Head Feature Geometric")
    log.info(f"{'='*55}")
    log.info(f"Output per siswa: [{seq_len}, {Config.FEATURE_DIM}]")
    log.info("")
    log.info("Layout 38 fitur per frame:")
    log.info("  [0:21]   Raw kp        : 7 kp × 3 (5 head + 2 shoulder)")
    log.info("  [21:24]  Geom. pose    : yaw, pitch, roll")
    log.info("  [24:26]  Head-body     : y_relative, size_ratio")
    log.info("  [26:28]  Visibility    : n_visible_norm, facing_back_flag")
    log.info("  [28:38]  Velocity      : Δxy untuk 5 head kp")
    log.info("")

    for split, videos in all_stats.items():
        n_st = sum(len(v) for v in videos.values())
        if n_st == 0: continue
        avg_valid = np.mean([
            s["valid_ratio"] for v in videos.values() for s in v.values()
        ])
        avg_back  = np.mean([
            s["facing_back_ratio"] for v in videos.values() for s in v.values()
        ])
        log.info(f"  {split:8s}: {n_st} siswa | pose detect avg: {avg_valid:.1%} | facing back: {avg_back:.1%}")
    log.info(f"{'='*55}\n")

    return all_stats


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 1 v3: Head Feature (YOLO Pose + Geometric + Temporal)"
    )
    parser.add_argument("--crop-dir",    default="crop")
    parser.add_argument("--feature-dir", default="features")
    parser.add_argument("--model",       default=Config.YOLO_MODEL)
    parser.add_argument("--seq-len",     type=int, default=Config.SEQ_LEN)
    parser.add_argument("--splits",      nargs="+",
                        default=["train", "valid", "test"])
    parser.add_argument("--overwrite",   action="store_true")
    parser.add_argument("--verbose",     action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_phase1_v3(
        crop_root=args.crop_dir,
        feature_root=args.feature_dir,
        model_path=args.model,
        seq_len=args.seq_len,
        splits=args.splits,
        overwrite=args.overwrite,
    )
