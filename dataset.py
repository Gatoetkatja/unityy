"""
dataset.py v4 — PyTorch Dataset dengan Label Per-Siswa yang BENAR
==================================================================
Membaca labels_per_student.json dari hasil Fase 0 v3,
bukan agregat dari file label YOLO.

Perubahan dari v3:
  - load_student_label tidak baca dari dataset/labels/ lagi
  - Sebagai gantinya baca dari crop/<split>/<video_id>/labels_per_student.json
  - Label per siswa berdasarkan tracking + IoU matching
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Literal
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────
SEQ_LEN     = 8     # ← sesuaikan dengan keputusan Fikar
FEATURE_DIM = 38


# ─────────────────────────────────────────────────────────────────
# Per-student label loading — INI YANG DIPERBAIKI
# ─────────────────────────────────────────────────────────────────
def load_per_student_labels(
    crop_video_dir: Path,
    student_id: str,
    labeling: Literal["any", "majority"] = "any"
) -> int:
    """
    Baca label per-siswa per-frame dari labels_per_student.json.

    Args:
        crop_video_dir : Path ke crop/<split>/<video_id>/
        student_id     : "student_001" misalnya
        labeling       : "any" = cheating jika ada minimal 1 frame cheating
                         "majority" = cheating jika >50% frame cheating

    Return:
        0 = cheating, 1 = not_cheating, -1 = tidak ada label
    """
    labels_json_path = crop_video_dir / "labels_per_student.json"
    if not labels_json_path.exists():
        return -1

    with open(labels_json_path, "r") as f:
        all_labels = json.load(f)

    student_labels = all_labels.get(student_id, [])
    if not student_labels:
        return -1

    # Filter out unmatched (-1) frames
    valid_labels = [c for c in student_labels if c >= 0]
    if not valid_labels:
        return -1

    # 0 = cheating, 1 = not_cheating di anotasi YOLO Fikar
    n_cheating = sum(1 for c in valid_labels if c == 0)
    n_total    = len(valid_labels)

    if labeling == "any":
        # Cheating jika ada minimal 1 frame berlabel cheating
        return 1 if n_cheating > 0 else 0
    else:  # majority
        return 1 if (n_cheating / n_total) > 0.5 else 0


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────
class ExamCheatingDataset(Dataset):
    """
    Setiap sampel = 1 siswa dengan label PER-SISWA yang benar
    (dari hasil tracking + IoU matching, bukan agregat video).

    Penggunaan:
        train_ds = ExamCheatingDataset(
            feature_root="features",
            crop_root="crop",          # ← BARU: butuh untuk baca labels
            split="train"
        )
    """

    def __init__(
        self,
        feature_root: str,
        crop_root: str,                 # ⚠️ BARU: butuh path ke crop/
        split: str = "train",
        seq_len: int = SEQ_LEN,
        feature_dim: int = FEATURE_DIM,
        labeling: Literal["any", "majority"] = "any",
        use_scaler: bool = False,
        scaler=None,
        verbose: bool = True,
    ):
        self.seq_len     = seq_len
        self.feature_dim = feature_dim
        self.split       = split
        self.use_scaler  = use_scaler

        feature_split_dir = Path(feature_root) / split
        crop_split_dir    = Path(crop_root)    / split

        # Kumpulkan sampel + label per-siswa
        self.samples = []  # list of (npy_path, label)
        for video_dir in sorted(feature_split_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            crop_video_dir = crop_split_dir / video_id

            for npy_file in sorted(video_dir.glob("*.npy")):
                student_id = npy_file.stem  # "student_001"
                label = load_per_student_labels(
                    crop_video_dir, student_id, labeling
                )
                self.samples.append((npy_file, label))

        # Filter out sampel tanpa label
        valid_samples = [s for s in self.samples if s[1] >= 0]
        n_dropped = len(self.samples) - len(valid_samples)
        self.samples = valid_samples

        if verbose:
            labels = [s[1] for s in self.samples]
            dist = Counter(labels)
            print(f"[Dataset/{split}] {len(self.samples)} sampel valid "
                  f"({n_dropped} di-drop karena tanpa label)")
            print(f"[Dataset/{split}] distribusi: "
                  f"not_cheat(0)={dist.get(0,0)}, cheat(1)={dist.get(1,0)}")

        # Scaler
        self.scaler = None
        if use_scaler:
            if scaler is not None:
                self.scaler = scaler
            elif split == "train":
                self.scaler = self._fit_scaler()
            else:
                raise ValueError("use_scaler=True tapi split bukan train, scaler kosong.")

    def _fit_scaler(self):
        from sklearn.preprocessing import StandardScaler
        all_data = []
        for npy_path, _ in self.samples:
            all_data.append(np.load(str(npy_path)))
        all_data = np.concatenate(all_data, axis=0)
        scaler = StandardScaler()
        scaler.fit(all_data)
        return scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        npy_path, label = self.samples[idx]
        feat = np.load(str(npy_path)).astype(np.float32)

        if feat.shape[0] != self.seq_len:
            if feat.shape[0] > self.seq_len:
                feat = feat[:self.seq_len]
            else:
                pad = np.zeros((self.seq_len - feat.shape[0], self.feature_dim),
                                dtype=np.float32)
                feat = np.concatenate([feat, pad], axis=0)

        if self.scaler is not None:
            feat = self.scaler.transform(feat).astype(np.float32)

        return (
            torch.from_numpy(feat),
            torch.tensor(label, dtype=torch.long)
        )

    def get_class_weights(self) -> torch.Tensor:
        labels = [s[1] for s in self.samples]
        n_total    = len(labels)
        n_cheating = sum(labels)
        n_not      = n_total - n_cheating
        if n_cheating == 0 or n_not == 0:
            return torch.ones(2)
        w_not      = n_total / (2 * n_not)
        w_cheating = n_total / (2 * n_cheating)
        return torch.tensor([w_not, w_cheating], dtype=torch.float32)


def build_dataloaders(
    feature_root: str,
    crop_root: str,                     # ⚠️ BARU
    seq_len: int      = SEQ_LEN,
    feature_dim: int  = FEATURE_DIM,
    batch_size: int   = 32,
    num_workers: int  = 0,
    use_scaler: bool  = False,
    labeling: str     = "any",
    pin_memory: bool  = True,
):
    train_ds = ExamCheatingDataset(
        feature_root=feature_root, crop_root=crop_root,
        split="train", seq_len=seq_len, feature_dim=feature_dim,
        labeling=labeling, use_scaler=use_scaler,
    )
    fitted_scaler = train_ds.scaler if use_scaler else None

    valid_ds = ExamCheatingDataset(
        feature_root=feature_root, crop_root=crop_root,
        split="valid", seq_len=seq_len, feature_dim=feature_dim,
        labeling=labeling, use_scaler=use_scaler, scaler=fitted_scaler,
    )
    test_ds = ExamCheatingDataset(
        feature_root=feature_root, crop_root=crop_root,
        split="test", seq_len=seq_len, feature_dim=feature_dim,
        labeling=labeling, use_scaler=use_scaler, scaler=fitted_scaler,
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)

    weights = train_ds.get_class_weights()
    print(f"\n[Dataset] Class weights (train): not={weights[0]:.3f}, cheat={weights[1]:.3f}")
    print(f"[Dataset] Dim per batch: ({batch_size}, {seq_len}, {feature_dim})")

    return train_dl, valid_dl, test_dl
