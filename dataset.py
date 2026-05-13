"""
dataset.py — PyTorch Dataset & DataLoader untuk Fase 2 (GRU)
=============================================================
Versi v3: Kompatibel dengan fitur 38-dim (YOLO Pose + Geometric + Temporal).

Setiap sampel:
  features : Tensor (SEQ_LEN=8, FEATURE_DIM=38)
  label    : 0 (not_cheating) atau 1 (cheating)

Layout 38 fitur per frame:
  [0:21]   Raw keypoints       : 7 kp × 3 (5 head + 2 shoulder)
  [21:24]  Geometric head pose : yaw, pitch, roll
  [24:26]  Head-body relation  : head_y_relative, head_size_ratio
  [26:28]  Visibility          : n_visible_norm, facing_back_flag
  [28:38]  Temporal velocity   : Δxy untuk 5 head keypoints

Anti-Leakage:
  - StandardScaler (opsional) di-fit HANYA pada train,
    lalu transform diterapkan ke valid dan test.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Literal

import torch
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────
# Konstanta — konsisten dengan feature_extractor_v3.py
# ─────────────────────────────────────────────────────────────────
SEQ_LEN     = 8
FEATURE_DIM = 38


# ─────────────────────────────────────────────────────────────────
# Label loading
# ─────────────────────────────────────────────────────────────────
def load_student_label(
    labels_dir: Path,
    student_npy: Path,
    labeling: Literal["any", "majority"] = "any"
) -> int:
    """
    Tentukan label binary untuk satu siswa berdasarkan anotasi YOLO.
    0 = cheating (class YOLO 0)
    1 = not_cheating (class YOLO 1)

    Strategi "any"      : cheating jika ada minimal 1 frame berlabel cheating
    Strategi "majority" : cheating jika >50% frame berlabel cheating
    """
    if not labels_dir.exists():
        return -1

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        return -1

    cheating_frames = 0
    total_frames    = 0

    for lf in label_files:
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                total_frames += 1
                if cls == 0:
                    cheating_frames += 1

    if total_frames == 0:
        return 0

    if labeling == "any":
        return 1 if cheating_frames > 0 else 0
    else:
        return 1 if (cheating_frames / total_frames) > 0.5 else 0


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────
class ExamCheatingDataset(Dataset):
    """
    Setiap sampel = 1 siswa dengan fitur head 38-dim.

    Penggunaan:
        train_ds = ExamCheatingDataset("features", "dataset", split="train")
        val_ds   = ExamCheatingDataset("features", "dataset", split="valid",
                                        scaler=train_ds.scaler)
    """

    def __init__(
        self,
        feature_root: str,
        dataset_root: str,
        split: str = "train",
        seq_len: int = SEQ_LEN,
        feature_dim: int = FEATURE_DIM,
        labeling: Literal["any", "majority"] = "any",
        use_scaler: bool = False,
        scaler=None,
    ):
        self.seq_len     = seq_len
        self.feature_dim = feature_dim
        self.split       = split
        self.use_scaler  = use_scaler

        feature_split_dir = Path(feature_root) / split
        labels_split_dir  = Path(dataset_root) / split / "labels"

        self.samples = []
        for video_dir in sorted(feature_split_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            labels_video_dir = labels_split_dir / video_id

            for npy_file in sorted(video_dir.glob("*.npy")):
                label = load_student_label(labels_video_dir, npy_file, labeling)
                self.samples.append((npy_file, label))

        self.scaler = None
        if use_scaler:
            if scaler is not None:
                self.scaler = scaler
            elif split == "train":
                self.scaler = self._fit_scaler()
            else:
                raise ValueError(
                    "use_scaler=True tapi split bukan 'train' dan scaler kosong."
                )

    def _fit_scaler(self):
        from sklearn.preprocessing import StandardScaler
        all_data = []
        for npy_path, _ in self.samples:
            arr = np.load(str(npy_path))
            all_data.append(arr)
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
                pad = np.zeros(
                    (self.seq_len - feat.shape[0], self.feature_dim),
                    dtype=np.float32
                )
                feat = np.concatenate([feat, pad], axis=0)

        if self.scaler is not None:
            feat = self.scaler.transform(feat).astype(np.float32)

        return (
            torch.from_numpy(feat),
            torch.tensor(label, dtype=torch.long)
        )

    def get_class_weights(self) -> torch.Tensor:
        labels = [s[1] for s in self.samples if s[1] >= 0]
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
    dataset_root: str,
    seq_len: int      = SEQ_LEN,
    feature_dim: int  = FEATURE_DIM,
    batch_size: int   = 32,
    num_workers: int  = 0,
    use_scaler: bool  = False,
    labeling: str     = "any",
    pin_memory: bool  = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = ExamCheatingDataset(
        feature_root=feature_root, dataset_root=dataset_root,
        split="train", seq_len=seq_len, feature_dim=feature_dim,
        labeling=labeling, use_scaler=use_scaler,
    )
    fitted_scaler = train_ds.scaler if use_scaler else None

    valid_ds = ExamCheatingDataset(
        feature_root=feature_root, dataset_root=dataset_root,
        split="valid", seq_len=seq_len, feature_dim=feature_dim,
        labeling=labeling, use_scaler=use_scaler, scaler=fitted_scaler,
    )
    test_ds = ExamCheatingDataset(
        feature_root=feature_root, dataset_root=dataset_root,
        split="test", seq_len=seq_len, feature_dim=feature_dim,
        labeling=labeling, use_scaler=use_scaler, scaler=fitted_scaler,
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)

    weights = train_ds.get_class_weights()
    print(f"\n[Dataset] Train  : {len(train_ds)} sampel")
    print(f"[Dataset] Valid  : {len(valid_ds)} sampel")
    print(f"[Dataset] Test   : {len(test_ds)} sampel")
    print(f"[Dataset] Class weights (train): not={weights[0]:.3f}, cheat={weights[1]:.3f}")
    print(f"[Dataset] Dim per batch: ({batch_size}, {seq_len}, {feature_dim})")

    return train_dl, valid_dl, test_dl
