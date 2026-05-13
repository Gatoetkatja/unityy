"""
train.py — Pipeline Training GRU untuk Deteksi Kecurangan Ujian
================================================================
Menggabungkan: model.py + dataset.py → training loop → model terbaik .pth

Fitur:
  - BCEWithLogitsLoss dengan pos_weight (class imbalance)
  - WeightedRandomSampler (opsional, untuk oversampling minority)
  - Early Stopping berbasis validation loss
  - Scheduler ReduceLROnPlateau
  - Simpan model terbaik otomatis
  - Visualisasi loss & accuracy dengan matplotlib
  - Reproducibility via random seed
"""

import os
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (aman di server)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import lokal
from model import CheatingGRU
from dataset import build_dataloaders, ExamCheatingDataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Konfigurasi Training (semua hyperparameter di satu tempat)
# ─────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Path
    feature_root:  str = "features"
    dataset_root:  str = "dataset"
    output_dir:    str = "output"

    # Dataset
    seq_len:       int   = 8
    labeling:      str   = "any"   # "any" | "majority"
    use_scaler:    bool  = False

    # Model
    hidden_dim:    int   = 128
    num_layers:    int   = 2
    fc_dim:        int   = 64
    dropout:       float = 0.3
    bidirectional: bool  = False
    use_attention: bool  = True

    # Training
    epochs:        int   = 50
    batch_size:    int   = 32
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-4
    num_workers:   int   = 0

    # Class imbalance
    use_pos_weight:       bool  = True   # BCEWithLogitsLoss pos_weight
    use_weighted_sampler: bool  = False  # WeightedRandomSampler di train loader

    # Early Stopping
    patience:      int   = 10
    min_delta:     float = 1e-4  # Perbaikan minimum agar dianggap "improved"

    # LR Scheduler
    lr_factor:     float = 0.5
    lr_patience:   int   = 5
    lr_min:        float = 1e-6

    # Lainnya
    seed:          int   = 42
    device:        str   = "auto"  # "auto" | "cpu" | "cuda" | "mps"


# ─────────────────────────────────────────────────────────────────
# Utilitas
# ─────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


# ─────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Hentikan training jika val_loss tidak membaik selama `patience` epoch.
    Simpan state model terbaik di memori (bukan hanya path file).
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float("inf")
        self.best_state = None    # state_dict model terbaik
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Return True jika val_loss membaik (model terbaik diupdate).
        Set self.should_stop = True jika harus berhenti.
        """
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss  = val_loss
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def restore_best(self, model: nn.Module):
        """Kembalikan bobot model terbaik."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            log.info("Model terbaik dipulihkan dari early stopping.")


# ─────────────────────────────────────────────────────────────────
# Metrics Helper
# ─────────────────────────────────────────────────────────────────
def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Hitung binary accuracy dari logits (sebelum sigmoid)."""
    preds   = (torch.sigmoid(logits) >= 0.5).long().squeeze(1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


# ─────────────────────────────────────────────────────────────────
# Training & Validation Step
# ─────────────────────────────────────────────────────────────────
def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scaler_amp: Optional[torch.cuda.amp.GradScaler] = None
) -> Tuple[float, float]:
    """
    Satu epoch training.
    Return: (avg_loss, avg_accuracy)
    """
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for features, labels in loader:
        features = features.to(device)             # (B, 60, 51)
        labels   = labels.float().to(device)       # (B,)
        labels   = labels.unsqueeze(1)             # (B, 1) ← BCEWithLogitsLoss

        optimizer.zero_grad()

        if scaler_amp is not None:
            # Mixed precision (GPU only)
            with torch.cuda.amp.autocast():
                logits = model(features)           # (B, 1)
                loss   = criterion(logits, labels)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            logits = model(features)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        acc = compute_accuracy(logits.detach(), labels.squeeze(1).long())
        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def validate_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device
) -> Tuple[float, float]:
    """
    Satu epoch validasi.
    Return: (avg_loss, avg_accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for features, labels in loader:
        features = features.to(device)
        labels   = labels.float().to(device).unsqueeze(1)

        logits = model(features)
        loss   = criterion(logits, labels)

        acc = compute_accuracy(logits, labels.squeeze(1).long())
        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ─────────────────────────────────────────────────────────────────
# Visualisasi
# ─────────────────────────────────────────────────────────────────
def plot_training_history(history: dict, output_path: str):
    """
    Buat grafik 2×2:
    - Training Loss vs Validation Loss
    - Training Accuracy vs Validation Accuracy
    - Learning Rate per epoch
    - (opsional) epoch best marker
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Training History — Cheating Detection GRU", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Plot 1: Loss ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], label="Train Loss",
             color="#2196F3", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",
             color="#F44336", linewidth=2, linestyle="--")
    if history.get("best_epoch"):
        ax1.axvline(history["best_epoch"], color="green",
                    linestyle=":", alpha=0.7, label=f"Best (ep {history['best_epoch']})")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Accuracy ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history["train_acc"], label="Train Acc",
             color="#2196F3", linewidth=2)
    ax2.plot(epochs, history["val_acc"],   label="Val Acc",
             color="#F44336", linewidth=2, linestyle="--")
    if history.get("best_epoch"):
        ax2.axvline(history["best_epoch"], color="green",
                    linestyle=":", alpha=0.7)
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Learning Rate ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(epochs, history["lr"], color="#9C27B0", linewidth=2)
    ax3.set_title("Learning Rate per Epoch")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("LR (log scale)")
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Val Loss (zoom) + early stop marker ────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, history["val_loss"], color="#FF9800", linewidth=2)
    ax4.fill_between(epochs, history["val_loss"], alpha=0.15, color="#FF9800")
    if history.get("best_epoch"):
        best_ep  = history["best_epoch"]
        best_val = history["val_loss"][best_ep - 1]
        ax4.scatter([best_ep], [best_val], color="green",
                    zorder=5, s=80, label=f"Best Val Loss: {best_val:.4f}")
        ax4.legend()
    ax4.set_title("Validation Loss (detail)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Val Loss")
    ax4.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Grafik training tersimpan: {output_path}")


# ─────────────────────────────────────────────────────────────────
# Fungsi build pos_weight dari class distribution
# ─────────────────────────────────────────────────────────────────
def compute_pos_weight(dataset: ExamCheatingDataset) -> torch.Tensor:
    """
    Hitung pos_weight untuk BCEWithLogitsLoss.
    pos_weight = n_negative / n_positive

    Jika n_negative >> n_positive (lebih banyak not_cheating),
    maka pos_weight > 1 sehingga loss untuk kelas cheating diberi bobot lebih.
    """
    labels = [s[1] for s in dataset.samples if s[1] >= 0]
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos

    if n_pos == 0:
        log.warning("Tidak ada sampel positif (cheating) di training set!")
        return torch.tensor([1.0])

    pw = n_neg / n_pos
    log.info(f"pos_weight = {pw:.3f}  (n_neg={n_neg}, n_pos={n_pos})")
    return torch.tensor([pw], dtype=torch.float32)


def build_weighted_sampler(dataset: ExamCheatingDataset) -> WeightedRandomSampler:
    """
    Buat WeightedRandomSampler agar tiap batch memiliki proporsi
    cheating/not_cheating yang seimbang.
    """
    labels  = [s[1] for s in dataset.samples]
    n_pos   = sum(l == 1 for l in labels)
    n_neg   = sum(l == 0 for l in labels)
    n_total = len(labels)

    if n_pos == 0 or n_neg == 0:
        log.warning("Salah satu kelas kosong, WeightedSampler tidak efektif.")
        weights = [1.0] * n_total
    else:
        w_pos = n_total / (2 * n_pos)
        w_neg = n_total / (2 * n_neg)
        weights = [w_pos if l == 1 else w_neg for l in labels]

    sampler = WeightedRandomSampler(
        weights     = torch.tensor(weights, dtype=torch.float64),
        num_samples = n_total,
        replacement = True
    )
    return sampler


# ─────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────
def train(cfg: TrainConfig):
    # ── Setup ────────────────────────────────────────────────────
    set_seed(cfg.seed)
    device    = get_device(cfg.device)
    out_dir   = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Device   : {device}")
    log.info(f"Output   : {out_dir.resolve()}")

    # ── DataLoaders ──────────────────────────────────────────────
    log.info("Memuat dataset...")
    train_ds = ExamCheatingDataset(
        feature_root = cfg.feature_root,
        dataset_root = cfg.dataset_root,
        split        = "train",
        seq_len      = cfg.seq_len,
        labeling     = cfg.labeling,
        use_scaler   = cfg.use_scaler
    )
    val_ds = ExamCheatingDataset(
        feature_root = cfg.feature_root,
        dataset_root = cfg.dataset_root,
        split        = "valid",
        seq_len      = cfg.seq_len,
        labeling     = cfg.labeling,
        use_scaler   = cfg.use_scaler,
        scaler       = train_ds.scaler   # ← Anti-leakage: scaler dari train
    )

    # Sampler (opsional, membantu imbalance)
    sampler = None
    if cfg.use_weighted_sampler:
        sampler = build_weighted_sampler(train_ds)
        log.info("WeightedRandomSampler aktif.")

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.batch_size,
        shuffle     = (sampler is None),   # shuffle=False jika pakai sampler
        sampler     = sampler,
        num_workers = cfg.num_workers,
        pin_memory  = (device.type == "cuda"),
        drop_last   = True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = (device.type == "cuda")
    )

    log.info(f"Train: {len(train_ds)} sampel | Val: {len(val_ds)} sampel")

    # ── Model ────────────────────────────────────────────────────
    model = CheatingGRU(
        input_dim     = 38,
        hidden_dim    = cfg.hidden_dim,
        num_layers    = cfg.num_layers,
        fc_dim        = cfg.fc_dim,
        dropout       = cfg.dropout,
        bidirectional = cfg.bidirectional,
        use_attention = cfg.use_attention
    ).to(device)
    model.summary()

    # ── Loss Function ────────────────────────────────────────────
    if cfg.use_pos_weight:
        pos_weight = compute_pos_weight(train_ds).to(device)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # BCEWithLogitsLoss = Sigmoid + BCELoss dalam satu operasi
    # Lebih stabil secara numerik daripada BCELoss terpisah

    # ── Optimizer & Scheduler ────────────────────────────────────
    optimizer = Adam(
        model.parameters(),
        lr           = cfg.learning_rate,
        weight_decay = cfg.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode      = "min",
        factor    = cfg.lr_factor,
        patience  = cfg.lr_patience,
        min_lr    = cfg.lr_min,
        verbose   = False
    )

    # ── Early Stopping ───────────────────────────────────────────
    early_stop = EarlyStopping(
        patience  = cfg.patience,
        min_delta = cfg.min_delta
    )

    # ── Mixed Precision (GPU only) ───────────────────────────────
    amp_scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Training Loop ────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr":         [], "best_epoch": None
    }

    log.info(f"\n{'='*55}")
    log.info("MULAI TRAINING")
    log.info(f"{'='*55}")

    for epoch in range(1, cfg.epochs + 1):
        t_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, amp_scaler
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # LR Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Catat history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t_start

        # Early Stopping check
        improved = early_stop.step(val_loss, model)
        marker = " ★" if improved else f"  (no improve {early_stop.counter}/{cfg.patience})"

        log.info(
            f"Ep {epoch:03d}/{cfg.epochs} | "
            f"TrainLoss={train_loss:.4f} Acc={train_acc:.3f} | "
            f"ValLoss={val_loss:.4f} Acc={val_acc:.3f} | "
            f"LR={current_lr:.2e} | {elapsed:.1f}s{marker}"
        )

        if improved:
            history["best_epoch"] = epoch

        if early_stop.should_stop:
            log.info(
                f"\nEarly stopping dipicu di epoch {epoch}. "
                f"Best val_loss={early_stop.best_loss:.4f} di epoch {history['best_epoch']}"
            )
            break

    # ── Restore & Simpan Model Terbaik ───────────────────────────
    early_stop.restore_best(model)

    model_path = out_dir / "best_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "best_val_loss": early_stop.best_loss,
        "best_epoch":    history["best_epoch"],
        "history":       history,
    }, str(model_path))
    log.info(f"\nModel terbaik tersimpan: {model_path}")

    # ── Visualisasi ──────────────────────────────────────────────
    plot_path = out_dir / "training_history.png"
    plot_training_history(history, str(plot_path))

    # ── Simpan history JSON ───────────────────────────────────────
    history_path = out_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"\n{'='*55}")
    log.info("RINGKASAN TRAINING")
    log.info(f"{'='*55}")
    log.info(f"Best epoch     : {history['best_epoch']}")
    log.info(f"Best val loss  : {early_stop.best_loss:.4f}")
    log.info(f"Best val acc   : {max(history['val_acc']):.4f}")
    log.info(f"Model tersimpan: {model_path}")
    log.info(f"Grafik          : {plot_path}")

    return model, history


# ─────────────────────────────────────────────────────────────────
# Fungsi Bantu untuk Inference / Fase 3
# ─────────────────────────────────────────────────────────────────
def load_best_model(
    checkpoint_path: str,
    device: str = "auto"
) -> Tuple[CheatingGRU, dict]:
    """
    Muat model terbaik dari file .pth.

    Return: (model, config_dict)

    Penggunaan di Fase 3:
        model, cfg = load_best_model("output/best_model.pth")
        probs = model.predict_proba(features_tensor)
    """
    _device = get_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    cfg_dict   = checkpoint["config"]

    model = CheatingGRU(
        input_dim     = 51,
        hidden_dim    = cfg_dict["hidden_dim"],
        num_layers    = cfg_dict["num_layers"],
        fc_dim        = cfg_dict["fc_dim"],
        dropout       = cfg_dict["dropout"],
        bidirectional = cfg_dict["bidirectional"],
        use_attention = cfg_dict["use_attention"],
    ).to(_device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    log.info(
        f"Model dimuat dari {checkpoint_path} "
        f"(best_epoch={checkpoint['best_epoch']}, "
        f"val_loss={checkpoint['best_val_loss']:.4f})"
    )
    return model, cfg_dict


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 2: Training GRU Cheating Detection"
    )
    # Path
    parser.add_argument("--feature-root",  default="features")
    parser.add_argument("--dataset-root",  default="dataset")
    parser.add_argument("--output-dir",    default="output")

    # Model
    parser.add_argument("--hidden-dim",    type=int,   default=128)
    parser.add_argument("--num-layers",    type=int,   default=2)
    parser.add_argument("--fc-dim",        type=int,   default=64)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--no-attention",  action="store_true")

    # Training
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--seq-len",       type=int,   default=60)

    # Imbalance
    parser.add_argument("--no-pos-weight",       action="store_true")
    parser.add_argument("--weighted-sampler",    action="store_true")

    # Early stopping
    parser.add_argument("--patience",      type=int,   default=10)
    parser.add_argument("--min-delta",     type=float, default=1e-4)

    # Lain
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--device",        default="auto")
    parser.add_argument("--labeling",      default="any",
                        choices=["any", "majority"])
    parser.add_argument("--num-workers",   type=int,   default=0)

    args = parser.parse_args()

    cfg = TrainConfig(
        feature_root         = args.feature_root,
        dataset_root         = args.dataset_root,
        output_dir           = args.output_dir,
        seq_len              = args.seq_len,
        labeling             = args.labeling,
        hidden_dim           = args.hidden_dim,
        num_layers           = args.num_layers,
        fc_dim               = args.fc_dim,
        dropout              = args.dropout,
        bidirectional        = args.bidirectional,
        use_attention        = not args.no_attention,
        epochs               = args.epochs,
        batch_size           = args.batch_size,
        learning_rate        = args.lr,
        weight_decay         = args.weight_decay,
        use_pos_weight       = not args.no_pos_weight,
        use_weighted_sampler = args.weighted_sampler,
        patience             = args.patience,
        min_delta            = args.min_delta,
        seed                 = args.seed,
        device               = args.device,
        num_workers          = args.num_workers
    )

    train(cfg)
