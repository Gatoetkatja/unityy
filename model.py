"""
model.py — Arsitektur GRU untuk Deteksi Kecurangan Ujian
=========================================================
Input  : (batch_size, seq_len=60, feat_dim=51)
Output : (batch_size, 1)  — logit sebelum sigmoid

Arsitektur:
  Input (51)
    → GRU Layer 1 (hidden_dim, bidirectional opsional)
    → Dropout
    → GRU Layer 2 (hidden_dim)
    → Dropout
    → Attention Pooling (opsional, lebih baik dari last-hidden saja)
    → FC Layer (hidden_dim → fc_dim)
    → ReLU + Dropout
    → FC Output (fc_dim → 1)
    → (Sigmoid diterapkan di luar saat inference, tidak di model
       karena BCEWithLogitsLoss lebih stabil secara numerik)
"""

import torch
import torch.nn as nn
from typing import Optional


class TemporalAttention(nn.Module):
    """
    Additive attention untuk meringkas sequence GRU.
    Memberi bobot lebih pada frame yang paling "informatif"
    (misalnya frame saat siswa sedang melihat ke samping).

    Input : (batch, seq_len, hidden_dim)
    Output: (batch, hidden_dim)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, gru_out: torch.Tensor) -> torch.Tensor:
        # gru_out: (B, T, H)
        scores  = self.attn(gru_out)            # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        context = (gru_out * weights).sum(dim=1)  # (B, H)
        return context


class CheatingGRU(nn.Module):
    """
    GRU 2-layer dengan Temporal Attention untuk klasifikasi biner.

    Args:
        input_dim    : Dimensi fitur per timestep (default 51)
        hidden_dim   : Ukuran hidden state GRU (default 128)
        num_layers   : Jumlah layer GRU (default 2)
        fc_dim       : Ukuran hidden layer FC (default 64)
        dropout      : Dropout probability (default 0.3)
        bidirectional: Gunakan BiGRU jika True (hidden_dim × 2 di output)
        use_attention: Gunakan temporal attention pooling (direkomendasikan)
    """

    def __init__(
        self,
        input_dim:     int   = 38,
        hidden_dim:    int   = 128,
        num_layers:    int   = 2,
        fc_dim:        int   = 64,
        dropout:       float = 0.3,
        bidirectional: bool  = False,
        use_attention: bool  = True,
    ):
        super().__init__()

        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1

        # ── Layer Normalisasi input ──────────────────────────
        self.input_norm = nn.LayerNorm(input_dim)

        # ── GRU Utama ────────────────────────────────────────
        self.gru = nn.GRU(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional
        )

        gru_out_dim = hidden_dim * self.num_directions

        # ── Attention / Pooling ──────────────────────────────
        if use_attention:
            self.attention = TemporalAttention(gru_out_dim)
        else:
            self.attention = None

        # ── Classifier Head ──────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1)
            # ↑ Tidak ada Sigmoid di sini
            #   → BCEWithLogitsLoss menangani ini secara numerik lebih stabil
        )

        self._init_weights()

    def _init_weights(self):
        """Inisialisasi bobot dengan Xavier/Orthogonal."""
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x               : (batch, seq_len, input_dim)
            return_attention : Jika True, kembalikan juga bobot attention

        Return:
            logits : (batch, 1) — sebelum sigmoid
            (opsional) attn_weights : (batch, seq_len, 1)
        """
        # Normalisasi input
        x = self.input_norm(x)  # (B, T, 51)

        # GRU forward
        gru_out, _ = self.gru(x)  # (B, T, H*D)

        # Pooling
        if self.attention is not None:
            # Attention pooling
            attn_scores = self.attention.attn(gru_out)        # (B, T, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
            context = (gru_out * attn_weights).sum(dim=1)     # (B, H*D)
        else:
            # Ambil hidden state terakhir
            context      = gru_out[:, -1, :]  # (B, H*D)
            attn_weights = None

        logits = self.classifier(context)  # (B, 1)

        if return_attention and attn_weights is not None:
            return logits, attn_weights
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilitas [0,1] (setelah sigmoid). Untuk inference."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Cetak ringkasan model."""
        print("=" * 50)
        print("CheatingGRU — Arsitektur")
        print("=" * 50)
        print(f"  Input dim      : {self.gru.input_size}")
        print(f"  Hidden dim     : {self.hidden_dim}")
        print(f"  GRU layers     : {self.num_layers}")
        print(f"  Bidirectional  : {self.bidirectional}")
        print(f"  Attention      : {self.use_attention}")
        print(f"  Total params   : {self.count_parameters():,}")
        print("=" * 50)


# ── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model = CheatingGRU(
        input_dim=51, hidden_dim=128, num_layers=2,
        fc_dim=64, dropout=0.3, use_attention=True
    )
    model.summary()

    dummy = torch.randn(8, 60, 51)
    out   = model(dummy)
    print(f"\nInput shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    assert out.shape == (8, 1), f"Output shape salah: {out.shape}"
    print("✓ Forward pass OK")
