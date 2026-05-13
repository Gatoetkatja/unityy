# Fase 0 v2: Student Tracking dengan ByteTrack + Bbox Ground Truth

## Perbedaan dengan Fase 0 Lama

| Aspek | Fase 0 Lama | Fase 0 v2 (ini) |
|-------|-------------|-----------------|
| Sumber bbox | YOLO inference (kadang miss) | **Label `.txt` ground truth** |
| Akurasi deteksi | Bergantung quality YOLO | **100% (ground truth)** |
| Tracker | Custom IoU + appearance | **ByteTrack (Kalman filter)** |
| Cocok untuk kamera bergerak | Sedang | **Bagus** (Kalman predict) |

---

## Kenapa ByteTrack?

1. **Kalman filter internal** memprediksi posisi siswa di frame berikutnya — sangat membantu saat kamera bergerak pelan, karena tracker tahu siswa "diharapkan" muncul di posisi tertentu
2. **Two-stage matching** (high-conf → low-conf detections) — bagus untuk skenario occlusion ringan saat siswa menunduk
3. **Track buffer** menjaga ID lama tetap "hidup" hingga 60 frame setelah hilang
4. Sudah tersedia di `ultralytics`, tidak perlu install package terpisah

---

## Cara Menjalankan

```bash
pip install -r requirements.txt

python tracker_bytetrack.py \
    --dataset ./dataset \
    --output  ./crop \
    --splits  train valid test
```

Tuning untuk kamera bergerak:
```bash
python tracker_bytetrack.py \
    --dataset ./dataset \
    --output  ./crop \
    --track-buffer 90 \         # Lebih besar untuk kamera yang sering bergerak
    --match-thresh 0.7          # Lebih longgar (default 0.8)
```

---

## Bagaimana Class `cheating`/`not_cheating` Ditangani?

Saat tracking, **semua siswa diperlakukan sebagai class yang sama** (`person`). Ini penting agar:
- Track tidak terpecah ketika seorang siswa berubah dari `not_cheating` → `cheating` di label
- Satu `student_id` mewakili satu orang, bukan satu "state perilaku"

Class `cheating`/`not_cheating` dari label asli **disimpan untuk Fase 2** (training GRU) lewat anotasi original, bukan lewat tracking.

---

## Output

```
crop/
├── train/
│   ├── 1/
│   │   ├── student_001/
│   │   │   ├── student_001_0001.jpg
│   │   │   └── ...
│   │   ├── student_002/
│   │   └── student_003/
│   └── 2/
└── ...
```

> `student_id` lokal per video, sesuai desain Fase 1 yang sudah ada.

---

## Tuning untuk Kondisi Berbeda

| Kondisi | Saran |
|---------|-------|
| Kamera statis sempurna | Default cukup |
| Kamera bergerak pelan (skenario Anda) | `--track-buffer 90` |
| Banyak occlusion (siswa sering menunduk panjang) | `--track-buffer 120` |
| Siswa duduk berdekatan | `--match-thresh 0.7` (longgar) |
| Ada sliding-zoom besar | Pertimbangkan tambah GMC (Global Motion Compensation) |

---

## Verifikasi Hasil

Pakai script verifikasi dari Fase 0 sebelumnya:
```bash
python ../phase0_tracking/verify_tracking.py --crop-dir ./crop
```

Catatan: Setelah dilakukan tracker_bytetrack, dilakukan review hasil crop dan perbaikan pengelompokkan student_id secara manual.

# Fase 1 v3: Head Feature — YOLO Pose + Geometric + Temporal

## Constraint Dunia Nyata yang Ditangani

| Constraint | Penanganan |
|------------|-----------|
| **Resolusi rendah (CCTV 640×640, crop ~60-120px)** | YOLO Pose (bukan MediaPipe yang butuh res tinggi untuk blendshapes) |
| **Siswi berkerudung menunduk** | Geometric pose dari nose↔shoulder, tidak bergantung ear/eye |
| **Siswa menghadap belakang** | `facing_back_flag=1` sebagai sinyal eksplisit |
| **Occlusion sementara** | Interpolasi linier sambil menjaga visibility flag |
| **Variasi gerakan kecil/halus** | Temporal velocity (Δxy frame-to-frame) |

---

## Layout 38 Fitur per Frame

```
[0:21]   Raw keypoints       : 7 keypoint × 3 (x_norm, y_norm, conf)
                                Urutan: nose, leye, reye, lear, rear,
                                        lshoulder, rshoulder
[21:24]  Geometric head pose : yaw, pitch, roll  ∈ [-1, 1]
                                • Tidak butuh 3D face model
                                • Dihitung dari relasi keypoint
[24:26]  Head-body relation  : head_y_relative, head_size_ratio
                                • Posisi & ukuran kepala relatif bahu
[26:28]  Visibility flags    : n_visible_norm, facing_back_flag
                                • Eksplisit menandai siswa hadap belakang
[28:38]  Temporal velocity   : Δxy untuk 5 head keypoints
                                • Sinyal gerakan halus (menoleh cepat dsb)
```

---

## Cara Kerja Geometric Pose

**Yaw (menoleh kiri/kanan)** — dari asimetri jarak hidung ke mata kiri vs kanan
```
Jika nose lebih dekat ke left_eye → menghadap kiri → yaw positif
Jika nose lebih dekat ke right_eye → menghadap kanan → yaw negatif
```

**Pitch (menunduk/mendongak)** — dari rasio Y_nose vs Y_shoulder
```
Normal frontal:    nose ~1.0 × shoulder_width di atas bahu → pitch 0
Menunduk:          nose mendekati Y bahu                   → pitch -1
Mendongak:         nose jauh di atas bahu                  → pitch +1
```

**Pitch tetap dihitung** meski hanya nose visible (kasus kerudung+menunduk!).

**Roll (kemiringan kepala)** — dari sudut garis mata atau bahu terhadap horizontal

---

## Cara Menjalankan

```bash
pip install -r requirements.txt

python feature_extractor_v3.py \
    --crop-dir ./crop \
    --feature-dir ./features \
    --model yolo11n-pose.pt

python verify_features_v3.py --feature-dir ./features
```

---

## Penanganan Edge Cases

| Skenario | Behavior |
|----------|----------|
| Frontal sehat | Semua fitur valid, facing_back=0 |
| Menoleh ke samping | Yaw bergerak, mata kanan/kiri berbeda confidence |
| Menunduk + berkerudung | Pitch dari nose↔shoulder, eye/ear tidak diperlukan |
| Hadap belakang total | facing_back_flag=1, raw_kp ~ 0, GRU dapat sinyal |
| Frame tunggal hilang | Interpolasi linier, velocity di-zero |
| Semua frame siswa hadap belakang | facing_back=1 di semua frame, sinyal untuk GRU |

---

## Integrasi Fase 2

File `dataset.py` di sini sudah update ke `FEATURE_DIM=38`. Ganti file lama.

Update di `model.py` Fase 2:
```python
model = CheatingGRU(input_dim=38, ...)   # ← bukan 51 lagi
```

---

## Tuning

Jika dataset CCTV-nya sangat rendah resolusi, gunakan model YOLO yang lebih besar untuk akurasi keypoint:
```bash
python feature_extractor_v3.py --model yolo11s-pose.pt   # small
python feature_extractor_v3.py --model yolo11m-pose.pt   # medium
```

Untuk visibility threshold (default 0.3):
- Lebih rendah (0.2): lebih banyak fitur dianggap valid tapi mungkin noisy
- Lebih tinggi (0.4): hanya keypoint yang sangat confident yang dipakai

# Fase 2: Model Training — GRU Cheating Detection

## File

| File | Keterangan |
|------|-----------|
| `model.py` | Arsitektur CheatingGRU |
| `train.py` | Pipeline training lengkap |
| `requirements.txt` | Dependensi |

---

## Cara Menjalankan

```bash
pip install -r requirements.txt

# Training standar
python train.py \
    --feature-root ./features \
    --dataset-root ./dataset \
    --output-dir   ./output

# Dengan semua opsi
python train.py \
    --feature-root ./features \
    --dataset-root ./dataset \
    --output-dir   ./output \
    --hidden-dim   128 \
    --epochs       50 \
    --batch-size   32 \
    --lr           1e-3 \
    --patience     10 \
    --weighted-sampler   # opsional, aktifkan jika imbalance parah
```

---

## Arsitektur Model

```
Input (batch, 60, 51)
  │
  ├─ LayerNorm(51)
  │
  ├─ GRU Layer 1 (hidden=128, dropout antar-layer)
  ├─ GRU Layer 2 (hidden=128)
  │
  ├─ Temporal Attention Pooling  ← memberi bobot lebih pada frame "mencurigakan"
  │     output: (batch, 128)
  │
  ├─ Linear(128 → 64) + ReLU + Dropout(0.3)
  └─ Linear(64 → 1)   ← logit (sebelum sigmoid)

Output: (batch, 1) logit
  → Sigmoid saat inference → probabilitas [0,1]
  → Threshold 0.5 → label (0=not_cheating, 1=cheating)
```

---

## Penanganan Class Imbalance

Tiga mekanisme tersedia, bisa dikombinasikan:

| Mekanisme | Default | Aktifkan |
|-----------|---------|----------|
| `pos_weight` di BCEWithLogitsLoss | ✅ ON | `--no-pos-weight` untuk mematikan |
| `WeightedRandomSampler` | ❌ OFF | `--weighted-sampler` |
| `use_scaler` (StandardScaler) | ❌ OFF | `--use-scaler` (fit hanya dari train) |

**Rekomendasi:** Untuk imbalance ringan–sedang, `pos_weight` saja sudah cukup.
Untuk imbalance ekstrem (>10:1), aktifkan `--weighted-sampler` juga.

`pos_weight` dihitung otomatis: `n_not_cheating / n_cheating`

---

## Output

```
output/
├── best_model.pth         ← checkpoint model terbaik
├── training_history.png   ← grafik loss & accuracy
└── training_history.json  ← log numerik per epoch
```

### Isi `best_model.pth`
```python
checkpoint = torch.load("output/best_model.pth")
checkpoint["model_state_dict"]   # bobot model
checkpoint["config"]             # semua hyperparameter
checkpoint["best_val_loss"]      # val loss terbaik
checkpoint["best_epoch"]         # epoch terbaik
checkpoint["history"]            # riwayat training
```

---

## Memuat Model untuk Fase 3 (Inference)

```python
from train import load_best_model

model, cfg = load_best_model("output/best_model.pth")

# Inference satu sequence siswa
features = torch.randn(1, 60, 51)  # contoh
prob     = model.predict_proba(features)  # → Tensor [[0.87]]
label    = (prob >= 0.5).int()            # → 1 (cheating)
```

---

## Early Stopping

- Monitor: `val_loss` (bukan val_acc, agar tidak overfitting ke accuracy metrik)
- `patience=10`: hentikan jika 10 epoch berturut tidak ada perbaikan ≥ `min_delta=0.0001`
- State model terbaik disimpan **di memori** selama training, lalu di-dump ke `.pth`
- LR dikurangi otomatis (`ReduceLROnPlateau`) jika val_loss stagnan selama `lr_patience=5` epoch

---

## Grafik Training

4 panel yang dihasilkan di `training_history.png`:
1. **Loss** — Train vs Val per epoch + penanda epoch terbaik
2. **Accuracy** — Train vs Val per epoch
3. **Learning Rate** — tampilan log-scale, terlihat kapan LR turun
4. **Val Loss Detail** — area chart + titik best epoch