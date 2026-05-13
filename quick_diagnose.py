"""
quick_diagnose.py — Diagnostic script
======================================
Untuk mengecek apakah masalah memang di pelabelan dan
apakah fix sudah benar.

Jalankan SETELAH menjalankan Fase 0 v3 dan Fase 1 v3 ulang.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
import argparse


def diagnose_old_labeling(feature_root: str, dataset_root: str):
    """Cek distribusi label dengan dataset.py LAMA (yang bug)."""
    print("\n" + "="*60)
    print("DIAGNOSIS LAMA (sebelum fix)")
    print("="*60)
    print("Membaca label dari dataset/labels/ secara agregat per-video...")

    dataset_path = Path(dataset_root)
    feature_path = Path(feature_root)

    for split in ["train", "valid", "test"]:
        feature_split = feature_path / split
        labels_split  = dataset_path / split / "labels"

        if not feature_split.exists():
            continue

        labels_list = []
        for video_dir in sorted(feature_split.iterdir()):
            if not video_dir.is_dir(): continue
            video_id = video_dir.name
            video_labels_dir = labels_split / video_id

            # Logika BUG dari dataset.py lama
            n_cheating, n_total = 0, 0
            if video_labels_dir.exists():
                for lf in video_labels_dir.glob("*.txt"):
                    with open(lf) as f:
                        for line in f:
                            parts = line.strip().split()
                            if not parts: continue
                            cls = int(parts[0])
                            n_total += 1
                            if cls == 0: n_cheating += 1

            # Semua siswa di video ini dapat label sama (bug)
            if n_total == 0:
                video_label = -1
            else:
                video_label = 1 if n_cheating > 0 else 0   # "any" strategy

            # Asumsi setiap .npy = 1 siswa
            n_students = len(list(video_dir.glob("*.npy")))
            labels_list.extend([video_label] * n_students)

        dist = Counter(labels_list)
        print(f"  {split:8s}: total={len(labels_list):4d}  | dist={dict(dist)}")


def diagnose_new_labeling(crop_root: str, feature_root: str):
    """Cek distribusi label dengan dataset.py BARU (yang fix)."""
    print("\n" + "="*60)
    print("DIAGNOSIS BARU (setelah fix dengan labels_per_student.json)")
    print("="*60)

    crop_path    = Path(crop_root)
    feature_path = Path(feature_root)

    for split in ["train", "valid", "test"]:
        feature_split = feature_path / split
        crop_split    = crop_path    / split

        if not feature_split.exists():
            continue

        labels_list = []
        unmatched_count = 0

        for video_dir in sorted(feature_split.iterdir()):
            if not video_dir.is_dir(): continue
            video_id = video_dir.name
            json_path = crop_split / video_id / "labels_per_student.json"

            if not json_path.exists():
                print(f"  ⚠️ {video_id}: labels_per_student.json tidak ada")
                continue

            with open(json_path) as f:
                all_labels = json.load(f)

            for npy_file in sorted(video_dir.glob("*.npy")):
                student_id = npy_file.stem
                student_labels = all_labels.get(student_id, [])
                valid_labels = [c for c in student_labels if c >= 0]

                if not valid_labels:
                    unmatched_count += 1
                    continue

                # "any" strategy: cheat jika ada minimal 1 frame cheat (class 0)
                has_cheat = any(c == 0 for c in valid_labels)
                labels_list.append(1 if has_cheat else 0)

        dist = Counter(labels_list)
        print(f"  {split:8s}: total={len(labels_list):4d}  | "
              f"dist={dict(dist)} | unmatched={unmatched_count}")

        # Cek per-frame distribusi
        per_frame_dist = Counter()
        for video_dir in sorted(feature_split.iterdir()):
            if not video_dir.is_dir(): continue
            json_path = crop_split / video_dir.name / "labels_per_student.json"
            if not json_path.exists(): continue
            with open(json_path) as f:
                all_labels = json.load(f)
            for student_classes in all_labels.values():
                per_frame_dist.update(student_classes)

        print(f"           per-frame: cheat(0)={per_frame_dist.get(0,0)}, "
              f"not_cheat(1)={per_frame_dist.get(1,0)}, "
              f"unmatched(-1)={per_frame_dist.get(-1,0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir",  default="features")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--crop-root",    default="crop")
    parser.add_argument("--mode",         default="both",
                        choices=["old", "new", "both"])
    args = parser.parse_args()

    if args.mode in ("old", "both"):
        diagnose_old_labeling(args.feature_dir, args.dataset_root)

    if args.mode in ("new", "both"):
        diagnose_new_labeling(args.crop_root, args.feature_dir)

    print()
