"""
verify_features_v3.py
======================
Verifikasi hasil Fase 1 v3:
- Cek dimensi (60, 38)
- Cek range nilai per kelompok fitur
- Cek statistik pose detection & facing-back ratio
"""

import numpy as np
from pathlib import Path
import argparse


def verify(feature_root: str, expected_seq_len: int = 8, expected_feat_dim: int = 38):
    feature_path = Path(feature_root)
    errors = []
    stats  = []

    print(f"\n{'='*55}")
    print(f"VERIFIKASI HEAD FEATURES v3 — {feature_root}")
    print(f"Dimensi: ({expected_seq_len}, {expected_feat_dim})")
    print(f"{'='*55}")

    total_files, ok_files = 0, 0

    for split_dir in sorted(feature_path.iterdir()):
        if not split_dir.is_dir():
            continue
        for video_dir in sorted(split_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            npy_files = sorted(video_dir.glob("*.npy"))
            if not npy_files:
                continue

            print(f"\n  [{split_dir.name}/{video_dir.name}] — {len(npy_files)} file")

            for npy_path in npy_files:
                total_files += 1
                try:
                    arr = np.load(str(npy_path))
                    if arr.shape != (expected_seq_len, expected_feat_dim):
                        errors.append(f"Shape salah {npy_path.name}: {arr.shape}")
                        continue

                    # Range checks
                    raw_kp_x = arr[:, [0,3,6,9,12,15,18]]   # x coords
                    raw_kp_y = arr[:, [1,4,7,10,13,16,19]]  # y coords
                    pose     = arr[:, 21:24]
                    body_rel = arr[:, 24:26]
                    vis      = arr[:, 26:28]
                    vel      = arr[:, 28:38]

                    kp_out = ((raw_kp_x < -0.01).any() or (raw_kp_x > 1.01).any() or
                              (raw_kp_y < -0.01).any() or (raw_kp_y > 1.01).any())
                    pose_out = (pose < -1.05).any() or (pose > 1.05).any()
                    vis_out  = (vis < -0.01).any() or (vis > 1.01).any()
                    vel_out  = (vel < -0.51).any() or (vel > 0.51).any()

                    issues = []
                    if kp_out:   issues.append("kp_range")
                    if pose_out: issues.append("pose_range")
                    if vis_out:  issues.append("vis_range")
                    if vel_out:  issues.append("vel_range")

                    facing_back_ratio = float(arr[:, 27].mean())
                    visible_ratio     = float(arr[:, 26].mean())

                    status = "✓" if not issues else "⚠"
                    label = ""
                    if facing_back_ratio > 0.5:
                        label = " [mostly facing back]"
                    elif visible_ratio < 0.3:
                        label = " [low visibility]"

                    print(
                        f"    {status} {npy_path.name:<25s} "
                        f"vis={visible_ratio:.2f} back={facing_back_ratio:.0%}"
                        f"{label}"
                    )

                    if issues:
                        errors.append(f"{npy_path.name}: {issues}")

                    stats.append({
                        "facing_back": facing_back_ratio,
                        "visibility":  visible_ratio
                    })
                    ok_files += 1

                except Exception as e:
                    errors.append(f"Error {npy_path}: {e}")

    print(f"\n{'='*55}")
    print(f"Total: {ok_files}/{total_files} file OK")
    if stats:
        avg_back = np.mean([s["facing_back"] for s in stats])
        avg_vis  = np.mean([s["visibility"]  for s in stats])
        print(f"Avg facing-back ratio : {avg_back:.1%}")
        print(f"Avg head visibility   : {avg_vis:.1%}")

    if errors:
        print(f"\n⛔ {len(errors)} ERROR (first 10):")
        for e in errors[:10]:
            print(f"   - {e}")
    else:
        print("\n✅ Semua file valid!")

    print(f"{'='*55}\n")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="features")
    parser.add_argument("--seq-len",     type=int, default=60)
    parser.add_argument("--feat-dim",    type=int, default=38)
    args = parser.parse_args()

    errors = verify(args.feature_dir, args.seq_len, args.feat_dim)
    exit(0 if not errors else 1)
