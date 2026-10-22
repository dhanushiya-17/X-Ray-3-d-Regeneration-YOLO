"""
ssm_oracle_test.py  — Diagnose SSM capacity by projecting GT shapes into PCA space.
This tells us the BEST POSSIBLE reconstruction IoU if the optimizer were perfect.
"""
from pathlib import Path
import csv, sys
import numpy as np

ROOT       = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "ssm_model.npz"
SPLIT_FILE = ROOT / "data" / "subject_split.csv"
OCC_DIR    = ROOT / "data" / "occupancy_targets"

def main():
    data = np.load(MODEL_FILE)
    mean   = data["mean_shape"]
    comps  = data["components"]        # (k, n_voxels)
    grid_shape = tuple(data["grid_shape"])
    ev     = data["explained_variance"]

    with open(SPLIT_FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    for split_name in ["train", "validation", "test"]:
        subjects = sorted(r["subject"] for r in rows if r["split"] == split_name)
        ious = []
        for subj in subjects:
            p = OCC_DIR / (subj + ".npy")
            if not p.exists():
                continue
            gt = np.load(p).astype(np.float32).ravel()
            centered = gt - mean
            for k in [10, 20, 30]:
                if k > comps.shape[0]:
                    continue
                coeffs = comps[:k] @ centered
                rec = mean + comps[:k].T @ coeffs
                rec = np.clip(rec, 0.0, 1.0)
                bp = (rec > 0.5)
                bg = (gt  > 0.5)
                inter = (bp & bg).sum()
                union = (bp | bg).sum()
                iou = float(inter) / float(union) if union > 0 else 0.0
                ious.append((subj, k, iou))

        print("=" * 55)
        print("Split:", split_name, " subjects:", len(subjects))
        for k in [10, 20, 30]:
            vals = [iou for (s, kk, iou) in ious if kk == k]
            if vals:
                print("  k={:2d}: mean IoU={:.4f}  min={:.4f}  max={:.4f}  std={:.4f}".format(
                    k, np.mean(vals), np.min(vals), np.max(vals), np.std(vals)))

if __name__ == "__main__":
    main()
