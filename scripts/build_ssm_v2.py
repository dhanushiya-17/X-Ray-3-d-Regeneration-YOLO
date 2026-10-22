"""
build_ssm_v2.py
===============
Build a high-resolution SSM from 256x96x96 occupancy grids.
Uses 60 PCA components to capture maximum shape variance.

Outputs:
  models/ssm_highres.npz
  models/ssm_highres_variance.png
  models/ssm_highres_oracle.csv  (oracle IoU for all splits)
"""

from pathlib import Path
import argparse
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT       = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data" / "subject_split.csv"
OCC_DIR    = ROOT / "data" / "occupancy_highres"
MODEL_DIR  = ROOT / "models"
MODEL_FILE = MODEL_DIR / "ssm_highres.npz"
VARIANCE_PLOT = MODEL_DIR / "ssm_highres_variance.png"
ORACLE_CSV = MODEL_DIR / "ssm_highres_oracle.csv"

N_COMPONENTS = 60


def load_split(split_file):
    splits = {}
    with split_file.open(newline="") as f:
        for row in csv.DictReader(f):
            splits.setdefault(row["split"], []).append(row["subject"])
    return {k: sorted(v) for k, v in splits.items()}


def load_occ(subject):
    path = OCC_DIR / (subject + ".npy")
    if not path.exists():
        raise FileNotFoundError(str(path))
    return np.load(path).astype(np.float32)


def voxel_iou(a, b, thr=0.5):
    pa, pb = (a > thr), (b > thr)
    inter = (pa & pb).sum()
    union = (pa | pb).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train the 70-subject high-resolution femur SSM")
    parser.add_argument("--force-overwrite", action="store_true", help="Allow replacement of an existing model, variance plot, or oracle CSV")
    args = parser.parse_args()
    collisions = [str(path) for path in (MODEL_FILE, VARIANCE_PLOT, ORACLE_CSV) if path.exists()]
    if collisions and not args.force_overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing model artifact(s): " + ", ".join(collisions)
            + ". Train in a clean copy or pass --force-overwrite deliberately."
        )
    print("=" * 60)
    print("BoneVision -- High-Resolution SSM Builder (v2)")
    print("=" * 60)

    splits = load_split(SPLIT_FILE)
    train_subjects = splits.get("train", [])

    print("Train subjects:", len(train_subjects))
    if len(train_subjects) != 70:
        print("ERROR: expected 70 training subjects")
        sys.exit(1)

    # Load training grids
    grids = []
    shape_ref = None
    for subj in train_subjects:
        g = load_occ(subj)
        if shape_ref is None:
            shape_ref = g.shape
            print("Grid shape:", shape_ref)
        elif g.shape != shape_ref:
            raise ValueError("Shape mismatch for " + subj)
        grids.append(g)
    print("Loaded all", len(grids), "training grids")

    n_voxels = int(np.prod(shape_ref))
    X = np.stack([g.ravel() for g in grids], axis=0).astype(np.float32)
    print("Data matrix:", X.shape, " voxels:", n_voxels)

    mean_shape = X.mean(axis=0)
    X_centered = X - mean_shape

    n_comp = min(N_COMPONENTS, len(train_subjects) - 1)
    print("Running PCA with", n_comp, "components...")

    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X_centered)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Print top-20 and key milestones
    for i in range(min(20, n_comp)):
        print("  PC{:02d}: {:.2f}%  cumulative {:.2f}%".format(
            i+1, explained[i]*100, cumulative[i]*100))

    # Find 90%, 95%, 99% thresholds
    for thr in [0.90, 0.95, 0.99]:
        idx = np.searchsorted(cumulative, thr)
        if idx < n_comp:
            print("  {} variance at {} components".format(
                str(round(thr*100)) + "%", idx + 1))

    print("  Total at {} components: {:.2f}%".format(n_comp, cumulative[-1]*100))

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL_FILE,
        mean_shape=mean_shape,
        components=pca.components_,
        singular_values=pca.singular_values_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        grid_shape=np.array(shape_ref),
        train_subjects=np.array(train_subjects),
        bounds=np.array([-470.0, 40.0, -60.0, 100.0, -70.0, 70.0]),
    )
    print("SSM saved:", MODEL_FILE)

    # Variance plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, len(explained)+1), explained*100, color="#4C72B0")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Per-Component Explained Variance")
    axes[0].grid(axis="y", alpha=0.4)

    axes[1].plot(range(1, len(cumulative)+1), cumulative*100,
                 marker="o", color="#DD8452", linewidth=2, markersize=3)
    for thr_val, color in [(95, "gray"), (99, "red")]:
        axes[1].axhline(thr_val, color=color, linestyle="--", linewidth=1,
                        label=str(thr_val) + "%")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend()
    axes[1].grid(alpha=0.4)

    plt.suptitle("BoneVision High-Res SSM -- PCA on 70 Training Femurs\n"
                 "Grid " + str(shape_ref) + " (" + str(n_voxels) + " voxels)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(VARIANCE_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print("Variance plot:", VARIANCE_PLOT)

    # Oracle test: project all subjects into PCA space and measure IoU
    print("\nOracle test (best possible IoU with perfect optimizer):")
    oracle_rows = []

    for split_name in ["train", "validation", "test"]:
        subjects = splits.get(split_name, [])
        for subj in subjects:
            try:
                gt = load_occ(subj).ravel()
            except FileNotFoundError:
                continue

            centered = gt - mean_shape
            for k in [10, 20, 30, 40, 50, n_comp]:
                if k > n_comp:
                    continue
                coeffs = pca.components_[:k] @ centered
                rec = mean_shape + pca.components_[:k].T @ coeffs
                rec = np.clip(rec, 0.0, 1.0)
                iou = voxel_iou(rec, gt)
                oracle_rows.append({
                    "subject": subj, "split": split_name,
                    "k": k, "iou": round(iou, 4)
                })

    with open(ORACLE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "split", "k", "iou"])
        writer.writeheader()
        writer.writerows(oracle_rows)

    # Summary by split and k
    for split_name in ["train", "validation", "test"]:
        print("  " + split_name + ":")
        for k in [10, 20, 30, 40, 50, n_comp]:
            vals = [r["iou"] for r in oracle_rows
                    if r["split"] == split_name and r["k"] == k]
            if vals:
                print("    k={:2d}: mean={:.4f} min={:.4f} max={:.4f}".format(
                    k, np.mean(vals), np.min(vals), np.max(vals)))

    print("\n" + "=" * 60)
    print("High-res SSM build complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
