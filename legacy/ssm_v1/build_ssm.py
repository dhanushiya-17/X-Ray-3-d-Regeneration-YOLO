from pathlib import Path
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data" / "subject_split.csv"
OCC_DIR    = ROOT / "data" / "occupancy_targets"
MODEL_DIR  = ROOT / "models"
MODEL_FILE = MODEL_DIR / "ssm_model.npz"
VARIANCE_PLOT = MODEL_DIR / "ssm_variance.png"
N_COMPONENTS = 30


def load_split(split_file):
    splits = {}
    with split_file.open(newline="") as f:
        for row in csv.DictReader(f):
            splits.setdefault(row["split"], []).append(row["subject"])
    return {k: sorted(v) for k, v in splits.items()}


def load_occupancy(subject, occ_dir):
    path = occ_dir / (subject + ".npy")
    if not path.exists():
        raise FileNotFoundError("Occupancy file missing: " + str(path))
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError("Expected 3-D array, got shape " + str(arr.shape))
    return arr


def main():
    print("=" * 60)
    print("BoneVision -- SSM Builder")
    print("=" * 60)

    splits = load_split(SPLIT_FILE)
    train_subjects = splits.get("train", [])
    val_subjects   = splits.get("validation", [])
    test_subjects  = splits.get("test", [])

    print("Train:", len(train_subjects),
          "  Val:", len(val_subjects),
          "  Test:", len(test_subjects))

    if len(train_subjects) != 70:
        print("ERROR: Expected 70 training subjects, found", len(train_subjects))
        sys.exit(1)

    print("Loading", len(train_subjects), "training occupancy grids...")

    grids = []
    failed = []
    shape_ref = None

    for subject in train_subjects:
        try:
            grid = load_occupancy(subject, OCC_DIR)
            if shape_ref is None:
                shape_ref = grid.shape
                print("  Grid shape (X,Y,Z):", shape_ref)
            elif grid.shape != shape_ref:
                raise ValueError("Shape mismatch: expected " + str(shape_ref)
                                 + ", got " + str(grid.shape))
            grids.append(grid)
        except Exception as exc:
            failed.append((subject, str(exc)))
            print("  FAILED", subject, ":", exc)

    if failed:
        print(len(failed), "grids failed. Cannot build SSM.")
        sys.exit(1)

    print("  All", len(grids), "grids loaded OK.")

    n_voxels = int(np.prod(shape_ref))
    X = np.stack([g.ravel() for g in grids], axis=0)   # (70, n_voxels)

    print("Data matrix:", X.shape)
    print("  Voxels/shape :", n_voxels)
    print("  Occ range    :", float(X.min()), "-", float(X.max()))
    print("  Mean occ     :", round(float(X.mean()), 4))

    mean_shape = X.mean(axis=0)
    X_centered = X - mean_shape

    n_components = min(N_COMPONENTS, len(train_subjects) - 1)
    print("Running PCA with", n_components, "components...")

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_centered)

    explained  = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    for i, (e, c) in enumerate(zip(explained, cumulative)):
        print("  PC" + str(i+1).zfill(2) + ": "
              + str(round(e * 100, 2)) + "%"
              + "  cumulative " + str(round(c * 100, 2)) + "%")
        if c > 0.995:
            print("  -->", i+1, "components explain >99.5% variance.")
            break

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
    )
    print("SSM saved to:", MODEL_FILE)

    # ------------------------------------------------------------------
    # Variance plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(1, len(explained) + 1), explained * 100, color="#4C72B0")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Per-Component Explained Variance")
    axes[0].grid(axis="y", alpha=0.4)

    axes[1].plot(range(1, len(cumulative) + 1), cumulative * 100,
                 marker="o", color="#DD8452", linewidth=2)
    axes[1].axhline(95, color="gray", linestyle="--", linewidth=1, label="95%")
    axes[1].axhline(99, color="red",  linestyle="--", linewidth=1, label="99%")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend()
    axes[1].grid(alpha=0.4)

    plt.suptitle(
        "BoneVision SSM -- PCA on " + str(len(train_subjects))
        + " Training Femur Grids  shape=" + str(shape_ref)
        + "  voxels=" + str(n_voxels),
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(VARIANCE_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print("Variance plot saved:", VARIANCE_PLOT)

    # ------------------------------------------------------------------
    # Sanity check: reconstruct a training shape from top-k modes
    # ------------------------------------------------------------------
    print("Sanity check -- reconstruct first training shape:")
    k_values = [5, 10, 20, n_components]
    x_true = X[0]

    for k in k_values:
        coords = pca.components_[:k] @ X_centered[0]
        x_rec  = mean_shape + pca.components_[:k].T @ coords
        bin_rec  = (x_rec  > 0.5).astype(np.float32)
        bin_true = (x_true > 0.5).astype(np.float32)
        inter = (bin_rec * bin_true).sum()
        union = ((bin_rec + bin_true) > 0).sum()
        iou   = inter / union if union > 0 else 0.0
        print("  k=" + str(k).rjust(2) + " components: IoU=" + str(round(iou, 4)))

    print("=" * 60)
    print("SSM build complete.")
    print("=" * 60)
    print("Next: run  scripts/ssm_reconstruct.py  to fit a query X-ray.")


if __name__ == "__main__":
    main()
