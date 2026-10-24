"""
build_ssm_sdf.py
================
Build a femur shape model by PCA on truncated signed distance fields (SDF)
instead of binary occupancy. Uses the same 256x96x96 occupancy targets and the
saved 70/15/16 subject split, so it can be run on the existing local data
without the raw CTs.

Why: PCA of binary occupancy averages 0/1 voxels, so every reconstruction is a
blurred blend that must be thresholded, and fine structures (neck, condyles)
wash out. SDF PCA keeps a smooth, metric representation whose zero level set
is a valid surface for any coefficient vector.

The oracle report compares both representations at the same number of
components on the validation and test subjects, so the change can be judged
on held-out data before switching the app to it.

Outputs:
  models/ssm_sdf.npz
  models/ssm_sdf_oracle.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from femur_grid import ROOT, OCC_DIR, SPLIT_FILE, grid_spacing, load_split, occupancy_to_sdf

MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "ssm_sdf.npz"
ORACLE_CSV = MODEL_DIR / "ssm_sdf_oracle.csv"


def voxel_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def project(pca_components, mean, x, k):
    coeffs = pca_components[:k] @ (x - mean)
    return mean + pca_components[:k].T @ coeffs


def main():
    parser = argparse.ArgumentParser(description="Train an SDF-based femur SSM on the training split")
    parser.add_argument("--occupancy-dir", default=str(OCC_DIR))
    parser.add_argument("--split-file", default=str(SPLIT_FILE))
    parser.add_argument("--extra-train-dir", default=None,
                        help="Optional directory of extra *.npy occupancy grids (same grid) used for training only, "
                             "e.g. output of import_external_femurs.py")
    parser.add_argument("--components", type=int, default=60)
    parser.add_argument("--truncation", type=float, default=20.0, help="SDF truncation in mm")
    parser.add_argument("--force-overwrite", action="store_true")
    args = parser.parse_args()

    collisions = [str(p) for p in (MODEL_FILE, ORACLE_CSV) if p.exists()]
    if collisions and not args.force_overwrite:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(collisions)
                              + ". Pass --force-overwrite deliberately.")

    occ_dir = Path(args.occupancy_dir)
    splits = load_split(args.split_file)
    train = splits.get("train", [])
    held_out = [(s, "validation") for s in splits.get("validation", [])] + \
               [(s, "test") for s in splits.get("test", [])]
    if not train:
        raise ValueError("No training subjects in split file")

    first = np.load(occ_dir / (train[0] + ".npy"))
    shape = first.shape
    spacing = grid_spacing(shape)
    print("Grid", shape, "spacing", tuple(round(s, 3) for s in spacing))

    train_files = [occ_dir / (s + ".npy") for s in train]
    train_names = list(train)
    if args.extra_train_dir:
        extra = sorted(Path(args.extra_train_dir).glob("*.npy"))
        train_files += extra
        train_names += [p.stem for p in extra]
        print("Extra training shapes:", len(extra))

    sdf_rows, occ_rows = [], []
    for path in train_files:
        occ = np.load(path)
        if occ.shape != shape:
            raise ValueError("Grid mismatch for " + str(path))
        sdf_rows.append(occupancy_to_sdf(occ, spacing, args.truncation).ravel())
        occ_rows.append(occ.astype(np.float32).ravel())
    S = np.stack(sdf_rows)
    O = np.stack(occ_rows)
    n_comp = min(args.components, len(train_files) - 1)
    print("Training shapes:", len(train_files), " components:", n_comp)

    sdf_mean = S.mean(axis=0)
    sdf_pca = PCA(n_components=n_comp, random_state=42).fit(S - sdf_mean)
    occ_mean = O.mean(axis=0)
    occ_pca = PCA(n_components=n_comp, random_state=42).fit(O - occ_mean)
    del S, O

    cumulative = np.cumsum(sdf_pca.explained_variance_ratio_)
    for thr in (0.90, 0.95, 0.99):
        idx = int(np.searchsorted(cumulative, thr))
        if idx < n_comp:
            print("  SDF {:.0f}% variance at {} components".format(thr * 100, idx + 1))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL_FILE,
        representation=np.array("sdf"),
        mean_shape=sdf_mean.astype(np.float32),
        components=sdf_pca.components_.astype(np.float32),
        singular_values=sdf_pca.singular_values_,
        explained_variance=sdf_pca.explained_variance_,
        explained_variance_ratio=sdf_pca.explained_variance_ratio_,
        grid_shape=np.array(shape),
        spacing=np.array(spacing),
        truncation=np.array(args.truncation),
        train_subjects=np.array(train_names),
        bounds=np.array([-470.0, 40.0, -60.0, 100.0, -70.0, 70.0]),
    )
    print("Saved", MODEL_FILE)

    # Oracle: best achievable reconstruction of held-out shapes with k modes.
    ks = sorted({k for k in (10, 20, 30, 40, 50, n_comp) if k <= n_comp})
    rows = []
    for subject, split in held_out:
        path = occ_dir / (subject + ".npy")
        if not path.exists():
            continue
        gt_occ = np.load(path)
        gt = gt_occ > 0.5
        sdf = occupancy_to_sdf(gt_occ, spacing, args.truncation).ravel()
        occ = gt_occ.astype(np.float32).ravel()
        for k in ks:
            rec_sdf = project(sdf_pca.components_, sdf_mean, sdf, k).reshape(shape) < 0
            rec_occ = project(occ_pca.components_, occ_mean, occ, k).reshape(shape) > 0.5
            rows.append({"subject": subject, "split": split, "k": k,
                         "iou_sdf": round(voxel_iou(rec_sdf, gt), 4),
                         "iou_occupancy": round(voxel_iou(rec_occ, gt), 4)})

    with ORACLE_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "split", "k", "iou_sdf", "iou_occupancy"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nHeld-out oracle IoU (mean over subjects)   SDF vs occupancy PCA")
    for split in ("validation", "test"):
        for k in ks:
            sel = [r for r in rows if r["split"] == split and r["k"] == k]
            if sel:
                print("  {:10s} k={:2d}  sdf={:.4f}  occupancy={:.4f}".format(
                    split, k, np.mean([r["iou_sdf"] for r in sel]),
                    np.mean([r["iou_occupancy"] for r in sel])))
    print("Oracle report:", ORACLE_CSV)


if __name__ == "__main__":
    main()
