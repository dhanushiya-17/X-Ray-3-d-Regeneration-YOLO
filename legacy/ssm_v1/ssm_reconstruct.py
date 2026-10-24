"""
ssm_reconstruct.py
==================
SSM-based 3D femur reconstruction from a single 2D X-ray (DRR).

Algorithm
---------
1. Load the pre-built SSM (mean shape + PCA components).
2. For a query subject, load its AP DRR (view_000.png) as the target image.
3. Initialise shape parameters (z-scores) from the retrieval baseline match
   if available, else start at zero (mean shape).
4. Render a 2D silhouette projection of the current shape hypothesis by
   summing voxel occupancies along the projection axis (axis 2 = Z, i.e.
   anterior-posterior), normalize, and compare to the query DRR with NCC.
5. Optimise shape parameters with scipy.optimize.minimize (L-BFGS-B),
   constraining each parameter within +/- 3 standard deviations.
6. Recover the reconstructed 3D occupancy grid from the optimised parameters.
7. Extract a surface mesh with marching cubes and save as STL.
8. Report NCC similarity and voxel IoU vs. ground truth occupancy.

Outputs (per subject, in reconstruction/ssm/)
---------------------------------------------
  {subject}_ssm_occ.npy      — reconstructed occupancy grid (128x48x48)
  {subject}_ssm_mesh.stl     — marching-cubes surface mesh
  {subject}_ssm_comparison.png — target DRR / projected SSM / GT projection
"""

from pathlib import Path
import csv, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from skimage.measure import marching_cubes
import vtk

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE  = ROOT / "models" / "ssm_model.npz"
SPLIT_FILE  = ROOT / "data" / "subject_split.csv"
OCC_DIR     = ROOT / "data" / "occupancy_targets"
DRR_DIR     = ROOT / "data" / "drr_dataset"
MATCH_FILE  = ROOT / "reconstruction" / "retrieval_baseline" / "multiview_validation_matches.csv"
OUTPUT_DIR  = ROOT / "reconstruction" / "ssm"

# Which DRR view to use as the query (0 = AP view).
QUERY_ANGLE = 0
QUERY_VIEW  = "view_000.png"

# Optimisation settings.
MAX_ITER    = 300
STD_LIMIT   = 3.0     # clamp shape params to +/- 3 std devs
SMOOTH_SIGMA = 1.0    # Gaussian smoothing sigma applied to projections before NCC


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_ssm(model_file):
    data = np.load(model_file)
    return {
        "mean_shape":             data["mean_shape"],           # (n_voxels,)
        "components":             data["components"],           # (k, n_voxels)
        "explained_variance":     data["explained_variance"],   # (k,)
        "grid_shape":             tuple(data["grid_shape"]),    # (X, Y, Z)
        "train_subjects":         list(data["train_subjects"]),
    }


# ---------------------------------------------------------------------------
# Projection (silhouette DRR substitute)
# ---------------------------------------------------------------------------

def project_occupancy(occ_3d, axis):
    """Sum along the given axis and normalise to [0,1]."""
    proj = occ_3d.sum(axis=axis).astype(np.float32)
    mx = proj.max()
    if mx > 0:
        proj /= mx
    return proj


def gaussian_smooth_2d(img, sigma):
    """Simple separable Gaussian blur without scipy dependency on GPU path."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img.astype(np.float64), sigma=sigma)


def ncc(a, b):
    """Normalised cross-correlation (scalar, higher = more similar)."""
    a = a.ravel()
    b = b.ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Shape from params
# ---------------------------------------------------------------------------

def decode_shape(params, ssm, std_limit=STD_LIMIT):
    """Convert shape-space z-scores to an occupancy grid."""
    stds = np.sqrt(ssm["explained_variance"])
    # clamp to allowed range
    p = np.clip(params, -std_limit, std_limit) * stds
    shape_vec = ssm["mean_shape"] + ssm["components"].T @ p
    shape_3d = shape_vec.reshape(ssm["grid_shape"])
    return np.clip(shape_3d, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(target_proj, ssm, proj_axis, sigma):
    """Return a callable that scipy.optimize.minimize will call."""
    target_smooth = gaussian_smooth_2d(target_proj, sigma)

    def objective(params):
        occ = decode_shape(params, ssm)
        proj = project_occupancy(occ, axis=proj_axis)
        proj_smooth = gaussian_smooth_2d(proj, sigma)
        return -ncc(proj_smooth, target_smooth)   # minimise → maximise NCC

    return objective


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def voxel_iou(pred, gt, threshold=0.5):
    p = (pred > threshold)
    g = (gt   > threshold)
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter) / float(union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Save STL from occupancy via marching cubes
# ---------------------------------------------------------------------------

def occ_to_stl(occ_3d, output_path, level=0.5):
    """Extract an isosurface and write an STL file."""
    try:
        verts, faces, _, _ = marching_cubes(occ_3d, level=level)
    except Exception:
        print("  marching_cubes failed (possibly empty volume); skipping STL.")
        return

    # Build VTK polydata
    points = vtk.vtkPoints()
    for v in verts:
        points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))

    polys = vtk.vtkCellArray()
    for f in faces:
        polys.InsertNextCell(3)
        for idx in f:
            polys.InsertCellPoint(int(idx))

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetPolys(polys)

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(pd)
    writer.Write()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_split():
    splits = {}
    with SPLIT_FILE.open(newline="") as f:
        for row in csv.DictReader(f):
            splits.setdefault(row["split"], []).append(row["subject"])
    return {k: sorted(v) for k, v in splits.items()}


def load_retrieval_init(subject):
    """Return the retrieval baseline match to warm-start optimisation."""
    if not MATCH_FILE.exists():
        return None
    with MATCH_FILE.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["query_subject"] == subject:
                return row.get("selected_template")
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SSM-based femur reconstruction")
    parser.add_argument("--subjects", nargs="*",
                        help="Subjects to reconstruct (default: all validation)")
    parser.add_argument("--split",   default="validation",
                        choices=["validation", "test"],
                        help="Which split to reconstruct")
    parser.add_argument("--max-components", type=int, default=20,
                        help="Number of PCA components to use (default: 20)")
    args = parser.parse_args()

    print("=" * 60)
    print("BoneVision -- SSM Reconstruction")
    print("=" * 60)

    # -- Load SSM --
    if not MODEL_FILE.exists():
        print("ERROR: SSM model not found. Run scripts/build_ssm.py first.")
        sys.exit(1)

    ssm = load_ssm(MODEL_FILE)
    n_comp = min(args.max_components, len(ssm["explained_variance"]))
    # Trim to requested number of components
    ssm["components"]         = ssm["components"][:n_comp]
    ssm["explained_variance"] = ssm["explained_variance"][:n_comp]

    print("SSM loaded:")
    print("  Grid shape    :", ssm["grid_shape"])
    print("  Components    :", n_comp)
    cum = np.cumsum(ssm["explained_variance"])
    total_ratio = cum[-1] / ssm["explained_variance"].sum() \
        if ssm["explained_variance"].sum() > 0 else 0.0
    # Note: explained_variance here is in absolute units; ratio must be computed
    # from full model
    full_ssm  = np.load(MODEL_FILE)
    full_ratio = np.cumsum(full_ssm["explained_variance_ratio"])
    print("  Cum. variance :", round(float(full_ratio[n_comp - 1]) * 100, 1), "%")

    # -- Which subjects --
    splits = load_split()
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = splits.get(args.split, [])

    print("Split          :", args.split)
    print("Subjects       :", len(subjects))

    # -- Verify no train subject contamination --
    train_set = set(splits.get("train", []))
    for s in subjects:
        if s in train_set:
            print("ERROR: subject", s, "is in the training set. Halting.")
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine projection axis.
    # In the anatomical grid (X=long axis, Y=mediolateral, Z=AP)
    # An AP X-ray projects along the Z axis → sum over axis 2.
    PROJ_AXIS = 2   # sum over Z to get the coronal/AP silhouette

    results = []

    for subject in subjects:
        print("\n" + "-" * 50)
        print("Subject:", subject)
        t0 = time.time()

        # -- Target DRR --
        drr_path = DRR_DIR / subject / QUERY_VIEW
        if not drr_path.exists():
            print("  WARNING: DRR not found:", drr_path)
            continue

        from PIL import Image
        target_img = np.asarray(
            Image.open(drr_path).convert("L"), dtype=np.float32
        ) / 255.0
        print("  Target DRR shape:", target_img.shape)

        # The SSM projection along PROJ_AXIS yields a (X, Y) image.
        # The SSM grid is (128, 48, 48) → projection = (128, 48).
        # The DRR is (512, 512). We need to resize the projection for comparison.
        # It's simpler to resize target to match the projected dimensions.
        from PIL import Image as PILImage
        gx, gy, gz = ssm["grid_shape"]
        # projecting over axis 2 (Z) → output is (gx, gy) = (128, 48)
        proj_h, proj_w = gx, gy
        target_resized = np.asarray(
            PILImage.fromarray((target_img * 255).astype(np.uint8)).resize(
                (proj_w, proj_h), PILImage.LANCZOS
            ), dtype=np.float32
        ) / 255.0
        print("  Target resized to projection dims:", target_resized.shape)

        # -- Initialise parameters --
        init_template = load_retrieval_init(subject)
        params0 = np.zeros(n_comp, dtype=np.float64)

        if init_template and init_template in ssm["train_subjects"]:
            idx = list(ssm["train_subjects"]).index(init_template)
            gt_occ = np.load(OCC_DIR / (init_template + ".npy")).astype(np.float32)
            x_vec = gt_occ.ravel() - ssm["mean_shape"]
            params0 = ssm["components"] @ x_vec   # project into PCA space
            params0 = params0 / np.sqrt(ssm["explained_variance"])  # z-scores
            params0 = np.clip(params0, -STD_LIMIT, STD_LIMIT)
            print("  Warm-started from retrieval match:", init_template)
        else:
            print("  Starting from mean shape (params=0)")

        # -- Objective function --
        objective = make_objective(target_resized, ssm, PROJ_AXIS, SMOOTH_SIGMA)

        print("  NCC at initialisation:", round(-objective(params0), 4))

        # -- Optimise --
        bounds = [(-STD_LIMIT, STD_LIMIT)] * n_comp
        result = minimize(
            objective,
            params0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": MAX_ITER, "ftol": 1e-8, "gtol": 1e-6},
        )

        params_opt = result.x
        ncc_opt = -result.fun
        t1 = time.time()

        print("  NCC after optimisation:", round(ncc_opt, 4),
              "  iterations:", result.nit,
              "  time:", round(t1 - t0, 1), "s")
        if not result.success:
            print("  Optimiser message:", result.message)

        # -- Decode reconstructed shape --
        occ_rec = decode_shape(params_opt, ssm)

        # -- Compare to ground truth --
        gt_occ_path = OCC_DIR / (subject + ".npy")
        gt_occ = None
        iou_val = float("nan")
        if gt_occ_path.exists():
            gt_occ = np.load(gt_occ_path).astype(np.float32)
            iou_val = voxel_iou(occ_rec, gt_occ)
            print("  Voxel IoU vs GT:", round(iou_val, 4))

        # -- Save occupancy --
        occ_path = OUTPUT_DIR / (subject + "_ssm_occ.npy")
        np.save(occ_path, occ_rec)

        # -- Save mesh --
        stl_path = OUTPUT_DIR / (subject + "_ssm_mesh.stl")
        occ_to_stl(occ_rec, stl_path)
        print("  Mesh saved:", stl_path)

        # -- Save comparison figure --
        proj_rec = project_occupancy(occ_rec, PROJ_AXIS)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(target_resized, cmap="gray", aspect="auto")
        axes[0].set_title("Query DRR (AP, resized)")
        axes[0].axis("off")

        axes[1].imshow(proj_rec, cmap="gray", aspect="auto")
        axes[1].set_title("SSM reconstruction projection\nNCC=" + str(round(ncc_opt, 3)))
        axes[1].axis("off")

        if gt_occ is not None:
            proj_gt = project_occupancy(gt_occ, PROJ_AXIS)
            axes[2].imshow(proj_gt, cmap="gray", aspect="auto")
            axes[2].set_title("GT occupancy projection\nIoU=" + str(round(iou_val, 3)))
        else:
            axes[2].set_title("GT not available")
        axes[2].axis("off")

        plt.suptitle("Subject: " + subject + "  Components: " + str(n_comp), fontsize=10)
        plt.tight_layout()
        fig_path = OUTPUT_DIR / (subject + "_ssm_comparison.png")
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close()

        results.append({
            "subject":       subject,
            "n_components":  n_comp,
            "ncc_init":      round(-objective(params0), 4),
            "ncc_optimised": round(ncc_opt, 4),
            "voxel_iou":     round(iou_val, 4) if not (iou_val != iou_val) else "nan",
            "iterations":    result.nit,
            "time_s":        round(t1 - t0, 1),
            "converged":     result.success,
            "retrieval_init": init_template or "",
        })

    # -- Save results CSV --
    if results:
        csv_path = OUTPUT_DIR / "ssm_validation_results.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("\nResults saved:", csv_path)

        iou_vals = [r["voxel_iou"] for r in results if r["voxel_iou"] != "nan"]
        ncc_vals = [r["ncc_optimised"] for r in results]
        print("Mean NCC:", round(float(np.mean(ncc_vals)), 4))
        if iou_vals:
            print("Mean voxel IoU:", round(float(np.mean(iou_vals)), 4))

    print("=" * 60)
    print("SSM reconstruction complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
