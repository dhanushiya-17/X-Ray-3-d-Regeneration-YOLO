"""
ssm_reconstruct_v2.py
=====================
Oracle SSM capacity experiment (not arbitrary-X-ray reconstruction).

Key improvements over v1:
  1. Uses high-res (256x96x96) SSM with 60 components
  2. Multi-view projection matching (all 8 views)
  3. Geometrically consistent projections (anatomical-frame orthographic)
  4. Dice + contour Chamfer combined objective
  5. CMA-ES global optimizer with L-BFGS-B local refinement
  6. Warm-start from retrieval baseline
  7. Output in millimetre coordinates with proper spacing

IMPORTANT: This script uses each evaluated subject's ground-truth occupancy
grid to generate target silhouettes. It measures shape-model capacity under
oracle projections; it does not reconstruct an arbitrary uploaded X-ray and
must not be reported as single-view inference.

Algorithm:
  - For each query subject, generate 8 ground-truth silhouette projections
    from the anatomical-frame occupancy grid (the "input X-rays").
  - Optimize SSM shape parameters so the SSM's own 8 projections best
    match these target silhouettes (multi-view Dice + contour loss).
  - Extract the reconstructed surface mesh via marching cubes.
  - Evaluate voxel IoU and surface distances vs. ground truth.
"""

from pathlib import Path
import csv, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from scipy.ndimage import rotate as ndrotate
from skimage.measure import marching_cubes
import vtk

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT        = Path(__file__).resolve().parents[1]
MODEL_FILE  = ROOT / "models" / "ssm_highres.npz"
SPLIT_FILE  = ROOT / "data" / "subject_split.csv"
OCC_DIR     = ROOT / "data" / "occupancy_highres"
MATCH_FILE  = ROOT / "reconstruction" / "retrieval_baseline" / "multiview_validation_matches.csv"
OUTPUT_DIR  = ROOT / "reconstruction" / "ssm_v2"

# Projection viewing angles (degrees around X-axis in anatomical frame)
VIEW_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

# Optimization settings
MAX_COMP     = 40      # use top-40 PCA modes (balance capacity vs. DoF)
STD_LIMIT    = 3.0     # parameter bounds in stddevs
DE_MAXITER   = 150     # differential evolution generations
DE_POPSIZE   = 20      # population size per parameter
LBFGSB_MAXITER = 200   # local refinement iterations

# Anatomical bounds (from build_highres_targets.py)
XMIN, XMAX = -470.0,  40.0
YMIN, YMAX =  -60.0, 100.0
ZMIN, ZMAX =  -70.0,  70.0


def load_ssm():
    data = np.load(MODEL_FILE)
    ssm = {
        "mean_shape":         data["mean_shape"],
        "components":         data["components"],
        "explained_variance": data["explained_variance"],
        "grid_shape":         tuple(int(x) for x in data["grid_shape"]),
        "train_subjects":     list(data["train_subjects"]),
    }
    return ssm


def load_split():
    splits = {}
    with open(SPLIT_FILE, newline="") as f:
        for row in csv.DictReader(f):
            splits.setdefault(row["split"], []).append(row["subject"])
    return {k: sorted(v) for k, v in splits.items()}


# ===================================================================
#  Projection functions
# ===================================================================

def project_ap(occ_3d):
    """Project along Z axis (AP view, angle=0). Returns 2D binary silhouette."""
    return (occ_3d.sum(axis=2) > 0).astype(np.float32)

def project_lateral(occ_3d):
    """Project along Y axis (lateral view, angle=90)."""
    return (occ_3d.sum(axis=1) > 0).astype(np.float32)

def project_at_angle(occ_3d, angle_deg):
    """
    Project occupancy grid at arbitrary angle around X-axis.
    angle_deg=0 -> AP (along Z), angle_deg=90 -> lateral (along Y).
    """
    if angle_deg == 0 or angle_deg == 360:
        return project_ap(occ_3d)
    if angle_deg == 90:
        return project_lateral(occ_3d)
    if angle_deg == 180:
        return (occ_3d.sum(axis=2) > 0).astype(np.float32)  # posterior
    if angle_deg == 270:
        return (occ_3d.sum(axis=1) > 0).astype(np.float32)  # lateral flipped

    # General case: rotate the YZ plane by angle_deg, then project along new Z
    # We rotate the grid around the X-axis (axis 0) in the YZ plane
    # scipy.ndimage.rotate rotates in the plane of the last two axes by default
    # For rotation around X: axes=(1,2)
    rotated = ndrotate(occ_3d.astype(np.float32), -angle_deg,
                       axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0)
    return (rotated.sum(axis=2) > 0).astype(np.float32)


def generate_multiview_silhouettes(occ_3d, angles):
    """Generate silhouette projections for all viewing angles."""
    silhouettes = {}
    for angle in angles:
        silhouettes[angle] = project_at_angle(occ_3d, angle)
    return silhouettes


# ===================================================================
#  Loss functions
# ===================================================================

def dice_2d(pred, target):
    """Dice coefficient between two binary 2D images. Higher is better."""
    p = (pred > 0.5).astype(np.float32)
    t = (target > 0.5).astype(np.float32)
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    if denom < 1.0:
        return 0.0
    return float(2.0 * inter / denom)


def contour_chamfer(pred, target):
    """
    Approximate Chamfer distance between contours of two binary silhouettes.
    Uses distance transforms for efficiency.
    """
    from scipy.ndimage import distance_transform_edt

    p = (pred > 0.5).astype(np.uint8)
    t = (target > 0.5).astype(np.uint8)

    # Contour = boundary pixels (dilated - original)
    p_contour = binary_dilation(p) & ~p.astype(bool)
    t_contour = binary_dilation(t) & ~t.astype(bool)

    if p_contour.sum() == 0 or t_contour.sum() == 0:
        return 100.0  # large penalty for empty silhouettes

    # Distance from each pred contour pixel to nearest target contour pixel
    dt_t = distance_transform_edt(~t_contour)
    dt_p = distance_transform_edt(~p_contour)

    d_p_to_t = dt_t[p_contour].mean()
    d_t_to_p = dt_p[t_contour].mean()

    return float(d_p_to_t + d_t_to_p)


# ===================================================================
#  Decode shape from PCA parameters
# ===================================================================

def decode_shape(params, ssm, n_comp):
    """Decode z-score parameters to 3D occupancy grid."""
    stds = np.sqrt(ssm["explained_variance"][:n_comp])
    p = np.clip(params[:n_comp], -STD_LIMIT, STD_LIMIT) * stds
    shape_vec = ssm["mean_shape"] + ssm["components"][:n_comp].T @ p
    return np.clip(shape_vec.reshape(ssm["grid_shape"]), 0.0, 1.0)


# ===================================================================
#  Objective function
# ===================================================================

def make_multiview_objective(target_silhouettes, ssm, n_comp, angles):
    """
    Returns objective function that:
    - Decodes shape parameters to 3D volume
    - Projects at all viewing angles
    - Compares to target silhouettes using Dice + contour loss
    - Returns negative composite score (to minimise)
    """
    def objective(params):
        occ = decode_shape(params, ssm, n_comp)
        occ_binary = (occ > 0.5).astype(np.float32)

        total_dice = 0.0
        total_chamfer = 0.0
        n_views = 0

        for angle in angles:
            pred_sil = project_at_angle(occ_binary, angle)
            tgt_sil = target_silhouettes[angle]

            d = dice_2d(pred_sil, tgt_sil)
            c = contour_chamfer(pred_sil, tgt_sil)

            total_dice += d
            total_chamfer += c
            n_views += 1

        mean_dice = total_dice / max(n_views, 1)
        mean_chamfer = total_chamfer / max(n_views, 1)

        # Combined loss: maximize Dice, minimize Chamfer
        # Scale Chamfer to be roughly comparable to Dice (empirically)
        loss = -(mean_dice - 0.01 * mean_chamfer)

        # Add mild regularisation on parameter magnitude
        reg = 0.001 * np.sum(params ** 2)

        return loss + reg

    return objective


# ===================================================================
#  IoU and surface distance evaluation
# ===================================================================

def voxel_iou(pred, gt, thr=0.5):
    p = (pred > thr)
    g = (gt > thr)
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def voxel_dice(pred, gt, thr=0.5):
    p = (pred > thr)
    g = (gt > thr)
    inter = (p & g).sum()
    return float(2 * inter) / float(p.sum() + g.sum()) if (p.sum() + g.sum()) > 0 else 0.0


def occ_to_stl(occ_3d, output_path, spacing, origin, level=0.5):
    """Marching cubes -> STL in mm coordinates."""
    try:
        verts, faces, _, _ = marching_cubes(occ_3d, level=level, spacing=spacing)
    except Exception:
        print("  marching_cubes failed; skipping STL.")
        return None

    # Shift to anatomical mm coordinates
    verts[:, 0] += origin[0]
    verts[:, 1] += origin[1]
    verts[:, 2] += origin[2]

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
    return pd


def surface_distances_mm(pred_mesh, gt_mesh, n_samples=10000, seed=42):
    """Symmetric sampled surface distance in mm."""
    def sample(mesh, n, s):
        rng = np.random.default_rng(s)
        triangles, areas = [], []
        for i in range(mesh.GetNumberOfCells()):
            cell = mesh.GetCell(i)
            if cell.GetNumberOfPoints() != 3:
                continue
            a = np.array(mesh.GetPoint(cell.GetPointId(0)))
            b = np.array(mesh.GetPoint(cell.GetPointId(1)))
            c = np.array(mesh.GetPoint(cell.GetPointId(2)))
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
            if area > 0:
                triangles.append((a, b, c))
                areas.append(area)
        if not triangles:
            return np.zeros((0, 3))
        probs = np.asarray(areas)
        probs /= probs.sum()
        sel = rng.choice(len(triangles), n, p=probs, replace=True)
        pts = np.empty((n, 3))
        for i, idx in enumerate(sel):
            a, b, c = triangles[idx]
            u, v = rng.random(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            pts[i] = a + u * (b - a) + v * (c - a)
        return pts

    def closest_dist(points, mesh):
        loc = vtk.vtkStaticCellLocator()
        loc.SetDataSet(mesh)
        loc.BuildLocator()
        result = np.empty(len(points))
        closest = [0.0, 0.0, 0.0]
        cid = vtk.reference(0)
        sid = vtk.reference(0)
        d2 = vtk.reference(0.0)
        for i, p in enumerate(points):
            loc.FindClosestPoint(p.tolist(), closest, cid, sid, d2)
            result[i] = float(d2) ** 0.5
        return result

    pts_p = sample(pred_mesh, n_samples, seed)
    pts_g = sample(gt_mesh,   n_samples, seed + 7777)
    if len(pts_p) == 0 or len(pts_g) == 0:
        return {"msd": float("nan"), "p95": float("nan"), "hd": float("nan")}

    d1 = closest_dist(pts_p, gt_mesh)
    d2 = closest_dist(pts_g, pred_mesh)
    combined = np.concatenate([d1, d2])
    return {
        "msd": float(combined.mean()),
        "median": float(np.median(combined)),
        "p95": float(np.percentile(combined, 95)),
        "hd":  float(combined.max()),
    }


# ===================================================================
#  Retrieval baseline warm-start
# ===================================================================

def load_retrieval_init(subject, ssm, n_comp):
    """Get PCA z-scores of the retrieval baseline match as warm-start."""
    if not MATCH_FILE.exists():
        return np.zeros(n_comp)
    with MATCH_FILE.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["query_subject"] == subject:
                template = row.get("selected_template")
                if template:
                    try:
                        gt = np.load(OCC_DIR / (template + ".npy")).astype(np.float32)
                        centered = gt.ravel() - ssm["mean_shape"]
                        coeffs = ssm["components"][:n_comp] @ centered
                        stds = np.sqrt(ssm["explained_variance"][:n_comp])
                        z = coeffs / np.maximum(stds, 1e-10)
                        return np.clip(z, -STD_LIMIT, STD_LIMIT)
                    except Exception:
                        pass
    return np.zeros(n_comp)


# ===================================================================
#  Main
# ===================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--split", default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--max-comp", type=int, default=MAX_COMP)
    parser.add_argument("--views", type=int, default=8,
                        choices=[1, 2, 4, 8],
                        help="Number of views (1=AP, 2=AP+LAT, 4, 8=all)")
    args = parser.parse_args()

    print("=" * 60)
    print("BoneVision -- SSM Oracle Capacity Experiment (v2)")
    print("WARNING: target projections come from ground-truth occupancy;")
    print("         this is not inference from an uploaded clinical X-ray.")
    print("=" * 60)

    if not MODEL_FILE.exists():
        print("ERROR: Run build_ssm_v2.py first.")
        sys.exit(1)

    ssm = load_ssm()
    n_comp = min(args.max_comp, ssm["components"].shape[0])
    gx, gy, gz = ssm["grid_shape"]

    # Voxel spacing in mm
    sx = (XMAX - XMIN) / (gx - 1)
    sy = (YMAX - YMIN) / (gy - 1)
    sz = (ZMAX - ZMIN) / (gz - 1)
    spacing = (sx, sy, sz)
    origin  = (XMIN, YMIN, ZMIN)

    print("Grid:", ssm["grid_shape"], " Spacing(mm):",
          tuple(round(s, 2) for s in spacing))
    print("Components:", n_comp)
    print("Views:", args.views)

    # Select viewing angles
    if args.views == 1:
        angles = [0]
    elif args.views == 2:
        angles = [0, 90]
    elif args.views == 4:
        angles = [0, 90, 180, 270]
    else:
        angles = VIEW_ANGLES

    splits = load_split()
    subjects = args.subjects if args.subjects else splits.get(args.split, [])
    train_set = set(splits.get("train", []))
    for s in subjects:
        if s in train_set:
            print("ERROR:", s, "is training data!")
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for subject in subjects:
        print("\n" + "-" * 55)
        print("Subject:", subject)
        t0 = time.time()

        gt_path = OCC_DIR / (subject + ".npy")
        if not gt_path.exists():
            print("  SKIP: no GT occupancy")
            continue

        gt_occ = np.load(gt_path).astype(np.float32)

        # Generate multi-view target silhouettes from GT
        gt_binary = (gt_occ > 0.5).astype(np.float32)
        target_sils = generate_multiview_silhouettes(gt_binary, angles)

        # Warm-start from retrieval baseline
        params_init = load_retrieval_init(subject, ssm, n_comp)
        objective = make_multiview_objective(target_sils, ssm, n_comp, angles)
        loss_init = objective(params_init)
        print("  Initial loss:", round(loss_init, 4))

        # --- Stage 1: Differential Evolution (global search) ---
        bounds_de = [(-STD_LIMIT, STD_LIMIT)] * n_comp
        print("  Stage 1: Differential Evolution (pop={}, maxiter={})...".format(
            DE_POPSIZE, DE_MAXITER))

        result_de = differential_evolution(
            objective,
            bounds=bounds_de,
            x0=params_init,
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            tol=1e-8,
            seed=42,
            disp=False,
            init="latinhypercube",
        )
        print("  DE result: loss={:.4f}  iterations={}  converged={}".format(
            result_de.fun, result_de.nit, result_de.success))

        # --- Stage 2: L-BFGS-B local refinement ---
        print("  Stage 2: L-BFGS-B local refinement...")
        result_lb = minimize(
            objective,
            result_de.x,
            method="L-BFGS-B",
            bounds=bounds_de,
            options={"maxiter": LBFGSB_MAXITER, "ftol": 1e-10, "gtol": 1e-8},
        )
        print("  L-BFGS-B result: loss={:.4f}  iterations={}".format(
            result_lb.fun, result_lb.nit))

        params_opt = result_lb.x if result_lb.fun <= result_de.fun else result_de.x
        loss_opt = min(result_lb.fun, result_de.fun)
        t1 = time.time()

        # Decode final shape
        occ_rec = decode_shape(params_opt, ssm, n_comp)
        iou = voxel_iou(occ_rec, gt_occ)
        dice = voxel_dice(occ_rec, gt_occ)

        print("  Voxel IoU:", round(iou, 4))
        print("  Voxel Dice:", round(dice, 4))
        print("  Time:", round(t1 - t0, 1), "s")

        # Save occupancy
        np.save(OUTPUT_DIR / (subject + "_occ.npy"), occ_rec)

        # Save mesh and compute surface distances
        stl_path = OUTPUT_DIR / (subject + "_pred.stl")
        pred_mesh = occ_to_stl(occ_rec, stl_path, spacing, origin)

        gt_stl_path = OUTPUT_DIR / (subject + "_gt.stl")
        gt_mesh = occ_to_stl(gt_occ, gt_stl_path, spacing, origin)

        surf = {"msd": float("nan"), "median": float("nan"),
                "p95": float("nan"), "hd": float("nan")}
        if pred_mesh is not None and gt_mesh is not None:
            if pred_mesh.GetNumberOfCells() > 0 and gt_mesh.GetNumberOfCells() > 0:
                surf = surface_distances_mm(pred_mesh, gt_mesh, n_samples=10000)
                print("  Surface MSD (mm):", round(surf["msd"], 2))
                print("  Surface P95 (mm):", round(surf["p95"], 2))
                print("  Surface HD  (mm):", round(surf["hd"], 2))

        # Save comparison figure (AP + lateral + 3D slice)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i, angle in enumerate(angles[:4]):
            pred_sil = project_at_angle((occ_rec > 0.5).astype(np.float32), angle)
            gt_sil = target_sils.get(angle, np.zeros_like(pred_sil))

            # Overlay: green=GT, red=pred, yellow=overlap
            overlay = np.zeros((*pred_sil.shape, 3))
            overlay[:, :, 0] = pred_sil
            overlay[:, :, 1] = gt_sil

            axes[0][i].imshow(overlay, aspect="auto")
            axes[0][i].set_title("View " + str(angle) + " deg\nDice=" +
                                str(round(dice_2d(pred_sil, gt_sil), 3)))
            axes[0][i].axis("off")

        for i, angle in enumerate(angles[4:8] if len(angles) > 4 else []):
            pred_sil = project_at_angle((occ_rec > 0.5).astype(np.float32), angle)
            gt_sil = target_sils.get(angle, np.zeros_like(pred_sil))
            overlay = np.zeros((*pred_sil.shape, 3))
            overlay[:, :, 0] = pred_sil
            overlay[:, :, 1] = gt_sil
            axes[1][i].imshow(overlay, aspect="auto")
            axes[1][i].set_title("View " + str(angle) + " deg\nDice=" +
                                str(round(dice_2d(pred_sil, gt_sil), 3)))
            axes[1][i].axis("off")

        # Fill remaining axes with axial slices
        remaining = list(range(len(angles[4:8] if len(angles) > 4 else []), 4))
        for j, ax_idx in enumerate(remaining):
            sl = gx // (len(remaining) + 1) * (j + 1)
            axes[1][ax_idx].imshow(
                np.hstack([(occ_rec[sl, :, :] > 0.5).astype(float),
                           (gt_occ[sl, :, :] > 0.5).astype(float)]),
                cmap="gray", aspect="auto")
            axes[1][ax_idx].set_title("Slice X=" + str(sl) + "\nPred | GT")
            axes[1][ax_idx].axis("off")

        plt.suptitle(subject + "  IoU=" + str(round(iou, 3))
                     + "  Dice=" + str(round(dice, 3))
                     + "  MSD=" + str(round(surf["msd"], 1)) + "mm"
                     + "  (" + str(args.views) + " views, "
                     + str(n_comp) + " comp, "
                     + str(round(t1 - t0, 0)) + "s)", fontsize=10)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / (subject + "_comparison.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()

        results.append({
            "subject": subject,
            "n_comp": n_comp,
            "n_views": args.views,
            "loss_init": round(float(loss_init), 4),
            "loss_final": round(float(loss_opt), 4),
            "voxel_iou": round(iou, 4),
            "voxel_dice": round(dice, 4),
            "msd_mm": round(surf["msd"], 2) if not np.isnan(surf["msd"]) else "nan",
            "p95_mm": round(surf["p95"], 2) if not np.isnan(surf["p95"]) else "nan",
            "hd_mm": round(surf["hd"], 2) if not np.isnan(surf["hd"]) else "nan",
            "time_s": round(t1 - t0, 1),
        })

    # Save results
    if results:
        csv_path = OUTPUT_DIR / "reconstruction_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("\n" + "=" * 55)
        print("Results saved:", csv_path)

        iou_vals = [r["voxel_iou"] for r in results]
        dice_vals = [r["voxel_dice"] for r in results]
        msd_vals = [r["msd_mm"] for r in results if r["msd_mm"] != "nan"]
        print("Aggregate ({} subjects):".format(len(results)))
        print("  Mean IoU:  ", round(float(np.mean(iou_vals)), 4))
        print("  Mean Dice: ", round(float(np.mean(dice_vals)), 4))
        if msd_vals:
            print("  Mean MSD:  ", round(float(np.mean(msd_vals)), 2), "mm")
            print("  Mean P95:  ", round(float(np.mean(
                [r["p95_mm"] for r in results if r["p95_mm"] != "nan"])), 2), "mm")

    print("=" * 60)
    print("High-accuracy reconstruction complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
