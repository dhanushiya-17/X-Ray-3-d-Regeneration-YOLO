"""
ssm_evaluate_3d.py
==================
Compute 3D surface-distance metrics for SSM-reconstructed femur meshes.

For each validation (or test) subject with a reconstructed STL:
  1. Load predicted mesh from reconstruction/ssm/{subject}_ssm_mesh.stl
     (voxel-space coordinates).
  2. Load the ground-truth occupancy grid and build its GT mesh via
     marching cubes in the same voxel-space coordinates.
  3. Compute symmetric mean surface distance (MSD) and the 95th percentile
     Hausdorff distance using random surface point sampling.
  4. Also compute voxel IoU from the stored occupancy grids.
  5. Compare to the retrieval-baseline MSD from multiview_validation_3d_metrics.csv.

Outputs
-------
  reconstruction/ssm/ssm_3d_metrics.csv
  reconstruction/ssm/ssm_vs_baseline_comparison.png
"""

from pathlib import Path
import csv, sys
import numpy as np
import vtk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE    = ROOT / "data" / "subject_split.csv"
OCC_DIR       = ROOT / "data" / "occupancy_targets"
SSM_OUT_DIR   = ROOT / "reconstruction" / "ssm"
BASELINE_FILE = ROOT / "reconstruction" / "retrieval_baseline" / "multiview_validation_3d_metrics.csv"

N_SAMPLES  = 5000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split(split_file):
    splits = {}
    with split_file.open(newline="") as f:
        for row in csv.DictReader(f):
            splits.setdefault(row["split"], []).append(row["subject"])
    return {k: sorted(v) for k, v in splits.items()}


def occ_to_vtk_mesh(occ_3d, level=0.5):
    """Convert occupancy grid to a VTK polydata via marching cubes."""
    try:
        verts, faces, _, _ = marching_cubes(occ_3d, level=level)
    except Exception:
        return None

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
    return pd


def sample_surface(mesh, n_samples, seed):
    """Sample n_samples points uniformly over the mesh surface."""
    rng = np.random.default_rng(seed)
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

    probs = np.asarray(areas, dtype=np.float64)
    probs /= probs.sum()
    sel = rng.choice(len(triangles), n_samples, p=probs, replace=True)

    pts = np.empty((n_samples, 3))
    for i, idx in enumerate(sel):
        a, b, c = triangles[idx]
        u, v = rng.random(2)
        if u + v > 1:
            u, v = 1 - u, 1 - v
        pts[i] = a + u * (b - a) + v * (c - a)
    return pts


def distances_to_surface(points, mesh):
    """Closest point distance from each point in `points` to `mesh`."""
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    result = np.empty(len(points))
    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id  = vtk.reference(0)
    dist2   = vtk.reference(0.0)

    for i, p in enumerate(points):
        locator.FindClosestPoint(p.tolist(), closest, cell_id, sub_id, dist2)
        result[i] = float(dist2) ** 0.5
    return result


def voxel_iou(pred, gt, threshold=0.5):
    p = (pred > threshold)
    g = (gt   > threshold)
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter) / float(union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    args = parser.parse_args()

    print("=" * 60)
    print("BoneVision -- SSM 3D Metric Evaluation")
    print("=" * 60)

    splits = load_split(SPLIT_FILE)
    subjects = splits.get(args.split, [])
    train_set = set(splits.get("train", []))

    # Load baseline for comparison
    baseline = {}
    if BASELINE_FILE.exists():
        with BASELINE_FILE.open(newline="") as f:
            for row in csv.DictReader(f):
                baseline[row["query_subject"]] = float(row["mean_surface_distance_mm"])

    results = []

    for subject in subjects:
        print("\n" + "-" * 45)
        print("Subject:", subject)

        if subject in train_set:
            print("  SKIP: subject is in training set.")
            continue

        pred_occ_path = SSM_OUT_DIR / (subject + "_ssm_occ.npy")
        if not pred_occ_path.exists():
            print("  SKIP: reconstructed occupancy not found:", pred_occ_path)
            continue

        gt_occ_path = OCC_DIR / (subject + ".npy")
        if not gt_occ_path.exists():
            print("  SKIP: GT occupancy not found:", gt_occ_path)
            continue

        pred_occ = np.load(pred_occ_path).astype(np.float32)
        gt_occ   = np.load(gt_occ_path).astype(np.float32)

        iou = voxel_iou(pred_occ, gt_occ)
        print("  Voxel IoU:", round(iou, 4))

        # Build meshes for surface distance
        pred_mesh = occ_to_vtk_mesh(pred_occ)
        gt_mesh   = occ_to_vtk_mesh(gt_occ)

        if pred_mesh is None or gt_mesh is None:
            print("  SKIP: could not build mesh from occupancy.")
            continue

        if pred_mesh.GetNumberOfCells() == 0 or gt_mesh.GetNumberOfCells() == 0:
            print("  SKIP: empty mesh.")
            continue

        # Sample surfaces
        pred_pts = sample_surface(pred_mesh, N_SAMPLES, RANDOM_SEED)
        gt_pts   = sample_surface(gt_mesh,   N_SAMPLES, RANDOM_SEED + 1000)

        if len(pred_pts) == 0 or len(gt_pts) == 0:
            print("  SKIP: zero surface samples.")
            continue

        d1 = distances_to_surface(pred_pts, gt_mesh)
        d2 = distances_to_surface(gt_pts,   pred_mesh)
        combined = np.concatenate([d1, d2])

        msd  = float(combined.mean())
        med  = float(np.median(combined))
        p95  = float(np.percentile(combined, 95))
        mxd  = float(combined.max())

        # NOTE: distances are in voxel units, not mm.
        # The occupancy grid has dimensions (128, 48, 48) spanning
        # (510, 160, 140) mm in anatomical space → voxel spacing is approx
        # (510/127, 160/47, 140/47) = (4.02, 3.40, 2.98) mm per voxel.
        # We report voxel-space distances with this caveat.

        baseline_msd = baseline.get(subject, float("nan"))

        print("  MSD (vox)   :", round(msd, 3))
        print("  Median (vox):", round(med, 3))
        print("  P95 (vox)   :", round(p95, 3))
        if not (baseline_msd != baseline_msd):
            print("  Baseline MSD (mm):", round(baseline_msd, 3))

        results.append({
            "subject":           subject,
            "voxel_iou":         round(iou, 4),
            "msd_voxels":        round(msd, 3),
            "median_vox":        round(med, 3),
            "p95_vox":           round(p95, 3),
            "max_vox":           round(mxd, 3),
            "baseline_msd_mm":   round(baseline_msd, 3) if not (baseline_msd != baseline_msd) else "nan",
        })

    if not results:
        print("\nNo valid results. Check that ssm_reconstruct.py has been run first.")
        sys.exit(0)

    # Save CSV
    csv_path = SSM_OUT_DIR / "ssm_3d_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nMetrics saved:", csv_path)

    iou_vals = [r["voxel_iou"]  for r in results]
    msd_vals = [r["msd_voxels"] for r in results]
    print("\nAggregate (", len(results), "subjects ):")
    print("  Mean voxel IoU      :", round(float(np.mean(iou_vals)), 4))
    print("  Mean MSD (voxels)   :", round(float(np.mean(msd_vals)), 3))
    print("  Median MSD (voxels) :", round(float(np.median(msd_vals)), 3))

    # Comparison plot vs. baseline
    baseline_vals = [r["baseline_msd_mm"] for r in results
                     if r["baseline_msd_mm"] != "nan"]
    if baseline_vals:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        subj_labels = [r["subject"] for r in results]
        x = np.arange(len(subj_labels))

        axes[0].bar(x, msd_vals, color="#4C72B0", label="SSM MSD (vox)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(subj_labels, rotation=45, ha="right", fontsize=7)
        axes[0].set_ylabel("Mean Surface Distance (voxels)")
        axes[0].set_title("SSM Reconstruction MSD per Subject")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.4)

        axes[1].bar(x, iou_vals, color="#DD8452", label="Voxel IoU")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(subj_labels, rotation=45, ha="right", fontsize=7)
        axes[1].set_ylabel("Voxel IoU")
        axes[1].set_title("SSM Reconstruction Voxel IoU per Subject")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.4)

        plt.suptitle("BoneVision SSM Reconstruction -- " + args.split + " set", fontsize=10)
        plt.tight_layout()
        plot_path = SSM_OUT_DIR / "ssm_metrics_per_subject.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Per-subject plot saved:", plot_path)

    print("=" * 60)
    print("Evaluation complete.")
    print("=" * 60)
    print("\nIMPORTANT: Surface distances are in voxel units (not mm).")
    print("The occupancy grid covers ~510x160x140 mm in anatomical space.")
    print("Approximate voxel sizes: X~4.0 mm, Y~3.4 mm, Z~3.0 mm.")
    print("To convert MSD to mm, multiply voxel MSD by the appropriate spacing.")


if __name__ == "__main__":
    main()
