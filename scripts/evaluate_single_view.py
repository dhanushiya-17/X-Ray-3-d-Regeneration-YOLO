"""Held-out single-view SSM vs 1-view retrieval evaluation on synthetic silhouettes.

Two input modes:

* ``ideal`` (default, the protocol behind the README numbers): the exact AP
  silhouette projected from each held-out CT occupancy, already in the SSM
  frame at its true size and position.
* ``app``: the same silhouette passed through the app's own standardization
  (rotation to the long axis, physical rescaling to the population-mean length,
  re-centring) before fitting. This removes the true size/position that the
  ideal mode leaks and is the closer proxy for an uncalibrated upload.

Both modes fit all four orientation hypotheses and score the posed occupancy
(the same geometry the exported mesh uses). Neither mode tests real-radiograph
segmentation, detector geometry or clinical domain shift.
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "subject_split.csv"
OCC = ROOT / "data" / "occupancy_highres"
MODEL = ROOT / "models" / "ssm_highres.npz"
OUT = ROOT / "reconstruction" / "single_view_test_metrics.csv"


def read_splits():
    rows = list(csv.DictReader(SPLIT.open(newline="")))
    train = sorted(r["subject"] for r in rows if r["split"] == "train")
    validation = sorted(r["subject"] for r in rows if r["split"] == "validation")
    test = sorted(r["subject"] for r in rows if r["split"] == "test")
    if len(train) != 70 or len(validation) != 15 or len(test) != 16:
        raise ValueError(f"Expected the saved 70/15/16 split, got train={len(train)} validation={len(validation)} test={len(test)}")
    return train, validation, test


def metrics(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float], seed: int) -> dict:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    dice = 2.0 * inter / max(1, int(pred.sum() + gt.sum()))
    iou = inter / max(1, union)
    structure = ndi.generate_binary_structure(3, 1)
    pred_surface = pred & ~ndi.binary_erosion(pred, structure=structure, border_value=0)
    gt_surface = gt & ~ndi.binary_erosion(gt, structure=structure, border_value=0)
    p = np.argwhere(pred_surface)
    g = np.argwhere(gt_surface)
    if not len(p) or not len(g):
        raise ValueError("Prediction or reference has an empty surface")
    rng = np.random.default_rng(seed)
    if len(p) > 25000:
        p = p[rng.choice(len(p), 25000, replace=False)]
    if len(g) > 25000:
        g = g[rng.choice(len(g), 25000, replace=False)]
    d_to_gt = ndi.distance_transform_edt(~gt_surface, sampling=spacing)[tuple(p.T)]
    d_to_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)[tuple(g.T)]
    both = np.concatenate((d_to_gt, d_to_pred))
    pooled_p95 = float(np.percentile(both, 95))
    symmetric_hausdorff95 = float(max(np.percentile(d_to_gt, 95), np.percentile(d_to_pred, 95)))
    return {
        "dice": float(dice), "iou": float(iou),
        "mean_symmetric_surface_distance_mm": float(both.mean()),
        "median_surface_distance_mm": float(np.median(both)),
        "p95_surface_distance_mm": pooled_p95,
        "hausdorff95_symmetric_mm": symmetric_hausdorff95,
        "chamfer_mean_squared_mm2": float(np.mean(np.square(both))),
    }


def silhouette_dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    return float(2 * inter / max(1, int(a.sum() + b.sum())))


def plot_metrics(csv_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in csv.DictReader(csv_path.open(newline="")) if r["status"] == "ok"]
    if not rows:
        raise ValueError("No successful test rows are available for plotting")
    series = [
        ([float(r["ssm_mean_symmetric_surface_distance_mm"]) for r in rows], [float(r["retrieval_mean_symmetric_surface_distance_mm"]) for r in rows]),
        ([float(r["ssm_dice"]) for r in rows], [float(r["retrieval_dice"]) for r in rows]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, values, title, ylabel in zip(axes, series, ("Surface distance", "Voxel overlap"), ("Mean symmetric surface distance (mm)", "Dice")):
        ax.boxplot(values, showfliers=False)
        ax.set_xticks((1, 2), ("SSM fit", "1-view template"))
        for x, vals in enumerate(values, start=1):
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.zeros(1)
            ax.scatter(np.full(len(vals), x) + jitter, vals, s=18, alpha=0.75, color=("#2b7a9b", "#d18c40")[x-1])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("BoneVision held-out single-view synthetic silhouette test")
    fig.text(0.5, 0.01, "16 held-out subjects; ideal mask and canonical AP projection. Not real-X-ray validation.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    output = csv_path.with_name(csv_path.stem + "_summary.png")
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100, help="Maximum Adam fit iterations (default: 100)")
    parser.add_argument("--split", choices=("validation", "test"), default="test", help="Evaluate validation for tuning or test once for final held-out reporting")
    parser.add_argument("--prior-weight", type=float, default=0.02, help="L2 penalty on standardized PCA coefficients; choose using validation only")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test cap; 0 evaluates all 16 test subjects")
    parser.add_argument("--output", type=Path, default=None, help="Result CSV path; defaults to a split-specific path")
    parser.add_argument("--force-overwrite", action="store_true", help="Allow replacement of an existing CSV or summary figure")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--input-mode", choices=("ideal", "app"), default="ideal", help="ideal: exact canonical silhouette; app: through the app's standardization")
    parser.add_argument("--plot-only", action="store_true", help="Generate the summary figure from an existing metrics CSV without rerunning the evaluation")
    args = parser.parse_args()
    default_name = OUT.name if args.split == "test" else "single_view_validation_metrics.csv"
    if args.input_mode == "app":
        default_name = default_name.replace(".csv", "_appmode.csv")
    args.output = args.output or OUT.with_name(default_name)
    figure_path = args.output.with_name(args.output.stem + "_summary.png")
    if args.plot_only:
        if figure_path.exists() and not args.force_overwrite:
            raise FileExistsError(f"Refusing to overwrite {figure_path}; choose another --output or pass --force-overwrite")
        figure = plot_metrics(args.output)
        print(f"Saved summary figure: {figure}")
        return
    collisions = [str(path) for path in (args.output, figure_path) if path.exists()]
    if collisions and not args.force_overwrite:
        raise FileExistsError("Refusing to overwrite existing output(s): " + ", ".join(collisions) + ". Choose a new --output or pass --force-overwrite.")
    if not MODEL.exists():
        raise FileNotFoundError(f"Missing model: {MODEL}")
    train, validation, test = read_splits()
    subjects = validation if args.split == "validation" else test
    if args.limit > 0:
        subjects = subjects[:args.limit]
    import sys
    sys.path.insert(0, str(ROOT))
    from bonevision.pipeline import fit_hypotheses, hypothesis_targets, load_model, pose_volume, standardize_mask_with_transform
    model = load_model(MODEL, None if args.device == "auto" else args.device)
    if set(model.train_subjects) != set(train):
        raise ValueError("SSM training subjects do not exactly match the saved training split")
    spacing = tuple(model.spacing)
    train_projections, train_volumes = [], []
    for subject in train:
        vol = np.load(OCC / f"{subject}.npy") > 0
        train_volumes.append(vol)
        train_projections.append(vol.any(axis=2))
    train_projections = np.stack(train_projections)
    rows = []
    for index, subject in enumerate(subjects):
        started = time.perf_counter()
        try:
            gt = np.load(OCC / f"{subject}.npy") > 0
            silhouette = gt.any(axis=2)
            if args.input_mode == "app":
                target, _ = standardize_mask_with_transform(silhouette, model.grid_shape[:2], model.reference_mask, model.spacing[:2])
            else:
                target = silhouette
            targets = hypothesis_targets(target)
            fit = fit_hypotheses(model, targets, args.iterations, args.prior_weight)
            choice = fit["choice"]
            pred = pose_volume(model.decode(fit["params"]), fit["pose"]) > 0.5
            ssm_metrics = metrics(pred, gt, spacing, seed=1000+index)
            query_for_templates = targets[choice]
            ssm_proj_dice = silhouette_dice(fit["projection"] >= 0.5, query_for_templates)
            scores = np.asarray([silhouette_dice(candidate, query_for_templates) for candidate in train_projections])
            selected = int(np.argmax(scores))
            retrieval_metrics = metrics(train_volumes[selected], gt, spacing, seed=2000+index)
            row = {"subject": subject, "status": "ok", "error": "", "retrieval_template": train[selected], "runtime_ssm_seconds": time.perf_counter()-started, "projection_soft_dice_ssm": fit["soft_dice"][choice], "projection_dice_ssm": ssm_proj_dice, "projection_dice_retrieval": float(scores[selected]), "orientation": choice, "input_mode": args.input_mode, "iterations": args.iterations, "prior_weight": args.prior_weight}
            row.update({"ssm_"+k:v for k,v in ssm_metrics.items()})
            row.update({"retrieval_"+k:v for k,v in retrieval_metrics.items()})
        except Exception as exc:
            row = {"subject": subject, "status": "error", "error": f"{type(exc).__name__}: {exc}", "retrieval_template": "", "runtime_ssm_seconds": time.perf_counter()-started, "iterations": args.iterations, "prior_weight": args.prior_weight}
        rows.append(row)
        if row["status"] == "ok":
            print(f"{subject}: SSM Dice={ssm_metrics['dice']:.3f} IoU={ssm_metrics['iou']:.3f} MSD={ssm_metrics['mean_symmetric_surface_distance_mm']:.2f} mm; 1-view template={train[selected]} Dice={retrieval_metrics['dice']:.3f}", flush=True)
        else:
            print(f"{subject}: FAILED: {row['error']}", flush=True)
    successful = [r for r in rows if r["status"] == "ok"]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    summary = {"subject": "MEAN", "status": "aggregate", "error": "", "retrieval_template": "", "runtime_ssm_seconds": float(np.mean([r["runtime_ssm_seconds"] for r in rows])), "failed_subjects": len(rows)-len(successful), "iterations": args.iterations, "prior_weight": args.prior_weight}
    for key in keys:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float, np.number))]
        if values and key not in summary:
            summary[key] = float(np.mean(values))
    rows.append(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(subjects)} {args.split} rows plus aggregate mean to {args.output}")
    print(f"Input mode: {args.input_mode}. Limit: synthetic CT silhouettes, not real X-ray or automatic segmentation accuracy.")
    if successful:
        print(f"Summary figure: {plot_metrics(args.output)}")
    else:
        print("No summary figure generated because every subject failed; failure rows remain in the CSV.")


if __name__ == "__main__":
    main()
