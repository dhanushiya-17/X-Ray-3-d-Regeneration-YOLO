"""
run_retraining.py
=================
Run the femur retraining steps in order and stop at the first result that
looks wrong, instead of training on bad data:

  1. check     GPU, input data, left/right sides: right femurs are mirrored onto
               the left, and the check stops if that does not make them consistent
  2. ssm       SDF shape model + held-out oracle (build_ssm_sdf.py)
  3. dataset   AP + lateral training pairs (build_biplanar_dataset.py),
               with an alignment check of each image against its 3D target
  4. smoke     2-epoch training run to catch memory or data errors early
  5. train     multi-view network (train_multiview_net.py)
  6. evaluate  validation gate, then the held-out test split, compared with
               the current single-view SSM numbers from the README

Steps whose outputs already exist are skipped, so the command can be re-run
after fixing a problem. Each step's console output is also saved in logs/.
The final summary is written to models/retraining_report.md.

Examples
  python scripts/run_retraining.py
  python scripts/run_retraining.py --epochs 300 --also-ap
  python scripts/run_retraining.py --from train     # redo training onwards
"""

import argparse
import csv
import subprocess
import sys
import time

import numpy as np

from femur_grid import ROOT, CANONICAL_SIDE, OCC_DIR, LANDMARK_ROOT, grid_spacing, load_split

SCRIPTS = ROOT / "scripts"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
DATA_DIR = ROOT / "data" / "biplanar"
STEPS = ("check", "ssm", "dataset", "smoke", "train", "evaluate")

# Current single-view SSM on the same test subjects (README, ideal AP silhouette).
SSM_BASELINE = {"dice": 0.687, "mean_surface_mm": 4.73, "worst_dice": 0.373}
MIN_VAL_DICE = 0.80


class StepFailed(Exception):
    pass


def banner(text):
    print("\n" + "=" * 70 + "\n" + text + "\n" + "=" * 70, flush=True)


def run(step, args):
    """Run a script with live output, saving a copy in logs/<step>.log."""
    LOG_DIR.mkdir(exist_ok=True)
    cmd = [sys.executable, "-u", str(SCRIPTS / args[0])] + args[1:]
    print("$", " ".join(cmd[2:]), flush=True)
    lines = []
    with (LOG_DIR / (step + ".log")).open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            lines.append(line)
        proc.wait()
    if proc.returncode != 0:
        raise StepFailed("{} exited with code {}. Full output: logs/{}.log".format(
            args[0], proc.returncode, step))
    return lines


def step_check(opts):
    import torch
    if not torch.cuda.is_available():
        raise StepFailed("PyTorch cannot see the GPU. Activate .venv-train and re-run the GPU check.")
    print("GPU:", torch.cuda.get_device_name(0), " torch", torch.__version__)

    splits = load_split()
    counts = {k: len(v) for k, v in splits.items()}
    print("Split:", counts)
    subjects = [s for v in splits.values() for s in v]
    missing_occ = [s for s in subjects if not (OCC_DIR / (s + ".npy")).exists()]
    missing_ct = [s for s in subjects if not (LANDMARK_ROOT / s / (s + ".nrrd")).exists()]
    missing_lm = [s for s in subjects if not (LANDMARK_ROOT / s / (s + "_palp.inp")).exists()]
    for name, miss in (("occupancy targets", missing_occ), ("CT volumes", missing_ct), ("landmarks", missing_lm)):
        print("{:18s} {}/{} present{}".format(name, len(subjects) - len(miss), len(subjects),
                                              "  missing e.g. " + ", ".join(miss[:3]) if miss else ""))
    if len(missing_occ) > 5 or len(missing_lm) > 5 or len(missing_ct) > 5:
        raise StepFailed("More than 5 subjects are missing input files (see above). The scripts expect "
                         "data/occupancy_highres/<S>.npy and data/raw/Subjects/<S>/<S>.nrrd + <S>_palp.inp.")

    from check_femur_sides import collect, summarize
    rows = collect()
    s = summarize(rows)
    raw_pos, raw_neg, _ = s["bow"]
    pos, neg, median = s["bow_canonical"]
    print("Side from landmarks: L={} R={}   bow sign as stored: +{} / -{}   "
          "after mirroring to {}: +{} / -{} (median {:.2f} mm)".format(
              s["left"], s["right"], raw_pos, raw_neg, CANONICAL_SIDE, pos, neg, median))
    with (MODEL_DIR / "femur_sides.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Mirroring must make the bow direction consistent; if it does not, the
    # side labels or the targets are not what we assume, so stop.
    minority, total = min(pos, neg), pos + neg
    if minority > max(3, 0.1 * total):
        raise StepFailed("After mirroring the right femurs, {} of {} still bow the opposite way, so "
                         "mirroring does not make the shapes consistent. See models/femur_sides.csv".format(
                             minority, total))
    n_mirror = s["right"] if CANONICAL_SIDE == "L" else s["left"]
    return "{} subjects ({} L, {} R; {} mirrored to {}), bow consistent in {} of {}".format(
        len(rows), s["left"], s["right"], n_mirror, CANONICAL_SIDE, max(pos, neg), total)


def step_ssm(opts):
    oracle = MODEL_DIR / "ssm_sdf_oracle.csv"
    if not (MODEL_DIR / "ssm_sdf.npz").exists() or not oracle.exists():
        run("ssm", ["build_ssm_sdf.py"])
    with oracle.open(newline="") as f:
        rows = list(csv.DictReader(f))
    k_max = max(int(r["k"]) for r in rows)
    val = [r for r in rows if r["split"] == "validation" and int(r["k"]) == k_max]
    sdf = float(np.mean([float(r["iou_sdf"]) for r in val]))
    occ = float(np.mean([float(r["iou_occupancy"]) for r in val]))
    print("Validation oracle IoU at k={}: SDF {:.4f}  occupancy {:.4f}".format(k_max, sdf, occ))
    if sdf + 0.01 < occ:
        # The network does not depend on the SSM, so this is a warning, not a stop.
        print("WARNING: the SDF model is not better than occupancy PCA on held-out shapes.")
    return "SDF oracle IoU {:.3f} vs occupancy {:.3f} (k={})".format(sdf, occ, k_max)


def _highpass(a, sigma=6.0):
    from scipy import ndimage as ndi
    a = a.astype(np.float64)
    return a - ndi.gaussian_filter(a, sigma)


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))


def view_margins(sample):
    """How much better each view lines up with its target than with shifted copies of it.

    The image and the projected target silhouette are high-pass filtered,
    which removes the smooth soft-tissue background and keeps the bone edges,
    then correlated. A matching pair correlates best unshifted. The shifts are
    16 mm across the shaft and 60 mm along it: the shaft looks the same after a
    short shift along its length, so only a long one tests the head and knee.
    Returns {"ap": margin, "lat": margin}, each the aligned correlation minus
    the best shifted one (> 0 means the target is where the image shows the femur).
    """
    d = np.load(sample)
    sx, sy, sz = grid_spacing(d["occ"].shape)
    out = {}
    for name, mask, s_across in (("ap", d["occ"].any(axis=2), sy), ("lat", d["occ"].any(axis=1), sz)):
        img = d[name]
        if not np.isfinite(img).all() or img.max() <= 0 or not mask.any():
            out[name] = -1.0
            continue
        across, along = int(round(16.0 / s_across)), int(round(60.0 / sx))
        hp = _highpass(img)
        aligned = _corr(hp, _highpass(mask))
        shifted = max(_corr(hp, _highpass(np.roll(mask, k, axis=ax)))
                      for ax, n in ((0, along), (1, across)) for k in (-n, n))
        out[name] = round(aligned - shifted, 4)
    return out


def step_dataset(opts):
    splits = load_split()
    expected = [s for v in splits.values() for s in v]
    if any(not (DATA_DIR / (s + ".npz")).exists() for s in expected):
        run("dataset", ["build_biplanar_dataset.py"])
    have = {s: (DATA_DIR / (s + ".npz")) for s in expected if (DATA_DIR / (s + ".npz")).exists()}
    per_split = {k: sum(s in have for s in v) for k, v in splits.items()}
    print("Samples per split:", per_split)
    for name, n in splits.items():
        if per_split[name] < 0.9 * len(n):
            raise StepFailed("Only {}/{} {} samples were built. Errors: data/biplanar/dataset_report.csv".format(
                per_split[name], len(n), name))
    per_view = {s: view_margins(p) for s, p in have.items()}
    with (MODEL_DIR / "alignment.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "ap_margin", "lat_margin"])
        for subj, m in sorted(per_view.items()):
            writer.writerow([subj, m["ap"], m["lat"]])
    for view in ("ap", "lat"):
        v = np.array([m[view] for m in per_view.values()])
        print("  {:3s} margin: median {:.3f}, below 0: {}".format(view, float(np.median(v)), int((v <= 0).sum())))
    scores = {subj: min(m.values()) for subj, m in per_view.items()}
    ranked = sorted((v, s) for s, v in scores.items())
    bad = [s for v, s in ranked if v <= 0.0]
    median = float(np.median(list(scores.values())))
    print("Image/target alignment (true silhouette vs shifted, >0 is aligned): median {:.2f}, lowest {}".format(
        median, ", ".join("{} {:.2f}".format(s, v) for v, s in ranked[:3])))
    if bad:
        print("Weakly aligned samples:", ", ".join(bad))
    if len(bad) > 0.1 * len(scores):
        raise StepFailed("{} samples line up with their 3D target no better than with a shifted copy of it. "
                         "Per-view numbers: models/alignment.csv. To look at them: "
                         "python scripts/plot_alignment.py".format(len(bad)))
    return "{} samples, alignment median {:.2f}{}".format(
        len(have), median, ", weak: " + " ".join(bad) if bad else "")


def checkpoint(views, opts):
    return MODEL_DIR / "multiview_{}_d{}.pt".format("_".join(views), opts.downsample)


def step_smoke(opts):
    if checkpoint(("ap", "lat"), opts).exists():
        return "skipped, the model is already trained"
    ckpt = MODEL_DIR / "_smoke_test.pt"
    for p in (ckpt, ckpt.with_suffix(".log.csv")):
        p.unlink(missing_ok=True)
    lines = run("smoke", ["train_multiview_net.py", "--downsample", str(opts.downsample), "--epochs", "2",
                          "--val-every", "1", "--batch-size", str(opts.batch_size), "--checkpoint", str(ckpt)])
    with ckpt.with_suffix(".log.csv").open(newline="") as f:
        log = list(csv.DictReader(f))
    epoch_s = float(log[-1]["seconds"])
    for p in (ckpt, ckpt.with_suffix(".log.csv")):
        p.unlink(missing_ok=True)
    if not any("val dice" in line for line in lines):
        raise StepFailed("The smoke run finished without a validation result. See logs/smoke.log")
    est_min = epoch_s * opts.epochs * (2 if opts.also_ap else 1) / 60 * 1.1
    print("One epoch takes {:.1f} s, so training should take about {:.0f} minutes.".format(epoch_s, est_min))
    return "{:.1f} s per epoch, about {:.0f} min of training".format(epoch_s, est_min)


def train_one(views, opts):
    ckpt = checkpoint(views, opts)
    if ckpt.exists():
        print("Using existing", ckpt.name)
    else:
        run("train_" + "_".join(views), ["train_multiview_net.py", "--views", ",".join(views),
                                          "--downsample", str(opts.downsample), "--epochs", str(opts.epochs),
                                          "--batch-size", str(opts.batch_size)])
    with ckpt.with_suffix(".log.csv").open(newline="") as f:
        log = list(csv.DictReader(f))
    losses = [float(r["train_loss"]) for r in log]
    vals = [float(r["val_dice"]) for r in log if r["val_dice"]]
    window = max(1, len(losses) // 10)
    first, last = np.mean(losses[:window]), np.mean(losses[-window:])
    print("{}: train loss {:.4f} -> {:.4f}, best validation Dice {:.4f}".format(
        ckpt.name, first, last, max(vals)))
    if len(losses) >= 20 and last > 0.8 * first:
        raise StepFailed("Training loss barely went down ({:.4f} -> {:.4f}). See {}".format(
            first, last, ckpt.with_suffix(".log.csv").name))
    return max(vals)


def step_train(opts):
    results = {"ap_lat": train_one(("ap", "lat"), opts)}
    if opts.also_ap:
        results["ap"] = train_one(("ap",), opts)
    return ", ".join("{} val Dice {:.3f}".format(k, v) for k, v in results.items())


def evaluate_one(views, split, opts):
    ckpt = checkpoint(views, opts)
    out = MODEL_DIR / "{}_{}_metrics.csv".format(ckpt.stem, split)
    if not out.exists():
        run("eval_{}_{}".format("_".join(views), split),
            ["train_multiview_net.py", "--evaluate", split, "--checkpoint", str(ckpt)])
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if r["status"] == "OK"]
    dice = [float(r["dice"]) for r in rows]
    worst = min(rows, key=lambda r: float(r["dice"]))
    return {"n": len(rows), "failed": len(rows) - len(ok), "dice": float(np.mean(dice)),
            "mean_surface_mm": float(np.mean([float(r["mean_symmetric_surface_distance_mm"]) for r in ok])),
            "hd95_mm": float(np.mean([float(r["hausdorff95_symmetric_mm"]) for r in ok])),
            "worst": "{} ({:.3f})".format(worst["subject"], float(worst["dice"])), "worst_dice": float(worst["dice"])}


def step_evaluate(opts):
    models = [("ap", "lat")] + ([("ap",)] if opts.also_ap and checkpoint(("ap",), opts).exists() else [])
    val = evaluate_one(("ap", "lat"), "validation", opts)
    print("Validation (AP + lateral):", val)
    if val["dice"] < MIN_VAL_DICE and not opts.force_test:
        raise StepFailed("Validation Dice {:.3f} is below {:.2f}. Not touching the test split yet, so it stays "
                         "a fair final check. Re-run with --force-test to evaluate anyway.".format(
                             val["dice"], MIN_VAL_DICE))
    results = {"_".join(v): evaluate_one(v, "test", opts) for v in models}
    voxel_mm = 2 * opts.downsample
    lines = ["# Femur retraining report", "",
             "Test split, synthetic parallel-beam DRRs. Network metrics are computed on a "
             "{} mm grid; the SSM baseline used 2 mm, so surface distances are not exactly comparable.".format(
                 voxel_mm), "",
             "| Model | Subjects | Mean Dice | Mean surface distance | HD95 | Worst case |",
             "|---|---|---|---|---|---|",
             "| Current SSM, single AP view (README) | 16 | {:.3f} | {:.2f} mm | | Pat099 ({:.3f}) |".format(
                 SSM_BASELINE["dice"], SSM_BASELINE["mean_surface_mm"], SSM_BASELINE["worst_dice"])]
    names = {"ap_lat": "Network, AP + lateral", "ap": "Network, AP only"}
    for key, r in results.items():
        lines.append("| {} | {}{} | {:.3f} | {:.2f} mm | {:.2f} mm | {} |".format(
            names[key], r["n"], " ({} failed)".format(r["failed"]) if r["failed"] else "",
            r["dice"], r["mean_surface_mm"], r["hd95_mm"], r["worst"]))
    lines += ["", "Validation Dice (AP + lateral): {:.3f}".format(val["dice"]),
              "Per-subject results: models/*_test_metrics.csv. Training curves: models/*.log.csv."]
    report = MODEL_DIR / "retraining_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    best = results["ap_lat"]
    return "test Dice {:.3f} vs SSM {:.3f}; report {}".format(best["dice"], SSM_BASELINE["dice"], report.name)


def main():
    parser = argparse.ArgumentParser(description="Run the femur retraining pipeline with checks between steps")
    parser.add_argument("--from", dest="start", choices=STEPS, default="check",
                        help="start at this step (earlier steps are skipped, not re-checked)")
    parser.add_argument("--to", dest="stop", choices=STEPS, default="evaluate")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--downsample", type=int, default=2, choices=(1, 2))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--also-ap", action="store_true", help="also train and test a single-view AP model")
    parser.add_argument("--force-test", action="store_true",
                        help="evaluate on the test split even if validation Dice is below the gate")
    opts = parser.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    steps = STEPS[STEPS.index(opts.start):STEPS.index(opts.stop) + 1]
    funcs = {"check": step_check, "ssm": step_ssm, "dataset": step_dataset,
             "smoke": step_smoke, "train": step_train, "evaluate": step_evaluate}
    summary = []
    for i, step in enumerate(steps, 1):
        banner("Step {}/{}: {}".format(i, len(steps), step))
        t0 = time.time()
        try:
            result = funcs[step](opts)
        except StepFailed as exc:
            summary.append((step, "STOPPED", str(exc)))
            break
        summary.append((step, "OK", "{} ({:.0f} s)".format(result, time.time() - t0)))
    banner("Summary")
    for step, status, text in summary:
        print("{:9s} {:8s} {}".format(step, status, text))
    if summary and summary[-1][1] == "STOPPED":
        print("\nFix the problem above, then re-run. Finished steps are skipped automatically.")
        sys.exit(1)


if __name__ == "__main__":
    main()
