"""Build a small SYNTHETIC femur-like shape model for UI and pipeline smoke tests.

The shapes are procedural (sphere head, neck, trochanter, bowed shaft, two
condyles) with random proportions. They are not anatomy and say nothing about
reconstruction accuracy. The file is flagged `synthetic_demo=True`; the web UI
shows a banner whenever it is loaded. Use scripts/build_ssm_v2.py with the
HFValid-derived targets for the real model.

    python scripts/make_demo_model.py --output models/ssm_demo_synthetic.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BOUNDS = (-470.0, 40.0, -60.0, 100.0, -70.0, 70.0)


def femur_like(grid_shape, rng) -> np.ndarray:
    xs = np.linspace(BOUNDS[0], BOUNDS[1], grid_shape[0])
    ys = np.linspace(BOUNDS[2], BOUNDS[3], grid_shape[1])
    zs = np.linspace(BOUNDS[4], BOUNDS[5], grid_shape[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    length = rng.uniform(380, 450)
    head_r = rng.uniform(20, 26)
    shaft_r = rng.uniform(12, 16)
    neck_y = rng.uniform(30, 40)
    bow = rng.uniform(4, 12)
    occ = (X**2 + Y**2 + Z**2) <= head_r**2                                  # head
    t = np.clip(-X / 45.0, 0, 1)                                              # neck toward shaft
    neck_c = np.stack((-45 * t, neck_y * t), 0)
    occ |= ((X - neck_c[0])**2 * 0.25 + (Y - neck_c[1])**2 + Z**2 <= (head_r * 0.62)**2) & (X <= 0) & (X >= -50)
    occ |= ((X + 35)**2 / 22**2 + (Y - neck_y - 18)**2 / 16**2 + Z**2 / 18**2) <= 1     # greater trochanter
    shaft_t = np.clip((-X - 40) / (length - 80), 0, 1)
    shaft_y = neck_y - 6 - 10 * shaft_t
    shaft_z = bow * np.sin(np.pi * shaft_t)
    flare = 1 + 0.9 * np.clip((shaft_t - 0.8) / 0.2, 0, 1) ** 2
    occ |= ((Y - shaft_y)**2 + (Z - shaft_z)**2 <= (shaft_r * flare)**2) & (X <= -40) & (X >= -length + 25)
    for dy in (-14, 16):                                                      # condyles
        occ |= ((X + length - 20)**2 / 26**2 + (Y - shaft_y - dy)**2 / 18**2 + (Z + 6)**2 / 24**2) <= 1
    return occ.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=Path("models/ssm_demo_synthetic.npz"))
    ap.add_argument("--subjects", type=int, default=40)
    ap.add_argument("--components", type=int, default=12)
    ap.add_argument("--grid", type=int, nargs=3, default=(128, 48, 48))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force-overwrite", action="store_true")
    args = ap.parse_args()
    if args.output.exists() and not args.force_overwrite:
        raise SystemExit(f"Refusing to overwrite {args.output}; pass --force-overwrite")
    rng = np.random.default_rng(args.seed)
    shapes = np.stack([femur_like(tuple(args.grid), rng).ravel() for _ in range(args.subjects)])
    mean = shapes.mean(axis=0)
    _, s, vt = np.linalg.svd(shapes - mean, full_matrices=False)
    k = min(args.components, args.subjects - 1)
    variance = (s[:k] ** 2) / (args.subjects - 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, mean_shape=mean.astype(np.float32), components=vt[:k].astype(np.float32),
        explained_variance=variance.astype(np.float32),
        explained_variance_ratio=(variance / ((s ** 2).sum() / (args.subjects - 1))).astype(np.float32),
        grid_shape=np.asarray(args.grid), bounds=np.asarray(BOUNDS),
        train_subjects=np.asarray([f"Synthetic{i:03d}" for i in range(args.subjects)]),
        synthetic_demo=np.asarray(True))
    print(f"Saved synthetic demo model ({args.subjects} shapes, {k} modes) to {args.output}")


if __name__ == "__main__":
    main()
