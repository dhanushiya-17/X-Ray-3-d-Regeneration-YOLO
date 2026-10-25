"""Train tiny SYNTHETIC multi-view networks for UI and pipeline smoke tests.

The training shapes are the procedural femur-like solids from
make_demo_model.py, with parallel-beam line integrals as their "X-rays". They
are not anatomy and say nothing about reconstruction accuracy; checkpoints are
flagged `synthetic_demo` and the web UI shows a banner while they are loaded.
The real networks come from scripts/run_retraining.py on HFValid data.

    python scripts/make_demo_network.py --output-dir models/demo
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from femur_grid import DIMENSIONS, grid_spacing
from make_demo_model import femur_like
from multiview_net import MultiViewFemurNet, dice_bce_loss
from train_multiview_net import normalize, pool2d, pool3d


def synthetic_set(n: int, seed: int):
    rng = np.random.default_rng(seed)
    _, sy, sz = grid_spacing()
    occs = [femur_like(DIMENSIONS, rng) for _ in range(n)]
    return occs, [{"ap": occ.sum(axis=2) * sz, "lat": occ.sum(axis=1) * sy} for occ in occs]


def train(views, occs, drrs, factor, steps, seed, log=print) -> dict:
    torch.manual_seed(seed)
    ny, nz = DIMENSIONS[1] // factor, DIMENSIONS[2] // factor
    x = {v: torch.from_numpy(np.stack([normalize(pool2d(d[v], factor)) for d in drrs]))[:, None] for v in views}
    y = torch.from_numpy(np.stack([pool3d(o, factor) for o in occs]))[:, None]
    n_val = max(1, len(occs) // 6)
    net = MultiViewFemurNet(views=views)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for step in range(steps):
        idx = torch.randint(n_val, len(occs), (4,))
        loss = dice_bce_loss(net({v: x[v][idx] for v in views}, (ny, nz)), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            log(f"  {'+'.join(views)} step {step + 1}/{steps} loss {loss.item():.3f}")
    net.eval()
    with torch.no_grad():
        pred = torch.sigmoid(net({v: x[v][:n_val] for v in views}, (ny, nz))) >= 0.5
    tgt = y[:n_val] >= 0.5
    dice = float((2 * (pred & tgt).sum() + 1) / (pred.sum() + tgt.sum() + 1))
    return {"model": net.state_dict(), "epoch": steps, "val": {"dice": dice},
            "config": {"views": list(views), "downsample": factor}, "synthetic_demo": True}


def build(output_dir: Path, subjects: int = 24, steps: int = 300, factor: int = 4, seed: int = 3, log=print) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    occs, drrs = synthetic_set(subjects, seed)
    mean_occ = np.mean(occs, axis=0) >= 0.5
    _, rows = np.nonzero(mean_occ.any(axis=(1, 2))[None])
    reference = {"ap": torch.from_numpy(mean_occ.any(axis=2)), "lat": torch.from_numpy(mean_occ.any(axis=1)),
                 "length_mm": float(np.ptp(rows) + 1) * grid_spacing()[0]}
    paths = {}
    for views, name in ((("ap", "lat"), "multiview_ap_lat_demo.pt"), (("ap",), "multiview_ap_demo.pt")):
        state = train(views, occs, drrs, factor, steps, seed, log) | {"reference": reference}
        torch.save(state, output_dir / name)
        paths[views] = output_dir / name
        log(f"Saved {output_dir / name} (synthetic validation Dice {state['val']['dice']:.3f})")
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("models/demo"))
    ap.add_argument("--subjects", type=int, default=24)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    t0 = time.perf_counter()
    build(args.output_dir, args.subjects, args.steps)
    print(f"Done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
