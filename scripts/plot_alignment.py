"""
plot_alignment.py
=================
Save AP and lateral DRRs from data/biplanar with the outline of their 3D
target drawn on top, to check by eye that images and targets line up.

  python scripts/plot_alignment.py                 # 4 worst + 2 best by models/alignment.csv
  python scripts/plot_alignment.py Pat001 Pat096   # chosen subjects

Output: models/alignment_overlays.png
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from femur_grid import ROOT

DATA_DIR = ROOT / "data" / "biplanar"
ALIGN_CSV = ROOT / "models" / "alignment.csv"
OUT = ROOT / "models" / "alignment_overlays.png"


def pick_subjects():
    if len(sys.argv) > 1:
        return sys.argv[1:], {}
    with ALIGN_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    margins = {r["subject"]: (float(r["ap_margin"]), float(r["lat_margin"])) for r in rows}
    ranked = sorted(margins, key=lambda s: min(margins[s]))
    return ranked[:4] + ranked[-2:], margins


def main():
    subjects, margins = pick_subjects()
    fig, axes = plt.subplots(len(subjects), 2, figsize=(7, 3.2 * len(subjects)), squeeze=False)
    for row, subject in zip(axes, subjects):
        d = np.load(DATA_DIR / (subject + ".npz"))
        for ax, view, mask in ((row[0], "ap", d["occ"].any(axis=2)), (row[1], "lat", d["occ"].any(axis=1))):
            img = d[view]
            # Shaft runs along the first axis; show it vertically with the head at the top.
            ax.imshow(img[::-1], cmap="gray", aspect="auto",
                      vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
            ax.contour(mask[::-1].astype(float), levels=[0.5], colors="red", linewidths=0.8)
            m = margins.get(subject)
            label = "" if m is None else "  margin {:.3f}".format(m[0] if view == "ap" else m[1])
            ax.set_title("{} {}{}".format(subject, view.upper(), label), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print("Saved", OUT)


if __name__ == "__main__":
    main()
