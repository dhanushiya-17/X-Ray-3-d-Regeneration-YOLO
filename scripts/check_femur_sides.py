"""
check_femur_sides.py
====================
Check whether the HFValid subjects mix left and right femurs, and whether
mirroring the right ones (femur_grid.CANONICAL_SIDE) makes them consistent.

The landmark frame uses z = x × y, which is right-handed for every subject.
A left and a right femur are mirror images, so in that frame one side has
its anterior bow pointing to +Z and the other to -Z. If both sides are in
the data, a shape model averages mirror-image shapes.

Two independent checks per subject:
  side          from the landmarks in CT physical space (ITK/LPS, +X = patient
                left): the greater trochanter is lateral, so its X offset from
                the femoral head gives the side.
  bow_mm        from data/occupancy_highres/<S>.npy as stored: mid-shaft
                centroid offset in Z relative to the line through two shaft
                points at 25% and 75% of the femur's length.
  bow_canonical_mm  the same after mirroring onto CANONICAL_SIDE. Its sign
                should be the same for every subject.

Output: data/femur_sides.csv and a summary on the console.
"""

import csv

import numpy as np

from femur_grid import (ROOT, BOUNDS, CANONICAL_SIDE, OCC_DIR, femur_side, grid_spacing,
                        load_split)


def anterior_bow(occ):
    """Mid-shaft Z centroid minus the mean of two shaft centroids, in mm (NaN if empty)."""
    zmin, sz = BOUNDS[4], grid_spacing()[2]
    z_mm = zmin + sz * np.arange(occ.shape[2])
    rows = np.flatnonzero(occ.any(axis=(1, 2)))
    if len(rows) < 20:
        return float("nan")
    lo, hi = rows[0], rows[-1]

    def centroid(frac):
        i = int(round(lo + frac * (hi - lo)))
        profile = occ[max(i - 2, 0):i + 3].sum(axis=(0, 1)).astype(np.float64)
        return float((profile * z_mm).sum() / profile.sum()) if profile.sum() else float("nan")

    # 25-75% of the length stays on the diaphysis, away from the head and condyles.
    return centroid(0.5) - 0.5 * (centroid(0.25) + centroid(0.75))


def collect():
    """Return one row per subject in the split file."""
    subjects = sorted(s for group in load_split().values() for s in group)
    rows = []
    for subject in subjects:
        row = {"subject": subject, "side": "", "bow_mm": "", "bow_canonical_mm": "", "note": ""}
        try:
            row["side"] = femur_side(subject)
        except (OSError, KeyError, ValueError) as exc:
            row["note"] = "landmarks: " + str(exc)
        occ_path = OCC_DIR / (subject + ".npy")
        if occ_path.exists():
            bow = anterior_bow(np.load(occ_path))
            if np.isfinite(bow):
                row["bow_mm"] = round(bow, 2)
                if row["side"]:
                    row["bow_canonical_mm"] = round(bow if row["side"] == CANONICAL_SIDE else -bow, 2)
            else:
                row["note"] += " empty or short occupancy"
        else:
            row["note"] += " missing occupancy"
        rows.append(row)
    return rows


def summarize(rows):
    def signs(key):
        v = np.array([r[key] for r in rows if r[key] != ""], dtype=float)
        return int((v > 0).sum()), int((v < 0).sum()), float(np.median(v)) if len(v) else float("nan")

    sides = [r["side"] for r in rows if r["side"]]
    return {"left": sides.count("L"), "right": sides.count("R"),
            "bow": signs("bow_mm"), "bow_canonical": signs("bow_canonical_mm")}


def main():
    rows = collect()
    out = ROOT / "data" / "femur_sides.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    s = summarize(rows)
    print(f"subjects: {len(rows)}   written: {out}")
    print(f"side from landmarks: L={s['left']}  R={s['right']}")
    print("bow sign as stored:      +{} / -{}   median {:.2f} mm".format(*s["bow"]))
    print("bow sign after mirroring: +{} / -{}   median {:.2f} mm".format(*s["bow_canonical"]))
    for r in rows:
        if r["note"]:
            print("  problem:", r["subject"], r["note"].strip())


if __name__ == "__main__":
    main()
