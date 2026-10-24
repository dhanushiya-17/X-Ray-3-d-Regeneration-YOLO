"""
check_femur_sides.py
====================
Check whether the HFValid subjects mix left and right femurs.

The landmark frame uses z = x × y, which is right-handed for every subject.
A left and a right femur are mirror images, so in that frame one side has
its anterior bow pointing to +Z and the other to -Z. If both sides are in
the data, a shape model averages mirror-image shapes.

Two independent checks per subject:
  side      from the landmarks in CT physical space (ITK/LPS, +X = patient
            left): the greater trochanter is lateral, so its X offset from
            the femoral head gives the side.
  bow_mm    from data/occupancy_highres/<S>.npy: mid-shaft centroid offset
            in Z relative to the line through the shaft ends. Its sign
            should be the same for every subject if the sides are not mixed.

Output: data/femur_sides.csv and a summary on the console.
"""

import csv

import numpy as np

from femur_grid import (ROOT, BOUNDS, OCC_DIR, LANDMARK_ROOT, grid_spacing, load_split)


def read_points(subject):
    path = LANDMARK_ROOT / subject / (subject + "_palp.inp")
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = float(value.strip())
    return {name: np.array([values[name + "X"], values[name + "Y"], values[name + "Z"]])
            for name in ("femurHead", "headDir", "greatTroch")}


def anterior_bow(occ):
    """Mid-shaft Z centroid minus the mean of the two shaft-end centroids, in mm."""
    xmin, _, _, _, zmin, _ = BOUNDS
    sx, _, sz = grid_spacing()
    z_mm = zmin + sz * np.arange(occ.shape[2])

    def centroid(x_mm):
        i = int(round((x_mm - xmin) / sx))
        profile = occ[max(i - 2, 0):i + 3].sum(axis=(0, 1)).astype(np.float64)
        if profile.sum() == 0:
            return np.nan
        return float((profile * z_mm).sum() / profile.sum())

    # Shaft between the lesser trochanter and the supracondylar region.
    proximal, middle, distal = centroid(-100.0), centroid(-230.0), centroid(-360.0)
    return middle - 0.5 * (proximal + distal)


def collect():
    """Return one row per subject in the split file."""
    subjects = sorted(s for group in load_split().values() for s in group)
    rows = []
    for subject in subjects:
        row = {"subject": subject, "side": "", "troch_dx_mm": "", "bow_mm": "", "note": ""}
        try:
            p = read_points(subject)
            dx = float(p["greatTroch"][0] - p["femurHead"][0])
            row["troch_dx_mm"] = round(dx, 1)
            row["side"] = "L" if dx > 0 else "R"
        except (OSError, KeyError, ValueError) as exc:
            row["note"] = "landmarks: " + str(exc)
        occ_path = OCC_DIR / (subject + ".npy")
        if occ_path.exists():
            row["bow_mm"] = round(anterior_bow(np.load(occ_path)), 2)
        else:
            row["note"] += " missing occupancy"
        rows.append(row)
    return rows


def main():
    rows = collect()
    out = ROOT / "data" / "femur_sides.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sides = [r["side"] for r in rows if r["side"]]
    bows = np.array([r["bow_mm"] for r in rows if r["bow_mm"] != "" and np.isfinite(r["bow_mm"])])
    print(f"subjects: {len(rows)}   written: {out}")
    print(f"side from landmarks: L={sides.count('L')}  R={sides.count('R')}")
    print(f"bow sign: positive={(bows > 0).sum()}  negative={(bows < 0).sum()}  "
          f"median={np.median(bows):.2f} mm")
    for side in ("L", "R"):
        b = [r["bow_mm"] for r in rows if r["side"] == side and r["bow_mm"] != ""]
        if b:
            print(f"  {side}: n={len(b)}  median bow={np.median(b):.2f} mm  "
                  f"range {min(b):.2f} .. {max(b):.2f}")
    problems = [r for r in rows if r["note"]]
    for r in problems:
        print("  problem:", r["subject"], r["note"].strip())


if __name__ == "__main__":
    main()
