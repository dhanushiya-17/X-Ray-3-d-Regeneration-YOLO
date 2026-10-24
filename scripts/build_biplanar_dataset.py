"""
build_biplanar_dataset.py
=========================
Create paired (AP image, lateral image, 3D occupancy) training samples for the
multi-view network, with the images and the target in the SAME anatomical
grid. This removes the camera-to-SSM-frame ambiguity documented in
docs/multiview-reconstruction-audit.md: every pixel row/column of the AP and
lateral images corresponds exactly to an X/Y (AP) or X/Z (lateral) column of
the 3D target.

For each subject the CT is resampled into the landmark frame
(q = R.T @ (p - origin), same as build_highres_targets.py). The projection
direction is integrated over a wider slab than the target grid (default
+/-160 mm) so overlying soft tissue and the pelvis appear as in a radiograph.
Projections are parallel-beam line integrals of linear attenuation; real
radiographs are cone-beam (magnification ~1.1-1.2 at 1 m SID), so keep the
pixel-scale augmentation in train_multiview_net.py switched on.

Input:  data/raw/Subjects/<S>/<S>.nrrd (full CT, default) or <S>_femur_only.nrrd
        data/raw/Subjects/<S>/<S>_palp.inp
        data/occupancy_highres/<S>.npy
Output: data/biplanar/<S>.npz with ap (256x96), lat (256x96), occ (256x96x96)
        data/biplanar/dataset_report.csv

With --manifest data/external_occupancy/manifest.csv the same rendering is
applied to external CTs imported by import_external_femurs.py (rows with
status OK and a CT path), using the stored rigid transform. Write those to a
separate folder (default data/biplanar_external) and pass it to
train_multiview_net.py --extra-train-dir: external cases are training-only.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from femur_grid import (ROOT, BOUNDS, DIMENSIONS, OCC_DIR, LANDMARK_ROOT, SPLIT_FILE,
                        grid_spacing, load_split, read_landmark_frame)

OUTPUT_DIR = ROOT / "data" / "biplanar"
MU_WATER = 0.02  # 1/mm, approximate linear attenuation of water at ~70 keV


def resample_box(image, origin, R, box_min, size, spacing):
    """Resample a CT on an axis-aligned box of the anatomical frame."""
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(tuple(R.flatten()))
    transform.SetTranslation(tuple(origin))
    out = sitk.Resample(image, [int(s) for s in size], transform, sitk.sitkLinear,
                        tuple(float(v) for v in box_min), tuple(float(s) for s in spacing),
                        (1, 0, 0, 0, 1, 0, 0, 0, 1), -1000.0, sitk.sitkFloat32)
    # SimpleITK arrays are (Z, Y, X); return (X, Y, Z)
    return np.transpose(sitk.GetArrayFromImage(out), (2, 1, 0))


def line_integral(hu, axis, step_mm):
    mu = np.clip(hu + 1000.0, 0.0, None) / 1000.0 * MU_WATER
    return (mu.sum(axis=axis) * step_mm).astype(np.float32)


def project_subject(ct_path, origin, R, slab_mm, slab_step):
    image = sitk.ReadImage(str(ct_path))
    xmin, xmax, ymin, ymax, zmin, zmax = BOUNDS
    sx, sy, sz = grid_spacing()
    nx, ny, nz = DIMENSIONS
    n_slab = int(round(2 * slab_mm / slab_step)) + 1
    # AP: grid in X and Y, wide slab in Z, integrate along Z.
    ap_vol = resample_box(image, origin, R, (xmin, ymin, -slab_mm),
                          (nx, ny, n_slab), (sx, sy, slab_step))
    ap = line_integral(ap_vol, axis=2, step_mm=slab_step)
    # Lateral: grid in X and Z, wide slab in Y centred on the shaft, integrate along Y.
    y_centre = 0.5 * (ymin + ymax)
    lat_vol = resample_box(image, origin, R, (xmin, y_centre - slab_mm, zmin),
                           (nx, n_slab, nz), (sx, slab_step, sz))
    lat = line_integral(lat_vol, axis=1, step_mm=slab_step)
    return ap, lat


def main():
    parser = argparse.ArgumentParser(description="Build grid-aligned AP/lateral DRR + occupancy pairs")
    parser.add_argument("--ct", choices=["full", "femur_only"], default="full",
                        help="full CT (realistic overlap) or femur-only CT")
    parser.add_argument("--slab-mm", type=float, default=160.0)
    parser.add_argument("--slab-step", type=float, default=2.0)
    parser.add_argument("--output", default=None, help="default data/biplanar (data/biplanar_external with --manifest)")
    parser.add_argument("--manifest", default=None, help="external manifest.csv from import_external_femurs.py")
    parser.add_argument("--force-overwrite", action="store_true")
    args = parser.parse_args()

    default_out = ROOT / "data" / ("biplanar_external" if args.manifest else "biplanar")
    out_dir = Path(args.output) if args.output else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []  # (subject, split, ct_path, occ_path, frame)
    if args.manifest:
        manifest = Path(args.manifest)
        with manifest.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] == "OK" and row["ct_path"]:
                    frame = (np.array(json.loads(row["origin"])), np.array(json.loads(row["R"])).reshape(3, 3))
                    jobs.append((row["name"], "train_external", Path(row["ct_path"]),
                                 manifest.parent / (row["name"] + ".npy"), frame))
    else:
        suffix = ".nrrd" if args.ct == "full" else "_femur_only.nrrd"
        for split, subjects in load_split(SPLIT_FILE).items():
            for subject in subjects:
                jobs.append((subject, split, LANDMARK_ROOT / subject / (subject + suffix),
                             OCC_DIR / (subject + ".npy"), None))
    report = []
    for subject, split, ct_path, occ_path, frame in jobs:
        out = out_dir / (subject + ".npz")
        if out.exists() and not args.force_overwrite:
            report.append({"subject": subject, "split": split, "status": "EXISTS"})
            continue
        try:
            if not ct_path.exists() or not occ_path.exists():
                raise FileNotFoundError(ct_path if not ct_path.exists() else occ_path)
            origin, R = frame if frame is not None else read_landmark_frame(subject)
            ap, lat = project_subject(ct_path, origin, R, args.slab_mm, args.slab_step)
            occ = np.load(occ_path).astype(np.uint8)
            if occ.shape != DIMENSIONS:
                raise ValueError("occupancy grid " + str(occ.shape))
            np.savez_compressed(out, ap=ap, lat=lat, occ=occ, ct=np.array(args.ct))
            report.append({"subject": subject, "split": split, "status": "OK"})
            print(subject, split, "OK  ap max {:.2f} lat max {:.2f}".format(ap.max(), lat.max()))
        except Exception as exc:
            report.append({"subject": subject, "split": split, "status": "ERROR: " + str(exc)})
            print(subject, "ERROR:", exc)

    with (out_dir / "dataset_report.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "split", "status"])
        writer.writeheader()
        writer.writerows(report)
    ok = sum(r["status"] in ("OK", "EXISTS") for r in report)
    print("Done: {}/{} subjects available in {}".format(ok, len(report), out_dir))


if __name__ == "__main__":
    main()
