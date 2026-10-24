"""
femur_grid.py
=============
Shared definitions for the anatomical femur grid used by the occupancy
targets (scripts/build_highres_targets.py), the SDF shape model, the biplanar
dataset builder and the multi-view network.

Grid convention: arrays are indexed (X, Y, Z) in the landmark frame
  X  along the femoral shaft      (-470 .. 40 mm)
  Y  towards the greater trochanter (-60 .. 100 mm)
  Z  anteroposterior               (-70 .. 70 mm)
A sum along Z is an AP projection, a sum along Y is a lateral projection.
"""

from pathlib import Path
import csv

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data" / "subject_split.csv"
OCC_DIR = ROOT / "data" / "occupancy_highres"
LANDMARK_ROOT = ROOT / "data" / "raw" / "Subjects"

BOUNDS = (-470.0, 40.0, -60.0, 100.0, -70.0, 70.0)
DIMENSIONS = (256, 96, 96)


def grid_spacing(dimensions=DIMENSIONS, bounds=BOUNDS):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return ((xmax - xmin) / (dimensions[0] - 1),
            (ymax - ymin) / (dimensions[1] - 1),
            (zmax - zmin) / (dimensions[2] - 1))


def load_split(split_file=SPLIT_FILE):
    splits = {}
    with Path(split_file).open(newline="") as f:
        for row in csv.DictReader(f):
            splits.setdefault(row["split"], []).append(row["subject"])
    return {k: sorted(v) for k, v in splits.items()}


def read_landmark_frame(subject, landmark_root=LANDMARK_ROOT):
    """Return (origin, R) so that anatomical q = R.T @ (p - origin)."""
    path = Path(landmark_root) / subject / (subject + "_palp.inp")
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = float(value.strip())

    def point(prefix):
        return np.array([values[prefix + "X"], values[prefix + "Y"],
                         values[prefix + "Z"]], dtype=np.float64)

    origin, head_dir, troch = point("femurHead"), point("headDir"), point("greatTroch")
    x = head_dir - origin
    x /= np.linalg.norm(x)
    y = troch - origin
    y -= np.dot(y, x) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return origin, np.column_stack((x, y, z))


def occupancy_to_sdf(occ, spacing, truncation=20.0):
    """Truncated signed distance in mm: negative inside, positive outside.

    PCA on a signed distance field interpolates shapes smoothly, unlike PCA
    on binary occupancy, whose linear blends produce fuzzy, non-anatomical
    surfaces. The zero level set is the bone surface.
    """
    inside = np.asarray(occ) > 0.5
    if not inside.any():
        raise ValueError("Empty occupancy grid")
    outside_dist = ndi.distance_transform_edt(~inside, sampling=spacing)
    inside_dist = ndi.distance_transform_edt(inside, sampling=spacing)
    # Half-voxel offset puts the zero crossing between the boundary voxels.
    sdf = np.where(inside, -(inside_dist - 0.5 * min(spacing)),
                   outside_dist - 0.5 * min(spacing))
    return np.clip(sdf, -truncation, truncation).astype(np.float32)


def downsample_occupancy(occ, factor):
    """Average-pool an (X, Y, Z) grid by an integer factor and re-binarize."""
    if factor == 1:
        return occ
    x, y, z = (s // factor for s in occ.shape)
    pooled = occ[:x * factor, :y * factor, :z * factor].reshape(
        x, factor, y, factor, z, factor).mean(axis=(1, 3, 5))
    return (pooled >= 0.5).astype(np.uint8)
