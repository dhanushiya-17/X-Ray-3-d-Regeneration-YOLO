from pathlib import Path
import csv
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_multiview_3d import load_aligned_mesh

SPLIT_FILE = ROOT / "data" / "subject_split.csv"
OUTPUT_DIR = ROOT / "data" / "occupancy_targets"

# Anatomical coordinate bounds in millimetres.
# These are provisional and will be checked against all meshes.
BOUNDS = (-470.0, 40.0, -60.0, 100.0, -70.0, 70.0)

# Z is the femur's long axis in the original CT frame, but after
# landmark alignment the long axis is X. Use a finer X resolution.
DIMENSIONS = (128, 48, 48)  # X, Y, Z

def rasterize_mesh(mesh, bounds, dimensions):
    image = vtk.vtkImageData()
    image.SetDimensions(*dimensions)

    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    spacing = (
        (xmax - xmin) / (dimensions[0] - 1),
        (ymax - ymin) / (dimensions[1] - 1),
        (zmax - zmin) / (dimensions[2] - 1),
    )
    image.SetSpacing(*spacing)
    image.SetOrigin(xmin, ymin, zmin)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    image.GetPointData().GetScalars().Fill(0)

    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(mesh)
    stencil.SetOutputOrigin(xmin, ymin, zmin)
    stencil.SetOutputSpacing(*spacing)
    stencil.SetOutputWholeExtent(image.GetExtent())
    stencil.Update()

    mask_filter = vtk.vtkImageStencil()
    mask_filter.SetInputData(image)
    mask_filter.SetStencilConnection(stencil.GetOutputPort())
    mask_filter.ReverseStencilOn()
    mask_filter.SetBackgroundValue(1)
    mask_filter.Update()

    arr = vtk_to_numpy(
        mask_filter.GetOutput().GetPointData().GetScalars()
    )
    # VTK stores X fastest; reshape to Z,Y,X then transpose to X,Y,Z.
    arr = arr.reshape(dimensions[2], dimensions[1], dimensions[0])
    arr = np.transpose(arr, (2, 1, 0))
    return arr.astype(np.uint8), spacing

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SPLIT_FILE.open(newline="") as f:
        rows = list(csv.DictReader(f))

    report = []
    for row in rows:
        subject = row["subject"]
        try:
            mesh = load_aligned_mesh(subject)
            bounds = mesh.GetBounds()

            xmin, xmax, ymin, ymax, zmin, zmax = BOUNDS
            outside = (
                bounds[0] < xmin or bounds[1] > xmax
                or bounds[2] < ymin or bounds[3] > ymax
                or bounds[4] < zmin or bounds[5] > zmax
            )
            if outside:
                report.append({
                    "subject": subject, "split": row["split"],
                    "status": "OUTSIDE_BOUNDS", "occupied_voxels": 0,
                    "mesh_bounds": str(tuple(round(v, 2) for v in bounds)),
                })
                continue

            occupancy, spacing = rasterize_mesh(mesh, BOUNDS, DIMENSIONS)
            count = int(occupancy.sum())
            if count == 0:
                status = "EMPTY"
            else:
                status = "OK"

            np.save(OUTPUT_DIR / f"{subject}.npy", occupancy)
            report.append({
                "subject": subject, "split": row["split"],
                "status": status, "occupied_voxels": count,
                "mesh_bounds": str(tuple(round(v, 2) for v in bounds)),
            })
            print(f"{subject}: {status}, occupied voxels={count}")

        except Exception as exc:
            report.append({
                "subject": subject, "split": row["split"],
                "status": f"ERROR: {exc}", "occupied_voxels": 0,
                "mesh_bounds": "",
            })
            print(f"{subject}: ERROR: {exc}")

    with (OUTPUT_DIR / "target_report.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "subject", "split", "status",
                "occupied_voxels", "mesh_bounds"
            ]
        )
        writer.writeheader()
        writer.writerows(report)

    from collections import Counter
    print("\nSummary:", dict(Counter(r["status"] for r in report)))
    print("Report:", OUTPUT_DIR / "target_report.csv")

if __name__ == "__main__":
    main()
