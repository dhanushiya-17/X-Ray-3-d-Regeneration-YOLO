"""
build_highres_targets.py
========================
Generate higher-resolution occupancy grids (256 x 96 x 96) for all 101
subjects.  These replace the original 128x48x48 grids for SSM construction
and give approximately 2 mm isotropic voxels.

Input:  data/meshes_qc/*.stl  +  data/raw/Subjects/*/Pat*_palp.inp
Output: data/occupancy_highres/*.npy  +  data/occupancy_highres/target_report.csv
"""

from pathlib import Path
import csv, sys
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT         = Path(__file__).resolve().parents[1]
SPLIT_FILE   = ROOT / "data" / "subject_split.csv"
MESH_ROOT    = ROOT / "data" / "meshes_qc"
LANDMARK_ROOT = ROOT / "data" / "raw" / "Subjects"
OUTPUT_DIR   = ROOT / "data" / "occupancy_highres"

# Anatomical bounds (mm) — same as original targets but validated
XMIN, XMAX = -470.0,  40.0    # 510 mm along femoral shaft
YMIN, YMAX =  -60.0, 100.0    # 160 mm mediolateral
ZMIN, ZMAX =  -70.0,  70.0    # 140 mm anteroposterior

# Higher resolution grid
DIMENSIONS = (256, 96, 96)     # X, Y, Z  (~2mm voxels)


def read_landmarks(subject):
    path = LANDMARK_ROOT / subject / (subject + "_palp.inp")
    values = {}
    with path.open() as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = float(value.strip())
    def point(prefix):
        return np.array([values[prefix + "X"],
                         values[prefix + "Y"],
                         values[prefix + "Z"]], dtype=np.float64)
    return point("femurHead"), point("headDir"), point("greatTroch")


def load_aligned_mesh(subject):
    """Load mesh and transform to anatomical coordinate frame."""
    origin, head_dir, troch = read_landmarks(subject)
    x = head_dir - origin
    x /= np.linalg.norm(x)
    y = troch - origin
    y -= np.dot(y, x) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    R = np.column_stack((x, y, z))

    matrix = vtk.vtkMatrix4x4()
    for r in range(3):
        for c in range(3):
            matrix.SetElement(r, c, R[c, r])
        matrix.SetElement(r, 3, -np.dot(R[:, r], origin))

    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(MESH_ROOT / (subject + ".stl")))
    reader.Update()
    if reader.GetOutput().GetNumberOfCells() == 0:
        raise ValueError("Empty mesh for " + subject)

    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetInputConnection(reader.GetOutputPort())
    filt.SetTransform(transform)
    filt.Update()

    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(filt.GetOutput())
    return mesh


def rasterize_mesh(mesh, bounds, dimensions):
    """Rasterize a closed mesh surface into a binary occupancy grid."""
    image = vtk.vtkImageData()
    image.SetDimensions(*dimensions)

    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    spacing = ((xmax - xmin) / (dimensions[0] - 1),
               (ymax - ymin) / (dimensions[1] - 1),
               (zmax - zmin) / (dimensions[2] - 1))
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

    arr = vtk_to_numpy(mask_filter.GetOutput().GetPointData().GetScalars())
    # VTK stores X-fastest; reshape to (Z, Y, X) then transpose to (X, Y, Z)
    arr = arr.reshape(dimensions[2], dimensions[1], dimensions[0])
    arr = np.transpose(arr, (2, 1, 0))
    return arr.astype(np.uint8), spacing


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SPLIT_FILE.open(newline="") as f:
        rows = list(csv.DictReader(f))

    bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
    report = []
    n_ok = 0

    for row in rows:
        subject = row["subject"]
        try:
            mesh = load_aligned_mesh(subject)
            mb = mesh.GetBounds()
            outside = (mb[0] < XMIN or mb[1] > XMAX or
                       mb[2] < YMIN or mb[3] > YMAX or
                       mb[4] < ZMIN or mb[5] > ZMAX)
            if outside:
                report.append({"subject": subject, "split": row["split"],
                               "status": "OUTSIDE_BOUNDS", "occupied": 0})
                print(subject, ": OUTSIDE_BOUNDS")
                continue

            occ, spacing = rasterize_mesh(mesh, bounds, DIMENSIONS)
            count = int(occ.sum())
            status = "OK" if count > 0 else "EMPTY"
            np.save(OUTPUT_DIR / (subject + ".npy"), occ)
            report.append({"subject": subject, "split": row["split"],
                           "status": status, "occupied": count})
            if status == "OK":
                n_ok += 1
            print(subject, ":", status, " voxels:", count,
                  " spacing:", tuple(round(s, 2) for s in spacing))

        except Exception as exc:
            report.append({"subject": subject, "split": row["split"],
                           "status": "ERROR: " + str(exc), "occupied": 0})
            print(subject, ": ERROR:", exc)

    csv_path = OUTPUT_DIR / "target_report.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "split", "status", "occupied"])
        writer.writeheader()
        writer.writerows(report)

    print("\n" + "=" * 50)
    print("High-res targets complete. OK:", n_ok, "/", len(rows))
    print("Grid:", DIMENSIONS, " voxels")
    sx = (XMAX - XMIN) / (DIMENSIONS[0] - 1)
    sy = (YMAX - YMIN) / (DIMENSIONS[1] - 1)
    sz = (ZMAX - ZMIN) / (DIMENSIONS[2] - 1)
    print("Spacing: {:.2f} x {:.2f} x {:.2f} mm".format(sx, sy, sz))
    print("Saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
