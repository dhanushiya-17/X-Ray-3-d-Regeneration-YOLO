from pathlib import Path
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = "Pat001"

INPUT = ROOT / "data/raw/Subjects" / SUBJECT / f"{SUBJECT}_femur_only.nrrd"
LANDMARKS = ROOT / "data/raw/Subjects" / SUBJECT / f"{SUBJECT}_palp.inp"
OUTPUT_DIR = ROOT / "data/drr_anatomical"
OUTPUT = OUTPUT_DIR / f"{SUBJECT}_anatomical_ct.nrrd"

# Same anatomical coordinate bounds as the occupancy targets.
XMIN, XMAX = -470.0, 40.0
YMIN, YMAX = -60.0, 100.0
ZMIN, ZMAX = -70.0, 70.0

# Higher-resolution grid for generating DRRs.
SPACING = (1.5, 1.5, 1.5)


def read_landmarks(path):
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = float(value.strip())

    def point(name):
        return np.array([
            values[name + "X"],
            values[name + "Y"],
            values[name + "Z"],
        ], dtype=float)

    return (
        point("femurHead"),
        point("headDir"),
        point("greatTroch"),
    )


def main():
    if not INPUT.exists() or not LANDMARKS.exists():
        raise FileNotFoundError("Input CT or landmark file is missing.")

    origin, head_dir, troch = read_landmarks(LANDMARKS)

    # Construct the same anatomical coordinate frame as the project.
    x = head_dir - origin
    x /= np.linalg.norm(x)

    y = troch - origin
    y -= np.dot(y, x) * x
    y /= np.linalg.norm(y)

    z = np.cross(x, y)
    R = np.column_stack((x, y, z))

    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise ValueError("Anatomical rotation matrix is not orthonormal.")

    # Output grid in anatomical coordinates.
    bounds_min = np.array([XMIN, YMIN, ZMIN])
    bounds_max = np.array([XMAX, YMAX, ZMAX])
    spacing = np.array(SPACING)

    # Include enough voxels to cover the specified upper bounds.
    size_values = np.ceil(
        (bounds_max - bounds_min) / spacing
    ).astype(int) + 1
    size = tuple(int(v) for v in size_values)

    print("Input CT:", INPUT)
    print("Output grid size (X,Y,Z):", size)
    print("Output spacing (mm):", SPACING)
    print("Anatomical origin:", bounds_min.tolist())

    image = sitk.ReadImage(str(INPUT))

    # SimpleITK's resampling transform maps output points to input points.
    # Anatomical q -> original physical point p = origin + R @ q.
    output_to_input = sitk.AffineTransform(3)
    output_to_input.SetMatrix(tuple(R.flatten()))
    output_to_input.SetTranslation(tuple(origin))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    anatomical_ct = sitk.Resample(
        image,
        size,
        output_to_input,
        sitk.sitkLinear,
        tuple(bounds_min),
        SPACING,
        (1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0),
        -1000.0,
        sitk.sitkFloat32,
    )

    sitk.WriteImage(anatomical_ct, str(OUTPUT), useCompression=True)

    arr = sitk.GetArrayFromImage(anatomical_ct)
    print("\nSaved:", OUTPUT)
    print("Array shape (Z,Y,X):", arr.shape)
    print("Intensity range:", float(arr.min()), "to", float(arr.max()))
    print("Voxels above -500 HU:", int((arr > -500).sum()))

    # Report whether the expected anatomical landmarks fit the grid.
    for name, point in [
        ("Femur head", origin),
        ("Head direction", head_dir),
        ("Greater trochanter", troch),
    ]:
        q = R.T @ (point - origin)
        inside = np.all(q >= bounds_min) and np.all(q <= bounds_max)
        print(f"{name}: aligned={np.round(q, 2)}, inside_grid={inside}")


if __name__ == "__main__":
    main()
