
from pathlib import Path
import csv
import numpy as np
import vtk

ROOT = Path(__file__).resolve().parents[1]
MATCH_FILE = (
    ROOT / "reconstruction" / "retrieval_baseline"
    / "multiview_validation_matches.csv"
)
SPLIT_FILE = ROOT / "data" / "subject_split.csv"
MESH_ROOT = ROOT / "data" / "meshes_qc"
LANDMARK_ROOT = ROOT / "data" / "raw" / "Subjects"
OUTPUT_DIR = ROOT / "reconstruction" / "retrieval_baseline"

N_SAMPLES = 10000
SEED = 42


def read_landmarks(subject):
    path = LANDMARK_ROOT / subject / f"{subject}_palp.inp"
    values = {}

    with path.open() as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = float(value.strip())

    def point(prefix):
        return np.array([
            values[prefix + "X"],
            values[prefix + "Y"],
            values[prefix + "Z"]
        ], dtype=float)

    return point("femurHead"), point("headDir"), point("greatTroch")


def load_aligned_mesh(subject):
    origin, head_dir, troch = read_landmarks(subject)

    x = head_dir - origin
    x /= np.linalg.norm(x)

    y = troch - origin
    y -= np.dot(y, x) * x
    y /= np.linalg.norm(y)

    z = np.cross(x, y)
    rotation = np.column_stack((x, y, z))

    # Local coordinates: R.T @ (point - origin)
    matrix = vtk.vtkMatrix4x4()
    for r in range(3):
        for c in range(3):
            matrix.SetElement(r, c, rotation[c, r])
        matrix.SetElement(r, 3, -np.dot(rotation[:, r], origin))

    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(MESH_ROOT / f"{subject}.stl"))
    reader.Update()

    if reader.GetOutput().GetNumberOfCells() == 0:
        raise ValueError(f"Empty mesh for {subject}")

    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetInputConnection(reader.GetOutputPort())
    filt.SetTransform(transform)
    filt.Update()

    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(filt.GetOutput())
    return mesh


def sample_surface(mesh, n, seed):
    rng = np.random.default_rng(seed)
    triangles, areas = [], []

    for i in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        if cell.GetNumberOfPoints() != 3:
            continue

        a, b, c = [
            np.array(mesh.GetPoint(cell.GetPointId(j)))
            for j in range(3)
        ]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))

        if area > 0:
            triangles.append((a, b, c))
            areas.append(area)

    if not triangles:
        raise ValueError("No valid surface triangles")

    probs = np.asarray(areas, dtype=float)
    probs /= probs.sum()
    selected = rng.choice(len(triangles), n, p=probs)

    points = np.empty((n, 3))
    for i, idx in enumerate(selected):
        a, b, c = triangles[idx]
        u, v = rng.random(2)

        if u + v > 1:
            u, v = 1 - u, 1 - v

        points[i] = a + u * (b - a) + v * (c - a)

    return points


def distances_to_surface(points, mesh):
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    result = np.empty(len(points))
    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)

    for i, p in enumerate(points):
        locator.FindClosestPoint(
            p.tolist(), closest, cell_id, sub_id, dist2
        )
        result[i] = np.sqrt(float(dist2))

    return result


def main():
    with SPLIT_FILE.open(newline="") as f:
        splits = list(csv.DictReader(f))

    validation = {
        r["subject"] for r in splits if r["split"] == "validation"
    }
    training = {
        r["subject"] for r in splits if r["split"] == "train"
    }

    with MATCH_FILE.open(newline="") as f:
        matches = list(csv.DictReader(f))

    if len(matches) != len(validation):
        raise ValueError("Expected exactly one match per validation subject")

    cache = {}
    results = []

    for row in matches:
        query = row["query_subject"]
        template = row["selected_template"]

        if query not in validation:
            raise ValueError(f"Unexpected query subject: {query}")
        if template not in training:
            raise ValueError(f"Template is not in training split: {template}")

        for subject in (query, template):
            if subject not in cache:
                cache[subject] = load_aligned_mesh(subject)

        gt_mesh = cache[query]
        pred_mesh = cache[template]

        gt_points = sample_surface(gt_mesh, N_SAMPLES, SEED)
        pred_points = sample_surface(pred_mesh, N_SAMPLES, SEED + 1000)

        d1 = distances_to_surface(gt_points, pred_mesh)
        d2 = distances_to_surface(pred_points, gt_mesh)
        combined = np.concatenate((d1, d2))

        result = {
            "query_subject": query,
            "selected_template": template,
            "mean_surface_distance_mm": combined.mean(),
            "median_surface_distance_mm": np.median(combined),
            "p95_surface_distance_mm": np.percentile(combined, 95),
            "max_sampled_distance_mm": combined.max(),
        }
        results.append(result)

        print(
            f"{query} <- {template}: "
            f"mean={result['mean_surface_distance_mm']:.2f} mm, "
            f"P95={result['p95_surface_distance_mm']:.2f} mm"
        )

    output = OUTPUT_DIR / "multiview_validation_3d_metrics.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nCompleted:", len(results), "validation subjects")
    print("Saved:", output)
    print("These are template shape-comparison metrics, not clinical validation.")


if __name__ == "__main__":
    main()
