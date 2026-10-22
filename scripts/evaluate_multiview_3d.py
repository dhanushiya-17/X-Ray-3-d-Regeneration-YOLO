
from pathlib import Path
import argparse
import csv
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data" / "subject_split.csv"
MESH_ROOT = ROOT / "data" / "meshes_qc"
LANDMARK_ROOT = ROOT / "data" / "raw" / "Subjects"
OUTPUT_DIR = ROOT / "reconstruction" / "retrieval_baseline"

N_SAMPLES = 2000
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
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64, copy=False)
    cells = vtk_to_numpy(mesh.GetPolys().GetData())
    if cells.size == 0 or cells.size % 4:
        raise ValueError("No valid surface triangles")
    cells = cells.reshape(-1, 4)
    if not np.all(cells[:, 0] == 3):
        raise ValueError("Expected a triangulated surface mesh")

    triangles = points[cells[:, 1:]]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    valid = areas > 0
    triangles = triangles[valid]
    areas = areas[valid]
    if not len(areas):
        raise ValueError("No non-degenerate surface triangles")

    selected = rng.choice(len(triangles), n, p=areas / areas.sum())
    selected_triangles = triangles[selected]
    uv = rng.random((n, 2))
    reflect = uv.sum(axis=1) > 1
    uv[reflect] = 1 - uv[reflect]
    return (
        selected_triangles[:, 0]
        + uv[:, :1] * (selected_triangles[:, 1] - selected_triangles[:, 0])
        + uv[:, 1:] * (selected_triangles[:, 2] - selected_triangles[:, 0])
    )


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
    parser = argparse.ArgumentParser(
        description="Measure multi-view template reconstruction on held-out subjects."
    )
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation",
        help="Query split; selected templates must always be training subjects.",
    )
    args = parser.parse_args()

    match_file = (
        OUTPUT_DIR / f"multiview_{args.split}_matches.csv"
    )
    with SPLIT_FILE.open(newline="") as f:
        splits = list(csv.DictReader(f))

    queries = {
        r["subject"] for r in splits if r["split"] == args.split
    }
    training = {
        r["subject"] for r in splits if r["split"] == "train"
    }

    with match_file.open(newline="") as f:
        matches = list(csv.DictReader(f))

    if len(matches) != len(queries):
        raise ValueError(f"Expected exactly one match per {args.split} subject")

    cache = {}
    results = []

    for row in matches:
        query = row["query_subject"]
        template = row["selected_template"]

        if query not in queries:
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
            "surface_samples_per_mesh": N_SAMPLES,
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

    output = OUTPUT_DIR / f"multiview_{args.split}_3d_metrics.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCompleted: {len(results)} {args.split} subjects")
    print("Saved:", output)
    print("These are template shape-comparison metrics, not clinical validation.")


if __name__ == "__main__":
    main()
