
from pathlib import Path
import csv
import numpy as np
import vtk


ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data" / "subject_split.csv"
MATCH_FILE = (
    ROOT / "reconstruction" / "retrieval_baseline"
    / "validation_matches.csv"
)
MESH_ROOT = ROOT / "data" / "meshes_qc"
LANDMARK_ROOT = ROOT / "data" / "raw" / "Subjects"
OUTPUT_DIR = ROOT / "reconstruction" / "retrieval_baseline"

N_SAMPLES = 10000
RANDOM_SEED = 42


def read_landmarks(subject):
    path = LANDMARK_ROOT / subject / f"{subject}_palp.inp"
    values = {}

    with path.open() as file:
        for line in file:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = float(value.strip())

    def point(prefix):
        return np.array([
            values[f"{prefix}X"],
            values[f"{prefix}Y"],
            values[f"{prefix}Z"],
        ], dtype=np.float64)

    return {
        "head": point("femurHead"),
        "head_dir": point("headDir"),
        "troch": point("greatTroch"),
    }


def anatomical_transform(subject):
    """Return a transform from CT coordinates to local anatomical coordinates."""
    points = read_landmarks(subject)
    origin = points["head"]

    x = points["head_dir"] - origin
    x /= np.linalg.norm(x)

    y = points["troch"] - origin
    y -= np.dot(y, x) * x
    y /= np.linalg.norm(y)

    z = np.cross(x, y)
    z /= np.linalg.norm(z)

    rotation = np.column_stack((x, y, z))

    # p_local = R.T @ (p_CT - origin)
    matrix = vtk.vtkMatrix4x4()

    for r in range(3):
        for c in range(3):
            matrix.SetElement(r, c, rotation[c, r])
        matrix.SetElement(r, 3, -np.dot(rotation[:, r], origin))

    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    return transform


def load_aligned_mesh(subject):
    path = MESH_ROOT / f"{subject}.stl"

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()

    mesh = reader.GetOutput()
    if mesh.GetNumberOfPoints() == 0 or mesh.GetNumberOfCells() == 0:
        raise ValueError(f"Empty mesh: {path}")

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(mesh)
    transform_filter.SetTransform(anatomical_transform(subject))
    transform_filter.Update()

    aligned = vtk.vtkPolyData()
    aligned.DeepCopy(transform_filter.GetOutput())
    return aligned


def sample_surface(mesh, n_samples, seed):
    """Sample mesh triangles proportional to surface area."""
    rng = np.random.default_rng(seed)
    triangles = []
    areas = []

    for i in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        if cell.GetNumberOfPoints() != 3:
            continue

        a = np.array(mesh.GetPoint(cell.GetPointId(0)))
        b = np.array(mesh.GetPoint(cell.GetPointId(1)))
        c = np.array(mesh.GetPoint(cell.GetPointId(2)))

        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        if area > 0:
            triangles.append((a, b, c))
            areas.append(area)

    if not triangles:
        raise ValueError("No valid triangles found")

    probabilities = np.asarray(areas, dtype=np.float64)
    probabilities /= probabilities.sum()

    selected = rng.choice(
        len(triangles), size=n_samples, replace=True, p=probabilities
    )

    points = np.empty((n_samples, 3), dtype=np.float64)

    for i, tri_index in enumerate(selected):
        a, b, c = triangles[tri_index]
        u, v = rng.random(2)

        if u + v > 1:
            u = 1 - u
            v = 1 - v

        points[i] = a + u * (b - a) + v * (c - a)

    return points


def directional_distances(source_points, target_mesh):
    """Distance from each sampled source point to target surface."""
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(target_mesh)
    locator.BuildLocator()

    distances = np.empty(len(source_points), dtype=np.float64)
    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)

    for i, point in enumerate(source_points):
        locator.FindClosestPoint(
            point.tolist(), closest, cell_id, sub_id, dist2
        )
        distances[i] = np.sqrt(float(dist2))

    return distances


def main():
    with SPLIT_FILE.open(newline="") as file:
        split_rows = list(csv.DictReader(file))

    validation_subjects = {
        row["subject"] for row in split_rows
        if row["split"] == "validation"
    }

    with MATCH_FILE.open(newline="") as file:
        matches = list(csv.DictReader(file))

    results = []
    mesh_cache = {}

    for row in matches:
        query_subject = row["query_subject"]
        template_subject = row["matched_train_subject"]
        angle = int(row["angle_degrees"])

        if query_subject not in validation_subjects:
            raise ValueError(
                f"Non-validation query found: {query_subject}"
            )

        if template_subject == query_subject:
            raise ValueError("A subject matched itself")

        for subject in (query_subject, template_subject):
            if subject not in mesh_cache:
                mesh_cache[subject] = load_aligned_mesh(subject)

        query_mesh = mesh_cache[query_subject]
        template_mesh = mesh_cache[template_subject]

        query_points = sample_surface(
            query_mesh, N_SAMPLES, RANDOM_SEED + angle
        )
        template_points = sample_surface(
            template_mesh, N_SAMPLES, RANDOM_SEED + angle + 1000
        )

        # Both directions are needed for a symmetric metric.
        query_to_template = directional_distances(
            query_points, template_mesh
        )
        template_to_query = directional_distances(
            template_points, query_mesh
        )

        combined = np.concatenate([
            query_to_template, template_to_query
        ])

        results.append({
            "query_subject": query_subject,
            "angle_degrees": angle,
            "matched_train_subject": template_subject,
            "mean_surface_distance_mm": float(combined.mean()),
            "median_surface_distance_mm": float(np.median(combined)),
            "p95_surface_distance_mm": float(np.percentile(combined, 95)),
            "max_sampled_distance_mm": float(combined.max()),
            "query_to_template_mean_mm": float(query_to_template.mean()),
            "template_to_query_mean_mm": float(template_to_query.mean()),
        })

        print(
            f"{query_subject} view={angle:3d} "
            f"template={template_subject} "
            f"mean={combined.mean():.2f} mm "
            f"P95={np.percentile(combined, 95):.2f} mm"
        )

    output_csv = OUTPUT_DIR / "validation_3d_surface_metrics.csv"

    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nEvaluation completed.")
    print("Evaluated validation matches:", len(results))
    print("Results saved to:", output_csv)
    print(
        "Note: these are sampled surface distances after landmark alignment, "
        "not clinical validation."
    )


if __name__ == "__main__":
    main()
