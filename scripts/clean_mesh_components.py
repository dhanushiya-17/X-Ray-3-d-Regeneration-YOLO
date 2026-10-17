
import csv
import os

import vtk

INPUT_ROOT = "data/meshes"
OUTPUT_ROOT = "data/meshes_qc"
REPORT_PATH = "data/meshes_qc_report.csv"


def read_mesh(path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()

    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(reader.GetOutput())

    if mesh.GetNumberOfPoints() == 0 or mesh.GetNumberOfCells() == 0:
        raise ValueError(f"Empty or unreadable mesh: {path}")

    return mesh


def extract_largest_component(mesh):
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(mesh)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()

    component_count = connectivity.GetNumberOfExtractedRegions()
    if component_count == 0:
        raise ValueError("No connected components found")

    largest_id = 0
    largest_triangles = -1
    component_sizes = []

    for region_id in range(component_count):
        connectivity.InitializeSpecifiedRegionList()
        connectivity.AddSpecifiedRegion(region_id)
        connectivity.SetExtractionModeToSpecifiedRegions()
        connectivity.Update()

        triangle_count = connectivity.GetOutput().GetNumberOfCells()
        component_sizes.append(triangle_count)

        if triangle_count > largest_triangles:
            largest_triangles = triangle_count
            largest_id = region_id

    connectivity.InitializeSpecifiedRegionList()
    connectivity.AddSpecifiedRegion(largest_id)
    connectivity.SetExtractionModeToSpecifiedRegions()
    connectivity.Update()

    # Deep copy so the output is independent of the filter.
    largest = vtk.vtkPolyData()
    largest.DeepCopy(connectivity.GetOutput())

    # Remove unused points from the extracted component.
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(largest)
    cleaner.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(cleaner.GetOutput())

    removed_triangles = (
        mesh.GetNumberOfCells() - result.GetNumberOfCells()
    )

    return result, component_count, removed_triangles, component_sizes


def write_mesh(mesh, path):
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(path)
    writer.SetFileTypeToBinary()
    writer.SetInputData(mesh)

    if writer.Write() != 1:
        raise IOError(f"Failed to write: {path}")

    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise IOError(f"Output missing or empty: {path}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    files = sorted(
        name for name in os.listdir(INPUT_ROOT)
        if name.lower().endswith(".stl")
    )

    if not files:
        raise RuntimeError(f"No STL files found in {INPUT_ROOT}")

    report_rows = []

    for filename in files:
        source = os.path.join(INPUT_ROOT, filename)
        destination = os.path.join(OUTPUT_ROOT, filename)

        mesh = read_mesh(source)
        cleaned, components, removed, sizes = (
            extract_largest_component(mesh)
        )

        write_mesh(cleaned, destination)

        report_rows.append({
            "filename": filename,
            "original_points": mesh.GetNumberOfPoints(),
            "original_triangles": mesh.GetNumberOfCells(),
            "original_components": components,
            "component_triangle_counts": ";".join(map(str, sizes)),
            "removed_triangles": removed,
            "output_points": cleaned.GetNumberOfPoints(),
            "output_triangles": cleaned.GetNumberOfCells(),
            "output_file": destination,
        })

        print(
            f"{filename}: components={components}, "
            f"removed_triangles={removed}, "
            f"output_triangles={cleaned.GetNumberOfCells()}"
        )

    fieldnames = list(report_rows[0].keys())

    with open(REPORT_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print("\nMesh cleanup finished.")
    print("Meshes processed:", len(report_rows))
    print("Output directory:", OUTPUT_ROOT)
    print("Audit report:", REPORT_PATH)


if __name__ == "__main__":
    main()
