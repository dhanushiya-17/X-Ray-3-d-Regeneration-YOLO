import os
import vtk


DATA_ROOT = "data/raw/Subjects"
OUTPUT_ROOT = "data/meshes"

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def load_stl(path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def clean_mesh(mesh):
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(mesh)
    cleaner.Update()
    return cleaner.GetOutput()


def save_stl(mesh, path):
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(path)
    writer.SetInputData(mesh)
    writer.Write()


def main():

    subjects = sorted(
        [
            name
            for name in os.listdir(DATA_ROOT)
            if name.startswith("Pat")
            and os.path.isdir(os.path.join(DATA_ROOT, name))
        ]
    )

    print("Subjects found:", len(subjects))

    successful = 0

    for subject in subjects:

        stl_path = os.path.join(
            DATA_ROOT,
            subject,
            f"{subject}.stl"
        )

        if not os.path.exists(stl_path):
            print(f"ERROR: Missing STL for {subject}")
            continue

        print(f"Processing {subject}...")

        mesh = load_stl(stl_path)

        print(
            "  Original points:",
            mesh.GetNumberOfPoints()
        )

        print(
            "  Original triangles:",
            mesh.GetNumberOfCells()
        )

        mesh = clean_mesh(mesh)

        output_path = os.path.join(
            OUTPUT_ROOT,
            f"{subject}.stl"
        )

        save_stl(mesh, output_path)

        print(
            "  Saved:",
            output_path
        )

        successful += 1

    print()
    print("========================================")
    print("Mesh preparation complete")
    print("Successful:", successful)
    print("========================================")


if __name__ == "__main__":
    main()
