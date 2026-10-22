"""
build_mesh_ssm.py
=================
Build a mesh-space SSM using Coherent Point Drift (CPD) non-rigid registration
to establish point correspondence between all training femur meshes.

This gives a fundamentally better shape model than voxel-space PCA because:
  1. PCA operates on vertex coordinates (continuous, geometrically meaningful)
  2. Linear interpolation of vertex positions produces valid shapes
  3. Sub-voxel accuracy is possible
  4. Better generalisation to unseen subjects

Steps:
  1. Decimate all aligned meshes to ~5000 vertices (uniform remeshing)
  2. Select the mesh closest to population centroid as the reference template
  3. Non-rigid CPD: deform reference template to each training mesh
  4. Collect all deformed templates (same topology, N_vertices x 3)
  5. PCA on the stacked vertex coordinate matrices

Outputs:
  models/mesh_ssm.npz
  models/mesh_ssm_variance.png
  models/mesh_ssm_oracle.csv
"""

from pathlib import Path
import csv, sys, time
import numpy as np
import vtk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT          = Path(__file__).resolve().parents[1]
SPLIT_FILE    = ROOT / "data" / "subject_split.csv"
MESH_ROOT     = ROOT / "data" / "meshes_qc"
LANDMARK_ROOT = ROOT / "data" / "raw" / "Subjects"
MODEL_DIR     = ROOT / "models"
MODEL_FILE    = MODEL_DIR / "mesh_ssm.npz"
CACHE_DIR     = ROOT / "data" / "mesh_registered"

TARGET_VERTICES = 3000   # decimate to this many vertices
N_PCA_COMPONENTS = 50
CPD_TOLERANCE = 1e-5
CPD_MAX_ITERATIONS = 150
CPD_ALPHA = 2.0          # smoothness (higher = smoother deformation)
CPD_BETA = 2.0           # width of Gaussian filter


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
    """Load and align mesh to anatomical coordinates, return VTK polydata."""
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


def decimate_mesh(mesh, target_n):
    """Decimate mesh to approximately target_n vertices."""
    current_n = mesh.GetNumberOfPoints()
    if current_n <= target_n:
        return mesh

    ratio = 1.0 - (target_n / current_n)
    ratio = min(ratio, 0.99)

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(mesh)
    decimate.SetTargetReduction(ratio)
    decimate.PreserveTopologyOn()
    decimate.BoundaryVertexDeletionOff()
    decimate.Update()

    # Clean up
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(decimate.GetOutputPort())
    cleaner.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(cleaner.GetOutput())
    return result


def vtk_to_numpy_points(mesh):
    """Extract vertex positions as Nx3 numpy array."""
    n = mesh.GetNumberOfPoints()
    pts = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        pts[i] = mesh.GetPoint(i)
    return pts


def numpy_to_vtk_mesh(points, reference_mesh):
    """Create a VTK mesh with new vertex positions but same topology as reference."""
    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(reference_mesh)

    vtk_points = vtk.vtkPoints()
    for p in points:
        vtk_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    mesh.SetPoints(vtk_points)
    return mesh


def save_stl(mesh, path):
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(mesh)
    writer.Write()


def cpd_register(source_pts, target_pts):
    """Non-rigid CPD: deform source points to match target point cloud."""
    from pycpd import DeformableRegistration

    reg = DeformableRegistration(
        X=target_pts.copy(),
        Y=source_pts.copy(),
        alpha=CPD_ALPHA,
        beta=CPD_BETA,
        tolerance=CPD_TOLERANCE,
        max_iterations=CPD_MAX_ITERATIONS,
    )
    result = reg.register()
    # result is (TY, params) where TY is the deformed source
    deformed = result[0]
    return deformed


def sample_surface_vtk(mesh, n_samples, seed):
    """Sample points uniformly on mesh surface."""
    rng = np.random.default_rng(seed)
    triangles, areas = [], []
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
        return np.zeros((0, 3))
    probs = np.asarray(areas)
    probs /= probs.sum()
    sel = rng.choice(len(triangles), n_samples, p=probs, replace=True)
    pts = np.empty((n_samples, 3))
    for i, idx in enumerate(sel):
        a, b, c = triangles[idx]
        u, v = rng.random(2)
        if u + v > 1:
            u, v = 1 - u, 1 - v
        pts[i] = a + u * (b - a) + v * (c - a)
    return pts


def main():
    print("=" * 60)
    print("BoneVision -- Mesh-Space SSM Builder (CPD Registration)")
    print("=" * 60)

    with open(SPLIT_FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    train_subjects = sorted(r["subject"] for r in rows if r["split"] == "train")
    val_subjects   = sorted(r["subject"] for r in rows if r["split"] == "validation")
    test_subjects  = sorted(r["subject"] for r in rows if r["split"] == "test")

    print("Train:", len(train_subjects), " Val:", len(val_subjects),
          " Test:", len(test_subjects))

    # ------------------------------------------------------------------
    # Step 1: Load and decimate all training meshes
    # ------------------------------------------------------------------
    print("\nStep 1: Loading and decimating training meshes...")

    decimated = {}
    point_clouds = {}

    for subj in train_subjects:
        mesh = load_aligned_mesh(subj)
        dec = decimate_mesh(mesh, TARGET_VERTICES)
        decimated[subj] = dec
        point_clouds[subj] = vtk_to_numpy_points(dec)
        print("  {} : {} -> {} vertices".format(
            subj, mesh.GetNumberOfPoints(), dec.GetNumberOfPoints()))

    # ------------------------------------------------------------------
    # Step 2: Select reference template (closest to centroid)
    # ------------------------------------------------------------------
    print("\nStep 2: Selecting reference template...")

    # Use dense surface samples for centroid calculation (since vertex counts differ)
    N_SURFACE_SAMPLES = 5000
    surface_samples = {}
    for subj in train_subjects:
        surface_samples[subj] = sample_surface_vtk(decimated[subj], N_SURFACE_SAMPLES, 42)

    # Compute pairwise mean distances to find centroid mesh
    mean_dists = []
    for i, si in enumerate(train_subjects):
        total = 0.0
        for j, sj in enumerate(train_subjects):
            if i == j:
                continue
            diff = surface_samples[si].mean(axis=0) - surface_samples[sj].mean(axis=0)
            total += np.linalg.norm(diff)
        mean_dists.append(total / (len(train_subjects) - 1))

    ref_idx = int(np.argmin(mean_dists))
    ref_subject = train_subjects[ref_idx]
    ref_mesh = decimated[ref_subject]
    ref_pts = point_clouds[ref_subject]
    n_ref = ref_mesh.GetNumberOfPoints()

    print("  Reference template:", ref_subject, "  vertices:", n_ref)

    # ------------------------------------------------------------------
    # Step 3: CPD non-rigid registration (template -> each training shape)
    # ------------------------------------------------------------------
    print("\nStep 3: Non-rigid CPD registration ({} subjects)...".format(
        len(train_subjects)))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    registered_vertices = []
    reg_subjects = []

    for idx, subj in enumerate(train_subjects):
        t0 = time.time()

        # Target is a dense sample from the subject mesh (for CPD matching)
        target_pts = sample_surface_vtk(decimated[subj], N_SURFACE_SAMPLES, 42)

        if subj == ref_subject:
            # Reference is its own registration
            deformed = ref_pts.copy()
        else:
            # Deform reference template to match this subject
            deformed = cpd_register(ref_pts.copy(), target_pts)

        t1 = time.time()
        registered_vertices.append(deformed.flatten())
        reg_subjects.append(subj)

        # Save registered mesh
        reg_mesh = numpy_to_vtk_mesh(deformed, ref_mesh)
        save_stl(reg_mesh, CACHE_DIR / (subj + "_registered.stl"))

        print("  [{:2d}/{:2d}] {} : {:.1f}s  deformation: mean={:.2f} max={:.2f} mm".format(
            idx + 1, len(train_subjects), subj, t1 - t0,
            float(np.linalg.norm(deformed - ref_pts, axis=1).mean()),
            float(np.linalg.norm(deformed - ref_pts, axis=1).max())))

    # ------------------------------------------------------------------
    # Step 4: PCA on registered vertex coordinates
    # ------------------------------------------------------------------
    print("\nStep 4: PCA on registered vertex coordinates...")

    X = np.stack(registered_vertices, axis=0).astype(np.float64)
    print("  Data matrix:", X.shape, " (subjects x vertex_coords)")

    mean_shape = X.mean(axis=0)
    X_centered = X - mean_shape

    from sklearn.decomposition import PCA
    n_comp = min(N_PCA_COMPONENTS, len(train_subjects) - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X_centered)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    for i in range(min(20, n_comp)):
        print("  PC{:02d}: {:.2f}%  cumulative {:.2f}%".format(
            i+1, explained[i]*100, cumulative[i]*100))

    for thr in [0.90, 0.95, 0.99]:
        idx = np.searchsorted(cumulative, thr)
        if idx < n_comp:
            print("  {}% variance at {} components".format(
                int(thr*100), idx + 1))

    # ------------------------------------------------------------------
    # Step 5: Save model
    # ------------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        MODEL_FILE,
        mean_shape=mean_shape,              # (n_vertices * 3,)
        components=pca.components_,         # (n_comp, n_vertices * 3)
        singular_values=pca.singular_values_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        n_vertices=np.array([n_ref]),
        ref_subject=np.array([ref_subject]),
        train_subjects=np.array(reg_subjects),
    )
    print("\nMesh SSM saved:", MODEL_FILE)

    # Save reference mesh topology
    save_stl(ref_mesh, MODEL_DIR / "mesh_ssm_reference.stl")
    print("Reference mesh:", MODEL_DIR / "mesh_ssm_reference.stl")

    # Variance plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, len(explained)+1), explained*100, color="#4C72B0")
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Per-Component Variance")
    axes[0].grid(axis="y", alpha=0.4)

    axes[1].plot(range(1, len(cumulative)+1), cumulative*100,
                 marker="o", color="#DD8452", linewidth=2, markersize=3)
    axes[1].axhline(95, color="gray", linestyle="--", label="95%")
    axes[1].axhline(99, color="red", linestyle="--", label="99%")
    axes[1].set_xlabel("Components")
    axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend()
    axes[1].grid(alpha=0.4)

    plt.suptitle("BoneVision Mesh-Space SSM (CPD Registration)\n"
                 + str(n_ref) + " vertices, "
                 + str(len(train_subjects)) + " training shapes", fontsize=10)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "mesh_ssm_variance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Step 6: Oracle test — register validation shapes and project into PCA
    # ------------------------------------------------------------------
    print("\nStep 6: Oracle test on validation + test sets...")

    oracle_rows = []
    for split_name, subjects in [("validation", val_subjects), ("test", test_subjects)]:
        for subj in subjects:
            try:
                mesh = load_aligned_mesh(subj)
                target_pts = sample_surface_vtk(mesh, N_SURFACE_SAMPLES, 42)

                # Register reference to this subject
                deformed = cpd_register(ref_pts.copy(), target_pts)
                subj_vec = deformed.flatten()

                centered = subj_vec - mean_shape
                for k in [10, 20, 30, 40, n_comp]:
                    if k > n_comp:
                        continue
                    coeffs = pca.components_[:k] @ centered
                    rec = mean_shape + pca.components_[:k].T @ coeffs
                    # Compute vertex RMSE in mm
                    rec_pts = rec.reshape(-1, 3)
                    def_pts = deformed.reshape(-1, 3)
                    rmse = float(np.sqrt(np.mean(np.sum((rec_pts - def_pts)**2, axis=1))))
                    oracle_rows.append({
                        "subject": subj, "split": split_name,
                        "k": k, "vertex_rmse_mm": round(rmse, 3),
                    })

                print("  {} : registered OK".format(subj))
            except Exception as exc:
                print("  {} : ERROR: {}".format(subj, exc))

    if oracle_rows:
        with open(MODEL_DIR / "mesh_ssm_oracle.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=oracle_rows[0].keys())
            writer.writeheader()
            writer.writerows(oracle_rows)

        for split_name in ["validation", "test"]:
            print("  " + split_name + ":")
            for k in [10, 20, 30, 40, n_comp]:
                vals = [r["vertex_rmse_mm"] for r in oracle_rows
                        if r["split"] == split_name and r["k"] == k]
                if vals:
                    print("    k={:2d}: mean RMSE={:.2f} mm  min={:.2f}  max={:.2f}".format(
                        k, np.mean(vals), np.min(vals), np.max(vals)))

    print("\n" + "=" * 60)
    print("Mesh-space SSM build complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
