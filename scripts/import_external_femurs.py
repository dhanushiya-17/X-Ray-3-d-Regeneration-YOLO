"""
import_external_femurs.py
=========================
Bring femurs from other public datasets into the project's anatomical grid so
they can be used as EXTRA TRAINING shapes (never validation/test, which stay
HFValid-only so results remain comparable with earlier runs).

Supported sources
  --totalseg-root  TotalSegmentator-style folders: <case>/ct.nii.gz and
                   <case>/segmentations/femur_left.nii.gz, femur_right.nii.gz
                   (TotalSegmentator v2 dataset, Zenodo, CC BY 4.0; also any CT
                   you segment yourself with the TotalSegmentator tool)
  --meshes         a folder of femur STL files, e.g. the VSD lower-extremity
                   meshes (CC BY-NC-SA 4.0)

Per femur:
  1. reject partial femurs (label touching the scan edge along the shaft, or
     shaft length below --min-length-mm); many abdominal CTs crop the femur
  2. rigidly align to the mean HFValid training femur: principal-axis
     initialisation over all sign flips, both the original and the mirrored
     shape, refined with point-to-point ICP; the best residual wins, so left
     femurs are mirrored onto the HFValid side automatically
  3. reject alignments whose RMS residual exceeds --max-rms-mm, or that fall
     outside the grid bounds
  4. rasterize into the same 256x96x96 grid as data/occupancy_highres

Output
  data/external_occupancy/<source>_<case>_<side>.npy
  data/external_occupancy/manifest.csv  (status, residual, transform, CT path)

The manifest's transform (p = origin + R @ q, R may include a mirror) lets
build_biplanar_dataset.py --manifest render grid-aligned DRRs from the
external CTs. Check the per-case status and spot-check overlays before
training; an automated rigid fit is not a substitute for landmark QC.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from femur_grid import ROOT, BOUNDS, DIMENSIONS, OCC_DIR, SPLIT_FILE, grid_spacing, load_occupancy, load_split

OUTPUT_DIR = ROOT / "data" / "external_occupancy"


# ---------------------------------------------------------------- geometry ---

def reference_points(n=6000):
    """Surface points of the mean training femur, in the anatomical frame."""
    train = load_split(SPLIT_FILE)["train"]
    mean = None
    for s in train:
        occ = load_occupancy(s, OCC_DIR).astype(np.float32)
        mean = occ if mean is None else mean + occ
    mean /= len(train)
    verts, _, _, _ = marching_cubes(mean, 0.5, spacing=grid_spacing())
    verts += np.array(BOUNDS[0::2])
    rng = np.random.default_rng(0)
    return verts[rng.choice(len(verts), min(n, len(verts)), replace=False)]


def principal_frame(points):
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    return c, vt  # rows are axes, longest first


def icp(src, ref_tree, ref, A, t, iterations=60):
    """Point-to-point rigid ICP; A may be a reflection and stays one."""
    det = np.sign(np.linalg.det(A))
    for _ in range(iterations):
        moved = src @ A.T + t
        dist, idx = ref_tree.query(moved)
        keep = dist < np.percentile(dist, 90)  # trim outliers
        p, q = moved[keep], ref[idx[keep]]
        pc, qc = p.mean(0), q.mean(0)
        u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
        d = np.sign(np.linalg.det(vt.T @ u.T))
        dR = vt.T @ np.diag([1, 1, d]) @ u.T
        A, t = dR @ A, dR @ (t - pc) + qc
    assert np.sign(np.linalg.det(A)) == det
    dist, _ = ref_tree.query(src @ A.T + t)
    return A, t, float(np.sqrt(np.mean(dist ** 2)))


def align(points, ref, ref_tree):
    """Return (A, t, rms, mirrored) mapping physical points p to q = A p + t."""
    rc, raxes = principal_frame(ref)
    pc, paxes = principal_frame(points)
    sub = points[np.random.default_rng(1).choice(len(points), min(4000, len(points)), replace=False)]
    best = None
    for signs in itertools.product((1, -1), repeat=3):
        A0 = raxes.T @ np.diag(signs) @ paxes   # maps p-axes onto ref axes
        t0 = rc - A0 @ pc
        A, t, rms = icp(sub, ref_tree, ref, A0, t0)
        if best is None or rms < best[2]:
            best = (A, t, rms, bool(np.linalg.det(A) < 0))
    return best


# ---------------------------------------------------------------- sources ---

def label_surface(label_path):
    """Physical (LPS) surface points of a binary label + completeness info."""
    import SimpleITK as sitk
    img = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayFromImage(img) > 0          # (z, y, x)
    if arr.sum() < 1000:
        return None, img, "EMPTY"
    zz = np.where(arr.any(axis=(1, 2)))[0]
    touches = zz[0] == 0 or zz[-1] == arr.shape[0] - 1
    verts, _, _, _ = marching_cubes(arr.astype(np.uint8), 0.5)
    idx = verts[:, ::-1]                             # -> (x, y, z) index
    spacing = np.array(img.GetSpacing())
    direction = np.array(img.GetDirection()).reshape(3, 3)
    pts = np.array(img.GetOrigin()) + (idx * spacing) @ direction.T
    return pts, img, "TOUCHES_SCAN_EDGE" if touches else "OK"


def mesh_points(stl_path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    reader.Update()
    return vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).astype(np.float64)


def rasterize_label(img, A, t):
    """Resample a label image into the anatomical grid (nearest neighbour)."""
    import SimpleITK as sitk
    R, origin = A.T, -A.T @ t                     # p = origin + R q
    tr = sitk.AffineTransform(3)
    tr.SetMatrix(tuple(R.flatten()))
    tr.SetTranslation(tuple(origin))
    out = sitk.Resample(img, list(DIMENSIONS), tr, sitk.sitkNearestNeighbor,
                        tuple(BOUNDS[0::2]), grid_spacing(), (1, 0, 0, 0, 1, 0, 0, 0, 1), 0, sitk.sitkUInt8)
    return (np.transpose(sitk.GetArrayFromImage(out), (2, 1, 0)) > 0).astype(np.uint8)


def rasterize_mesh(stl_path, A, t):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    m = vtk.vtkMatrix4x4()
    for r in range(3):
        for c in range(3):
            m.SetElement(r, c, A[r, c])
        m.SetElement(r, 3, t[r])
    tf = vtk.vtkTransform()
    tf.SetMatrix(m)
    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetInputConnection(reader.GetOutputPort())
    filt.SetTransform(tf)
    if np.linalg.det(A) < 0:   # a reflection flips triangle winding
        rev = vtk.vtkReverseSense()
        rev.SetInputConnection(filt.GetOutputPort())
        rev.Update()
        mesh = rev.GetOutput()
    else:
        filt.Update()
        mesh = filt.GetOutput()
    spacing = grid_spacing()
    image = vtk.vtkImageData()
    image.SetDimensions(*DIMENSIONS)
    image.SetSpacing(*spacing)
    image.SetOrigin(*BOUNDS[0::2])
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    image.GetPointData().GetScalars().Fill(0)
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(mesh)
    stencil.SetOutputOrigin(*BOUNDS[0::2])
    stencil.SetOutputSpacing(*spacing)
    stencil.SetOutputWholeExtent(image.GetExtent())
    mask = vtk.vtkImageStencil()
    mask.SetInputData(image)
    mask.SetStencilConnection(stencil.GetOutputPort())
    mask.ReverseStencilOn()
    mask.SetBackgroundValue(1)
    mask.Update()
    arr = vtk_to_numpy(mask.GetOutput().GetPointData().GetScalars())
    return np.transpose(arr.reshape(DIMENSIONS[::-1]), (2, 1, 0)).astype(np.uint8)


def candidates(args):
    if args.totalseg_root:
        for case in sorted(Path(args.totalseg_root).iterdir()):
            for side in ("left", "right"):
                label = case / "segmentations" / ("femur_" + side + ".nii.gz")
                if label.exists():
                    ct = case / "ct.nii.gz"
                    yield {"source": "totalseg", "case": case.name, "side": side, "kind": "label",
                           "path": label, "ct": ct if ct.exists() else None}
    if args.meshes:
        for stl in sorted(Path(args.meshes).glob("*.stl")):
            yield {"source": args.mesh_source, "case": stl.stem, "side": "", "kind": "mesh",
                   "path": stl, "ct": None}


def main():
    parser = argparse.ArgumentParser(description="Import external femurs as extra training shapes")
    parser.add_argument("--totalseg-root", default=None)
    parser.add_argument("--meshes", default=None)
    parser.add_argument("--mesh-source", default="vsd")
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--min-length-mm", type=float, default=340.0)
    parser.add_argument("--max-rms-mm", type=float, default=6.0)
    parser.add_argument("--limit", type=int, default=0, help="stop after N candidates (for a trial run)")
    args = parser.parse_args()
    if not args.totalseg_root and not args.meshes:
        parser.error("give --totalseg-root and/or --meshes")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = reference_points()
    ref_tree = cKDTree(ref)
    lo, hi = np.array(BOUNDS[0::2]), np.array(BOUNDS[1::2])
    rows = []
    for n, cand in enumerate(candidates(args)):
        if args.limit and n >= args.limit:
            break
        name = "{}_{}{}".format(cand["source"], cand["case"], "_" + cand["side"] if cand["side"] else "")
        row = {"name": name, "source": cand["source"], "file": str(cand["path"]),
               "ct_path": str(cand["ct"]) if cand["ct"] else "", "mirrored": "", "length_mm": "",
               "icp_rms_mm": "", "status": "", "origin": "", "R": ""}
        try:
            if cand["kind"] == "label":
                pts, img, status = label_surface(cand["path"])
            else:
                pts, img, status = mesh_points(cand["path"]), None, "OK"
            if pts is not None:
                _, axes = principal_frame(pts)
                proj = (pts - pts.mean(0)) @ axes[0]
                row["length_mm"] = round(float(proj.max() - proj.min()), 1)
                if status == "OK" and row["length_mm"] < args.min_length_mm:
                    status = "TOO_SHORT"
            if status == "OK":
                A, t, rms, mirrored = align(pts, ref, ref_tree)
                row.update(icp_rms_mm=round(rms, 2), mirrored=mirrored,
                           origin=json.dumps(np.round(-A.T @ t, 4).tolist()),
                           R=json.dumps(np.round(A.T, 6).flatten().tolist()))
                q = pts @ A.T + t
                if rms > args.max_rms_mm:
                    status = "HIGH_RESIDUAL"
                elif (q.min(0) < lo - 2).any() or (q.max(0) > hi + 2).any():
                    status = "OUTSIDE_BOUNDS"
                else:
                    occ = rasterize_label(img, A, t) if cand["kind"] == "label" else rasterize_mesh(cand["path"], A, t)
                    if occ.sum() < 1000:
                        status = "EMPTY_RASTER"
                    else:
                        np.save(out_dir / (name + ".npy"), occ)
            row["status"] = status
        except Exception as exc:
            row["status"] = "ERROR: " + str(exc)
        rows.append(row)
        print(name, row["status"], "rms", row["icp_rms_mm"], "len", row["length_mm"])

    with (out_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(r["status"] == "OK" for r in rows)
    print("Imported {}/{} femurs into {}".format(ok, len(rows), out_dir))


if __name__ == "__main__":
    main()
