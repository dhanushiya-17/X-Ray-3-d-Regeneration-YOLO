"""3D Slicer export helpers for BoneVision reconstructions.

Every reconstruction is written in the canonical SSM anatomical frame in
millimetres (x: femoral head towards the proximal end, y: towards the greater
trochanter, z: x cross y). The files declare that frame as Slicer's RAS world
space so Slicer loads the coordinates unchanged, and the mesh and labelmap line
up exactly. Slicer would otherwise treat a headerless STL as LPS and flip x/y.

The canonical frame is not a patient frame. ``display_transforms`` gives
rotations that stand the femur upright in Slicer's anatomical views; they are
for viewing only and assume the femur side, which a single X-ray does not
establish. Nothing here registers the estimate to a patient.
"""
from __future__ import annotations

import gzip
import io
import struct
import zipfile
from pathlib import Path

import numpy as np

LOADER_PATH = Path(__file__).resolve().parents[1] / "slicer" / "BoneVisionLoader" / "BoneVisionLoader.py"
MESH_NAME = "femur_reconstruction.stl"
LABELMAP_NAME = "femur_labelmap.nrrd"
FRAME_DESCRIPTION = (
    "canonical SSM anatomical frame in millimetres (x: head to proximal, y: towards greater trochanter, "
    "z: x cross y), written as Slicer RAS so it loads unchanged; not registered to patient RAS/LPS"
)


def stl_bytes(vertices: np.ndarray, faces: np.ndarray, title: str = "BoneVision femur estimate") -> bytes:
    """Binary STL whose header carries Slicer's ``SPACE=RAS`` tag."""
    tri = np.asarray(vertices, dtype=np.float32)[np.asarray(faces)]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    records = np.empty(len(tri), dtype=np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]))
    records["normal"] = normals
    records["vertices"] = tri
    records["attr"] = 0
    header = f"{title} SPACE=RAS".encode("ascii", "replace")[:80]
    return header.ljust(80, b"\0") + struct.pack("<I", len(tri)) + records.tobytes()


def write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray, title: str = "BoneVision femur estimate") -> None:
    Path(path).write_bytes(stl_bytes(vertices, faces, title))


def nrrd_bytes(volume: np.ndarray, spacing, origin, extra: dict | None = None) -> bytes:
    """Gzip NRRD in RAS space; ``volume[i, j, k]`` is the sample at ``origin + (i, j, k) * spacing``."""
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError("Expected a 3D volume")
    types = {np.dtype(np.uint8): "uint8", np.dtype(np.int16): "int16", np.dtype(np.float32): "float"}
    if volume.dtype not in types:
        raise ValueError(f"Unsupported NRRD dtype {volume.dtype}")
    sx, sy, sz = (float(s) for s in spacing)
    lines = [
        "NRRD0004",
        f"type: {types[volume.dtype]}",
        "dimension: 3",
        "space: right-anterior-superior",
        "sizes: " + " ".join(str(n) for n in volume.shape),
        f"space directions: ({sx!r},0,0) (0,{sy!r},0) (0,0,{sz!r})",
        "kinds: domain domain domain",
        "endian: little",
        "encoding: gzip",
        "space origin: (" + ",".join(repr(float(v)) for v in origin) + ")",
    ]
    for key, value in (extra or {}).items():
        lines.append(f"{key}:={value}")
    # NRRD stores the first axis fastest, which is numpy's Fortran order.
    payload = gzip.compress(volume.astype(volume.dtype.newbyteorder("<"), copy=False).tobytes(order="F"))
    return ("\n".join(lines) + "\n\n").encode("ascii") + payload


def write_nrrd(path: Path, volume: np.ndarray, spacing, origin, extra: dict | None = None) -> None:
    Path(path).write_bytes(nrrd_bytes(volume, spacing, origin, extra))


def labelmap_bytes(occupancy: np.ndarray, bounds, threshold: float = 0.5) -> bytes:
    """The thresholded occupancy as a uint8 labelmap NRRD on the SSM grid (1 = femur)."""
    spacing, origin = grid_geometry(bounds, occupancy.shape)
    labelmap = (np.asarray(occupancy) >= threshold).astype(np.uint8)
    return nrrd_bytes(labelmap, spacing, origin, {"BoneVision_label_1": "femur estimate"})


def grid_geometry(bounds, grid_shape) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Spacing and origin of the SSM occupancy grid (samples include both bounds)."""
    spacing = tuple(float((bounds[2*i+1] - bounds[2*i]) / (grid_shape[i] - 1)) for i in range(3))
    origin = (float(bounds[0]), float(bounds[2]), float(bounds[4]))
    return spacing, origin


def display_transforms() -> dict[str, list[list[float]]]:
    """Canonical-to-RAS rotations that show the femur upright (proximal = superior).

    For a right femur the trochanter side (+y) is patient right (+R) and x cross y
    points anterior; for a left femur +y is patient left and x cross y points
    posterior. Both matrices are proper rotations, so the shape is never mirrored.
    """
    return {
        "canonical": np.eye(4).tolist(),
        "upright_right_femur": [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
        "upright_left_femur": [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
    }


def export_for_slicer(output_dir: Path, vertices_mm: np.ndarray, faces: np.ndarray, occupancy: np.ndarray, bounds, threshold: float = 0.5) -> dict:
    """Write the mesh and labelmap; return the ``slicer`` section for the run report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_stl(output_dir / MESH_NAME, vertices_mm, faces)
    (output_dir / LABELMAP_NAME).write_bytes(labelmap_bytes(occupancy, bounds, threshold))
    return slicer_info(threshold)


def slicer_info(threshold: float = 0.5) -> dict:
    """The ``slicer`` section of a run report for files named MESH_NAME and LABELMAP_NAME."""
    return {
        "mesh": MESH_NAME,
        "labelmap": LABELMAP_NAME,
        "labelmap_values": {"0": "background", "1": "femur estimate"},
        "labelmap_threshold": threshold,
        "units": "mm",
        "coordinate_frame": FRAME_DESCRIPTION,
        "display_transforms": display_transforms(),
        "display_transform_note": "Rotations for viewing only; choose the one matching the femur side. Not a patient registration.",
        "load_instructions": "In 3D Slicer use File > Add Data on the STL (and tick LabelMap for the NRRD), or run slicer/BoneVisionLoader/BoneVisionLoader.py on this folder.",
    }


def slicer_bundle(output_dir: Path) -> bytes:
    """Zip a reconstruction folder together with the Slicer loader script."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(Path(output_dir).iterdir()):
            if path.is_file():
                zf.write(path, f"bonevision_femur/{path.name}")
        if LOADER_PATH.exists():
            zf.write(LOADER_PATH, f"bonevision_femur/{LOADER_PATH.name}")
    return buffer.getvalue()
