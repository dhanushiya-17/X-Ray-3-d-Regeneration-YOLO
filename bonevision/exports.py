"""Mesh, image and report writers. Every function returns bytes; callers decide where they go."""
from __future__ import annotations

import io
import json
import struct
import zipfile
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from .pipeline import overlay_on_input

COORDINATE_FRAME = (
    "Canonical SSM anatomical frame, millimetres. Origin at the femoral-head landmark; "
    "+X along the femoral long axis (distal is negative X), +Y toward the greater trochanter, "
    "Z = X × Y. The fitted in-plane scale/shift is applied. This is NOT patient RAS/LPS space."
)
LIMITATIONS = [
    "Research estimate only; not for diagnosis, treatment or surgical planning.",
    "One projection does not determine hidden depth; depth comes from the population shape prior.",
    "Ordinary PNG/JPEG carries no physical scale; size is assumed from the prior unless calibration was supplied.",
    "Silhouette agreement measures 2D overlap with the traced outline, not 3D accuracy.",
    "Segmentation is classical thresholding and can select the wrong region.",
    "Accuracy on real clinical radiographs has not been validated.",
]

FILES = {
    "femur_estimate.stl": ("model/stl", "Binary STL mesh for 3D Slicer (File → Add Data)."),
    "femur_estimate.obj": ("text/plain", "Wavefront OBJ mesh for Blender, MeshLab and most 3D tools."),
    "femur_estimate_spread.ply": ("application/octet-stream", "PLY mesh with per-vertex shape-spread colours and values."),
    "report.json": ("application/json", "Settings, model identity, metrics, warnings, coordinate frame and limits."),
    "silhouette_fit.png": ("image/png", "Canonical-frame overlap of the traced outline and fitted projection."),
    "input_overlay.png": ("image/png", "Your X-ray with the selected mask and fitted outline drawn on it."),
    "mask.png": ("image/png", "The binary bone mask used as the fitting target (input pixel grid)."),
}


def stl_bytes(vertices: np.ndarray, faces: np.ndarray, header: bytes = b"BoneVision single-view SSM estimate") -> bytes:
    tri = vertices[faces].astype(np.float32)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    records = np.empty(len(faces), dtype=np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]))
    records["normal"] = normals
    records["vertices"] = tri
    records["attr"] = 0
    return header[:80].ljust(80, b"\0") + struct.pack("<I", len(faces)) + records.tobytes()


def obj_bytes(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    buf = io.StringIO()
    buf.write("# BoneVision single-view SSM estimate (millimetres, canonical SSM frame)\n")
    np.savetxt(buf, vertices, fmt="v %.4f %.4f %.4f")
    np.savetxt(buf, faces + 1, fmt="f %d %d %d")
    return buf.getvalue().encode("ascii")


SPREAD_STOPS = np.array([[0.93, 0.89, 0.82], [0.95, 0.71, 0.42], [0.89, 0.34, 0.18]])


def spread_colors(values: np.ndarray | None, n: int) -> np.ndarray:
    if values is None:
        return np.tile((SPREAD_STOPS[0] * 255).astype(np.uint8), (n, 1))
    t = np.clip(values / 0.5, 0.0, 1.0) * 2.0      # occupancy std 0.5 is the maximum possible
    i = np.minimum(t.astype(int), 1)
    f = (t - i)[:, None]
    rgb = SPREAD_STOPS[i] * (1 - f) + SPREAD_STOPS[i + 1] * f
    return (rgb * 255).astype(np.uint8)


def ply_bytes(vertices: np.ndarray, faces: np.ndarray, spread: np.ndarray | None) -> bytes:
    n = len(vertices)
    colors = spread_colors(spread, n)
    values = spread if spread is not None else np.zeros(n, dtype=np.float32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment BoneVision estimate; quality = silhouette-consistent occupancy std (heuristic)\n"
        f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty float quality\n"
        f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    vdt = np.dtype([("p", "<f4", (3,)), ("c", "u1", (3,)), ("q", "<f4")])
    v = np.empty(n, dtype=vdt)
    v["p"], v["c"], v["q"] = vertices, colors, values
    fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    f = np.empty(len(faces), dtype=fdt)
    f["n"], f["i"] = 3, faces
    return header + v.tobytes() + f.tobytes()


def png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def silhouette_png(result: dict, upscale: int = 3) -> bytes:
    """Green = overlap, warm = traced outline only, blue = fitted projection only (canonical frame)."""
    tgt = result["target_projection"].astype(bool)
    pred = result["projection"] >= 0.5
    rgb = np.full(tgt.shape + (3,), (14, 20, 27), dtype=np.uint8)
    rgb[tgt & pred] = (120, 200, 160)
    rgb[tgt & ~pred] = (237, 158, 89)
    rgb[~tgt & pred] = (120, 190, 245)
    return png_bytes(np.kron(rgb, np.ones((upscale, upscale, 1), dtype=np.uint8)))


def mask_png(result: dict) -> bytes:
    return png_bytes((result["mask"].astype(np.uint8) * 255))


def report_dict(result: dict, input_meta: dict | None, crop_box=None) -> dict:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "software": {"name": "BoneVision", "pipeline_version": result["pipeline_version"]},
        "method": "single-view AP silhouette fit of a voxel PCA statistical shape model, 4 orientation hypotheses",
        "input": {k: v for k, v in (input_meta or {}).items() if k in ("source_name", "format", "sha256", "shape", "pixel_spacing_mm", "spacing_source")}
                 | {"crop_box_rows_cols": list(crop_box) if crop_box else None},
        "model": result["model"],
        "settings": result["settings"],
        "fit_status": result["fit_status"],
        "metrics": {
            "silhouette_soft_dice": result["projection_dice"],
            "silhouette_soft_dice_meaning": "2D overlap between the traced outline and the projected estimate. Not 3D accuracy.",
            "final_objective": result["objective"],
            "orientation_margin": result["orientation_margin"],
            "coefficient_norm": result["coefficient_norm"],
            "coefficient_norm_expected_for_typical_shape": result["coefficient_norm_expected"],
        },
        "orientation": {"selected": result["hypothesis"]["id"], "hypotheses": [
            {k: h[k] for k in ("id", "label", "meaning", "loss", "soft_dice")} for h in result["hypotheses"]]},
        "segmentation": result["segmentation"],
        "standardization": result["transform"],
        "fitted_pose_log_scale_shift": result["fitted_pose_log_scale_shift"],
        "pose_applied_to_mesh": result["pose_applied_to_mesh"],
        "shape_coefficients_z": result["coefficients_z"],
        "shape_spread": result["spread"],
        "mesh": {"vertices": int(len(result["vertices_mm"])), "faces": int(len(result["faces"])),
                 "dimensions_mm": result["dimensions_mm"], "units": "mm", "coordinate_frame": COORDINATE_FRAME},
        "grid": {"shape": list(result["grid_shape"]), "spacing_mm": list(result["spacing_mm"])},
        "timings_s": result["timings"] | {"total": result["elapsed_seconds"]}, "device": result["device"],
        "optimization_loss_history": result["history"],
        "warnings": result["warnings"],
        "limitations": LIMITATIONS,
    }


def build_file(name: str, result: dict, image: np.ndarray, input_meta: dict | None = None, crop_box=None) -> bytes:
    v, f = result["vertices_mm"], result["faces"]
    if name == "femur_estimate.stl":
        return stl_bytes(v, f)
    if name == "femur_estimate.obj":
        return obj_bytes(v, f)
    if name == "femur_estimate_spread.ply":
        return ply_bytes(v, f, result.get("vertex_spread"))
    if name == "report.json":
        return json.dumps(report_dict(result, input_meta, crop_box), indent=2, default=float).encode("utf-8")
    if name == "silhouette_fit.png":
        return silhouette_png(result)
    if name == "input_overlay.png":
        return png_bytes(overlay_on_input(result, image))
    if name == "mask.png":
        return mask_png(result)
    raise KeyError(name)


def bundle_zip(result: dict, image: np.ndarray, input_meta: dict | None = None, crop_box=None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            z.writestr(name, build_file(name, result, image, input_meta, crop_box))
        z.writestr("README.txt", "BoneVision export\n\n" + COORDINATE_FRAME + "\n\nOpen femur_estimate.stl in 3D Slicer with File > Add Data.\n\n"
                   + "\n".join("- " + x for x in LIMITATIONS) + "\n")
    return buf.getvalue()


def write_outputs(result: dict, image: np.ndarray, output_dir, input_meta=None, crop_box=None, force: bool = False) -> list:
    """Write the standard files to a directory, refusing to overwrite unless forced."""
    from pathlib import Path
    output_dir = Path(output_dir)
    paths = [output_dir / n for n in FILES]
    clash = [str(p) for p in paths if p.exists()]
    if clash and not force:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(clash))
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        p.write_bytes(build_file(p.name, result, image, input_meta, crop_box))
    return paths

