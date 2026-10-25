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


def stl_bytes(vertices: np.ndarray, faces: np.ndarray, header: bytes = b"BoneVision single-view SSM estimate SPACE=RAS") -> bytes:
    """Binary STL. The SPACE=RAS header tag makes 3D Slicer load the coordinates unchanged (it reads
    untagged STLs as LPS and flips x and y); the frame is still the canonical one, not the patient's."""
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
    if is_network(result):
        return build_network_file(name, result, input_meta)
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
    if is_network(result):
        return network_bundle_zip(result, input_meta)
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



# ------------------------------------------------------------- learned network

NETWORK_FRAME = (
    "Femur landmark frame in millimetres, the network's training grid: origin at the femoral head, "
    "+X along the femoral long axis (distal is negative X), +Y toward the greater trochanter, Z = X × Y. "
    "Always a LEFT femur (right legs are mirrored onto the left). The STL and NRRD are tagged as Slicer RAS so "
    "they load unchanged and overlap exactly. This is NOT patient RAS/LPS space."
)
NETWORK_LIMITATIONS = [
    "Research estimate only; not for diagnosis, treatment or surgical planning.",
    "The network was trained on simulated X-rays (DRRs) from 101 HFValid CTs and has not been validated on real radiographs.",
    "Femur length is set to the training mean; uploaded images carry no reliable physical scale.",
    "Views are assumed to be true AP and true lateral of the same femur. An AP-only estimate takes depth from what the network learned.",
    "Segmentation is classical thresholding and can select the wrong region; check each mask before trusting the shape.",
    "The voxel grid is about 4 mm, so fine surface detail is smoothed.",
]
VIEW_ARRAYS = ("mask", "target_projection", "projection", "grid_to_image", "input_image")
MESH_NAME, LABELMAP_NAME, NETWORK_REPORT = "femur_reconstruction.stl", "femur_labelmap.nrrd", "reconstruction.json"


def is_network(result: dict) -> bool:
    return result.get("backend") == "multiview-net"


def network_files(result: dict) -> dict:
    files = {
        MESH_NAME: ("model/stl", "Binary STL mesh tagged as RAS for 3D Slicer (File → Add Data)."),
        LABELMAP_NAME: ("application/octet-stream", "Voxel labelmap (1 = femur) on the same grid, for Slicer slice views and Segment Editor."),
        "femur_estimate.obj": ("text/plain", "Wavefront OBJ mesh for Blender, MeshLab and most 3D tools."),
        NETWORK_REPORT: ("application/json", "Checkpoint, views, per-view agreement, warnings, coordinate frame and limits."),
        "view_fit.png": ("image/png", "Each view on the network grid: traced outline (red) against the estimate's silhouette (cyan)."),
    }
    for view in result["views"]:
        files[f"input_overlay_{view['kind']}.png"] = ("image/png", f"Your {'AP' if view['kind'] == 'ap' else 'lateral'} X-ray with its mask and the estimate's outline mapped back.")
    return files


def network_report(result: dict, input_meta: dict | None) -> dict:
    from pathlib import Path

    from . import network  # noqa: F401  (puts scripts/ on the import path)
    from slicer_export import slicer_info
    slicer = slicer_info() | {"coordinate_frame": NETWORK_FRAME}
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "software": {"name": "BoneVision", "pipeline_version": result["pipeline_version"]},
        "method": result["method_label"], "backend": result["backend"],
        "inputs": (input_meta or {}).get("views", []),
        "checkpoint": Path(result["checkpoint"]).name, "model_views": result["model_views"],
        "checkpoint_validation_synthetic": result["checkpoint_validation"],
        "fit_status": result["fit_status"],
        "metrics": {
            "mean_view_silhouette_dice": result["projection_dice"], "min_view_silhouette_dice": result["min_view_dice"],
            "meaning": "2D overlap between each traced outline and the estimate's silhouette on the network grid. Not 3D accuracy.",
        },
        "orientation_hypotheses_tested": result["hypotheses_tested"],
        "views": [{k: v for k, v in view.items() if k not in VIEW_ARRAYS} for view in result["views"]],
        "ignored_views": result["ignored_views"],
        "mesh": {"vertices": int(len(result["vertices_mm"])), "faces": int(len(result["faces"])),
                 "dimensions_mm": result["dimensions_mm"], "units": "mm", "coordinate_frame": NETWORK_FRAME},
        "grid": {"shape": list(result["grid_shape"]), "spacing_mm": list(result["spacing_mm"]), "bounds_mm": list(result["bounds"])},
        "canonical_side": result["canonical_side"],
        "output_mesh": MESH_NAME, "output_labelmap": LABELMAP_NAME, "slicer": slicer,
        "timings_s": {"total": result["elapsed_seconds"]}, "device": result["device"],
        "warnings": result["warnings"],
        "limitations": NETWORK_LIMITATIONS,
    }


def view_fit_png(result: dict, upscale: int = 3) -> bytes:
    """Per-view canvases on the network grid: green = overlap, warm = outline only, blue = estimate only."""
    panels = []
    for view in result["views"]:
        tgt = view["target_projection"].astype(bool)
        pred = view["projection"] >= 0.5
        rgb = np.full(tgt.shape + (3,), (14, 20, 27), dtype=np.uint8)
        rgb[tgt & pred] = (120, 200, 160)
        rgb[tgt & ~pred] = (237, 158, 89)
        rgb[~tgt & pred] = (120, 190, 245)
        panels.append(rgb[::-1])             # grid row 0 is the knee; show the head at the top
    height = max(p.shape[0] for p in panels)
    padded = [np.pad(p, ((0, height - p.shape[0]), (0, 6), (0, 0)), constant_values=40) for p in panels]
    return png_bytes(np.kron(np.concatenate(padded, axis=1), np.ones((upscale, upscale, 1), dtype=np.uint8)))


def build_network_file(name: str, result: dict, input_meta: dict | None = None) -> bytes:
    from .network import overlay_view
    from slicer_export import labelmap_bytes, stl_bytes as ras_stl_bytes
    v, f = result["vertices_mm"], result["faces"]
    if name == MESH_NAME:
        return ras_stl_bytes(v, f, "BoneVision network femur estimate")
    if name == LABELMAP_NAME:
        return labelmap_bytes(result["occupancy"], result["bounds"])
    if name == "femur_estimate.obj":
        return obj_bytes(v, f).replace(b"single-view SSM estimate (millimetres, canonical SSM frame)",
                                       b"network estimate (millimetres, femur landmark frame, left femur)", 1)
    if name == NETWORK_REPORT:
        return json.dumps(network_report(result, input_meta), indent=2, default=float).encode("utf-8")
    if name == "view_fit.png":
        return view_fit_png(result)
    for view in result["views"]:
        if name == f"input_overlay_{view['kind']}.png":
            return png_bytes(overlay_view(view, view["input_image"]))
    raise KeyError(name)


def network_bundle_zip(result: dict, input_meta: dict | None = None) -> bytes:
    """All network files plus the Slicer loader in one folder: run BoneVisionLoader.py from it to open everything."""
    from pathlib import Path
    loader = Path(__file__).resolve().parents[1] / "slicer" / "BoneVisionLoader" / "BoneVisionLoader.py"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in network_files(result):
            z.writestr(f"bonevision_femur/{name}", build_network_file(name, result, input_meta))
        if loader.exists():
            z.write(loader, f"bonevision_femur/{loader.name}")
        z.writestr("bonevision_femur/README.txt", "BoneVision network export\n\n" + NETWORK_FRAME
                   + "\n\nIn 3D Slicer: File > Add Data on femur_reconstruction.stl (tick LabelMap for the NRRD), or run\n"
                   + "  Slicer --python-script BoneVisionLoader.py\nfrom this folder.\n\n"
                   + "\n".join("- " + x for x in NETWORK_LIMITATIONS) + "\n")
    return buf.getvalue()


def files_for(result: dict) -> dict:
    return network_files(result) if is_network(result) else FILES
