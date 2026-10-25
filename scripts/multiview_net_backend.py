"""Reconstruction backend for the trained multi-view network.

Wraps checkpoints from train_multiview_net.py (AP + lateral, and AP only) in
the ReconstructionBackend interface used by the app, so a retrained network
replaces the old one by swapping the checkpoint file.

The network was trained on images that are aligned with the anatomical grid
(femur_grid.py): rows run along the shaft from the knee (row 0) to the head,
AP columns run towards the greater trochanter, lateral columns run
posterior to anterior, and every femur is a left femur. An uploaded X-ray
has none of that, so each view is
  1. segmented (the app's threshold segmentation),
  2. placed on the grid: the femur's long axis is found by PCA of the mask,
     its proximal end is put where the femoral head sits in the grid, its
     length is scaled to the mean training femur, and across the shaft it is
     centred on the mean training silhouette,
  3. tried in four orientations (long axis up or down x image mirrored or
     not). The orientation whose silhouette best matches the mean training
     silhouette is used. (The network's own reprojection cannot decide this:
     it reproduces almost any input silhouette, including mirrored ones.)
Absolute size therefore comes from the training population, as in the SSM.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

from femur_grid import BOUNDS, CANONICAL_SIDE, DIMENSIONS, grid_spacing
from reconstruct_multiview_xray import projection_panel
from reconstruct_single_xray import ROOT, segment_single_bone
from slicer_export import FRAME_DESCRIPTION, LABELMAP_NAME, MESH_NAME, export_for_slicer

MODEL_DIR = ROOT / "models"
DEFAULT_CHECKPOINTS = {("ap", "lat"): MODEL_DIR / "multiview_ap_lat_d2.pt", ("ap",): MODEL_DIR / "multiview_ap_d2.pt"}
REFERENCE_SSM = MODEL_DIR / "ssm_sdf.npz"
ANGLE_TOLERANCE_DEG = 20.0


def classify_view(angle_deg: float) -> str | None:
    """'ap' for 0/180 degrees, 'lat' for 90/270, None for anything else."""
    a = float(angle_deg) % 180.0
    if min(a, 180.0 - a) <= ANGLE_TOLERANCE_DEG:
        return "ap"
    if abs(a - 90.0) <= ANGLE_TOLERANCE_DEG:
        return "lat"
    return None


def reference_silhouettes(checkpoint: dict, ssm_path: Path = REFERENCE_SSM) -> dict:
    """Mean training silhouettes on the full grid, {"ap": (X, Y) bool, "lat": (X, Z) bool,
    "length_mm": mean femur length}."""
    ref = checkpoint.get("reference")
    if ref is not None:
        out = {"ap": np.asarray(ref["ap"]).astype(bool), "lat": np.asarray(ref["lat"]).astype(bool)}
        if "length_mm" in ref:
            out["length_mm"] = float(ref["length_mm"])
        return out
    if not Path(ssm_path).exists():
        raise FileNotFoundError(f"{ssm_path} is needed for the mean femur silhouette; build it with scripts/build_ssm_sdf.py")
    data = np.load(ssm_path)
    occ = data["mean_shape"].reshape(tuple(int(v) for v in data["grid_shape"])) < 0
    return {"ap": occ.any(axis=2), "lat": occ.any(axis=1)}


def _principal_frame(points: np.ndarray):
    """Centroid, long axis u and cross axis v (u rotated +90 degrees) of 2D points."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    u = vt[0] / np.linalg.norm(vt[0])
    return centroid, u, np.array([-u[1], u[0]])


def place_on_grid(image: np.ndarray, mask: np.ndarray, ref: dict, kind: str, s_rows: float, s_cols: float,
                  flip: bool, mirror: bool, return_affine: bool = False):
    """Resample an X-ray and its femur mask onto the network's grid for one orientation hypothesis.

    The femur's principal axis is matched to the principal axis of the mean
    training silhouette (in millimetres, so the small tilt of the neck is
    kept). Its proximal end goes where the mean femur's proximal end is (the
    grid's origin is the femoral head), its length is scaled to the mean
    training femur, and across the shaft the centroids are matched.
    Returns (grid_image, grid_mask, mm_per_pixel), plus with return_affine the
    map (M, t) from grid (row, col) to image (row, col): p = M @ q + t.
    """
    ref_mask = ref[kind]
    ref_pts = np.argwhere(ref_mask) * np.array([s_rows, s_cols])
    ref_c, ref_u, ref_v = _principal_frame(ref_pts)
    if ref_u[0] < 0:                       # point the reference axis towards the head (increasing rows)
        ref_u, ref_v = -ref_u, -ref_v
    ref_along = (ref_pts - ref_c) @ ref_u
    ref_len_mm = ref.get("length_mm", float(np.ptp(ref_along)))

    pts = np.argwhere(mask).astype(np.float64)
    centroid, u, v = _principal_frame(pts)
    if flip:
        u, v = -u, -v
    if mirror:
        v = -v
    along = (pts - centroid) @ u
    mm_per_px = ref_len_mm / max(float(np.ptp(along)), 1.0)

    rr, cc = np.meshgrid(np.arange(ref_mask.shape[0]), np.arange(ref_mask.shape[1]), indexing="ij")
    g = np.stack([rr * s_rows, cc * s_cols], axis=-1) - ref_c
    alpha, beta = g @ ref_u, g @ ref_v
    a = (alpha - ref_along.max()) / mm_per_px + along.max()   # pixels along the femur from its centroid
    b = beta / mm_per_px                                       # pixels across it
    coords = centroid[:, None, None] + u[:, None, None] * a + v[:, None, None] * b
    background = float(np.percentile(image, 1))
    grid_image = ndi.map_coordinates(image.astype(np.float32), coords, order=1, mode="constant", cval=background)
    grid_mask = ndi.map_coordinates(mask.astype(np.float32), coords, order=0, mode="constant", cval=0.0) > 0.5
    if not return_affine:
        return grid_image, grid_mask, mm_per_px
    t = coords[:, 0, 0]
    M = np.stack([coords[:, 1, 0] - t, coords[:, 0, 1] - t], axis=1)   # the map is affine, so two steps fix it
    return grid_image, grid_mask, mm_per_px, (M, t)


def _dice(a, b):
    """Dice of two silhouettes; probabilities are thresholded at 0.5."""
    a, b = np.asarray(a) >= 0.5, np.asarray(b) >= 0.5
    return float((2.0 * (a & b).sum() + 1.0) / (a.sum() + b.sum() + 1.0))


def surface_from_occupancy(occupancy: np.ndarray):
    """Keep the largest solid component of a grid occupancy and mesh its 0.5 level in millimetres.
    Returns (cleaned occupancy, vertices_mm, faces)."""
    solid = ndi.binary_fill_holes(occupancy >= 0.5)
    labels, n = ndi.label(solid)
    if n == 0:
        raise ValueError("The network predicted an empty volume; check the segmentation masks and view labels")
    if n > 1:
        solid = labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    occupancy = np.where(solid, np.maximum(occupancy, 0.5), np.minimum(occupancy, 0.49)).astype(np.float32)
    verts, faces, _, _ = marching_cubes(occupancy, level=0.5, spacing=grid_spacing())
    verts = verts + np.asarray(BOUNDS[0::2], dtype=np.float32)[None, :]
    return occupancy, verts.astype(np.float32), faces.astype(np.int32)


class MultiViewNetBackend:
    """Learned AP (+ lateral) to 3D femur network; see module docstring."""
    name = "multiview-net"
    label = "Multi-view network, AP + lateral (trained on HFValid DRRs)"
    max_views = 2

    def __init__(self, checkpoints: dict | None = None):
        self.checkpoints = dict(checkpoints or DEFAULT_CHECKPOINTS)
        if os.environ.get("BONEVISION_NET_AP_LAT"):
            self.checkpoints[("ap", "lat")] = Path(os.environ["BONEVISION_NET_AP_LAT"])
        if os.environ.get("BONEVISION_NET_AP"):
            self.checkpoints[("ap",)] = Path(os.environ["BONEVISION_NET_AP"])
        self._cache: dict = {}

    def availability(self) -> str | None:
        if not any(Path(p).exists() for p in self.checkpoints.values()):
            names = ", ".join(str(p) for p in self.checkpoints.values())
            return f"No trained network found ({names}). Train one with scripts/run_retraining.py."
        return None

    def _load(self, views: tuple, device):
        import torch
        from multiview_net import MultiViewFemurNet
        path = Path(self.checkpoints[views])
        key = (str(path), path.stat().st_mtime, str(device))
        if key not in self._cache:
            state = torch.load(path, map_location=device)
            cfg = state["config"]
            net = MultiViewFemurNet(views=tuple(cfg["views"])).to(device)
            net.load_state_dict(state["model"])
            net.eval()
            self._cache = {key: (net, cfg, reference_silhouettes(state), path, state.get("val", {}))}
        return self._cache[key]

    def _pick_model(self, kinds: set) -> tuple:
        wanted = ("ap", "lat") if "lat" in kinds else ("ap",)
        if Path(self.checkpoints.get(wanted, "")).exists():
            return wanted
        if wanted == ("ap", "lat") and Path(self.checkpoints[("ap",)]).exists():
            return ("ap",)
        raise FileNotFoundError(f"The {' + '.join(wanted).upper()} network checkpoint is missing: {self.checkpoints.get(wanted)}")

    def reconstruct(self, views, iterations=250, prior_weight=0.02, device=None, refine_iterations=100):
        # iterations / prior_weight / refine_iterations belong to the SSM fit and are not used here.
        import torch
        import torch.nn.functional as F
        from train_multiview_net import normalize, pool2d
        start = time.perf_counter()
        problem = self.availability()
        if problem:
            raise FileNotFoundError(problem)
        if not views or len(views) > self.max_views:
            raise ValueError("The network takes one AP view, optionally with one lateral view")
        kinds = []
        for view in views:
            kind = classify_view(view.angle_deg)
            if kind is None:
                raise ValueError(f"{view.name}: the network only accepts AP (0°) and lateral (90°) views, got {view.angle_deg:.0f}°")
            kinds.append(kind)
        if kinds.count("ap") != 1 or kinds.count("lat") > 1:
            raise ValueError("Provide exactly one AP view and at most one lateral view")
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_views = self._pick_model(set(kinds))
        net, cfg, ref, ckpt_path, val = self._load(model_views, torch_device)
        f = int(cfg["downsample"])
        sx, sy, sz = grid_spacing()
        col_spacing = {"ap": sy, "lat": sz}
        used = [(view, kind) for view, kind in zip(views, kinds) if kind in model_views]

        # Segment each view, place it on the grid in each orientation, and keep the
        # orientation whose silhouette best matches the mean training femur: head
        # versus knee and trochanter side are distinct enough to tell them apart.
        per_view, scores = [], []
        for view, kind in used:
            image = np.asarray(view.image, dtype=np.float32)
            try:
                mask, seg = segment_single_bone(image, view.polarity, view.threshold_bias)
            except ValueError as exc:
                raise ValueError(f"{view.name}: {exc}") from exc
            bright = image if seg["polarity"] == "bright" else 1.0 - image
            mirrors = {"auto": (False, True), "no": (False,), "yes": (True,)}[view.mirror]
            hyps = []
            for flip, mirror in itertools.product((False, True), mirrors):
                g_img, g_mask, mm_px, affine = place_on_grid(bright, mask, ref, kind, sx, col_spacing[kind], flip, mirror,
                                                             return_affine=True)
                hyps.append({"flip": flip, "mirror": mirror, "grid_image": g_img, "grid_mask": g_mask,
                             "mm_per_pixel": mm_px, "grid_to_image": affine, "reference_dice": _dice(g_mask, ref[kind])})
            hyps.sort(key=lambda h: -h["reference_dice"])
            scores += [h["reference_dice"] for h in hyps]
            per_view.append({"view": view, "kind": kind, "mask": mask, "seg": seg, "hyps": hyps, "chosen": hyps[0]})

        ny, nz = DIMENSIONS[1] // f, DIMENSIONS[2] // f
        images = {p["kind"]: torch.from_numpy(normalize(pool2d(p["chosen"]["grid_image"] - p["chosen"]["grid_image"].min(), f)))[None, None].to(torch_device)
                  for p in per_view}
        with torch.no_grad(), torch.autocast(device_type=torch_device.type, enabled=torch_device.type == "cuda"):
            prob = torch.sigmoid(net(images, (ny, nz)).float())[0, 0]
        combos = [len(p["hyps"]) for p in per_view]
        occupancy = F.interpolate(prob[None, None], scale_factor=f, mode="trilinear", align_corners=False)[0, 0]
        occupancy = occupancy.clamp(0, 1).cpu().numpy()
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
        occupancy, verts, faces = surface_from_occupancy(occupancy)
        spacing = grid_spacing()

        view_results = []
        for p_i, p in enumerate(per_view):
            h = p["chosen"]
            proj = occupancy.max(axis=2) if p["kind"] == "ap" else occupancy.max(axis=1)
            target = h["grid_mask"]
            view_results.append({
                "name": p["view"].name, "angle_deg": float(p["view"].angle_deg), "view_type": p["kind"],
                "mask": p["mask"], "segmentation": p["seg"], "target_projection": target, "projection": proj,
                "soft_dice": _dice(proj, target), "mirrored": bool(h["mirror"]), "flipped_long_axis": bool(h["flip"]),
                "orientation_match": round(h["reference_dice"], 4),
                "orientation_margin": round(h["reference_dice"] - (p["hyps"][1]["reference_dice"] if len(p["hyps"]) > 1 else 0.0), 4),
                "assumed_mm_per_pixel": round(h["mm_per_pixel"], 4), "grid_to_image": h["grid_to_image"],
                "input_image_shape_pixels": list(np.asarray(p["view"].image).shape),
            })
        ignored = [v.name for v, k in zip(views, kinds) if k not in model_views]
        warnings = []
        for r in view_results:
            if r["orientation_margin"] < 0.02:
                warnings.append(f"{r['name']}: the orientation is uncertain (the femur mask barely prefers one "
                                "orientation). Check that the green mask covers only the femur, or set the "
                                "left/right orientation by hand.")
            if r["soft_dice"] < 0.8:
                warnings.append(f"{r['name']}: the 3D estimate reproduces this view's silhouette poorly "
                                f"(Dice {r['soft_dice']:.2f}). The mask, the crop or the view label may be wrong.")
        dices = [r["soft_dice"] for r in view_results]
        return {
            "backend": self.name,
            "method": "learned {} network ({})".format(" + ".join(model_views).upper(), ckpt_path.name),
            "views": view_results, "n_views": len(view_results), "ignored_views": ignored,
            "occupancy": occupancy, "vertices_mm": verts.astype(np.float32), "faces": faces.astype(np.int32),
            "projection_dice": float(np.mean(dices)), "min_view_dice": float(np.min(dices)),
            "history": scores, "hypotheses_tested": int(sum(combos)),
            "flipped_long_axis": bool(view_results[0]["flipped_long_axis"]),
            "device": torch_device.type, "elapsed_seconds": time.perf_counter() - start,
            "spacing_mm": spacing, "bounds": tuple(float(b) for b in BOUNDS), "grid_shape": tuple(occupancy.shape),
            "dimensions_mm": tuple(float(x) for x in np.ptp(verts, axis=0)),
            "grid_dimensions_mm": tuple(float(b - a) for a, b in zip(BOUNDS[::2], BOUNDS[1::2])),
            "checkpoint": str(ckpt_path), "model_views": list(model_views), "network_downsample": f,
            "checkpoint_validation": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in val.items()},
            "canonical_side": CANONICAL_SIDE, "fit_status": "completed", "warnings": warnings,
            "details": [
                f"Model: {' + '.join(model_views).upper()} network, checkpoint {ckpt_path.name}",
                f"Validation Dice at training time: {val.get('dice', float('nan')):.3f} (synthetic DRRs)",
                "Orientation match with the mean femur (margin over the next best): " + ", ".join(
                    f"{r['view_type'].upper()} {r['orientation_match']:.2f} (+{r['orientation_margin']:.2f})" for r in view_results),
                "Output is a left femur; femur length is set to the training mean",
            ] + ([f"Ignored views: {', '.join(ignored)}"] if ignored else []),
        }

    def save(self, result, output_dir):
        output_dir = Path(output_dir)
        slicer = export_for_slicer(output_dir, result["vertices_mm"], result["faces"], result["occupancy"], result["bounds"])
        Image.fromarray(projection_panel(result)).save(output_dir / "projection_fit.png")
        report = {
            "method": result["method"], "backend": self.name, "fit_status": result["fit_status"],
            "checkpoint": result["checkpoint"], "model_views": result["model_views"],
            "checkpoint_validation_synthetic": result["checkpoint_validation"],
            "mean_projection_dice": result["projection_dice"], "min_view_projection_dice": result["min_view_dice"],
            "orientation_hypotheses_tested": result["hypotheses_tested"],
            "runtime_seconds": result["elapsed_seconds"], "device": result["device"],
            "grid_shape": result["grid_shape"], "spacing_mm": result["spacing_mm"],
            "dimensions_mm": result["dimensions_mm"], "canonical_side": result["canonical_side"],
            "ignored_views": result["ignored_views"], "warnings": result["warnings"],
            "views": [{k: v for k, v in view.items() if k not in ("mask", "target_projection", "projection", "grid_to_image")}
                      for view in result["views"]],
            "coordinate_frame": FRAME_DESCRIPTION, "output_mesh": MESH_NAME, "output_labelmap": LABELMAP_NAME,
            "slicer": slicer,
            "warning": "Research estimate only, not validated for clinical use. The network was trained on simulated "
                       "X-rays (DRRs) from 101 CTs and has not been validated on real radiographs. Femur length is set "
                       "to the training mean, the views are assumed to be true AP and lateral, and the mesh is a left "
                       "femur; mirror it for a right leg.",
        }
        (output_dir / "reconstruction.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
