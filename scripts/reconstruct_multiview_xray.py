"""Multi-view femur shape-prior reconstruction.

Fits one set of CT-trained voxel SSM coefficients jointly to the silhouettes
of several radiographs of the same femur (for example AP + lateral). Each view
is described by its rotation about the femur's long axis (0 = AP, 90 =
lateral) and keeps its own 2D nuisance pose (scale and shift), because plain
image files carry no detector calibration. The 3D shape is shared, so extra
views constrain depth that a single AP image leaves to the population prior.

One view reproduces the single-view fit in reconstruct_single_xray.py. The
result is still a research estimate in the canonical SSM frame: view angles
are user-declared, not measured, and absolute scale is not recovered.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

from reconstruct_single_xray import BOUNDS_DEFAULT, MODEL_PATH, read_image, segment_single_bone, standardize_mask
from slicer_export import FRAME_DESCRIPTION, LABELMAP_NAME, MESH_NAME, export_for_slicer

MAX_VIEWS = 6
VIEW_PRESETS = {"AP": 0.0, "lateral": 90.0}


@dataclass
class ViewInput:
    """One radiograph of the femur plus how it was acquired and segmented."""
    image: np.ndarray
    angle_deg: float = 0.0
    name: str = "view"
    polarity: str = "auto"
    threshold_bias: float = 0.0
    mirror: str = "auto"  # "auto" tries both horizontal orientations; "no" / "yes" fix it


def load_model(model_path: Path = MODEL_PATH) -> dict:
    """Load the SSM arrays once so callers can reuse them across fits."""
    data = np.load(model_path)
    shape = tuple(int(v) for v in data["grid_shape"])
    return {
        "shape": shape,
        "bounds": tuple(float(v) for v in data["bounds"]) if "bounds" in data else BOUNDS_DEFAULT,
        "mean": data["mean_shape"].astype(np.float32, copy=False),
        "components": data["components"].astype(np.float32, copy=False),
        "explained_variance": data["explained_variance"].astype(np.float32, copy=False),
        "n_training_subjects": int(len(data["train_subjects"])) if "train_subjects" in data else None,
        "path": str(model_path),
    }


def _rotation_grid(shape, bounds, angle_deg, device):
    """Sampling grid for rays at angle_deg about the long axis, plus step weight.

    Output pixels span the wider transverse extent; depth samples span the
    transverse diagonal. The returned weight converts sums over samples into
    the same units as an AP sum over native Z voxels.
    """
    import torch
    _, ny, nz = shape
    y0, y1, z0, z1 = bounds[2], bounds[3], bounds[4], bounds[5]
    cy, cz = (y0 + y1) / 2, (z0 + z1) / 2
    width = max(y1 - y0, z1 - z0)
    depth = math.hypot(y1 - y0, z1 - z0)
    n_u = max(ny, nz)
    n_d = int(math.ceil(depth / ((z1 - z0) / (nz - 1))))
    theta = math.radians(angle_deg)
    u_dir = (math.cos(theta), -math.sin(theta))
    d_dir = (math.sin(theta), math.cos(theta))
    u = torch.linspace(-width / 2, width / 2, n_u, device=device)
    t = torch.linspace(-depth / 2, depth / 2, n_d, device=device)
    tt, uu = torch.meshgrid(t, u, indexing="ij")
    py = cy + uu * u_dir[0] + tt * d_dir[0]
    pz = cz + uu * u_dir[1] + tt * d_dir[1]
    grid = torch.stack(((pz - z0) / (z1 - z0) * 2 - 1, (py - y0) / (y1 - y0) * 2 - 1), dim=-1)
    step_weight = (depth / (n_d - 1)) / ((z1 - z0) / (nz - 1))
    return grid[None], step_weight


def project_basis(volumes, shape, bounds, angle_deg: float, device, chunk: int = 8):
    """Project flattened volumes (N, L*Y*Z) along rays at angle_deg.

    Returns (sum projections, max projections), each (N, L, W). Exact
    quarter-turns use axis sums; other angles resample with grid_sample.
    """
    import torch
    import torch.nn.functional as F
    vols = torch.as_tensor(volumes, dtype=torch.float32, device=device).reshape((-1,) + tuple(shape))
    quarter = round(angle_deg / 90.0)
    if abs(angle_deg - 90.0 * quarter) < 1e-6:
        quarter %= 4
        axis = 3 if quarter in (0, 2) else 2
        sums, maxes = vols.sum(dim=axis), vols.amax(dim=axis)
        if quarter in (2, 3):  # opposite viewing direction mirrors the image
            sums, maxes = sums.flip(-1), maxes.flip(-1)
        return sums, maxes
    grid, weight = _rotation_grid(shape, bounds, angle_deg, device)
    sums, maxes = [], []
    for start in range(0, len(vols), chunk):
        part = vols[start:start + chunk]  # (n, L, Y, Z): long-axis slices act as channels
        sampled = F.grid_sample(part, grid.expand(len(part), -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=True)
        sums.append(sampled.sum(dim=2) * weight)
        maxes.append(sampled.amax(dim=2))
    return torch.cat(sums), torch.cat(maxes)


def prepare_view(model: dict, angle_deg: float, device):
    """Project the SSM mean and modes once for one view angle."""
    import torch
    mean_sum, mean_max = project_basis(model["mean"][None], model["shape"], model["bounds"], angle_deg, device)
    comp_sum, _ = project_basis(model["components"], model["shape"], model["bounds"], angle_deg, device)
    image_shape = tuple(mean_sum.shape[1:])
    return {
        "angle_deg": float(angle_deg),
        "image_shape": image_shape,
        "mean_2d": mean_sum.reshape(-1),
        "comp_2d": comp_sum.reshape(len(comp_sum), -1),
        "reference_mask": (mean_max[0] > 0.35).cpu().numpy(),
    }


def _warp(pred, pose):
    """Apply per-hypothesis 2D scale/shift nuisance pose, as in the single-view fit."""
    import torch
    import torch.nn.functional as F
    log_scale = pose[:, 0].clamp(-0.18, 0.18)
    shift = pose[:, 1:].clamp(-0.22, 0.22)
    theta = torch.zeros((len(pred), 2, 3), device=pred.device)
    theta[:, 0, 0] = torch.exp(-log_scale)
    theta[:, 1, 1] = theta[:, 0, 0]
    theta[:, 0, 2] = -shift[:, 1]
    theta[:, 1, 2] = -shift[:, 0]
    grid = F.affine_grid(theta, pred[:, None].shape, align_corners=False)
    return F.grid_sample(pred[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]


def _hypotheses(views: list[ViewInput]) -> list[tuple[bool, tuple[bool, ...]]]:
    """Long-axis flip (shared by all views) x per-view horizontal mirror choices."""
    mirror_options = [(False, True) if v.mirror == "auto" else ((v.mirror == "yes"),) for v in views]
    return [(flip, mirrors) for flip in (False, True) for mirrors in itertools.product(*mirror_options)]


SHARPNESS = 12.0  # logistic slope used to turn decoded occupancy into a near-binary solid


def _silhouette_loss(pred, tgt, pose):
    import torch.nn.functional as F
    inter = (pred * tgt).sum(dim=(1, 2))
    dice_loss = 1.0 - (2.0 * inter + 1.0) / (pred.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2)) + 1.0)
    bce = F.binary_cross_entropy(pred, tgt, reduction="none").mean(dim=(1, 2))
    return dice_loss + 0.08 * bce + 0.04 * pose.square().mean(dim=1)


def _solid_projection(occ_flat, model, view, device):
    """Silhouette of the decoded solid: the ray maximum of sharpened occupancy.

    A ray is inside the silhouette when it crosses any voxel above the 0.5
    mesh level, so this matches the exported surface. A line integral would
    also light rays that only cross deep, partly occupied regions.
    """
    import torch
    _, ray_max = project_basis(torch.sigmoid(SHARPNESS * (occ_flat - 0.5)), model["shape"], model["bounds"], view["angle_deg"], device)
    return ray_max


def _compose_poses(log_scale, shifts):
    """(H, 1) shared log-scale and (V, H, 2) shifts -> (V, H, 3) poses for _warp."""
    import torch
    return torch.cat((log_scale[None].expand(len(shifts), -1, -1), shifts), dim=2)


def fit_views(targets: list[np.ndarray], prepared_views: list[dict], model: dict, iterations: int, device, prior_weight: float = 0.02, refine_iterations: int = 100):
    """Jointly fit shared SSM coefficients to per-view targets of shape (H, h, w).

    Stage 1 fits every orientation hypothesis with precomputed linear
    projections of the PCA modes (fast, as in the single-view fit). Linear
    projections saturate on soft occupancy, so a silhouette can match while the
    0.5-level mesh is thinner than it; stage 2 therefore refines the chosen
    hypothesis against projections of the decoded, sharpened 3D solid.

    Views are normalized to a common bone length, so they share one 2D scale
    nuisance and only their shifts are per view.
    """
    import torch
    import torch.nn.functional as F
    n_hyp = targets[0].shape[0]
    n_comp = int(model["components"].shape[0])
    std = torch.as_tensor(np.sqrt(model["explained_variance"]), dtype=torch.float32, device=device)
    target_t = [F.avg_pool2d(torch.as_tensor(t.astype(np.float32), device=device)[:, None], 3, 1, 1)[:, 0] for t in targets]
    params = torch.nn.Parameter(torch.zeros((n_hyp, n_comp), device=device))
    log_scale = torch.nn.Parameter(torch.zeros((n_hyp, 1), device=device))
    shifts = torch.nn.Parameter(torch.zeros((len(targets), n_hyp, 2), device=device))
    opt = torch.optim.Adam(({"params": params, "lr": 0.04}, {"params": (log_scale, shifts), "lr": 0.015}))
    best_losses = torch.full((n_hyp,), float("inf"), device=device)
    best_params = torch.zeros_like(params)
    best_poses = torch.zeros((len(targets), n_hyp, 3), device=device)
    history = []
    for _ in range(iterations):
        opt.zero_grad(set_to_none=True)
        coeff = params.clamp(-2.5, 2.5) * std[None, :]
        poses = _compose_poses(log_scale, shifts)
        view_losses = []
        for v, view in enumerate(prepared_views):
            line = (view["mean_2d"][None, :] + coeff @ view["comp_2d"]).reshape((n_hyp,) + view["image_shape"]).clamp_min(0.0)
            pred = _warp((1.0 - torch.exp(-line)).clamp(1e-5, 1.0 - 1e-5), poses[v]).clamp(1e-5, 1.0 - 1e-5)
            view_losses.append(_silhouette_loss(pred, target_t[v], poses[v]))
        losses = torch.stack(view_losses).mean(dim=0) + float(prior_weight) * params.square().mean(dim=1)
        losses.mean().backward()
        opt.step()
        with torch.no_grad():
            improved = losses < best_losses
            best_losses = torch.where(improved, losses, best_losses)
            best_params = torch.where(improved[:, None], params.detach().clamp(-2.5, 2.5), best_params)
            best_poses = torch.where(improved[None, :, None], poses.detach(), best_poses)
            history.append(losses.detach().clone())
    choice = int(torch.argmin(best_losses).item())
    history = torch.stack(history)[:, choice].cpu().numpy().astype(float).tolist()
    mean = torch.as_tensor(model["mean"], dtype=torch.float32, device=device)
    components = torch.as_tensor(model["components"], dtype=torch.float32, device=device)
    targets_chosen = [t[choice:choice + 1] for t in target_t]
    coeffs = best_params[choice:choice + 1].clone()
    pose_fit = best_poses[:, choice:choice + 1].clone()

    def solid_losses(c, p):
        occ = (mean + (c.clamp(-2.5, 2.5) * std) @ components).clamp(0.0, 1.0)
        per_view = []
        for v, view in enumerate(prepared_views):
            pred = _warp(_solid_projection(occ, model, view, device).clamp(1e-5, 1.0 - 1e-5), p[v]).clamp(1e-5, 1.0 - 1e-5)
            per_view.append(_silhouette_loss(pred, targets_chosen[v], p[v]))
        return torch.stack(per_view).mean(dim=0) + float(prior_weight) * c.square().mean(dim=1)

    refine_history = []
    if refine_iterations > 0:
        c = torch.nn.Parameter(coeffs.clone())
        r_scale = torch.nn.Parameter(pose_fit[0, :, :1].clone())
        r_shifts = torch.nn.Parameter(pose_fit[:, :, 1:].clone())
        opt = torch.optim.Adam(({"params": c, "lr": 0.02}, {"params": (r_scale, r_shifts), "lr": 0.008}))
        with torch.no_grad():
            best = solid_losses(c, pose_fit)[0]
        for _ in range(refine_iterations):
            opt.zero_grad(set_to_none=True)
            p = _compose_poses(r_scale, r_shifts)
            loss = solid_losses(c, p)
            loss.sum().backward()
            opt.step()
            with torch.no_grad():
                refine_history.append(float(loss[0]))
                if loss[0] < best:
                    best = loss[0].detach()
                    coeffs = c.detach().clamp(-2.5, 2.5).clone()
                    pose_fit = p.detach().clone()
    with torch.no_grad():
        objective = float(solid_losses(coeffs, pose_fit)[0])
        occ = (mean + (coeffs[0] * std) @ components).clamp(0.0, 1.0)
        per_view = []
        for v, view in enumerate(prepared_views):
            fitted = _warp(_solid_projection(occ[None], model, view, device), pose_fit[v])[0]
            tgt = targets_chosen[v][0]
            soft_dice = float((2.0 * (fitted * tgt).sum() + 1.0) / (fitted.sum() + tgt.sum() + 1.0))
            per_view.append({"projection": fitted.cpu().numpy(), "soft_dice": soft_dice, "pose": tuple(float(x) for x in pose_fit[v, 0].cpu())})
        occ_np = occ.reshape(model["shape"]).cpu().numpy()
    return {
        "occupancy": occ_np, "choice": choice, "objective": objective,
        "history": history + refine_history, "refine_iterations": len(refine_history),
        "views": per_view, "coefficients": coeffs[0].cpu().numpy().astype(float).tolist(),
    }


def reconstruct_multiview(views: list[ViewInput], model_path: Path = MODEL_PATH, iterations: int = 250, device: str | None = None, prior_weight: float = 0.02, model: dict | None = None, prepared_cache: dict | None = None, refine_iterations: int = 100) -> dict:
    """Segment each view, fit the SSM to all silhouettes, and extract a millimetre mesh.

    `model` and `prepared_cache` (keyed by angle) let a long-running app reuse
    the loaded SSM and its per-angle projections between requests.
    """
    import torch
    if not views:
        raise ValueError("Provide at least one X-ray view")
    if len(views) > MAX_VIEWS:
        raise ValueError(f"At most {MAX_VIEWS} views are supported")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if refine_iterations < 0:
        raise ValueError("refine_iterations must be nonnegative")
    if not np.isfinite(prior_weight) or prior_weight < 0:
        raise ValueError("prior_weight must be a finite, nonnegative number")
    for view in views:
        if np.asarray(view.image).ndim != 2:
            raise ValueError(f"{view.name}: expected one 2D grayscale X-ray image")
        if not np.isfinite(view.image).all():
            raise ValueError(f"{view.name}: X-ray contains NaN or infinite pixel values")
        if view.mirror not in ("auto", "no", "yes"):
            raise ValueError(f"{view.name}: mirror must be auto, no, or yes")
    start = time.perf_counter()
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model or load_model(model_path)
    cache = prepared_cache if prepared_cache is not None else {}
    prepared = []
    for view in views:
        key = (model["path"], round(float(view.angle_deg) % 360.0, 3), str(torch_device))
        if key not in cache:
            cache[key] = prepare_view(model, float(view.angle_deg) % 360.0, torch_device)
        prepared.append(cache[key])
    masks, segmentations, canonical = [], [], []
    for view, prep in zip(views, prepared):
        try:
            mask, seg = segment_single_bone(np.asarray(view.image, dtype=np.float32), view.polarity, view.threshold_bias)
            canon = standardize_mask(mask, prep["image_shape"], prep["reference_mask"], match_length=len(views) > 1)
        except ValueError as exc:
            raise ValueError(f"{view.name}: {exc}") from exc
        masks.append(mask)
        segmentations.append(seg)
        canonical.append(canon)
    hypotheses = _hypotheses(views)
    targets = []
    for v, canon in enumerate(canonical):
        stack = []
        for flip, mirrors in hypotheses:
            t = np.flip(canon, axis=0) if flip else canon
            stack.append(np.flip(t, axis=1) if mirrors[v] else t)
        targets.append(np.ascontiguousarray(np.stack(stack)))
    fit = fit_views(targets, prepared, model, iterations, torch_device, prior_weight, refine_iterations)
    flip, mirrors = hypotheses[fit["choice"]]
    shape, bounds = model["shape"], model["bounds"]
    spacing = tuple((bounds[2*i+1] - bounds[2*i]) / (shape[i] - 1) for i in range(3))
    verts, faces, _, _ = marching_cubes(fit["occupancy"], level=0.5, spacing=spacing)
    verts = verts + np.asarray([bounds[0], bounds[2], bounds[4]], dtype=np.float32)[None, :]
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    view_results = []
    for v, view in enumerate(views):
        view_results.append({
            "name": view.name, "angle_deg": float(view.angle_deg), "mask": masks[v],
            "target_projection": targets[v][fit["choice"]], "projection": fit["views"][v]["projection"],
            "soft_dice": fit["views"][v]["soft_dice"], "mirrored": bool(mirrors[v]),
            "fitted_pose_log_scale_shift": fit["views"][v]["pose"], "segmentation": segmentations[v],
            "input_image_shape_pixels": list(np.asarray(view.image).shape),
        })
    dices = [r["soft_dice"] for r in view_results]
    return {
        "method": "multi-view silhouette fit to voxel PCA statistical shape model" if len(views) > 1 else "single-view silhouette fit to voxel PCA statistical shape model",
        "views": view_results, "n_views": len(views), "flipped_long_axis": bool(flip),
        "occupancy": fit["occupancy"], "vertices_mm": verts.astype(np.float32), "faces": faces.astype(np.int32),
        "projection_dice": float(np.mean(dices)), "min_view_dice": float(np.min(dices)),
        "objective": fit["objective"], "history": fit["history"], "pca_coefficients_std": fit["coefficients"],
        "device": torch_device.type, "elapsed_seconds": time.perf_counter() - start, "spacing_mm": spacing,
        "dimensions_mm": tuple(float(x) for x in np.ptp(verts, axis=0)),
        "grid_dimensions_mm": tuple(float(b - a) for a, b in zip(bounds[::2], bounds[1::2])),
        "grid_shape": shape, "bounds": tuple(float(b) for b in bounds), "n_components": int(len(model["explained_variance"])),
        "n_training_subjects": model["n_training_subjects"], "prior_weight": float(prior_weight),
        "hypotheses_tested": len(hypotheses), "refine_iterations": fit["refine_iterations"], "fit_status": "completed" if fit["history"] else "failed",
    }


def projection_panel(result: dict) -> np.ndarray:
    """Side-by-side contour comparison for every view (red = input, cyan = fit)."""
    panels = []
    for view in result["views"]:
        target = view["target_projection"]
        pred = view["projection"] >= 0.5
        rgb = np.zeros(target.shape + (3,), dtype=np.uint8)
        rgb[..., 0] = (target & ~ndi.binary_erosion(target)) * 255
        pred_edge = pred & ~ndi.binary_erosion(pred)
        rgb[..., 1] = pred_edge * 255
        rgb[..., 2] = pred_edge * 255
        panels.append(rgb)
    height = max(p.shape[0] for p in panels)
    padded = [np.pad(p, ((0, height - p.shape[0]), (0, 4), (0, 0)), constant_values=40) for p in panels]
    return np.concatenate(padded, axis=1)


def build_report(result: dict, backend_name: str = "ssm-silhouette", slicer: dict | None = None) -> dict:
    return {
        "method": result["method"],
        "backend": backend_name,
        "fit_status": result["fit_status"],
        "mean_projection_dice": result["projection_dice"],
        "min_view_projection_dice": result["min_view_dice"],
        "final_objective": result["objective"],
        "runtime_seconds": result["elapsed_seconds"],
        "device": result["device"],
        "grid_shape": result["grid_shape"],
        "spacing_mm": result["spacing_mm"],
        "dimensions_mm": result["dimensions_mm"],
        "grid_dimensions_mm": result["grid_dimensions_mm"],
        "shape_prior_weight": result["prior_weight"],
        "n_components": result["n_components"],
        "n_training_subjects": result["n_training_subjects"],
        "flipped_long_axis": result["flipped_long_axis"],
        "hypotheses_tested": result["hypotheses_tested"],
        "solid_refinement_iterations": result["refine_iterations"],
        "pca_coefficients_std": result["pca_coefficients_std"],
        "views": [{k: v for k, v in view.items() if k not in ("mask", "target_projection", "projection")} for view in result["views"]],
        "coordinate_frame": FRAME_DESCRIPTION,
        "optimization_loss_history": result["history"],
        "output_mesh": MESH_NAME,
        "output_labelmap": LABELMAP_NAME,
        "slicer": slicer,
        "warning": "Research estimate only, not validated for clinical use. View angles are user-declared, not calibrated; each view's scale and position are normalized, so absolute size comes from the shape prior. Silhouette agreement is not a measurement of patient-specific 3D accuracy. Inputs were segmented with threshold-based image processing.",
    }


def save_multiview_results(result: dict, output_dir: Path, backend_name: str = "ssm-silhouette") -> None:
    slicer = export_for_slicer(output_dir, result["vertices_mm"], result["faces"], result["occupancy"], result["bounds"])
    Image.fromarray(projection_panel(result)).save(output_dir / "projection_fit.png")
    (output_dir / "reconstruction.json").write_text(json.dumps(build_report(result, backend_name, slicer), indent=2), encoding="utf-8")


def _parse_view(spec: str) -> tuple[Path, float]:
    """PATH or PATH:ANGLE or PATH:AP / PATH:lateral."""
    path, _, angle = spec.rpartition(":") if ":" in spec else (spec, "", "AP")
    if not path:
        path, angle = spec, "AP"
    try:
        value = VIEW_PRESETS[angle] if angle in VIEW_PRESETS else float(angle)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Unknown view angle {angle!r}; use AP, lateral, or degrees") from None
    return Path(path), value


def main():
    parser = argparse.ArgumentParser(description="Estimate a 3D femur from one or more X-rays of the same bone and export an STL for 3D Slicer.")
    parser.add_argument("views", nargs="+", type=_parse_view, help="Image path with optional view, e.g. ap.png:AP lat.png:lateral oblique.dcm:45")
    parser.add_argument("--output", type=Path, default=Path("reconstruction_output"))
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--iterations", type=int, default=250, help="Stage-1 linear-projection fit iterations over all orientation hypotheses")
    parser.add_argument("--refine-iterations", type=int, default=100, help="Stage-2 refinement against the decoded 3D solid; 0 disables")
    parser.add_argument("--polarity", choices=("auto", "bright", "dark"), default="auto")
    parser.add_argument("--threshold-bias", type=float, default=0.0)
    parser.add_argument("--prior-weight", type=float, default=0.02)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--force-overwrite", action="store_true", help="Allow replacement of existing output files")
    args = parser.parse_args()
    for path, _ in args.views:
        if not path.exists():
            parser.error(f"Input image does not exist: {path}")
    if not args.model.exists():
        parser.error(f"SSM model not found: {args.model}; build it with scripts/build_ssm_v2.py")
    collisions = [str(args.output / n) for n in (MESH_NAME, LABELMAP_NAME, "reconstruction.json", "projection_fit.png") if (args.output / n).exists()]
    if collisions and not args.force_overwrite:
        parser.error("Refusing to overwrite existing output(s): " + ", ".join(collisions) + ". Choose another --output or pass --force-overwrite.")
    bias = float(np.clip(args.threshold_bias, -0.25, 0.25))
    views = [ViewInput(read_image(path), angle, path.name, args.polarity, bias) for path, angle in args.views]
    result = reconstruct_multiview(views, args.model, args.iterations, None if args.device == "auto" else args.device, args.prior_weight, refine_iterations=args.refine_iterations)
    save_multiview_results(result, args.output)
    print(json.dumps({"fit_status": result["fit_status"], "per_view_dice": {v["name"]: round(v["soft_dice"], 3) for v in result["views"]}, "elapsed_seconds": result["elapsed_seconds"], "dimensions_mm": result["dimensions_mm"]}, indent=2))
    print(f"STL: {args.output / MESH_NAME}")
    print(f"Labelmap: {args.output / LABELMAP_NAME}")


if __name__ == "__main__":
    main()
