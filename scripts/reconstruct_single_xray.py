"""Single-view femur shape-prior reconstruction.

This is a research estimator for a single AP-like radiograph. It fits the
projected silhouette of a CT-trained voxel statistical shape model, then
exports the inferred surface in model-space millimetres. The hidden depth is
provided by the shape prior and is not uniquely determined by one X-ray.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, marching_cubes, regionprops
from skimage.morphology import disk

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ssm_highres.npz"
BOUNDS_DEFAULT = (-470.0, 40.0, -60.0, 100.0, -70.0, 70.0)


def read_image(path: Path) -> np.ndarray:
    """Read PNG/JPEG/TIFF or DICOM into normalized grayscale, preserving contrast."""
    if path.suffix.lower() in {".dcm", ".dicom"}:
        import pydicom
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
            arr = arr.max() - arr
    else:
        im = ImageOps.grayscale(Image.open(path))
        arr = np.asarray(im, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("Image has no finite pixel values")
    lo, hi = np.percentile(arr[finite], [0.5, 99.5])
    if hi <= lo:
        raise ValueError("Image has no usable contrast")
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _component_score(component: np.ndarray) -> float:
    ys, xs = np.where(component)
    area = len(xs)
    if not area:
        return -1e9
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    elongation = max(h, w) / max(1, min(h, w))
    fill = area / max(1, h * w)
    return math.log1p(elongation) + 0.35 * fill - 0.4 * abs(math.log(max(area / component.size, 1e-6) / 0.12))


def segment_single_bone(image: np.ndarray, polarity: str = "auto", threshold_bias: float = 0.0) -> tuple[np.ndarray, dict]:
    """Conservative threshold segmentation for an isolated, full-bone radiograph.

    The UI exposes polarity and threshold adjustment. This is classical image
    processing, not a learned X-ray segmentation model.
    """
    smooth = ndi.gaussian_filter(image.astype(np.float32), 1.0)
    otsu = float(threshold_otsu(smooth))
    cut = float(np.clip(otsu + threshold_bias, 0.02, 0.98))
    polarities = ("bright", "dark") if polarity == "auto" else (polarity,)
    candidates = []
    min_area = max(64, int(image.size * 0.0005))
    for pol in polarities:
        raw = smooth >= cut if pol == "bright" else smooth <= cut
        raw = ndi.binary_closing(raw, structure=disk(2))
        raw = ndi.binary_fill_holes(raw)
        labels_small, _ = ndi.label(raw)
        sizes = np.bincount(labels_small.ravel())
        keep = sizes >= min_area
        keep[0] = False
        raw = keep[labels_small]
        labels = label(raw)
        for region in regionprops(labels):
            area_ratio = region.area / image.size
            if area_ratio < 0.0005 or area_ratio > 0.65:
                continue
            comp = labels == region.label
            score = _component_score(comp)
            # Down-rank components touching the image edge; these are often background.
            ys, xs = region.coords[:, 0], region.coords[:, 1]
            border_fraction = np.mean((ys == 0) | (xs == 0) | (ys == image.shape[0]-1) | (xs == image.shape[1]-1))
            score -= 2.0 * float(border_fraction)
            candidates.append((score, comp, pol, area_ratio))
    if not candidates:
        raise ValueError("Could not isolate a bone-like component. Try changing threshold/polarity or crop to one femur.")
    score, mask, chosen, area = max(candidates, key=lambda x: x[0])
    props = regionprops(mask.astype(np.uint8))
    if not props or max(props[0].bbox[2] - props[0].bbox[0], props[0].bbox[3] - props[0].bbox[1]) < 32:
        raise ValueError("Detected region is too small. Use a full-femur image with better resolution.")
    return mask.astype(bool), {"polarity": chosen, "otsu_threshold": otsu, "threshold_used": cut, "component_area_fraction": area, "component_score": score}


def standardize_mask(mask: np.ndarray, output_shape: tuple[int, int], reference_mask: np.ndarray | None = None) -> np.ndarray:
    """Orient one isolated femur and place it in the SSM's canonical AP frame.

    Absolute scale and translation are absent from ordinary PNG/JPEG images,
    so length and center are normalized to the mean SSM projection. Relative
    silhouette proportions are preserved as much as the projection bounds allow.
    """
    coords = np.argwhere(mask)
    if len(coords) < 100:
        raise ValueError("The detected bone mask is too small to fit the shape model.")
    props0 = regionprops(label(mask.astype(np.uint8)))
    orientation = math.degrees(float(props0[0].orientation)) if props0 else 0.0
    from skimage.transform import rotate
    alternatives = []
    for angle in (-orientation, orientation):
        rotated_try = rotate(mask.astype(np.float32), angle=angle, resize=True, order=0, preserve_range=True) > 0.5
        props = regionprops(label(rotated_try))
        alignment_error = abs(float(props[0].orientation)) if props else float("inf")
        alternatives.append((alignment_error, rotated_try))
    rotated = min(alternatives, key=lambda item: item[0])[1]
    rr, cc = np.where(rotated)
    cropped = rotated[rr.min():rr.max()+1, cc.min():cc.max()+1]
    if cropped.shape[0] < cropped.shape[1]:
        cropped = cropped.T
    out_h, out_w = output_shape
    if reference_mask is not None:
        ry, rx = np.where(reference_mask)
        if len(ry):
            ref_h = int(ry.max() - ry.min() + 1)
            ref_w = int(rx.max() - rx.min() + 1)
            center_y, center_x = float(np.mean(ry)), float(np.mean(rx))
        else:
            ref_h, ref_w = out_h, out_w
            center_y, center_x = (out_h - 1) / 2, (out_w - 1) / 2
    else:
        ref_h, ref_w = out_h, out_w
        center_y, center_x = (out_h - 1) / 2, (out_w - 1) / 2
    scale = min(ref_h / cropped.shape[0], ref_w / cropped.shape[1])
    new_h = max(1, min(out_h, int(round(cropped.shape[0] * scale))))
    new_w = max(1, min(out_w, int(round(cropped.shape[1] * scale))))
    resized = np.asarray(Image.fromarray(cropped.astype(np.uint8) * 255).resize((new_w, new_h), Image.Resampling.NEAREST)) > 0
    canvas = np.zeros(output_shape, dtype=bool)
    y0 = int(round(center_y - (new_h - 1) / 2))
    x0 = int(round(center_x - (new_w - 1) / 2))
    sy0, sx0 = max(0, -y0), max(0, -x0)
    dy0, dx0 = max(0, y0), max(0, x0)
    hcopy = min(new_h - sy0, out_h - dy0)
    wcopy = min(new_w - sx0, out_w - dx0)
    if hcopy > 0 and wcopy > 0:
        canvas[dy0:dy0+hcopy, dx0:dx0+wcopy] = resized[sy0:sy0+hcopy, sx0:sx0+wcopy]
    return canvas

def prepare_ssm_fit(data, device_name: str | None = None):
    """Preload and project the SSM once; reusable for held-out batches."""
    import torch
    shape = tuple(int(v) for v in data["grid_shape"])
    bounds = tuple(float(v) for v in data["bounds"]) if "bounds" in data else BOUNDS_DEFAULT
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    n_comp = int(data["components"].shape[0])
    mean_3d = data["mean_shape"].astype(np.float32, copy=False).reshape(shape)
    comp_3d = data["components"].astype(np.float32, copy=False).reshape((n_comp,) + shape)
    return {
        "shape": shape, "bounds": bounds, "device": device,
        "mean_2d": torch.as_tensor(mean_3d.sum(axis=2).reshape(-1), dtype=torch.float32, device=device),
        "comp_2d": torch.as_tensor(comp_3d.sum(axis=3).reshape(n_comp, -1), dtype=torch.float32, device=device),
        "std": torch.as_tensor(np.sqrt(data["explained_variance"]), dtype=torch.float32, device=device),
        "mean": torch.as_tensor(data["mean_shape"], dtype=torch.float32, device=device),
        "components": torch.as_tensor(data["components"], dtype=torch.float32, device=device),
    }

def _fit_once(targets: np.ndarray, data, iterations: int, device_name: str | None = None, prepared=None, prior_weight: float = 0.02):
    """Fit both long-axis orientations in parallel using a projected SSM."""
    import torch
    import torch.nn.functional as F
    prepared = prepared or prepare_ssm_fit(data, device_name)
    shape, bounds, device = prepared["shape"], prepared["bounds"], prepared["device"]
    n_comp = int(prepared["components"].shape[0])
    mean_2d, comp_2d, std = prepared["mean_2d"], prepared["comp_2d"], prepared["std"]
    target_t = torch.as_tensor(targets.astype(np.float32), device=device)
    target_t = F.avg_pool2d(target_t[:, None], kernel_size=3, stride=1, padding=1)[:, 0]
    params = torch.nn.Parameter(torch.zeros((len(targets), n_comp), device=device))
    pose = torch.nn.Parameter(torch.zeros((len(targets), 3), device=device))
    opt = torch.optim.Adam(({"params": params, "lr": 0.04}, {"params": pose, "lr": 0.015}))
    best_losses = torch.full((len(targets),), float("inf"), device=device)
    best_params = torch.zeros_like(params)
    best_pose = torch.zeros_like(pose)
    loss_history = []
    for _ in range(iterations):
        opt.zero_grad(set_to_none=True)
        coeff = params.clamp(-2.5, 2.5) * std[None, :]
        line_integral = (mean_2d[None, :] + coeff @ comp_2d).reshape(len(targets), shape[0], shape[1]).clamp_min(0.0)
        pred = (1.0 - torch.exp(-line_integral)).clamp(1e-5, 1.0 - 1e-5)
        log_scale = pose[:, 0].clamp(-0.18, 0.18)
        shift = pose[:, 1:].clamp(-0.22, 0.22)
        theta = torch.zeros((len(targets), 2, 3), device=device)
        inv_scale = torch.exp(-log_scale)
        theta[:, 0, 0] = inv_scale
        theta[:, 1, 1] = inv_scale
        theta[:, 0, 2] = -shift[:, 1]
        theta[:, 1, 2] = -shift[:, 0]
        grid = F.affine_grid(theta, pred[:, None].shape, align_corners=False)
        pred = F.grid_sample(pred[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
        intersection = (pred * target_t).sum(dim=(1, 2))
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (pred.sum(dim=(1, 2)) + target_t.sum(dim=(1, 2)) + 1.0)
        bce = F.binary_cross_entropy(pred, target_t, reduction="none").mean(dim=(1, 2))
        prior = params.square().mean(dim=1)
        pose_prior = 0.04 * (pose.square().mean(dim=1))
        losses = dice_loss + 0.08 * bce + float(prior_weight) * prior + pose_prior
        losses.mean().backward()
        opt.step()
        with torch.no_grad():
            improved = losses < best_losses
            best_losses = torch.where(improved, losses, best_losses)
            best_params = torch.where(improved[:, None], params.detach().clamp(-2.5, 2.5), best_params)
            best_pose = torch.where(improved[:, None], pose.detach(), best_pose)
            loss_history.append(losses.detach().clone())
    choice = int(torch.argmin(best_losses).item())
    # Decode one selected solution at full resolution only once.
    mean = prepared["mean"]
    components = prepared["components"]
    with torch.no_grad():
        coeff = best_params[choice] * std
        occ = (mean + coeff @ components).clamp(0.0, 1.0).reshape(shape)
        projection = 1.0 - torch.exp(-occ.sum(dim=2))
        occ_np = occ.cpu().numpy()
        selected_pose = best_pose[choice]
        scale = torch.exp(selected_pose[0].clamp(-0.18, 0.18))
        shift = selected_pose[1:].clamp(-0.22, 0.22)
        theta = torch.zeros((1, 2, 3), device=device)
        theta[0, 0, 0] = torch.exp(-selected_pose[0].clamp(-0.18, 0.18))
        theta[0, 1, 1] = theta[0, 0, 0]
        theta[0, 0, 2] = -shift[1]
        theta[0, 1, 2] = -shift[0]
        grid = F.affine_grid(theta, projection[None, None].shape, align_corners=False)
        fit_projection = F.grid_sample(projection[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]
        proj_np = fit_projection.cpu().numpy()
        selected_target = target_t[choice]
        soft_dice = float((2.0 * (fit_projection * selected_target).sum() + 1.0) / (fit_projection.sum() + selected_target.sum() + 1.0))
        pose_np = selected_pose.detach().cpu().numpy()
    loss = float(best_losses[choice].detach().cpu())
    history = torch.stack(loss_history)[:, choice].detach().cpu().numpy().astype(float).tolist()
    del mean, components, comp_2d, mean_2d, std, params, pose, best_params, best_pose, best_losses, occ, projection, fit_projection
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return occ_np, proj_np, soft_dice, loss, history, device.type, bounds, choice, pose_np

def _write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    tri = vertices[faces].astype(np.float32)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    records = np.empty(len(faces), dtype=np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]))
    records["normal"] = normals
    records["vertices"] = tri
    records["attr"] = 0
    with path.open("wb") as f:
        f.write(b"BoneVision single-view SSM estimate".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(faces)))
        records.tofile(f)


def reconstruct(image: np.ndarray, model_path: Path = MODEL_PATH, iterations: int = 250, polarity: str = "auto", threshold_bias: float = 0.0, device: str | None = None, prior_weight: float = 0.02) -> dict:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Expected one 2D grayscale X-ray image")
    if not np.isfinite(image).all():
        raise ValueError("X-ray contains NaN or infinite pixel values")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if not np.isfinite(prior_weight) or prior_weight < 0:
        raise ValueError("prior_weight must be a finite, nonnegative number")
    start = time.perf_counter()
    mask, segmentation = segment_single_bone(image, polarity, threshold_bias)
    data = np.load(model_path)
    grid_shape = tuple(int(x) for x in data["grid_shape"])
    mean_projection = data["mean_shape"].reshape(grid_shape).max(axis=2)
    reference_mask = mean_projection > 0.35
    target0 = standardize_mask(mask, grid_shape[:2], reference_mask)
    targets = np.stack((target0, np.flip(target0, axis=0)))
    occ, projection, dice, loss, history, backend, bounds, choice, fitted_pose = _fit_once(
        targets, data, iterations, device, prior_weight=prior_weight
    )
    flipped_long_axis = bool(choice)
    target = targets[choice]
    spacing = tuple((bounds[2*i+1] - bounds[2*i]) / (grid_shape[i] - 1) for i in range(3))
    verts, faces, _, _ = marching_cubes(occ, level=0.5, spacing=spacing)
    origin = np.asarray([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
    verts = verts + origin[None, :]
    elapsed = time.perf_counter() - start
    return {
        "mask": mask, "target_projection": target, "projection": projection, "occupancy": occ,
        "flipped_long_axis": flipped_long_axis,
        "fitted_pose_log_scale_shift": tuple(float(x) for x in fitted_pose),
        "vertices_mm": verts.astype(np.float32), "faces": faces.astype(np.int32),
        "segmentation": segmentation, "projection_dice": dice,
        "objective": loss, "history": history, "device": backend,
        "elapsed_seconds": elapsed, "spacing_mm": spacing,
        "dimensions_mm": tuple(float(x) for x in np.ptp(verts, axis=0)),
        "grid_dimensions_mm": tuple(float(b-a) for a,b in zip(bounds[::2], bounds[1::2])),
        "grid_shape": grid_shape, "n_components": int(len(data["explained_variance"])),
        "n_training_subjects": int(len(data["train_subjects"])) if "train_subjects" in data else None,
        "prior_weight": float(prior_weight),
        "fit_status": "completed" if len(history) else "failed",
    }


def save_results(result: dict, output_dir: Path, input_image: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_binary_stl(output_dir / "femur_reconstruction.stl", result["vertices_mm"], result["faces"])
    from PIL import ImageDraw
    target = result["target_projection"]
    pred = result["projection"]
    # Red = target contour, cyan = fitted SSM silhouette.
    target_edge = target & ~ndi.binary_erosion(target)
    pred_edge = (pred >= 0.5) & ~ndi.binary_erosion(pred >= 0.5)
    h, w = target.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = target_edge.astype(np.uint8) * 255
    rgb[..., 1] = pred_edge.astype(np.uint8) * 255
    rgb[..., 2] = pred_edge.astype(np.uint8) * 255
    Image.fromarray(rgb).save(output_dir / "projection_fit.png")
    report = {
        "method": "single-view AP silhouette fit to voxel PCA statistical shape model",
        "fit_status": result["fit_status"],
        "projection_dice": result["projection_dice"],
        "final_objective": result["objective"],
        "runtime_seconds": result["elapsed_seconds"],
        "device": result["device"],
        "grid_shape": result["grid_shape"],
        "spacing_mm": result["spacing_mm"],
        "dimensions_mm": result["dimensions_mm"],
        "grid_dimensions_mm": result["grid_dimensions_mm"],
        "segmentation": result["segmentation"],
        "fitted_pose_log_scale_shift": result["fitted_pose_log_scale_shift"],
        "shape_prior_weight": result["prior_weight"],
        "n_training_subjects": result["n_training_subjects"],
        "coordinate_frame": "canonical SSM anatomical frame in millimetres; STL does not encode patient RAS/LPS orientation",
        "flipped_long_axis": result["flipped_long_axis"],
        "optimization_loss_history": result["history"],
        "input_image_shape_pixels": list(input_image.shape),
        "output_mesh": "femur_reconstruction.stl",
        "warning": "Research estimate only. Single-view depth is underdetermined; projection agreement is not a measurement of patient-specific 3D accuracy. Input was segmented with threshold-based image processing.",
    }
    (output_dir / "reconstruction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Estimate a 3D femur from one AP-like X-ray and export an STL for 3D Slicer.")
    parser.add_argument("image", type=Path, help="Input PNG/JPG/TIFF or DICOM X-ray")
    parser.add_argument("--output", type=Path, default=Path("reconstruction_output"))
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--polarity", choices=("auto", "bright", "dark"), default="auto")
    parser.add_argument("--threshold-bias", type=float, default=0.0, help="Offset from Otsu threshold in normalized intensity, range -0.25 to 0.25")
    parser.add_argument("--prior-weight", type=float, default=0.02, help="PCA shape-prior penalty (default 0.02; other values are not clinically validated)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--force-overwrite", action="store_true", help="Allow replacement of existing output files")
    args = parser.parse_args()
    if not args.image.exists():
        parser.error(f"Input image does not exist: {args.image}")
    if not args.model.exists():
        parser.error(f"SSM model not found: {args.model}; build it with scripts/build_ssm_v2.py")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if not np.isfinite(args.prior_weight) or args.prior_weight < 0:
        parser.error("--prior-weight must be finite and nonnegative")
    output_files = [args.output / name for name in ("femur_reconstruction.stl", "reconstruction.json", "projection_fit.png")]
    collisions = [str(path) for path in output_files if path.exists()]
    if collisions and not args.force_overwrite:
        parser.error("Refusing to overwrite existing output(s): " + ", ".join(collisions) + ". Choose another --output or pass --force-overwrite.")
    bias = float(np.clip(args.threshold_bias, -0.25, 0.25))
    image = read_image(args.image)
    result = reconstruct(image, args.model, args.iterations, args.polarity, bias, None if args.device == "auto" else args.device, args.prior_weight)
    save_results(result, args.output, image)
    print(json.dumps({k: result[k] for k in ("fit_status", "projection_dice", "elapsed_seconds", "device", "spacing_mm", "dimensions_mm")}, indent=2))
    print(f"STL: {args.output / 'femur_reconstruction.stl'}")


if __name__ == "__main__":
    main()
