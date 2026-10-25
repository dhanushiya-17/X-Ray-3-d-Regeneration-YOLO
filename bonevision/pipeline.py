"""Single-view femur shape-prior estimation.

Pipeline: read one AP-like radiograph -> classical threshold segmentation of a
single bone-like region -> place that silhouette in the shape model's canonical
projection frame -> fit a voxel PCA statistical shape model (SSM) to it under
four image-orientation hypotheses -> extract a millimetre surface mesh.

Hidden depth and absolute size come from the learned population prior. One
projection does not determine them, and silhouette agreement is not a measure
of 3D accuracy. Nothing here is a diagnostic or surgical-planning tool.
"""
from __future__ import annotations

import hashlib
import io
import math
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, marching_cubes, regionprops
from skimage.morphology import disk

from . import __version__

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ssm_highres.npz"
BOUNDS_DEFAULT = (-470.0, 40.0, -60.0, 100.0, -70.0, 70.0)

LOG_SCALE_LIMIT = 0.18   # in-plane nuisance scale, about +/-20 %
SHIFT_LIMIT = 0.22       # in-plane nuisance shift, normalized grid units
COEFF_LIMIT = 2.5        # PCA coefficients are clamped to +/-2.5 standard deviations

# The fit is run once per hypothesis; flips are applied to the canonical target.
# A left-right flip means the trochanter sits on the opposite image side to the
# model convention (the other leg, or a PA / mirrored image). Flipping both axes
# is a 180-degree rotation, which is not a mirror.
HYPOTHESES = (
    {"id": "as_oriented", "flip_rows": False, "flip_cols": False, "label": "As oriented",
     "meaning": "Trochanter side and head end match the model convention."},
    {"id": "rotated_180", "flip_rows": True, "flip_cols": True, "label": "Rotated 180°",
     "meaning": "Image is upside down relative to the model; same side convention."},
    {"id": "flipped_lr", "flip_rows": False, "flip_cols": True, "label": "Flipped left–right",
     "meaning": "Trochanter on the opposite image side: other leg, PA view, or a mirrored file."},
    {"id": "flipped_tb", "flip_rows": True, "flip_cols": False, "label": "Flipped top–bottom",
     "meaning": "Opposite side convention and upside down."},
)


class FitCancelled(RuntimeError):
    """Raised when a running fit is cancelled by the caller."""


# --------------------------------------------------------------------------- input

def _is_dicom(data: bytes, name: str | None) -> bool:
    if name and Path(name).suffix.lower() in {".dcm", ".dicom"}:
        return True
    return len(data) > 132 and data[128:132] == b"DICM"


def read_image_bytes(data: bytes, name: str | None = None) -> tuple[np.ndarray, dict]:
    """Decode PNG/JPEG/TIFF/DICOM bytes into normalized grayscale plus metadata.

    Nothing is written to disk. Pixel spacing is only reported for DICOM; it is
    the stored spacing and does not include radiographic magnification.
    """
    meta: dict = {"source_name": name, "format": None, "pixel_spacing_mm": None,
                  "spacing_source": None, "photometric": None,
                  "sha256": hashlib.sha256(data).hexdigest()}
    if _is_dicom(data, name):
        try:
            import pydicom
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ValueError("Reading DICOM requires pydicom (pip install pydicom)") from exc
        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
            arr = arr[0]                      # multi-frame: use the first frame
        elif arr.ndim == 3:
            arr = arr[..., :3].mean(axis=-1)
        arr = arr * float(getattr(ds, "RescaleSlope", 1.0) or 1.0) + float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
        if photometric == "MONOCHROME1":
            arr = arr.max() - arr
        for tag in ("PixelSpacing", "ImagerPixelSpacing"):
            value = getattr(ds, tag, None)
            if value is not None and len(value) >= 2 and float(value[0]) > 0 and float(value[1]) > 0:
                meta["pixel_spacing_mm"] = (float(value[0]), float(value[1]))
                meta["spacing_source"] = tag
                break
        meta.update(format="DICOM", photometric=photometric)
    else:
        try:
            im = Image.open(io.BytesIO(data))
            im.seek(0)
        except Exception as exc:
            raise ValueError("This file is not a readable PNG, JPEG, TIFF or DICOM image") from exc
        meta["format"] = im.format
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.asarray(im, dtype=np.float32)
        else:
            arr = np.asarray(im.convert("L"), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected a single 2D grayscale image")
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("Image has no finite pixel values")
    lo, hi = np.percentile(arr[finite], [0.5, 99.5])
    if hi <= lo:
        raise ValueError("Image has no usable contrast")
    out = np.clip((np.where(finite, arr, lo) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    meta["shape"] = [int(out.shape[0]), int(out.shape[1])]
    return out, meta


def read_image(path: Path) -> np.ndarray:
    """Read an image file into normalized grayscale (kept for script compatibility)."""
    path = Path(path)
    return read_image_bytes(path.read_bytes(), path.name)[0]


def apply_crop(image: np.ndarray, crop: dict | None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop by edge fractions {top,bottom,left,right} in [0, 0.45]. Returns (image, (y0,y1,x0,x1))."""
    h, w = image.shape
    crop = crop or {}
    f = {k: float(np.clip(float(crop.get(k, 0.0)), 0.0, 0.45)) for k in ("top", "bottom", "left", "right")}
    y0, y1 = int(round(h * f["top"])), int(round(h * (1 - f["bottom"])))
    x0, x1 = int(round(w * f["left"])), int(round(w * (1 - f["right"])))
    if y1 - y0 < 96 or x1 - x0 < 48:
        raise ValueError("The crop leaves too little image. Reduce the crop margins.")
    return image[y0:y1, x0:x1], (y0, y1, x0, x1)


# --------------------------------------------------------------------- segmentation

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


def _elongation(mask: np.ndarray) -> float:
    coords = np.argwhere(mask).astype(np.float64)
    if len(coords) < 3:
        return 0.0
    ev = np.linalg.eigvalsh(np.cov(coords.T))
    return float(math.sqrt(max(ev[-1], 1e-9) / max(ev[0], 1e-9)))


def segment_single_bone(image: np.ndarray, polarity: str = "auto", threshold_bias: float = 0.0) -> tuple[np.ndarray, dict]:
    """Classical Otsu threshold segmentation of one isolated, full-length bone.

    This is image processing, not a learned radiograph segmentation model. It
    can pick the wrong region on overlapping bones, implants or low contrast.
    """
    if polarity not in ("auto", "bright", "dark"):
        raise ValueError("polarity must be auto, bright or dark")
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
            ys, xs = region.coords[:, 0], region.coords[:, 1]
            border = (ys == 0) | (xs == 0) | (ys == image.shape[0] - 1) | (xs == image.shape[1] - 1)
            score -= 2.0 * float(np.mean(border))
            candidates.append((score, comp, pol, area_ratio, bool(border.any())))
    if not candidates:
        raise ValueError("Could not isolate a bone-like region. Try the other contrast polarity, adjust the threshold, or crop to one femur.")
    candidates.sort(key=lambda c: c[0], reverse=True)
    score, mask, chosen, area, touches_border = candidates[0]
    rows, cols = np.where(mask)
    if max(np.ptp(rows), np.ptp(cols)) + 1 < 32:
        raise ValueError("The detected region is too small. Use a full-femur image with better resolution.")
    margin = float(score - candidates[1][0]) if len(candidates) > 1 else None
    return mask.astype(bool), {
        "method": "Otsu threshold + morphology + elongation/area component score",
        "polarity": chosen, "otsu_threshold": otsu, "threshold_used": cut,
        "component_area_fraction": float(area), "component_score": float(score),
        "runner_up_margin": margin, "n_candidates": len(candidates),
        "touches_border": touches_border, "elongation": _elongation(mask),
        "bbox_rows_cols": [int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())],
    }


def input_quality_warnings(image: np.ndarray, seg: dict | None) -> list[dict]:
    """Checks that can be made before fitting. Levels: info, warn, error."""
    out = []
    h, w = image.shape
    if h < 384 or w < 128:
        out.append({"level": "warn", "code": "low_resolution",
                    "message": f"Image is {w}×{h} px, which is low for a full-length femur."})
    if float(image.std()) < 0.08:
        out.append({"level": "warn", "code": "low_contrast", "message": "Low overall contrast; the threshold may follow soft tissue or background."})
    if seg is None:
        return out
    if seg["touches_border"]:
        out.append({"level": "warn", "code": "touches_border",
                    "message": "The selected region touches the image edge, so the femur may be cut off or merged with background."})
    if seg["elongation"] < 3.0:
        out.append({"level": "warn", "code": "not_elongated",
                    "message": f"Selected region is not long and thin (elongation {seg['elongation']:.1f}); it may not be a full femur."})
    if seg["component_area_fraction"] > 0.35:
        out.append({"level": "warn", "code": "large_region", "message": "Selected region covers a large part of the image; it may include background."})
    if seg["runner_up_margin"] is not None and seg["runner_up_margin"] < 0.3:
        out.append({"level": "info", "code": "ambiguous_region",
                    "message": "Another region scored almost as well. Check that the highlighted mask is the femur."})
    return out


# ------------------------------------------------------------------ standardization

def _principal_rotation(coords: np.ndarray) -> np.ndarray:
    """Rotation matrix (row, col) that maps the major axis onto the row axis."""
    cov = np.cov(coords.T)
    _, vecs = np.linalg.eigh(cov)
    v = vecs[:, -1]
    if v[0] < 0:
        v = -v
    vr, vc = float(v[0]), float(v[1])
    return np.array([[vr, vc], [-vc, vr]], dtype=np.float64)


def standardize_mask_with_transform(mask: np.ndarray, output_shape: tuple[int, int],
                                    reference_mask: np.ndarray | None = None,
                                    spacing_rc: tuple[float, float] = (1.0, 1.0),
                                    mm_per_pixel: float | None = None) -> tuple[np.ndarray, dict]:
    """Place one isolated femur silhouette in the SSM's canonical projection frame.

    The long axis is aligned with grid rows by a rotation (never a reflection).
    Scaling is physical: grid rows and columns can have different millimetre
    spacing, so pixels are first converted to millimetres and then to grid
    indices. Without calibration (`mm_per_pixel=None`) the silhouette is sized
    to the reference (mean) projection, so absolute length is assumed from the
    population prior, not measured.

    Returns the canonical boolean target and the affine map
    q = A @ p + b from input pixel (row, col) to canonical grid (row, col).
    """
    coords = np.argwhere(mask).astype(np.float64)
    if len(coords) < 100:
        raise ValueError("The detected bone mask is too small to fit the shape model.")
    out_h, out_w = output_shape
    sp0, sp1 = float(spacing_rc[0]), float(spacing_rc[1])
    centroid = coords.mean(axis=0)
    R = _principal_rotation(coords)
    rotated = (coords - centroid) @ R.T
    h_px = float(np.ptp(rotated[:, 0]) + 1.0)
    w_px = float(np.ptp(rotated[:, 1]) + 1.0)
    if reference_mask is not None and reference_mask.any():
        ref = np.argwhere(reference_mask).astype(np.float64)
        ref_centroid = ref.mean(axis=0)
        ref_h_mm = (float(np.ptp(ref[:, 0])) + 1.0) * sp0
        ref_w_mm = (float(np.ptp(ref[:, 1])) + 1.0) * sp1
    else:
        ref_centroid = np.array([(out_h - 1) / 2.0, (out_w - 1) / 2.0])
        ref_h_mm, ref_w_mm = out_h * sp0, out_w * sp1
    calibrated = mm_per_pixel is not None and mm_per_pixel > 0
    k = float(mm_per_pixel) if calibrated else min(ref_h_mm / h_px, ref_w_mm / w_px)
    A = k * np.diag([1.0 / sp0, 1.0 / sp1]) @ R
    b = ref_centroid - A @ centroid
    A_inv = np.linalg.inv(A)
    canvas = ndi.affine_transform(mask.astype(np.float32), A_inv, offset=-A_inv @ b,
                                  output_shape=(out_h, out_w), order=1, mode="constant", cval=0.0) >= 0.5
    kept = float(canvas.sum() * abs(np.linalg.det(A_inv)) / max(1.0, float(mask.sum())))
    return canvas, {
        "A": A.tolist(), "b": b.tolist(), "mm_per_pixel": k, "calibrated": bool(calibrated),
        "rotation_deg": float(math.degrees(math.atan2(R[0, 1], R[0, 0]))),
        "length_mm_assumed": float(h_px * k), "width_mm_assumed": float(w_px * k),
        "fraction_inside_grid": min(1.0, kept),
    }


def standardize_mask(mask: np.ndarray, output_shape: tuple[int, int], reference_mask: np.ndarray | None = None,
                     spacing_rc: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    return standardize_mask_with_transform(mask, output_shape, reference_mask, spacing_rc)[0]


def hypothesis_targets(target: np.ndarray) -> np.ndarray:
    out = []
    for hyp in HYPOTHESES:
        t = target
        if hyp["flip_rows"]:
            t = t[::-1, :]
        if hyp["flip_cols"]:
            t = t[:, ::-1]
        out.append(np.ascontiguousarray(t))
    return np.stack(out)


# ------------------------------------------------------------------------- model

class ShapeModel:
    """A loaded voxel PCA SSM with projection tensors prepared on one device."""

    def __init__(self, path: Path, device: str | None = None):
        import torch
        path = Path(path)
        data = np.load(path)
        self.path = path
        self.grid_shape = tuple(int(v) for v in data["grid_shape"])
        self.bounds = tuple(float(v) for v in data["bounds"]) if "bounds" in data else BOUNDS_DEFAULT
        self.spacing = tuple((self.bounds[2 * i + 1] - self.bounds[2 * i]) / (self.grid_shape[i] - 1) for i in range(3))
        self.origin = np.asarray([self.bounds[0], self.bounds[2], self.bounds[4]], dtype=np.float32)
        self.train_subjects = [str(x) for x in data["train_subjects"]] if "train_subjects" in data else []
        self.synthetic_demo = bool(data["synthetic_demo"]) if "synthetic_demo" in data else False
        mean = data["mean_shape"].astype(np.float32, copy=False)
        comps = data["components"].astype(np.float32, copy=False)
        variance = data["explained_variance"].astype(np.float32, copy=False)
        self.n_components = int(comps.shape[0])
        if "explained_variance_ratio" in data:
            self.explained_ratio = [float(x) for x in data["explained_variance_ratio"]]
        else:
            self.explained_ratio = (variance / max(float(variance.sum()), 1e-12)).tolist()
        self.mean_volume = mean.reshape(self.grid_shape)
        self.reference_mask = self.mean_volume.max(axis=2) > 0.35
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        comp_3d = comps.reshape((self.n_components,) + self.grid_shape)
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=self.device)
        self.mean_2d = t(self.mean_volume.sum(axis=2).reshape(-1))
        self.comp_2d = t(comp_3d.sum(axis=3).reshape(self.n_components, -1))
        self.std = t(np.sqrt(variance))
        self.mean = t(mean)
        self.components = t(comps)
        self.lock = threading.Lock()
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        self.sha256 = h.hexdigest()

    def summary(self) -> dict:
        return {
            "file": self.path.name, "sha256": self.sha256, "n_components": self.n_components,
            "n_training_subjects": len(self.train_subjects) or None, "grid_shape": list(self.grid_shape),
            "bounds_mm": list(self.bounds), "spacing_mm": [round(s, 4) for s in self.spacing],
            "device": self.device.type, "explained_variance_ratio": self.explained_ratio,
            "kind": "voxel-occupancy PCA statistical shape model",
            "synthetic_demo": self.synthetic_demo,
        }

    # --- differentiable projection ---------------------------------------
    def project(self, z, pose):
        """Soft AP silhouette for standardized coefficients z (B,n) and pose (B,3)."""
        import torch
        import torch.nn.functional as F
        H, W = self.grid_shape[0], self.grid_shape[1]
        B = z.shape[0]
        coeff = z.clamp(-COEFF_LIMIT, COEFF_LIMIT) * self.std[None, :]
        line = (self.mean_2d[None, :] + coeff @ self.comp_2d).reshape(B, H, W).clamp_min(0.0)
        pred = (1.0 - torch.exp(-line)).clamp(1e-5, 1.0 - 1e-5)
        log_scale = pose[:, 0].clamp(-LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)
        shift = pose[:, 1:].clamp(-SHIFT_LIMIT, SHIFT_LIMIT)
        inv = torch.exp(-log_scale)
        theta = torch.zeros((B, 2, 3), device=z.device, dtype=z.dtype)
        theta[:, 0, 0] = inv
        theta[:, 1, 1] = inv
        theta[:, 0, 2] = -shift[:, 1]
        theta[:, 1, 2] = -shift[:, 0]
        grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
        return F.grid_sample(pred[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]

    def decode(self, z) -> np.ndarray:
        """Occupancy volume (unposed, canonical grid) for one coefficient vector."""
        import torch
        with torch.no_grad():
            zt = torch.as_tensor(np.asarray(z, dtype=np.float32), device=self.device).clamp(-COEFF_LIMIT, COEFF_LIMIT)
            occ = (self.mean + (zt * self.std) @ self.components).clamp(0.0, 1.0)
            return occ.reshape(self.grid_shape).cpu().numpy()


_MODEL_CACHE: dict = {}
_MODEL_CACHE_LOCK = threading.Lock()


def load_model(path: Path = MODEL_PATH, device: str | None = None) -> ShapeModel:
    """Load (once) and cache a shape model keyed by file, modification time and device."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Shape model not found: {path}. Build it with scripts/build_ssm_v2.py.")
    key = (str(path), path.stat().st_mtime_ns, device or "auto")
    with _MODEL_CACHE_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE.clear()
            _MODEL_CACHE[key] = ShapeModel(path, device)
        return _MODEL_CACHE[key]


# --------------------------------------------------------------------------- fit

def fit_hypotheses(model: ShapeModel, targets: np.ndarray, iterations: int, prior_weight: float = 0.02,
                   progress=None, cancel=None) -> dict:
    """Fit the SSM to every target hypothesis in one batch and keep each one's best state."""
    import torch
    import torch.nn.functional as F
    device = model.device
    B = len(targets)
    target_t = torch.as_tensor(targets.astype(np.float32), device=device)
    target_t = F.avg_pool2d(target_t[:, None], kernel_size=3, stride=1, padding=1)[:, 0]
    params = torch.nn.Parameter(torch.zeros((B, model.n_components), device=device))
    pose = torch.nn.Parameter(torch.zeros((B, 3), device=device))
    opt = torch.optim.Adam(({"params": params, "lr": 0.04}, {"params": pose, "lr": 0.015}))
    best_losses = torch.full((B,), float("inf"), device=device)
    best_params = torch.zeros_like(params)
    best_pose = torch.zeros_like(pose)
    history = []
    for it in range(iterations):
        if cancel is not None and cancel():
            raise FitCancelled("Fit cancelled")
        opt.zero_grad(set_to_none=True)
        pred = model.project(params, pose)
        intersection = (pred * target_t).sum(dim=(1, 2))
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (pred.sum(dim=(1, 2)) + target_t.sum(dim=(1, 2)) + 1.0)
        bce = F.binary_cross_entropy(pred, target_t, reduction="none").mean(dim=(1, 2))
        losses = dice_loss + 0.08 * bce + float(prior_weight) * params.square().mean(dim=1) + 0.04 * pose.square().mean(dim=1)
        losses.mean().backward()
        opt.step()
        with torch.no_grad():
            improved = losses < best_losses
            best_losses = torch.where(improved, losses, best_losses)
            best_params = torch.where(improved[:, None], params.detach().clamp(-COEFF_LIMIT, COEFF_LIMIT), best_params)
            best_pose = torch.where(improved[:, None], pose.detach(), best_pose)
            history.append(losses.detach().clone())
        if progress is not None and (it % 5 == 0 or it == iterations - 1):
            progress(it + 1, iterations, float(best_losses.min()))
    with torch.no_grad():
        final_pred = model.project(best_params, best_pose)
        inter = (final_pred * target_t).sum(dim=(1, 2))
        soft_dice = (2.0 * inter + 1.0) / (final_pred.sum(dim=(1, 2)) + target_t.sum(dim=(1, 2)) + 1.0)
    choice = int(torch.argmin(best_losses).item())
    hist = torch.stack(history).cpu().numpy()
    result = {
        "choice": choice,
        "losses": best_losses.cpu().numpy().astype(float).tolist(),
        "soft_dice": soft_dice.cpu().numpy().astype(float).tolist(),
        "params": best_params[choice].cpu().numpy().astype(np.float32),
        "pose": best_pose[choice].cpu().numpy().astype(np.float32),
        "projection": final_pred[choice].cpu().numpy().astype(np.float32),
        "blurred_target": target_t[choice].cpu().numpy().astype(np.float32),
        "history": hist[:, choice].astype(float).tolist(),
        "history_all": hist.T.astype(float).tolist(),
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def clamp_pose(pose) -> tuple[float, float, float]:
    p = np.asarray(pose, dtype=np.float64)
    return (float(np.clip(p[0], -LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)),
            float(np.clip(p[1], -SHIFT_LIMIT, SHIFT_LIMIT)), float(np.clip(p[2], -SHIFT_LIMIT, SHIFT_LIMIT)))


def pose_index_points(points: np.ndarray, pose, grid_shape) -> np.ndarray:
    """Apply the fitted in-plane pose to grid-index points (N,3) so the mesh matches the shown fit.

    Rows/cols follow the same normalized affine as the 2D fit; depth is scaled
    about the grid centre with the same factor, so the size change is isotropic.
    """
    log_s, sh0, sh1 = clamp_pose(pose)
    s = math.exp(log_s)
    H, W, D = grid_shape
    out = points.astype(np.float64).copy()
    yn = (2.0 * out[:, 0] + 1.0) / H - 1.0
    xn = (2.0 * out[:, 1] + 1.0) / W - 1.0
    out[:, 0] = ((s * (yn + sh0) + 1.0) * H - 1.0) / 2.0
    out[:, 1] = ((s * (xn + sh1) + 1.0) * W - 1.0) / 2.0
    c = (D - 1) / 2.0
    out[:, 2] = s * (out[:, 2] - c) + c
    return out


def pose_volume(volume: np.ndarray, pose, order: int = 1) -> np.ndarray:
    """Resample a canonical volume with the fitted pose (used for 3D evaluation)."""
    shape = volume.shape
    idx = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    fwd = pose_index_points(idx, pose, shape)
    a = fwd[1] - fwd[0]          # forward map is diagonal-affine: out = a*in + c
    c = fwd[0]
    matrix = 1.0 / a
    offset = -c / a
    return ndi.affine_transform(volume.astype(np.float32), matrix, offset=offset, output_shape=shape, order=order, mode="constant")


def silhouette_consistent_spread(model: ShapeModel, z_best: np.ndarray, pose: np.ndarray, blurred_target: np.ndarray,
                                 base_dice: float, tolerance: float = 0.02, max_keep: int = 24,
                                 n_candidates: int = 192, seed: int = 0) -> dict:
    """Heuristic per-voxel spread among shapes whose silhouettes fit about as well as the estimate.

    Coefficients near the optimum are perturbed and kept when their projected
    soft Dice stays within `tolerance` of the estimate. The voxel standard
    deviation of the kept shapes shows where the one view leaves geometry
    loosely constrained. It is not a calibrated confidence interval.
    """
    import torch
    rng = np.random.default_rng(seed)
    device = model.device
    target = torch.as_tensor(blurred_target, device=device)[None]
    kept, used_sigma = [], None
    with torch.no_grad():
        for sigma in (0.6, 0.35, 0.2):
            cand = z_best[None, :] + sigma * rng.standard_normal((n_candidates, len(z_best))).astype(np.float32)
            zt = torch.as_tensor(cand, device=device)
            pt = torch.as_tensor(np.repeat(pose[None, :], n_candidates, axis=0), device=device)
            dice_all = []
            for s in range(0, n_candidates, 64):
                pred = model.project(zt[s:s + 64], pt[s:s + 64])
                inter = (pred * target).sum(dim=(1, 2))
                d = (2.0 * inter + 1.0) / (pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1.0)
                dice_all.append(d.cpu().numpy())
            dice_all = np.concatenate(dice_all)
            ok = np.where(dice_all >= base_dice - tolerance)[0]
            kept.extend(cand[i] for i in ok[: max_keep - len(kept)])
            used_sigma = sigma
            if len(kept) >= max_keep:
                break
        if len(kept) < 3:
            return {"available": False, "n_samples": len(kept), "reason": "Too few silhouette-consistent samples."}
        acc_sum = np.zeros(model.grid_shape, dtype=np.float64)
        acc_sq = np.zeros(model.grid_shape, dtype=np.float64)
        for z in kept:
            occ = model.decode(z)
            acc_sum += occ
            acc_sq += occ.astype(np.float64) ** 2
        n = len(kept)
        var = np.clip(acc_sq / n - (acc_sum / n) ** 2, 0.0, None)
    return {"available": True, "std_volume": np.sqrt(var).astype(np.float32), "n_samples": n,
            "sigma": used_sigma, "tolerance": tolerance}


# ----------------------------------------------------------------------- top level

def _surface(occ: np.ndarray, model: ShapeModel, pose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if float(occ.max()) <= 0.5:
        raise ValueError("The fitted occupancy is empty; the silhouette fit failed.")
    verts_idx, faces, _, _ = marching_cubes(occ, level=0.5)
    posed = pose_index_points(verts_idx, pose, model.grid_shape)
    verts_mm = posed * np.asarray(model.spacing)[None, :] + model.origin[None, :]
    return verts_idx, verts_mm.astype(np.float32), faces.astype(np.int32)


def mean_shape_mesh(model: ShapeModel) -> tuple[np.ndarray, np.ndarray]:
    _, verts, faces = _surface(model.mean_volume, model, (0.0, 0.0, 0.0))
    return verts, faces


def reconstruct(image: np.ndarray, model: ShapeModel | Path | None = None, iterations: int = 250,
                polarity: str = "auto", threshold_bias: float = 0.0, device: str | None = None,
                prior_weight: float = 0.02, mm_per_pixel: float | None = None, spread: bool = True,
                progress=None, cancel=None) -> dict:
    """Run segmentation, standardization, the 4-hypothesis SSM fit and surface extraction."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Expected one 2D grayscale X-ray image")
    if not np.isfinite(image).all():
        raise ValueError("X-ray contains NaN or infinite pixel values")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if not np.isfinite(prior_weight) or prior_weight < 0:
        raise ValueError("prior_weight must be a finite, nonnegative number")
    if not isinstance(model, ShapeModel):
        model = load_model(model or MODEL_PATH, device)
    stage = (lambda name: progress(0, iterations, None, stage=name)) if progress else (lambda name: None)
    timings = {}
    start = time.perf_counter()

    stage("segmenting")
    mask, segmentation = segment_single_bone(image, polarity, threshold_bias)
    warnings = input_quality_warnings(image, segmentation)
    target, transform = standardize_mask_with_transform(
        mask, model.grid_shape[:2], model.reference_mask, model.spacing[:2], mm_per_pixel)
    if transform["fraction_inside_grid"] < 0.97:
        warnings.append({"level": "warn", "code": "outside_grid",
                         "message": f"Only {100 * transform['fraction_inside_grid']:.0f}% of the silhouette fits inside the model grid; check calibration or crop."})
    targets = hypothesis_targets(target)
    timings["segment_s"] = time.perf_counter() - start

    stage("fitting")
    t0 = time.perf_counter()
    with model.lock:
        fit_progress = (lambda i, n, loss: progress(i, n, loss, stage="fitting")) if progress else None
        fit = fit_hypotheses(model, targets, iterations, prior_weight, progress=fit_progress, cancel=cancel)
        timings["fit_s"] = time.perf_counter() - t0
        choice = fit["choice"]
        occ = model.decode(fit["params"])
        stage("surface")
        verts_idx, verts_mm, faces = _surface(occ, model, fit["pose"])
        spread_info = {"available": False, "reason": "Disabled"}
        vertex_spread = None
        if spread:
            if cancel is not None and cancel():
                raise FitCancelled("Fit cancelled")
            stage("spread")
            t1 = time.perf_counter()
            spread_info = silhouette_consistent_spread(model, fit["params"], fit["pose"], fit["blurred_target"], fit["soft_dice"][choice])
            if spread_info.get("available"):
                vertex_spread = ndi.map_coordinates(spread_info["std_volume"], verts_idx.T, order=1, mode="nearest").astype(np.float32)
            timings["spread_s"] = time.perf_counter() - t1

    # ---- diagnostics -------------------------------------------------------
    losses = np.asarray(fit["losses"])
    order = np.argsort(losses)
    margin = float(losses[order[1]] - losses[order[0]])
    dice = float(fit["soft_dice"][choice])
    pose = clamp_pose(fit["pose"])
    z = fit["params"].astype(float)
    if choice != 0:
        warnings.append({"level": "info", "code": "orientation",
                         "message": f"Best fit used the '{HYPOTHESES[choice]['label']}' hypothesis: {HYPOTHESES[choice]['meaning']}"})
    if margin < 0.01:
        warnings.append({"level": "warn", "code": "orientation_ambiguous",
                         "message": f"Orientation is ambiguous: '{HYPOTHESES[order[1]]['label']}' fit almost as well."})
    if dice < 0.70:
        warnings.append({"level": "error", "code": "fit_failed", "message": f"Silhouette agreement is very low ({dice:.2f}). Treat this estimate as a failed fit."})
    elif dice < 0.85:
        warnings.append({"level": "warn", "code": "weak_fit", "message": f"Silhouette agreement is weak ({dice:.2f}). Check the mask before trusting the shape."})
    if abs(pose[0]) >= LOG_SCALE_LIMIT * 0.95 or max(abs(pose[1]), abs(pose[2])) >= SHIFT_LIMIT * 0.95:
        warnings.append({"level": "warn", "code": "pose_at_limit", "message": "The fitted scale or shift reached its allowed limit; the silhouette may be mis-sized or off-centre."})
    if np.any(np.abs(z) >= COEFF_LIMIT * 0.99):
        warnings.append({"level": "warn", "code": "shape_at_limit", "message": "At least one shape mode hit its ±2.5σ limit; the outline may be outside the training population."})

    spread_summary = {k: v for k, v in spread_info.items() if k != "std_volume"}
    if vertex_spread is not None:
        long_axis = verts_mm[:, 0]
        lo, hi = float(long_axis.min()), float(long_axis.max())
        thirds = np.digitize(long_axis, [lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3])
        names = {2: "proximal", 1: "shaft", 0: "distal"}
        spread_summary.update({
            "mean": float(vertex_spread.mean()), "p90": float(np.percentile(vertex_spread, 90)),
            "by_region": {names[i]: float(vertex_spread[thirds == i].mean()) for i in (2, 1, 0) if np.any(thirds == i)},
        })

    elapsed = time.perf_counter() - start
    return {
        "pipeline_version": __version__,
        "mask": mask, "segmentation": segmentation, "transform": transform,
        "target_projection": targets[choice], "projection": fit["projection"], "occupancy": occ,
        "hypothesis": {**HYPOTHESES[choice], "index": choice},
        "hypotheses": [{**h, "loss": float(fit["losses"][i]), "soft_dice": float(fit["soft_dice"][i])} for i, h in enumerate(HYPOTHESES)],
        "orientation_margin": margin,
        "fitted_pose_log_scale_shift": pose, "pose_applied_to_mesh": True,
        "coefficients_z": z.tolist(),
        "coefficient_norm": float(np.linalg.norm(z)), "coefficient_norm_expected": float(math.sqrt(len(z))),
        "vertices_mm": verts_mm, "faces": faces, "vertex_spread": vertex_spread, "spread": spread_summary,
        "projection_dice": dice, "objective": float(fit["losses"][choice]),
        "history": fit["history"], "history_all": fit["history_all"],
        "device": model.device.type, "elapsed_seconds": elapsed, "timings": timings,
        "spacing_mm": tuple(model.spacing), "dimensions_mm": tuple(float(x) for x in np.ptp(verts_mm, axis=0)),
        "grid_dimensions_mm": tuple(float(b - a) for a, b in zip(model.bounds[::2], model.bounds[1::2])),
        "grid_shape": model.grid_shape, "n_components": model.n_components,
        "n_training_subjects": len(model.train_subjects) or None,
        "model": model.summary() | {"explained_variance_ratio": None},
        "prior_weight": float(prior_weight), "iterations": int(iterations),
        "settings": {"polarity": polarity, "threshold_bias": float(threshold_bias), "iterations": int(iterations),
                     "prior_weight": float(prior_weight), "mm_per_pixel": mm_per_pixel, "spread": bool(spread)},
        "warnings": warnings,
        "fit_status": "completed",
    }


def overlay_on_input(result: dict, image: np.ndarray) -> np.ndarray:
    """RGB uint8: input X-ray with the selected mask (warm) and the fitted silhouette outline (ice blue)."""
    A = np.asarray(result["transform"]["A"])
    b = np.asarray(result["transform"]["b"])
    H, W = result["projection"].shape
    hyp = result["hypothesis"]
    F = np.diag([-1.0 if hyp["flip_rows"] else 1.0, -1.0 if hyp["flip_cols"] else 1.0])
    f0 = np.array([H - 1.0 if hyp["flip_rows"] else 0.0, W - 1.0 if hyp["flip_cols"] else 0.0])
    T = F @ A
    t = F @ b + f0
    fitted = ndi.affine_transform(result["projection"], T, offset=t, output_shape=image.shape, order=1, mode="constant") >= 0.5
    mask = result["mask"]
    rgb = np.repeat(np.clip(image, 0, 1)[..., None], 3, axis=2) * 0.85
    rgb[mask] = rgb[mask] * 0.55 + np.array([0.93, 0.62, 0.35]) * 0.45
    thick = max(1, int(round(max(image.shape) / 500)))
    edge = fitted & ~ndi.binary_erosion(fitted, iterations=thick)
    rgb[edge] = np.array([0.55, 0.85, 1.0])
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
