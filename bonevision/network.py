"""Learned AP (+ optional lateral) femur network for the web app.

The network itself lives in scripts/ (multiview_net.py, trained by
train_multiview_net.py through run_retraining.py); this module adapts
scripts/multiview_net_backend.py to the server's case flow. Checkpoints:
models/multiview_ap_lat_d2.pt and models/multiview_ap_d2.pt, or the files named
by BONEVISION_NET_AP_LAT / BONEVISION_NET_AP. A retrained network replaces the
current one by swapping those files; nothing here changes.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import __version__
from .pipeline import ROOT, FitCancelled

sys.path.insert(0, str(ROOT / "scripts"))

from multiview_net_backend import MultiViewNetBackend  # noqa: E402
from reconstruct_multiview_xray import ViewInput  # noqa: E402

VIEW_ANGLES = {"ap": 0.0, "lat": 90.0}
VIEW_LABELS = {"ap": "AP", "lat": "Lateral"}

# Held-out test results of the checkpoints shipped with this branch
# (16 HFValid subjects, synthetic DRRs, models/retraining_report.md).
REPORTED_TEST = {
    "source": "models/retraining_report.md from the 2026-10-25 training run (not re-computed on this machine)",
    "protocol": "16 held-out HFValid test subjects; synthetic parallel-beam DRRs placed on the training grid; 4 mm voxels.",
    "columns": ["Metric", "AP + lateral network", "AP-only network", "Old SSM fit (AP)"],
    "rows": [
        ["Mean voxel Dice", "0.943", "0.865", "0.687"],
        ["Mean symmetric surface distance", "0.95 mm", "2.19 mm", "4.73 mm"],
        ["Mean HD95", "3.45 mm", "5.56 mm", "n/a"],
        ["Worst subject Dice", "0.917 (Pat036)", "0.760 (Pat082)", "0.373 (Pat099)"],
    ],
    "caveat": "Inputs were simulated X-rays already aligned with the training grid at the true femur length. Real "
              "radiographs add segmentation errors, soft tissue, cone-beam magnification and unknown scale, so expect "
              "lower accuracy on uploads. Re-run scripts/run_retraining.py --from evaluate to refresh.",
}

_lock = threading.Lock()
_backend: MultiViewNetBackend | None = None
_val_cache: dict = {}


def backend() -> MultiViewNetBackend:
    global _backend
    with _lock:
        if _backend is None:
            _backend = MultiViewNetBackend()
        return _backend


def _checkpoint_validation(path: Path) -> dict | None:
    """Training-time validation metrics stored in a checkpoint (cached by mtime)."""
    key = (str(path), path.stat().st_mtime)
    if key not in _val_cache:
        import torch
        state = torch.load(path, map_location="cpu", weights_only=False)
        val = state.get("val") or {}
        _val_cache[key] = {"epoch": state.get("epoch"), "synthetic_demo": bool(state.get("synthetic_demo", False)),
                           **{k: float(v) for k, v in val.items() if isinstance(v, (int, float))}}
    return _val_cache[key]


def status() -> dict:
    b = backend()
    checkpoints = []
    for views, path in b.checkpoints.items():
        path = Path(path)
        item = {"views": list(views), "file": path.name, "available": path.exists()}
        if path.exists():
            try:
                item["validation"] = _checkpoint_validation(path)
            except Exception as exc:  # unreadable or incompatible file
                item["available"] = False
                item["error"] = f"{type(exc).__name__}: {exc}"
        checkpoints.append(item)
    usable = any(c["available"] for c in checkpoints)
    return {"available": usable,
            "synthetic_demo": any(c.get("validation", {}).get("synthetic_demo") for c in checkpoints if c["available"]), "problem": None if usable else b.availability() or "No readable network checkpoint",
            "checkpoints": checkpoints,
            "lateral_supported": any(c["available"] and "lat" in c["views"] for c in checkpoints)}


def reconstruct_views(views: list[dict], device: str | None = None, progress=None, cancel=None) -> dict:
    """views: [{"image", "kind" ("ap"/"lat"), "name", "polarity", "threshold_bias"}], images already cropped."""
    stage = (lambda name: progress(0, 0, None, stage=name)) if progress else (lambda name: None)
    if cancel is not None and cancel():
        raise FitCancelled("Cancelled")
    inputs = [ViewInput(image=np.asarray(v["image"], dtype=np.float32), angle_deg=VIEW_ANGLES[v["kind"]],
                        name=v.get("name") or VIEW_LABELS[v["kind"]], polarity=v.get("polarity", "auto"),
                        threshold_bias=float(v.get("threshold_bias", 0.0)), mirror=v.get("mirror", "auto"))
              for v in views]
    stage("network")
    result = backend().reconstruct(inputs, device=device)
    if cancel is not None and cancel():
        raise FitCancelled("Cancelled")
    stage("surface")
    for view, request in zip(result["views"], views):
        view["kind"] = request["kind"]
        view["input_image"] = np.asarray(request["image"], dtype=np.float32)
    result["warnings"] = [{"level": "warn", "code": "network_check", "message": w} for w in result["warnings"]]
    if result["ignored_views"]:
        result["warnings"].append({"level": "info", "code": "view_ignored",
                                   "message": "No AP + lateral checkpoint was found, so the lateral view was not used: "
                                              + ", ".join(result["ignored_views"])})
    result["warnings"].append({"level": "info", "code": "assumed_size",
                               "message": "Femur length is set to the training mean, and the output is shown as a left femur."})
    result["pipeline_version"] = __version__
    result["method_label"] = result.pop("method")
    return result


def overlay_view(view: dict, image: np.ndarray) -> np.ndarray:
    """RGB uint8: one input view with its mask (warm) and the network's silhouette mapped back (ice blue)."""
    M, t = (np.asarray(a, dtype=np.float64) for a in view["grid_to_image"])
    M_inv = np.linalg.inv(M)
    fitted = ndi.affine_transform(view["projection"].astype(np.float32), M_inv, offset=-M_inv @ t,
                                  output_shape=image.shape, order=1, mode="constant") >= 0.5
    mask = view["mask"]
    rgb = np.repeat(np.clip(image, 0, 1)[..., None], 3, axis=2) * 0.85
    rgb[mask] = rgb[mask] * 0.55 + np.array([0.93, 0.62, 0.35]) * 0.45
    thick = max(1, int(round(max(image.shape) / 500)))
    edge = fitted & ~ndi.binary_erosion(fitted, iterations=thick)
    rgb[edge] = np.array([0.55, 0.85, 1.0])
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
