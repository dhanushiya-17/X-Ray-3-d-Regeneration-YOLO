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


MAX_VIEWS_PER_KIND = 3


def _view_input(v: dict) -> ViewInput:
    return ViewInput(image=np.asarray(v["image"], dtype=np.float32), angle_deg=VIEW_ANGLES[v["kind"]],
                     name=v.get("name") or VIEW_LABELS[v["kind"]], polarity=v.get("polarity", "auto"),
                     threshold_bias=float(v.get("threshold_bias", 0.0)), mirror=v.get("mirror", "auto"))


def _dice(a, b) -> float:
    a, b = np.asarray(a) >= 0.5, np.asarray(b) >= 0.5
    return float((2.0 * (a & b).sum() + 1.0) / (a.sum() + b.sum() + 1.0))


def reconstruct_views(views: list[dict], device: str | None = None, progress=None, cancel=None) -> dict:
    """views: [{"image", "kind" ("ap"/"lat"), "name", "polarity", "threshold_bias"}], images already cropped.

    The network takes one AP and at most one lateral view. With several of
    either, it runs once for every AP x lateral pair and the occupancies are
    averaged; each view is then scored against the combined estimate, and a
    view that agrees clearly less than the others is flagged.
    """
    def stage(name, i=0, n=0):
        if progress:
            progress(i, n, None, stage=name)

    def check_cancel():
        if cancel is not None and cancel():
            raise FitCancelled("Cancelled")

    check_cancel()
    for i, v in enumerate(views):
        v.setdefault("key", f"{v['kind']}{1 + sum(w['kind'] == v['kind'] for w in views[:i])}")
    aps = [v for v in views if v["kind"] == "ap"]
    lats = [v for v in views if v["kind"] == "lat"]
    ignored = []
    if lats and not status()["lateral_supported"]:
        ignored, lats = [v["name"] for v in lats], []
    pairs = [(a, l) for a in aps for l in lats] or [(a, None) for a in aps]
    runs = []
    for i, (a, lat) in enumerate(pairs):
        stage("network", i + 1, len(pairs))
        runs.append(backend().reconstruct([_view_input(a)] + ([_view_input(lat)] if lat else []), device=device))
        check_cancel()
    stage("surface")

    result = runs[0]
    if len(runs) > 1:
        from multiview_net_backend import surface_from_occupancy
        mean_occ = np.mean([r["occupancy"] for r in runs], axis=0)
        occ, verts, faces = surface_from_occupancy(mean_occ)
        result = dict(runs[0], occupancy=occ, vertices_mm=verts, faces=faces,
                      dimensions_mm=tuple(float(x) for x in np.ptp(verts, axis=0)),
                      elapsed_seconds=float(sum(r["elapsed_seconds"] for r in runs)),
                      hypotheses_tested=int(sum(r["hypotheses_tested"] for r in runs)))
        result["run_agreement"] = [_dice(r["occupancy"], occ) for r in runs]

    # One entry per used view, scored against the final (possibly combined) occupancy.
    occ = result["occupancy"]
    out_views = []
    for v in aps + lats:
        pair_index = next(i for i, (a, lat) in enumerate(pairs) if v is a or v is lat)
        run_view = runs[pair_index]["views"][0 if v["kind"] == "ap" else 1]
        proj = occ.max(axis=2) if v["kind"] == "ap" else occ.max(axis=1)
        out_views.append(dict(run_view, projection=proj, soft_dice=_dice(proj, run_view["target_projection"]),
                              kind=v["kind"], key=v["key"], name=v["name"],
                              input_image=np.asarray(v["image"], dtype=np.float32)))
    dices = [v["soft_dice"] for v in out_views]
    result.update(views=out_views, n_views=len(out_views), projection_dice=float(np.mean(dices)),
                  min_view_dice=float(np.min(dices)), n_network_runs=len(runs),
                  ignored_views=ignored + [n for r in runs for n in r["ignored_views"]])

    warnings = []
    for w in dict.fromkeys(w for r in runs for w in r["warnings"]):
        warnings.append({"level": "warn", "code": "network_check", "message": w})
    if len(out_views) > 2:
        median = float(np.median(dices))
        for v in out_views:
            if v["soft_dice"] < median - 0.05:
                warnings.append({"level": "warn", "code": "view_disagrees",
                                 "message": f"{v['name']} agrees clearly less with the combined estimate (Dice {v['soft_dice']:.2f} "
                                            f"against a median of {median:.2f}). It may show a different femur, a different "
                                            "angle, or a wrong mask; try removing it."})
    if result["ignored_views"]:
        warnings.append({"level": "info", "code": "view_ignored",
                         "message": "No AP + lateral checkpoint was found, so these views were not used: "
                                    + ", ".join(result["ignored_views"])})
    if len(runs) > 1:
        warnings.append({"level": "info", "code": "combined",
                         "message": f"Combined {len(runs)} network runs ({len(aps)} AP x {max(1, len(lats))} lateral). "
                                    "Extra views of the same direction average out segmentation and placement noise; "
                                    "they do not add new angles."})
    warnings.append({"level": "info", "code": "assumed_size",
                     "message": "Femur length is set to the training mean, and the output is shown as a left femur."})
    result["warnings"] = warnings
    result["pipeline_version"] = __version__
    label = result.pop("method")
    result["method_label"] = label if len(runs) == 1 else f"{label}, mean of {len(runs)} runs"
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
