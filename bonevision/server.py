"""BoneVision local web backend (FastAPI).

Uploads, cases and exports live in process memory only: medical images are not
written to disk, and everything is gone when the server stops. Bind to
localhost (the default in serve.py) unless you have a reason not to.
"""
from __future__ import annotations

import base64
import csv
import io
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import __version__
from . import network
from .exports import build_file, bundle_zip, files_for
from .pipeline import (
    HYPOTHESES, MODEL_PATH, ROOT, FitCancelled, apply_crop, input_quality_warnings, load_model,
    mean_shape_mesh, read_image_bytes, reconstruct, segment_single_bone,
)

MAX_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_UPLOADS = 6
MAX_CASES = 6
WEB_DIR = ROOT / "web"

app = FastAPI(title="BoneVision", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bonevision-fit")
_lock = threading.Lock()
_uploads: "OrderedDict[str, dict]" = OrderedDict()
_cases: "OrderedDict[str, dict]" = OrderedDict()
_mean_mesh_cache: dict = {}


def model_path() -> Path:
    return Path(os.environ.get("BONEVISION_MODEL", str(MODEL_PATH)))


def model_device() -> str | None:
    value = os.environ.get("BONEVISION_DEVICE", "auto")
    return None if value == "auto" else value


def _model():
    try:
        return load_model(model_path(), model_device())
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


# ----------------------------------------------------------------------- helpers

def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _data_url(rgb: np.ndarray, max_side: int = 1100) -> str:
    im = Image.fromarray(rgb)
    scale = max_side / max(im.size)
    if scale < 1:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _mesh_payload(vertices: np.ndarray, faces: np.ndarray, spread=None) -> dict:
    return {"vertex_count": int(len(vertices)), "face_count": int(len(faces)),
            "positions": _b64(vertices.astype("<f4")), "indices": _b64(faces.astype("<u4")),
            "spread": _b64(spread.astype("<f4")) if spread is not None else None,
            "bounds_min": vertices.min(axis=0).tolist(), "bounds_max": vertices.max(axis=0).tolist()}


def _trim(store: OrderedDict, limit: int):
    while len(store) > limit:
        _, old = store.popitem(last=False)
        if "cancel" in old:
            old["cancel"].set()


def _get(store: OrderedDict, key: str, what: str) -> dict:
    with _lock:
        item = store.get(key)
    if item is None:
        raise HTTPException(404, f"{what} not found (the server keeps only the most recent {what.lower()}s in memory)")
    return item


# ------------------------------------------------------------------------ status

@app.get("/api/status")
def status():
    import torch
    path = model_path()
    info = {"version": __version__, "model_available": path.exists(), "model_file": path.name,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "storage": "memory-only; uploads and results are discarded when the server stops",
            "hypotheses": [{k: h[k] for k in ("id", "label", "meaning")} for h in HYPOTHESES]}
    if path.exists():
        try:
            info["model"] = _model().summary()
        except Exception as exc:  # corrupt or incompatible model file
            info["model_available"] = False
            info["model_error"] = f"{type(exc).__name__}: {exc}"
    net = network.status()
    info["methods"] = {
        "network": {"label": "Learned network", "views": "AP + optional lateral", **net},
        "ssm": {"label": "Shape-model fit", "views": "one AP", "available": info["model_available"],
                "problem": None if info["model_available"] else info.get("model_error") or f"No shape model at {path.name}"},
    }
    info["default_method"] = "network" if net["available"] else "ssm"
    return info


@app.get("/api/model/mean-mesh")
def mean_mesh():
    model = _model()
    key = model.sha256
    if key not in _mean_mesh_cache:
        verts, faces = mean_shape_mesh(model)
        _mean_mesh_cache.clear()
        _mean_mesh_cache[key] = _mesh_payload(verts, faces)
    return _mean_mesh_cache[key]


# ------------------------------------------------------------------------ uploads

@app.post("/api/uploads")
async def create_upload(file: UploadFile = File(...)):
    chunks, size = [], 0
    while True:
        chunk = await file.read(1 << 20)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File is larger than 80 MB")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(400, "The uploaded file is empty")
    try:
        image, meta = read_image_bytes(data, file.filename)
    except Exception as exc:
        raise HTTPException(422, f"Could not read this image: {exc}") from exc
    del data
    uid = uuid.uuid4().hex[:12]
    with _lock:
        _uploads[uid] = {"id": uid, "image": image, "meta": meta, "created": time.time()}
        _trim(_uploads, MAX_UPLOADS)
    rgb = (np.repeat(image[..., None], 3, axis=2) * 255).astype(np.uint8)
    return {"id": uid, "meta": _jsonable(meta), "preview": _data_url(rgb),
            "warnings": input_quality_warnings(image, None)}


class SegmentRequest(BaseModel):
    crop: dict = Field(default_factory=dict)
    polarity: str = "auto"
    threshold_bias: float = Field(0.0, ge=-0.25, le=0.25)


@app.post("/api/uploads/{uid}/segment")
def segment(uid: str, req: SegmentRequest):
    up = _get(_uploads, uid, "Upload")
    image = up["image"]
    try:
        cropped, box = apply_crop(image, req.crop)
        mask, seg = segment_single_bone(cropped, req.polarity, req.threshold_bias)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    y0, y1, x0, x1 = box
    rgb = np.repeat(image[..., None], 3, axis=2) * 0.35
    region = np.repeat(cropped[..., None], 3, axis=2) * 0.9
    region[mask] = region[mask] * 0.45 + np.array([0.93, 0.62, 0.35]) * 0.55
    rgb[y0:y1, x0:x1] = region
    return {"overlay": _data_url((np.clip(rgb, 0, 1) * 255).astype(np.uint8)), "crop_box": list(box),
            "segmentation": _jsonable(seg), "warnings": input_quality_warnings(cropped, seg),
            "mask_pixels": int(mask.sum())}


@app.delete("/api/uploads/{uid}")
def delete_upload(uid: str):
    with _lock:
        _uploads.pop(uid, None)
    return {"deleted": uid}


# -------------------------------------------------------------------------- cases

class ViewRequest(BaseModel):
    upload_id: str
    kind: str = "ap"
    crop: dict = Field(default_factory=dict)
    polarity: str = "auto"
    threshold_bias: float = Field(0.0, ge=-0.25, le=0.25)


class CaseRequest(BaseModel):
    method: str = "ssm"
    upload_id: str | None = None
    views: list[ViewRequest] = Field(default_factory=list, max_length=2)
    crop: dict = Field(default_factory=dict)
    polarity: str = "auto"
    threshold_bias: float = Field(0.0, ge=-0.25, le=0.25)
    iterations: int = Field(250, ge=10, le=1000)
    prior_weight: float = Field(0.02, ge=0.0, le=1.0)
    use_dicom_spacing: bool = False
    magnification: float = Field(1.0, gt=0.5, lt=2.0)
    spread: bool = True


def _network_summary(case: dict, r: dict) -> dict:
    files = files_for(r)
    return _jsonable({
        "method": "network", "method_label": r["method_label"], "score": r["projection_dice"],
        "silhouette_soft_dice": r["projection_dice"], "min_view_dice": r["min_view_dice"],
        "views": [{k: v[k] for k in ("kind", "view_type", "name", "soft_dice", "mirrored", "flipped_long_axis",
                                     "orientation_match", "orientation_margin", "assumed_mm_per_pixel", "segmentation")}
                  | {"crop_box": inp["crop_box"], "image_shape": list(inp["image"].shape), "input": {k: inp["meta"].get(k) for k in ("source_name", "format", "shape")}}
                  for v, inp in zip(r["views"], case["views_input"])],
        "ignored_views": r["ignored_views"], "checkpoint": Path(r["checkpoint"]).name, "model_views": r["model_views"],
        "checkpoint_validation": r["checkpoint_validation"], "hypotheses_tested": r["hypotheses_tested"],
        "network_downsample": r["network_downsample"], "canonical_side": r["canonical_side"],
        "warnings": r["warnings"], "elapsed_seconds": r["elapsed_seconds"], "device": r["device"],
        "dimensions_mm": r["dimensions_mm"], "spacing_mm": r["spacing_mm"], "grid_shape": r["grid_shape"],
        "mesh": {"vertices": len(r["vertices_mm"]), "faces": len(r["faces"])},
        "report_file": "reconstruction.json",
        "files": [{"name": n, "type": t, "description": d, "bytes": len(case["files"][n])} for n, (t, d) in files.items() if n in case["files"]],
    })


def _summary(case: dict) -> dict:
    out = {k: case[k] for k in ("id", "status", "stage", "iteration", "total", "loss", "error", "created", "started", "finished", "name")}
    out["method"] = case["request"]["method"]
    out["settings"] = case["request"]
    out["live_history"] = case["live_history"][-400:]
    r = case.get("result")
    if r is not None and out["method"] == "network":
        out["result"] = _network_summary(case, r)
    elif r is not None:
        files = files_for(r)
        out["result"] = _jsonable({
            "method": "ssm", "score": r["projection_dice"], "report_file": "report.json",
            "silhouette_soft_dice": r["projection_dice"], "objective": r["objective"],
            "hypothesis": {k: r["hypothesis"][k] for k in ("id", "label", "meaning", "index")},
            "hypotheses": [{k: h[k] for k in ("id", "label", "meaning", "loss", "soft_dice")} for h in r["hypotheses"]],
            "orientation_margin": r["orientation_margin"], "pose_log_scale_shift": r["fitted_pose_log_scale_shift"],
            "coefficients_z": r["coefficients_z"], "coefficient_norm": r["coefficient_norm"],
            "coefficient_norm_expected": r["coefficient_norm_expected"],
            "explained_variance_ratio": case["explained_ratio"],
            "spread": r["spread"], "segmentation": r["segmentation"],
            "transform": {k: r["transform"][k] for k in ("mm_per_pixel", "calibrated", "rotation_deg", "length_mm_assumed", "width_mm_assumed", "fraction_inside_grid")},
            "warnings": r["warnings"], "timings": r["timings"], "elapsed_seconds": r["elapsed_seconds"],
            "device": r["device"], "dimensions_mm": r["dimensions_mm"], "spacing_mm": r["spacing_mm"],
            "mesh": {"vertices": len(r["vertices_mm"]), "faces": len(r["faces"])},
            "history": r["history"], "history_all": r["history_all"], "model": r["model"],
            "input": {k: case["meta"].get(k) for k in ("source_name", "format", "sha256", "shape", "pixel_spacing_mm", "spacing_source")},
            "crop_box": case["crop_box"], "image_shape": list(case["image"].shape),
            "files": [{"name": n, "type": files[n][0], "description": files[n][1], "bytes": len(case["files"][n])} for n in files if n in case["files"]],
        })
    return out


def _run_case(cid: str):
    with _lock:
        case = _cases.get(cid)
    if case is None:
        return
    case.update(status="running", started=time.time(), stage="loading model")

    def progress(i, n, loss, stage=None):
        case["stage"] = stage or case["stage"]
        if i:
            case["iteration"], case["total"] = i, n
        if loss is not None:
            case["loss"] = loss
            case["live_history"].append([i, loss])

    try:
        req = case["request"]
        if req["method"] == "network":
            case["stage"] = "segmenting"
            result = network.reconstruct_views(case["views_input"], model_device(), progress=progress, cancel=case["cancel"].is_set)
        else:
            model = load_model(model_path(), model_device())
            case["explained_ratio"] = model.explained_ratio
            result = reconstruct(case["image"], model, req["iterations"], req["polarity"], req["threshold_bias"],
                                 prior_weight=req["prior_weight"], mm_per_pixel=case["mm_per_pixel"], spread=req["spread"],
                                 progress=progress, cancel=case["cancel"].is_set)
        case["stage"] = "writing exports"
        case["files"] = {n: build_file(n, result, case["image"], case["meta"], case["crop_box"]) for n in files_for(result)}
        case["result"] = result
        case.update(status="done", stage="done", finished=time.time())
    except FitCancelled:
        case.update(status="cancelled", stage="cancelled", finished=time.time())
    except Exception as exc:
        case.update(status="failed", stage="failed", error=str(exc), finished=time.time())


def _network_case(req: CaseRequest) -> dict:
    net = network.status()
    if not net["available"]:
        raise HTTPException(503, net["problem"])
    kinds = [v.kind for v in req.views]
    if any(k not in network.VIEW_ANGLES for k in kinds):
        raise HTTPException(422, "Each view must be 'ap' or 'lat'")
    if kinds.count("ap") != 1 or kinds.count("lat") > 1:
        raise HTTPException(422, "The network needs exactly one AP view and at most one lateral view")
    views = []
    for v in sorted(req.views, key=lambda v: v.kind != "ap"):   # AP first
        if v.polarity not in ("auto", "bright", "dark"):
            raise HTTPException(422, "polarity must be auto, bright or dark")
        up = _get(_uploads, v.upload_id, "Upload")
        try:
            image, box = apply_crop(up["image"], v.crop)
        except ValueError as exc:
            raise HTTPException(422, f"{network.VIEW_LABELS[v.kind]} view: {exc}") from exc
        views.append({"image": image, "kind": v.kind, "name": f"{network.VIEW_LABELS[v.kind]} view",
                      "polarity": v.polarity, "threshold_bias": v.threshold_bias, "crop_box": box, "meta": up["meta"]})
    ap = views[0]
    meta = dict(ap["meta"]) | {"views": [{"kind": v["kind"], "crop_box_rows_cols": list(v["crop_box"])}
                                         | {k: v["meta"].get(k) for k in ("source_name", "format", "sha256", "shape")} for v in views]}
    name = " + ".join(v["meta"].get("source_name") or v["kind"] for v in views)
    return {"name": name, "image": ap["image"], "meta": meta, "crop_box": ap["crop_box"], "mm_per_pixel": None,
            "views_input": views, "total": 0}


@app.post("/api/cases")
def create_case(req: CaseRequest):
    if req.method not in ("ssm", "network"):
        raise HTTPException(422, "method must be ssm or network")
    if req.method == "network":
        fields = _network_case(req)
    else:
        fields = _ssm_case(req)
    cid = uuid.uuid4().hex[:12]
    case = {"id": cid, "request": req.model_dump(), "status": "queued", "stage": "queued", "iteration": 0,
            "loss": None, "error": None, "created": time.time(), "started": None, "finished": None, "live_history": [],
            "cancel": threading.Event(), "files": {}, "result": None, "explained_ratio": None, **fields}
    with _lock:
        _cases[cid] = case
        _trim(_cases, MAX_CASES)
    _executor.submit(_run_case, cid)
    return {"id": cid}


def _ssm_case(req: CaseRequest) -> dict:
    if not req.upload_id:
        raise HTTPException(422, "upload_id is required for the shape-model fit")
    up = _get(_uploads, req.upload_id, "Upload")
    if not model_path().exists():
        raise HTTPException(503, f"No shape model at {model_path().name}. Build it with scripts/build_ssm_v2.py, or run serve.py --demo.")
    if req.polarity not in ("auto", "bright", "dark"):
        raise HTTPException(422, "polarity must be auto, bright or dark")
    try:
        image, box = apply_crop(up["image"], req.crop)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    mm_per_pixel = None
    if req.use_dicom_spacing:
        spacing = up["meta"].get("pixel_spacing_mm")
        if not spacing:
            raise HTTPException(422, "This image has no DICOM pixel spacing to use")
        if abs(spacing[0] - spacing[1]) > 1e-3:
            raise HTTPException(422, "Anisotropic DICOM pixel spacing is not supported yet")
        mm_per_pixel = spacing[0] / req.magnification
    return {"name": up["meta"].get("source_name") or "upload", "image": image, "meta": up["meta"], "crop_box": box,
            "mm_per_pixel": mm_per_pixel, "total": req.iterations}


@app.get("/api/cases")
def list_cases():
    with _lock:
        items = list(_cases.values())
    return [{k: c[k] for k in ("id", "name", "status", "created", "finished")} | {"method": c["request"]["method"]} | (
        {"silhouette_soft_dice": c["result"]["projection_dice"]} if c.get("result") else {}) for c in reversed(items)]


@app.get("/api/cases/{cid}")
def get_case(cid: str):
    return _summary(_get(_cases, cid, "Case"))


@app.post("/api/cases/{cid}/cancel")
def cancel_case(cid: str):
    case = _get(_cases, cid, "Case")
    case["cancel"].set()
    if case["status"] == "queued":
        case.update(status="cancelled", stage="cancelled")
    return {"id": cid, "status": case["status"]}


@app.delete("/api/cases/{cid}")
def delete_case(cid: str):
    with _lock:
        case = _cases.pop(cid, None)
    if case:
        case["cancel"].set()
    return {"deleted": cid}


def _done(cid: str) -> dict:
    case = _get(_cases, cid, "Case")
    if case["status"] != "done":
        raise HTTPException(409, f"Case is {case['status']}")
    return case


@app.get("/api/cases/{cid}/mesh")
def case_mesh(cid: str):
    r = _done(cid)["result"]
    return JSONResponse(_mesh_payload(r["vertices_mm"], r["faces"], r.get("vertex_spread")))


@app.get("/api/cases/{cid}/files/{name}")
def case_file(cid: str, name: str, inline: bool = False):
    case = _done(cid)
    files = files_for(case["result"])
    if name not in files:
        raise HTTPException(404, "Unknown export")
    data = case["files"].get(name) or build_file(name, case["result"], case["image"], case["meta"], case["crop_box"])
    disposition = "inline" if inline else "attachment"
    return Response(data, media_type=files[name][0], headers={"Content-Disposition": f'{disposition}; filename="{name}"', "Cache-Control": "no-store"})


@app.get("/api/cases/{cid}/bundle.zip")
def case_bundle(cid: str):
    case = _done(cid)
    data = bundle_zip(case["result"], case["image"], case["meta"], case["crop_box"])
    return Response(data, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="bonevision_estimate.zip"', "Cache-Control": "no-store"})


# --------------------------------------------------------------------- evaluation

REPORTED_SYNTHETIC = {
    "source": "README (project-reported; not re-computed on this machine)",
    "protocol": "16 held-out HFValid test subjects; exact canonical AP silhouette from each CT (ideal mask, known size and position); 100 iterations.",
    "rows": [
        ["Mean voxel Dice", "0.687", "0.613"], ["Mean voxel IoU", "0.543", "0.479"],
        ["Mean symmetric surface distance", "4.73 mm", "6.23 mm"], ["Median surface distance", "3.35 mm", "4.48 mm"],
        ["Mean subject P95 surface distance", "14.43 mm", "17.61 mm"], ["Symmetric mean-squared Chamfer", "53.77 mm²", "96.19 mm²"],
        ["Soft silhouette projection Dice", "0.919", "0.944"],
    ],
    "columns": ["Metric", "SSM silhouette fit", "One-view template baseline"],
    "worst_case": "Pat099: Dice 0.373, mean surface distance 10.09 mm",
    "caveat": "Measured before the orientation, aspect-ratio and pose fixes in pipeline 0.3.0, and with an input that keeps true size. Re-run scripts/evaluate_single_view.py (both --input-mode ideal and app) to refresh.",
}


@app.get("/api/evaluation")
def evaluation():
    found = []
    for path in sorted((ROOT / "reconstruction").glob("single_view_*metrics*.csv")):
        try:
            rows = list(csv.DictReader(path.open(newline="")))
        except OSError:
            continue
        mean = next((r for r in rows if r.get("subject") == "MEAN"), None)
        per = [r for r in rows if r.get("status") == "ok"]
        found.append({"file": path.name, "aggregate": mean, "n_ok": len(per),
                      "n_failed": sum(1 for r in rows if r.get("status") == "error"),
                      "subjects": [{k: r.get(k) for k in ("subject", "ssm_dice", "ssm_mean_symmetric_surface_distance_mm", "retrieval_dice")} for r in per]})
    return {"reported": REPORTED_SYNTHETIC, "network": network.REPORTED_TEST, "local_runs": found}


# ----------------------------------------------------------------------- frontend

if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
