"""Swappable reconstruction backends for the BoneVision app and CLI.

The web app only talks to a backend through this interface, so a retrained
model (a new SSM, or a learned X-ray-to-3D network) can replace the current
silhouette fit without touching the UI. A backend takes a list of ViewInput
radiographs of one femur and returns a result dict with at least:

    vertices_mm (V, 3) float32, faces (F, 3) int32, occupancy (L, Y, Z),
    bounds (6 floats), views (list of per-view dicts with name, angle_deg,
    mask, target_projection, projection, soft_dice), projection_dice,
    elapsed_seconds, device, method, fit_status, history.

`save(result, output_dir)` writes the Slicer-ready STL/NRRD, the JSON report
and the silhouette comparison. A backend may also return `details`, a list
of short lines the app shows under the fit status. Register a new backend
with `register_backend`; the app lists every registered backend, and the
BONEVISION_BACKEND environment variable sets the default.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Protocol

from multiview_net_backend import MultiViewNetBackend
from reconstruct_multiview_xray import (
    MAX_VIEWS, MODEL_PATH, ViewInput, load_model, reconstruct_multiview, save_multiview_results,
)


class ReconstructionBackend(Protocol):
    name: str
    label: str
    max_views: int

    def availability(self) -> str | None:
        """None when ready, otherwise a message explaining what is missing."""

    def reconstruct(self, views: list[ViewInput], iterations: int = 250, prior_weight: float = 0.02, device: str | None = None, refine_iterations: int = 100) -> dict:
        ...

    def save(self, result: dict, output_dir: Path) -> None:
        ...


class SSMSilhouetteBackend:
    """Current model: joint silhouette fit of the 70-subject voxel PCA SSM."""
    name = "ssm-silhouette"
    label = "Voxel PCA SSM, joint silhouette fit"
    max_views = MAX_VIEWS

    def __init__(self, model_path: Path | None = None):
        self.model_path = Path(os.environ.get("BONEVISION_MODEL", model_path or MODEL_PATH))
        self._model = None
        self._model_mtime = None
        self._prepared: dict = {}

    def availability(self) -> str | None:
        if not self.model_path.exists():
            return f"The trained SSM is missing at {self.model_path}. Build it with scripts/build_ssm_v2.py from the HFValid-derived data."
        return None

    def _loaded_model(self) -> dict:
        mtime = self.model_path.stat().st_mtime
        if self._model is None or mtime != self._model_mtime:
            self._model, self._model_mtime, self._prepared = load_model(self.model_path), mtime, {}
        return self._model

    def reconstruct(self, views, iterations=250, prior_weight=0.02, device=None, refine_iterations=100):
        problem = self.availability()
        if problem:
            raise FileNotFoundError(problem)
        result = reconstruct_multiview(views, self.model_path, iterations, device, prior_weight, model=self._loaded_model(), prepared_cache=self._prepared, refine_iterations=refine_iterations)
        result["backend"] = self.name
        return result

    def save(self, result, output_dir):
        save_multiview_results(result, Path(output_dir), self.name)


# Listed in order of preference: the first one whose model files exist is the default.
_BACKENDS: dict[str, Callable[[], ReconstructionBackend]] = {
    MultiViewNetBackend.name: MultiViewNetBackend,
    SSMSilhouetteBackend.name: SSMSilhouetteBackend,
}


def register_backend(name: str, factory: Callable[[], ReconstructionBackend]) -> None:
    _BACKENDS[name] = factory


def available_backends() -> list[str]:
    return list(_BACKENDS)


def default_backend_name() -> str:
    """BONEVISION_BACKEND if set, else the first registered backend that is ready to use."""
    if os.environ.get("BONEVISION_BACKEND"):
        return os.environ["BONEVISION_BACKEND"]
    for name in _BACKENDS:
        if _BACKENDS[name]().availability() is None:
            return name
    return SSMSilhouetteBackend.name


def get_backend(name: str | None = None) -> ReconstructionBackend:
    name = name or default_backend_name()
    if name not in _BACKENDS:
        raise KeyError(f"Unknown reconstruction backend {name!r}; registered: {', '.join(_BACKENDS)}")
    return _BACKENDS[name]()
