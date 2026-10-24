"""Multi-view fit checks against a small synthetic SSM (no trained model needed)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch
except ImportError:  # PyTorch is an optional app dependency
    torch = None

BOUNDS = (-470.0, 40.0, -60.0, 100.0, -70.0, 70.0)
SHAPE = (96, 40, 40)


def _box(y_half, z_half):
    """Shaft-like occupancy whose half-widths in Y and Z are set independently."""
    xs = np.linspace(BOUNDS[0], BOUNDS[1], SHAPE[0])[:, None, None]
    ys = np.linspace(BOUNDS[2], BOUNDS[3], SHAPE[1])[None, :, None]
    zs = np.linspace(BOUNDS[4], BOUNDS[5], SHAPE[2])[None, None, :]
    head = (xs > -60) & (xs < 20) & (np.abs(ys - 40) < y_half * 1.4)
    shaft = (xs > -420) & (xs < 0) & (np.abs(ys - 20) < y_half)
    return ((head | shaft) & (np.abs(zs) < z_half)).astype(np.float32)


def synthetic_model():
    """Mean shape plus two orthogonal modes: Y width (seen in AP) and Z depth (seen only laterally)."""
    mean = _box(18, 18)
    modes = np.stack([(_box(30, 18) - _box(8, 18)) / 2, (_box(18, 32) - _box(18, 6)) / 2]).reshape(2, -1)
    norms = np.linalg.norm(modes, axis=1, keepdims=True)
    return {
        "shape": SHAPE, "bounds": BOUNDS, "mean": mean.reshape(-1), "components": (modes / norms).astype(np.float32),
        "explained_variance": (norms[:, 0] ** 2).astype(np.float32), "n_training_subjects": 2, "path": "synthetic",
    }


def radiograph(silhouette, scale=3):
    """Upsample a binary silhouette into a bright-bone image on a dark background."""
    image = np.kron(silhouette.astype(np.float32), np.ones((scale, scale), dtype=np.float32))
    return np.pad(image, 24) * 0.8 + 0.1


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MultiViewReconstructionTests(unittest.TestCase):
    def setUp(self):
        from reconstruct_multiview_xray import ViewInput, project_basis, reconstruct_multiview
        self.ViewInput, self.project_basis, self.reconstruct = ViewInput, project_basis, reconstruct_multiview
        self.model = synthetic_model()
        self.truth = _box(18, 32)  # mean width in AP; about +1.6 std of the depth mode

    def silhouette(self, angle):
        line, _ = self.project_basis(self.truth.reshape(1, -1), SHAPE, BOUNDS, angle, torch.device("cpu"))
        return (line[0].numpy() > 0.5)

    def iou(self, result):
        pred, gt = result["occupancy"] > 0.5, self.truth > 0.5
        return (pred & gt).sum() / (pred | gt).sum()

    def test_general_rotation_matches_exact_quarter_turn(self):
        exact = self.silhouette(90.0)
        rotated = self.silhouette(90.0001)
        # Rotated output spans the wider transverse extent, so compare profile widths in mm.
        exact_mm = exact.sum(axis=1) * (BOUNDS[5] - BOUNDS[4]) / (SHAPE[2] - 1)
        rotated_mm = rotated.sum(axis=1) * max(BOUNDS[3] - BOUNDS[2], BOUNDS[5] - BOUNDS[4]) / (max(SHAPE[1:]) - 1)
        self.assertLess(np.abs(exact_mm - rotated_mm).mean(), 8.0)

    def test_lateral_view_recovers_depth_that_ap_alone_cannot(self):
        ap = self.ViewInput(radiograph(self.silhouette(0.0)), 0.0, "ap.png", "bright", mirror="no")
        lat = self.ViewInput(radiograph(self.silhouette(90.0)), 90.0, "lat.png", "bright", mirror="no")
        ap_only = self.reconstruct([ap], iterations=150, device="cpu", model=self.model)
        both = self.reconstruct([ap, lat], iterations=150, device="cpu", model=self.model)
        self.assertEqual(both["n_views"], 2)
        self.assertEqual(len(both["views"]), 2)
        self.assertGreater(both["views"][1]["soft_dice"], 0.85)
        # Only the lateral silhouette shows the depth mode; AP alone leaves it at the prior mean.
        self.assertGreater(both["pca_coefficients_std"][1], ap_only["pca_coefficients_std"][1] + 0.5)
        self.assertGreater(self.iou(both), self.iou(ap_only) + 0.2)

    def test_backend_saves_slicer_outputs_and_report(self):
        from reconstruction_backends import SSMSilhouetteBackend, available_backends, get_backend
        self.assertIn("ssm-silhouette", available_backends())
        with self.assertRaises(KeyError):
            get_backend("does-not-exist")
        backend = SSMSilhouetteBackend(Path("missing-model.npz"))
        self.assertIsNotNone(backend.availability())
        views = [self.ViewInput(radiograph(self.silhouette(a)), a, f"v{a:.0f}", "bright") for a in (0.0, 90.0)]
        result = self.reconstruct(views, iterations=20, device="cpu", model=self.model)
        self.assertEqual(result["hypotheses_tested"], 8)
        with tempfile.TemporaryDirectory() as temp_dir:
            backend.save(result, Path(temp_dir))
            names = sorted(p.name for p in Path(temp_dir).iterdir())
            report = json.loads((Path(temp_dir) / "reconstruction.json").read_text())
        self.assertEqual(names, ["femur_labelmap.nrrd", "femur_reconstruction.stl", "projection_fit.png", "reconstruction.json"])
        self.assertEqual([v["angle_deg"] for v in report["views"]], [0.0, 90.0])
        self.assertEqual(report["slicer"]["mesh"], "femur_reconstruction.stl")

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            self.reconstruct([], model=self.model)
        bad = self.ViewInput(np.zeros((10, 10, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            self.reconstruct([bad], model=self.model, device="cpu")


if __name__ == "__main__":
    unittest.main()
