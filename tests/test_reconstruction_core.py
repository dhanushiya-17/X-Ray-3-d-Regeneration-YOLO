"""Fast correctness checks for BoneVision's single-view pipeline, exports and API.

Run with:  python -m unittest discover -s tests -v
Tests that need the licensed real model skip when models/ssm_highres.npz is absent;
end-to-end tests build a tiny synthetic model in a temp directory instead.
"""
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from bonevision import exports  # noqa: E402
from bonevision.pipeline import (  # noqa: E402
    HYPOTHESES, MODEL_PATH, hypothesis_targets, pose_index_points, pose_volume, read_image_bytes,
    segment_single_bone, standardize_mask, standardize_mask_with_transform,
)
from evaluate_single_view import metrics  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

_TMP = tempfile.TemporaryDirectory()
DEMO_MODEL = Path(_TMP.name) / "demo.npz"


def build_demo_model():
    if not DEMO_MODEL.exists():
        from make_demo_model import femur_like
        rng = np.random.default_rng(1)
        grid = (96, 36, 36)
        shapes = np.stack([femur_like(grid, rng).ravel() for _ in range(24)])
        mean = shapes.mean(0)
        _, s, vt = np.linalg.svd(shapes - mean, full_matrices=False)
        np.savez(DEMO_MODEL, mean_shape=mean.astype(np.float32), components=vt[:8].astype(np.float32),
                 explained_variance=(s[:8] ** 2 / 23).astype(np.float32), grid_shape=np.asarray(grid),
                 bounds=np.asarray((-470.0, 40.0, -60.0, 100.0, -70.0, 70.0)),
                 train_subjects=np.asarray([f"S{i}" for i in range(24)]), synthetic_demo=np.asarray(True))
    return DEMO_MODEL


def synthetic_xray(mirror=True, angle=10.0):
    """Radiograph-like image of a procedural femur (seed differs from the model's)."""
    from make_demo_model import femur_like
    occ = femur_like((256, 96, 96), np.random.default_rng(99))
    img = 1 - np.exp(-occ.sum(axis=2) * (140 / 95) / 40)
    img = ndi.zoom(img, (2.0, 1.684), order=1)
    if mirror:
        img = img[:, ::-1]
    img = np.pad(ndi.rotate(img, angle, reshape=True, order=1), 30)
    return (img * 0.75 + 0.12 + np.random.default_rng(0).normal(0, 0.02, img.shape)).clip(0, 1).astype(np.float32)


class SegmentationAndStandardization(unittest.TestCase):
    def test_synthetic_bright_bone_segmentation_and_canonical_mask(self):
        image = np.zeros((512, 256), dtype=np.float32)
        for y in range(28, 485):
            center = 130 + int(8 * np.sin(y / 75))
            half_width = 14 if y < 90 or y > 410 else 7
            image[y, center - half_width:center + half_width + 1] = 1.0
        mask, info = segment_single_bone(image, polarity="bright")
        self.assertGreater(int(mask.sum()), 5000)
        self.assertEqual(info["polarity"], "bright")
        canonical = standardize_mask(mask, (256, 96))
        self.assertEqual(canonical.shape, (256, 96))
        self.assertGreater(int(canonical.sum()), 1000)

    def test_physical_aspect_is_preserved_on_anisotropic_grid(self):
        # A 300 x 40 px rectangle at 1 mm/px must become 150 rows (2.0 mm) x ~23.8 cols (1.684 mm).
        mask = np.zeros((400, 200), dtype=bool)
        mask[50:350, 80:120] = True
        canvas, tf = standardize_mask_with_transform(mask, (256, 96), None, (2.0, 1.684), mm_per_pixel=1.0)
        rows, cols = np.where(canvas)
        self.assertAlmostEqual(np.ptp(rows) + 1, 150, delta=2)
        self.assertAlmostEqual(np.ptp(cols) + 1, 40 / 1.684, delta=2)
        self.assertTrue(tf["calibrated"])

    def test_standardization_rotates_without_reflecting(self):
        mask = np.zeros((300, 300), dtype=bool)
        yy, xx = np.mgrid[:300, :300]
        mask[((yy - 150) * np.cos(0.4) + (xx - 150) * np.sin(0.4)) ** 2 / 120 ** 2
             + ((xx - 150) * np.cos(0.4) - (yy - 150) * np.sin(0.4)) ** 2 / 15 ** 2 <= 1] = True
        _, tf = standardize_mask_with_transform(mask, (128, 64), None, (2.0, 2.0))
        self.assertGreater(np.linalg.det(np.asarray(tf["A"])), 0)

    def test_four_orientation_hypotheses(self):
        t = np.zeros((6, 4), dtype=bool)
        t[0, 0] = True
        stack = hypothesis_targets(t)
        self.assertEqual(len(stack), len(HYPOTHESES))
        corners = [tuple(np.argwhere(s)[0]) for s in stack]
        self.assertEqual(sorted(corners), sorted([(0, 0), (5, 3), (0, 3), (5, 0)]))


class PoseAndMetrics(unittest.TestCase):
    def test_zero_pose_is_identity(self):
        pts = np.random.default_rng(0).uniform(0, 30, (50, 3))
        np.testing.assert_allclose(pose_index_points(pts, (0, 0, 0), (32, 32, 32)), pts, atol=1e-9)

    def test_volume_pose_matches_point_pose(self):
        vol = np.zeros((40, 40, 40), dtype=np.float32)
        vol[18:22, 10:14, 19:21] = 1.0
        pose = (0.1, 0.05, -0.08)
        centre = np.argwhere(vol > 0.5).mean(axis=0)
        moved = np.argwhere(pose_volume(vol, pose) > 0.5).mean(axis=0)
        np.testing.assert_allclose(moved, pose_index_points(centre[None], pose, vol.shape)[0], atol=0.6)

    def test_surface_metrics_are_zero_for_identical_geometry(self):
        a = np.zeros((16, 16, 16), dtype=bool)
        a[3:12, 5:10, 4:13] = True
        result = metrics(a, a, (2.0, 1.5, 1.0), seed=42)
        self.assertEqual(result["dice"], 1.0)
        self.assertAlmostEqual(result["mean_symmetric_surface_distance_mm"], 0.0)


class InputAndExports(unittest.TestCase):
    def test_garbage_bytes_are_rejected(self):
        with self.assertRaises(ValueError):
            read_image_bytes(b"not an image", "x.png")

    def test_16bit_png_keeps_contrast(self):
        arr = (np.linspace(0, 60000, 64 * 64).reshape(64, 64)).astype(np.uint16)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        img, meta = read_image_bytes(buf.getvalue(), "x.png")
        self.assertEqual(img.shape, (64, 64))
        self.assertGreater(len(np.unique(img)), 256)
        self.assertEqual(len(meta["sha256"]), 64)

    def test_binary_stl_export_has_valid_triangle_count_and_size(self):
        data = exports.stl_bytes(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32), np.asarray([[0, 1, 2]]))
        self.assertEqual(struct.unpack("<I", data[80:84])[0], 1)
        self.assertEqual(len(data), 84 + 50)

    def test_ply_header_matches_payload(self):
        v = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
        f = np.asarray([[0, 1, 2], [2, 3, 4]])
        data = exports.ply_bytes(v, f, np.linspace(0, 0.5, 10).astype(np.float32))
        head, body = data.split(b"end_header\n", 1)
        self.assertIn(b"element vertex 10", head)
        self.assertEqual(len(body), 10 * (12 + 3 + 4) + 2 * (1 + 12))


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed")
class EndToEndOnSyntheticModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bonevision.pipeline import load_model, reconstruct
        cls.image = synthetic_xray(mirror=True, angle=10.0)
        cls.model = load_model(build_demo_model(), "cpu")
        cls.result = reconstruct(cls.image, cls.model, iterations=80)

    def test_mirrored_input_selects_left_right_flip(self):
        self.assertEqual(self.result["hypothesis"]["id"], "flipped_lr")
        self.assertGreater(self.result["projection_dice"], 0.75)

    def test_rotation_is_recovered(self):
        self.assertAlmostEqual(abs(self.result["transform"]["rotation_deg"]), 10.0, delta=3.0)

    def test_mesh_and_spread(self):
        r = self.result
        self.assertEqual(r["vertices_mm"].shape[1], 3)
        self.assertGreater(len(r["faces"]), 500)
        self.assertTrue(r["pose_applied_to_mesh"])
        if r["spread"].get("available"):
            self.assertEqual(len(r["vertex_spread"]), len(r["vertices_mm"]))
            self.assertTrue(np.all((r["vertex_spread"] >= 0) & (r["vertex_spread"] <= 0.5 + 1e-6)))

    def test_bundle_contains_every_export_and_a_valid_report(self):
        data = exports.bundle_zip(self.result, self.image, {"source_name": "synthetic.png"})
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = set(z.namelist())
            self.assertTrue(set(exports.FILES) <= names)
            report = json.loads(z.read("report.json"))
        self.assertEqual(report["orientation"]["selected"], "flipped_lr")
        self.assertTrue(report["model"]["synthetic_demo"])
        self.assertIn("not patient ras/lps", report["mesh"]["coordinate_frame"].lower())

    def test_overlay_is_drawn_on_the_input_grid(self):
        from bonevision.pipeline import overlay_on_input
        rgb = overlay_on_input(self.result, self.image)
        self.assertEqual(rgb.shape, self.image.shape + (3,))


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed")
class Api(unittest.TestCase):
    def test_upload_segment_fit_and_download(self):
        try:
            from fastapi.testclient import TestClient
        except Exception:
            self.skipTest("fastapi/httpx not installed")
        import time
        os.environ["BONEVISION_MODEL"] = str(build_demo_model())
        os.environ["BONEVISION_DEVICE"] = "cpu"
        from bonevision.server import app
        c = TestClient(app)
        self.assertTrue(c.get("/api/status").json()["model_available"])
        buf = io.BytesIO()
        Image.fromarray((synthetic_xray() * 255).astype(np.uint8)).save(buf, format="PNG")
        up = c.post("/api/uploads", files={"file": ("x.png", buf.getvalue(), "image/png")}).json()
        self.assertEqual(c.post(f"/api/uploads/{up['id']}/segment", json={}).status_code, 200)
        self.assertEqual(c.post(f"/api/uploads/{up['id']}/segment", json={"crop": {"top": 0.45, "bottom": 0.45}}).status_code, 422)
        cid = c.post("/api/cases", json={"upload_id": up["id"], "iterations": 40, "spread": False}).json()["id"]
        for _ in range(300):
            st = c.get(f"/api/cases/{cid}").json()
            if st["status"] not in ("queued", "running"):
                break
            time.sleep(0.1)
        self.assertEqual(st["status"], "done", st.get("error"))
        stl = c.get(f"/api/cases/{cid}/files/femur_estimate.stl")
        self.assertEqual(stl.status_code, 200)
        self.assertIn("attachment", stl.headers["content-disposition"])
        self.assertEqual(c.get(f"/api/cases/{cid}/files/../../etc").status_code, 404)
        self.assertEqual(c.get("/api/cases/nope").status_code, 404)


@unittest.skipUnless(MODEL_PATH.exists(), "real SSM (models/ssm_highres.npz) is not available")
class RealModel(unittest.TestCase):
    def test_ssm_grid_spacing_is_positive_physical_mm(self):
        data = np.load(MODEL_PATH)
        dims = tuple(int(x) for x in data["grid_shape"])
        bounds = tuple(float(x) for x in data["bounds"])
        spacing = tuple((bounds[2 * i + 1] - bounds[2 * i]) / (dims[i] - 1) for i in range(3))
        self.assertTrue(all(x > 0 for x in spacing))
        self.assertAlmostEqual(spacing[0], 2.0, places=3)

    @unittest.skipUnless(os.environ.get("BONEVISION_RUN_INTEGRATION") == "1", "set BONEVISION_RUN_INTEGRATION=1")
    def test_single_xray_to_mesh_integration(self):
        from bonevision.pipeline import reconstruct
        gt = np.load(ROOT / "data" / "occupancy_highres" / "Pat001.npy").astype(np.float32)
        result = reconstruct(1.0 - np.exp(-gt.sum(axis=2)), iterations=10, spread=False)
        self.assertGreater(len(result["faces"]), 1000)


if __name__ == "__main__":
    unittest.main()
