"""Fast checks for the SDF shape model, external import alignment and multi-view network."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from femur_grid import occupancy_to_sdf, downsample_occupancy


def _ellipsoid(shape=(40, 24, 24), radii=(15, 8, 6)):
    idx = np.indices(shape).astype(float)
    centre = np.array(shape)[:, None, None, None] / 2.0
    r = np.array(radii, dtype=float)[:, None, None, None]
    return (((idx - centre) / r) ** 2).sum(axis=0) < 1.0


class SdfTests(unittest.TestCase):
    def test_sdf_sign_and_zero_level_recovers_shape(self):
        occ = _ellipsoid()
        sdf = occupancy_to_sdf(occ, (2.0, 1.5, 1.5), truncation=10.0)
        self.assertTrue((sdf[occ] < 0).all())
        self.assertTrue((sdf[~occ] > 0).all())
        self.assertTrue(np.array_equal(sdf < 0, occ))
        self.assertLessEqual(float(np.abs(sdf).max()), 10.0)

    def test_downsample_keeps_shape(self):
        occ = _ellipsoid().astype(np.uint8)
        small = downsample_occupancy(occ, 2)
        self.assertEqual(small.shape, (20, 12, 12))
        self.assertGreater(int(small.sum()), 0)


class AlignmentTests(unittest.TestCase):
    def test_icp_recovers_known_rigid_transform(self):
        from scipy.spatial import cKDTree
        from import_external_femurs import align
        rng = np.random.default_rng(0)
        # An asymmetric "bone": elongated, bent point cloud
        t = rng.uniform(0, 1, 4000)
        ref = np.stack([-400 * t, 30 * np.sin(3 * t) + rng.normal(0, 8, t.size),
                        rng.normal(0, 6, t.size) + 10 * t ** 2], axis=1)
        a = 0.4
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        moved = ref @ rot.T + np.array([20.0, -300.0, 900.0])
        A, tr, rms, _ = align(moved, ref, cKDTree(ref))
        back = moved @ A.T + tr
        self.assertLess(rms, 6.0)
        self.assertLess(float(np.abs(back - ref).mean()), 5.0)


class NetworkTests(unittest.TestCase):
    def test_forward_shapes_for_two_and_one_view(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from multiview_net import MultiViewFemurNet, dice_bce_loss
        x, y, z = 32, 16, 16
        for views in (("ap", "lat"), ("ap",)):
            net = MultiViewFemurNet(views=views, feat=4, base2d=4, base3d=4)
            images = {"ap": torch.rand(1, 1, x, y), "lat": torch.rand(1, 1, x, z)}
            out = net(images, (y, z))
            self.assertEqual(tuple(out.shape), (1, 1, x, y, z))
            loss = dice_bce_loss(out, torch.zeros_like(out))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
