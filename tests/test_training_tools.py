"""Fast checks for the SDF shape model, external import alignment and multi-view network."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from femur_grid import (DIMENSIONS, femur_side, load_occupancy, occupancy_to_sdf,
                        downsample_occupancy, read_landmark_frame)


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


class SideMirroringTests(unittest.TestCase):
    def _subject(self, root, name, troch_x):
        d = Path(root) / name
        d.mkdir()
        (d / (name + "_palp.inp")).write_text(
            "femurHeadX=0\nfemurHeadY=0\nfemurHeadZ=0\nheadDirX=0\nheadDirY=0\nheadDirZ=10\n"
            "greatTrochX={}\ngreatTrochY=0\ngreatTrochZ=0\n".format(troch_x))

    def test_right_femur_maps_onto_left_in_canonical_frame(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self._subject(tmp, "L1", 50)
            self._subject(tmp, "R1", -50)
            self.assertEqual((femur_side("L1", tmp), femur_side("R1", tmp)), ("L", "R"))
            o_l, r_l = read_landmark_frame("L1", tmp)
            o_r, r_r = read_landmark_frame("R1", tmp)
            # A right femur is the left one mirrored in patient X (LPS).
            p_left = np.array([12.0, -7.0, -150.0])
            p_right = p_left * np.array([-1.0, 1.0, 1.0])
            np.testing.assert_allclose(r_l.T @ (p_left - o_l), r_r.T @ (p_right - o_r), atol=1e-9)
            # Un-mirrored frames differ by the sign of z, which load_occupancy undoes.
            _, r_raw = read_landmark_frame("R1", tmp, canonical=False)
            np.testing.assert_allclose(r_raw[:, 2], -r_r[:, 2])
            occ = np.zeros(DIMENSIONS, np.uint8)
            occ[10, 20, 5] = 1
            np.save(Path(tmp) / "R1.npy", occ)
            np.save(Path(tmp) / "L1.npy", occ)
            self.assertEqual(load_occupancy("R1", tmp, tmp)[10, 20, DIMENSIONS[2] - 1 - 5], 1)
            self.assertEqual(load_occupancy("L1", tmp, tmp)[10, 20, 5], 1)


if __name__ == "__main__":
    unittest.main()
