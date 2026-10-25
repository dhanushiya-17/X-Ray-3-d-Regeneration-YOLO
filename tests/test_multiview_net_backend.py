"""Placement of an uploaded X-ray on the multi-view network's grid."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import rotate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from femur_grid import DIMENSIONS, grid_spacing
from multiview_net_backend import _dice, classify_view, place_on_grid


def _femur_silhouette():
    """A femur-like AP silhouette on the grid: shaft, offset head and wider condyles."""
    x, y = np.meshgrid(np.arange(DIMENSIONS[0]), np.arange(DIMENSIONS[1]), indexing="ij")
    shaft = (np.abs(y - 45) < 9) & (x > 30) & (x < 225)
    head = (x - 236) ** 2 + (y - 30) ** 2 < 14 ** 2
    neck = (x > 215) & (x < 236) & (y > 30) & (y < 50)
    troch = (x - 222) ** 2 / 9 ** 2 + (y - 58) ** 2 / 8 ** 2 < 1
    condyles = (x - 32) ** 2 / 14 ** 2 + (y - 45) ** 2 / 24 ** 2 < 1
    return shaft | head | neck | troch | condyles


class PlacementTests(unittest.TestCase):
    def test_view_classification(self):
        self.assertEqual([classify_view(a) for a in (0, 180, 10, 90, 270, -85, 45)],
                         ["ap", "ap", "ap", "lat", "lat", "lat", None])

    def test_rotated_rescaled_mirrored_xray_is_placed_back_on_the_grid(self):
        sx, sy, _ = grid_spacing()
        ref_mask = _femur_silhouette()
        ref = {"ap": ref_mask}
        pixel_mm = 0.7
        # Square-pixel "radiograph" of the same femur: mirrored, upside down, rotated 20 degrees.
        img = ndi.zoom(ref_mask.astype(np.float32), (sx / pixel_mm, sy / pixel_mm), order=1)
        img = np.pad(img, 50)[::-1, ::-1]
        img = rotate(img, 20, resize=True, order=1)
        mask = img > 0.5
        results = {}
        for flip in (False, True):
            for mirror in (False, True):
                g_img, g_mask, mm_px = place_on_grid(img, mask, ref, "ap", sx, sy, flip, mirror)
                results[(flip, mirror)] = (_dice(g_mask, ref_mask), mm_px)
        best = max(results, key=lambda k: results[k][0])
        self.assertGreater(results[best][0], 0.9)
        second = sorted(v[0] for v in results.values())[-2]
        self.assertGreater(results[best][0] - second, 0.05)
        self.assertAlmostEqual(results[best][1], pixel_mm, delta=0.05)


if __name__ == "__main__":
    unittest.main()
