"""Fast correctness checks for BoneVision's single-view pipeline."""
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reconstruct_single_xray import segment_single_bone, standardize_mask, _write_binary_stl
from evaluate_single_view import metrics
from slicer_export import display_transforms, export_for_slicer, slicer_bundle


class ReconstructionCoreTests(unittest.TestCase):
    def test_synthetic_bright_bone_segmentation_and_canonical_mask(self):
        image = np.zeros((512, 256), dtype=np.float32)
        # A connected tapered shaft with a wider head and condyle region.
        for y in range(28, 485):
            center = 130 + int(8 * np.sin(y / 75))
            half_width = 14 if y < 90 or y > 410 else 7
            image[y, center-half_width:center+half_width+1] = 1.0
        mask, info = segment_single_bone(image, polarity="bright")
        self.assertGreater(int(mask.sum()), 5000)
        self.assertEqual(info["polarity"], "bright")
        canonical = standardize_mask(mask, (256, 96))
        self.assertEqual(canonical.shape, (256, 96))
        self.assertGreater(int(canonical.sum()), 1000)

    def test_surface_metrics_are_zero_for_identical_geometry(self):
        a = np.zeros((16, 16, 16), dtype=bool)
        a[3:12, 5:10, 4:13] = True
        result = metrics(a, a, (2.0, 1.5, 1.0), seed=42)
        self.assertEqual(result["dice"], 1.0)
        self.assertEqual(result["iou"], 1.0)
        self.assertAlmostEqual(result["mean_symmetric_surface_distance_mm"], 0.0)
        self.assertAlmostEqual(result["chamfer_mean_squared_mm2"], 0.0)

    def test_binary_stl_export_has_valid_triangle_count_and_size(self):
        vertices = np.asarray([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float32)
        faces = np.asarray([[0,1,2]], dtype=np.int32)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "one_triangle.stl"
            _write_binary_stl(path, vertices, faces)
            content = path.read_bytes()
        self.assertEqual(struct.unpack("<I", content[80:84])[0], 1)
        self.assertEqual(len(content), 84 + 50)

    def test_slicer_export_aligns_mesh_and_labelmap_in_ras(self):
        import gzip
        import zipfile
        from skimage.measure import marching_cubes
        bounds = (-470.0, 40.0, -60.0, 100.0, -70.0, 70.0)
        occ = np.zeros((32, 12, 12), dtype=np.float32)
        occ[4:28, 3:9, 2:10] = 1.0
        spacing = tuple((bounds[2*i+1]-bounds[2*i])/(occ.shape[i]-1) for i in range(3))
        verts, faces, _, _ = marching_cubes(occ, level=0.5, spacing=spacing)
        verts = verts + np.asarray(bounds[::2])
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            info = export_for_slicer(out, verts, faces, occ, bounds)
            stl = (out / info["mesh"]).read_bytes()
            raw = (out / info["labelmap"]).read_bytes()
            names = zipfile.ZipFile(__import__("io").BytesIO(slicer_bundle(out))).namelist()
        self.assertIn(b"SPACE=RAS", stl[:80])
        header, payload = raw.split(b"\n\n", 1)
        fields = dict(line.split(": ", 1) for line in header.decode().splitlines()[1:] if ":=" not in line)
        self.assertEqual(fields["space"], "right-anterior-superior")
        self.assertEqual(fields["sizes"], "32 12 12")
        label = np.frombuffer(gzip.decompress(payload), dtype=np.uint8).reshape(occ.shape, order="F")
        np.testing.assert_array_equal(label, occ >= 0.5)
        origin = np.asarray([float(v) for v in fields["space origin"].strip("()").split(",")])
        idx = np.argwhere(label)
        lo, hi = origin + idx.min(0) * spacing, origin + idx.max(0) * spacing
        # The 0.5 isosurface sits half a voxel outside the labelled voxel centres.
        np.testing.assert_allclose(verts.min(0), lo - np.asarray(spacing) / 2, atol=1e-3)
        np.testing.assert_allclose(verts.max(0), hi + np.asarray(spacing) / 2, atol=1e-3)
        self.assertIn("bonevision_femur/BoneVisionLoader.py", names)

    def test_slicer_display_transforms_are_rotations_with_proximal_superior(self):
        for name, matrix in display_transforms().items():
            rot = np.asarray(matrix, dtype=float)[:3, :3]
            np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rot), 1.0)
            if name != "canonical":
                np.testing.assert_array_equal(rot @ [1, 0, 0], [0, 0, 1])

    def test_ssm_grid_spacing_is_positive_physical_mm(self):
        data = np.load(ROOT / "models" / "ssm_highres.npz")
        dims = tuple(int(x) for x in data["grid_shape"])
        bounds = tuple(float(x) for x in data["bounds"])
        spacing = tuple((bounds[2*i+1]-bounds[2*i])/(dims[i]-1) for i in range(3))
        self.assertTrue(all(x > 0 for x in spacing))
        self.assertAlmostEqual(spacing[0], 2.0, places=3)

    @unittest.skipUnless(os.environ.get("BONEVISION_RUN_INTEGRATION") == "1", "set BONEVISION_RUN_INTEGRATION=1 to run GPU integration")
    def test_single_xray_to_mesh_integration(self):
        from reconstruct_single_xray import MODEL_PATH, reconstruct, save_results
        if not MODEL_PATH.exists():
            self.skipTest("trained SSM is not available")
        gt = np.load(ROOT / "data" / "occupancy_highres" / "Pat001.npy").astype(np.float32)
        image = 1.0 - np.exp(-gt.sum(axis=2))
        result = reconstruct(image, iterations=10)
        self.assertEqual(result["vertices_mm"].shape[1], 3)
        self.assertGreater(len(result["faces"]), 1000)
        self.assertGreaterEqual(result["projection_dice"], 0.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            save_results(result, Path(temp_dir), image)
            self.assertTrue((Path(temp_dir) / "femur_reconstruction.stl").exists())
            self.assertTrue((Path(temp_dir) / "reconstruction.json").exists())
            self.assertTrue((Path(temp_dir) / "projection_fit.png").exists())
            import vtk
            reader = vtk.vtkSTLReader()
            reader.SetFileName(str(Path(temp_dir) / "femur_reconstruction.stl"))
            reader.Update()
            self.assertGreater(reader.GetOutput().GetNumberOfPolys(), 1000)


if __name__ == "__main__":
    unittest.main()
