"""Command-line single-view femur estimate: one AP-like X-ray in, STL/OBJ/PLY/JSON/PNG out.

The implementation lives in the `bonevision` package and is shared with the web
backend (`serve.py`). Depth and absolute size come from the shape prior; see
the report's `limitations` field.

    python scripts/reconstruct_single_xray.py femur.png --output runs/case01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bonevision.exports import FILES, stl_bytes, write_outputs  # noqa: E402
from bonevision.pipeline import (  # noqa: E402,F401  (re-exported for older scripts/tests)
    BOUNDS_DEFAULT, MODEL_PATH, ROOT, apply_crop, load_model, read_image, read_image_bytes, reconstruct,
    segment_single_bone, standardize_mask,
)


def _write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    Path(path).write_bytes(stl_bytes(vertices, faces))


def main():
    parser = argparse.ArgumentParser(description="Estimate a 3D femur from one AP-like X-ray and export files for 3D Slicer.")
    parser.add_argument("image", type=Path, help="Input PNG/JPG/TIFF or DICOM X-ray")
    parser.add_argument("--output", type=Path, default=Path("reconstruction_output"))
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--polarity", choices=("auto", "bright", "dark"), default="auto")
    parser.add_argument("--threshold-bias", type=float, default=0.0, help="Offset from Otsu threshold, clipped to -0.25..0.25")
    parser.add_argument("--prior-weight", type=float, default=0.02, help="PCA shape-prior penalty (default 0.02; other values are not validated)")
    parser.add_argument("--crop", type=float, nargs=4, metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"), default=None,
                        help="Edge crop fractions, each 0..0.45")
    parser.add_argument("--use-dicom-spacing", action="store_true",
                        help="Size the silhouette from DICOM pixel spacing instead of the population mean (no magnification correction)")
    parser.add_argument("--magnification", type=float, default=1.0, help="Divide DICOM spacing by this factor when --use-dicom-spacing is set")
    parser.add_argument("--no-spread", action="store_true", help="Skip the silhouette-consistent shape-spread estimate")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--force-overwrite", action="store_true", help="Allow replacement of existing output files")
    args = parser.parse_args()
    if not args.image.exists():
        parser.error(f"Input image does not exist: {args.image}")
    if not args.model.exists():
        parser.error(f"Shape model not found: {args.model}; build it with scripts/build_ssm_v2.py")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if not np.isfinite(args.prior_weight) or args.prior_weight < 0:
        parser.error("--prior-weight must be finite and nonnegative")
    collisions = [str(args.output / name) for name in FILES if (args.output / name).exists()]
    if collisions and not args.force_overwrite:
        parser.error("Refusing to overwrite existing output(s): " + ", ".join(collisions) + ". Choose another --output or pass --force-overwrite.")

    image, meta = read_image_bytes(args.image.read_bytes(), args.image.name)
    crop_box = None
    if args.crop:
        image, crop_box = apply_crop(image, dict(zip(("top", "bottom", "left", "right"), args.crop)))
    mm_per_pixel = None
    if args.use_dicom_spacing:
        if not meta.get("pixel_spacing_mm"):
            parser.error("--use-dicom-spacing was set but the file has no DICOM pixel spacing")
        sy, sx = meta["pixel_spacing_mm"]
        if abs(sy - sx) > 1e-3:
            parser.error("Anisotropic DICOM pixel spacing is not supported yet")
        mm_per_pixel = sy / max(args.magnification, 1e-6)
    model = load_model(args.model, None if args.device == "auto" else args.device)
    result = reconstruct(image, model, args.iterations, args.polarity, float(np.clip(args.threshold_bias, -0.25, 0.25)),
                         prior_weight=args.prior_weight, mm_per_pixel=mm_per_pixel, spread=not args.no_spread)
    paths = write_outputs(result, image, args.output, meta, crop_box, force=args.force_overwrite)
    print(json.dumps({"silhouette_soft_dice": result["projection_dice"], "orientation": result["hypothesis"]["id"],
                      "seconds": round(result["elapsed_seconds"], 2), "device": result["device"],
                      "dimensions_mm": [round(x, 1) for x in result["dimensions_mm"]],
                      "warnings": [w["message"] for w in result["warnings"]]}, indent=2))
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
