# Legacy code

Kept for history; not part of the current pipeline and not maintained.

- `yolo/`: the earlier 2D fracture-detection experiments (FracAtlas YOLO split, GRAZPEDWRI wrist annotation count). Unrelated to the femur 3D work.
- `ssm_v1/`: the first low-resolution SSM builder and its synthetic reconstruction experiment. Superseded by `scripts/build_ssm_v2.py` and the `bonevision` package. `ssm_reconstruct.py` has hard-coded views and target paths and must not be used as an X-ray-to-STL tool (see `docs/multiview-reconstruction-audit.md`).

The paths inside these scripts assume their old locations.
