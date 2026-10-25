# BoneVision: femur reconstruction from X-rays

BoneVision is a research prototype that estimates a 3D femur surface from an **AP radiograph, optionally with a lateral one**. It traces the bone in each image and reconstructs it with one of two methods, then returns a mesh you can orbit in the browser and open in 3D Slicer:

- **Learned network** (default when its checkpoints are present): a multi-view network trained on simulated X-rays of 101 HFValid CT femurs. Takes one AP view and, optionally, one lateral view.
- **Shape-model fit**: a CT-trained statistical shape model (SSM) fitted to one AP outline.

> **Research estimate, not a measurement.** Neither method has been validated on real clinical radiographs, and the network was trained only on simulated X-rays. Size comes from the training population, not from your image. Silhouette agreement is 2D overlap, not 3D accuracy. Do not use it for diagnosis, treatment or surgical planning.

## Quick start

```bash
pip install -r requirements-app.txt   # install a CUDA build of torch first for GPU inference
python serve.py --demo                # try the UI with SYNTHETIC demo models (trains a tiny demo network once, a few minutes on CPU)
python serve.py                       # use the trained models in models/ (see below)
```

`python serve.py` looks for:

| File | Used by | Where it comes from |
|---|---|---|
| `models/multiview_ap_lat_d2.pt` | network, AP + lateral | `scripts/run_retraining.py` (or `bash scripts/train_wsl.sh`) |
| `models/multiview_ap_d2.pt` | network, AP only | same run with `--also-ap` |
| `models/ssm_sdf.npz` | network, only for checkpoints saved before the mean silhouettes were stored in them | `scripts/build_ssm_sdf.py` (step `ssm` of the same run) |
| `models/ssm_highres.npz` | shape-model fit | `scripts/build_ssm_v2.py` |

Either method works on its own; the UI greys out whichever is missing. To try a different network, point the server at its checkpoints without touching code: `python serve.py --net-ap-lat path/to/new_ap_lat.pt --net-ap path/to/new_ap.pt` (or set `BONEVISION_NET_AP_LAT` / `BONEVISION_NET_AP`). `--device cuda` runs on the GPU.

Open <http://127.0.0.1:8000>. On Windows use the same commands with your environment's `python`.

`--demo` builds a small procedural shape model (`scripts/make_demo_model.py`) so a fresh clone can run the whole interface. Its shapes are not anatomy, and the UI shows a banner while it is loaded. The real model needs the licensed HFValid data (see [Training the shape model](#training-the-shape-model)).

## The web app

| Page | What it shows |
|---|---|
| **Case** | Model choice (learned network or shape-model fit). With the network, an **AP** and an optional **Lateral** tab (add more of either with **+ AP** / **+ Lateral**, up to 3 each), each with its own upload, frame and mask settings. Upload (PNG, JPEG, TIFF, DICOM; drag and drop works anywhere), a draggable frame, live mask preview with contrast/threshold controls, and input-quality checks. The shape-model fit adds its settings: effort, shape spread, DICOM-spacing sizing with a magnification factor, and the prior weight. |
| **Inferences** | Network: each view with its mask and the estimate's outline mapped back, every view on the network grid, per-view agreement and orientation choice, the checkpoint's validation Dice, and flags. Shape-model fit: your X-ray with the traced mask and the fitted outline mapped back onto it; the canonical-frame overlap; silhouette agreement, orientation, shape typicality and assumed length (each with its meaning stated); per-case flags; the 4 orientation hypotheses; the optimization trace; shape-mode coefficients; shape spread by region; segmentation, standardization and model details. |
| **3D model** | An orbitable surface with solid, wireframe and X-ray (fresnel) modes; colouring by shape spread; an overlay of the population mean; AP, lateral, axial and oblique cameras; extents; and screenshots. Without a case it shows the mean shape. |
| **Export** | Network: `femur_reconstruction.stl` and `femur_labelmap.nrrd` (both tagged RAS so they load unchanged and overlap in Slicer), OBJ, `reconstruction.json` and the overlays; the ZIP includes `BoneVisionLoader.py`, so `Slicer --python-script BoneVisionLoader.py` in the unzipped folder opens everything. Shape-model fit: STL, OBJ, a colour PLY carrying the spread values, `report.json`, overlay images and the mask. Slicer steps and the coordinate frame for both. |
| **Method** | How each method works, what it is and is not, the reported synthetic evidence with its caveats, and data licences. |

Uploads and results are held **in server memory only**: images are never written to disk, and everything is discarded when the server stops. The server binds to `127.0.0.1` by default. The API is documented at `/api/docs`.

## Command line

```bash
python scripts/reconstruct_single_xray.py femur.png --output runs/case01 [--crop T B L R] [--use-dicom-spacing --magnification 1.1] [--device cuda]
```

This writes the same seven files as the web export. It refuses to overwrite existing outputs unless you pass `--force-overwrite`.

## How the network path works (`bonevision/network.py`)

The network is `scripts/multiview_net.py` (2D encoder per view, back-projection, 3D decoder). It was trained by `scripts/run_retraining.py` on grid-aligned synthetic DRRs, with every right femur mirrored onto the left. For an upload, `scripts/multiview_net_backend.py`:

1. **Traces** each view with the same threshold segmentation as the shape-model path.
2. **Places** it on the training grid: the mask's principal axis is matched to the mean training silhouette, its proximal end goes to the femoral head, its length is scaled to the mean training femur, and it is centred across the shaft.
3. **Picks the orientation**: head up or down, mirrored or not. The placement whose outline best matches the mean training femur wins; a small margin raises a flag.
4. **Runs the network once** (the AP + lateral checkpoint when a lateral view is given, else the AP-only one), keeps the largest solid component, and extracts the surface.
5. **Several images per direction** (up to 3 AP and 3 lateral, added with **+ AP** / **+ Lateral**): the network runs once per AP × lateral pair and the occupancies are averaged. Each image is then scored against the combined estimate, and one that agrees clearly less than the others (Dice more than 0.05 below the median) is flagged, which catches a mislabelled view or a different femur. Extra images of the same direction average out segmentation and placement noise; they do not add new angles, because the network was trained on AP and lateral only.

Held-out synthetic test (16 HFValid subjects): AP + lateral Dice 0.943, 0.95 mm mean surface distance; AP only 0.865, 2.19 mm (old SSM fit 0.687, 4.73 mm). These inputs were simulated and already grid-aligned at true length, so expect less on real radiographs. The output is always a left femur at the training-mean length. Views must be true AP and lateral of the same femur.

### Adding a new model or more data

- **Retrain with more data:** import extra femurs as training-only shapes with `scripts/import_external_femurs.py` (TotalSegmentator, VSD, NMDID), then rerun `bash scripts/train_wsl.sh` (or `python scripts/run_retraining.py --also-ap`). Keep the HFValid validation/test split fixed so new numbers compare with the ones above.
- **Swap checkpoints:** save each run under its own name with its `models/retraining_report.md`, and start the server with `--net-ap-lat` / `--net-ap`. Nothing else changes.
- **A different kind of model:** add a module like `bonevision/network.py` that returns the same result keys, and a branch for it in `bonevision/server.py` (`create_case` and `_run_case`) and `bonevision/exports.py`.

## How the shape-model pipeline works (`bonevision/pipeline.py`)

1. **Trace**: Otsu threshold, morphology, and an elongation/area/border score select one bone-like region. This is classical image processing, not a learned segmenter.
2. **Standardize**: the outline is rotated to its principal axis (a rotation, never a reflection) and converted to the model's millimetre grid. Without calibration it is sized to the population-mean length.
3. **Fit**: the voxel PCA SSM's projected silhouette is fitted by Adam to the outline under **four orientation hypotheses** (as oriented, rotated 180°, flipped left–right, flipped top–bottom), together with an in-plane nuisance scale and shift. The lowest objective wins.
4. **Surface**: marching cubes on the fitted occupancy. The fitted pose is applied to the mesh, so the exported geometry is what the overlay shows.
5. **Shape spread (heuristic)**: coefficients near the optimum are perturbed. Samples whose silhouettes fit within 0.02 Dice are kept, and their per-voxel disagreement is mapped to the surface. This shows where one view leaves geometry loose. It is not a confidence interval.

### Fixes in pipeline 0.3.0

These changes alter outputs relative to earlier runs:

- **Left–right orientation:** earlier code tried only the original image and an upside-down flip, so a contralateral leg, a PA view or a mirrored file was fitted against a backwards silhouette. On one synthetic mirrored test image the old code reached Dice 0.708 and picked the wrong flip; the new search reaches 0.859 with the correct one.
- **Aspect ratio:** the model's projection grid is anisotropic (2.0 × ~1.68 mm). Earlier code scaled in pixels, which narrowed every uploaded femur by about 16% relative to its length.
- **Reflection:** a `transpose` in standardization reflected the mask; it is now a pure rotation.
- **Overlay and mesh:** the fitted 2D pose was applied to the overlay and score but not to the STL. It is now applied to both.
- **Evaluation path:** `evaluate_single_view.py --input-mode app` now runs held-out silhouettes through the app's own standardization, which discards true size and position as an uncalibrated upload does. The default `ideal` mode reproduces the original protocol. Both modes score the posed occupancy with all four hypotheses.

## Evidence so far (synthetic only)

These are README-reported results from the earlier pipeline on 16 held-out HFValid test subjects. The input was the exact AP silhouette of each test CT, already in the model frame at its true size and position (100 iterations).

| Metric | SSM silhouette fit | One-view template baseline |
|---|---:|---:|
| Mean voxel Dice | 0.687 | 0.613 |
| Mean voxel IoU | 0.543 | 0.479 |
| Mean symmetric surface distance | 4.73 mm | 6.23 mm |
| Median surface distance | 3.35 mm | 4.48 mm |
| Mean subject P95 surface distance | 14.43 mm | 17.61 mm |
| Symmetric mean-squared Chamfer | 53.77 mm² | 96.19 mm² |
| Soft silhouette projection Dice | 0.919 | 0.944 |

The worst case (Pat099: Dice 0.373, 10.09 mm) is kept in the average. This protocol bypasses segmentation, detector geometry and clinical domain shift, and it gives the fit size information an uncalibrated upload does not have. On toy synthetic shapes, switching to `--input-mode app` lowered 3D Dice from 0.92 to 0.65. Re-run both modes with the real model before quoting any of these numbers:

```bash
python scripts/evaluate_single_view.py --iterations 100 --input-mode ideal --output reconstruction/sv_ideal_v030.csv
python scripts/evaluate_single_view.py --iterations 100 --input-mode app   --output reconstruction/sv_app_v030.csv
```

The multiview template-retrieval baseline (8 synthetic DRR views, 70 candidates) is described in [`docs/multiview-reconstruction-audit.md`](docs/multiview-reconstruction-audit.md), together with the known mesh-SSM correspondence problem. That problem affects the **mesh** SSM and the multiview path, not the voxel SSM used by this app.

## Training the shape model

The current model is PCA on CT-derived occupancy grids (CPU, scikit-learn), not a CNN.

1. Obtain the HFValid Collection (CC BY 4.0) and stage raw meshes and landmarks under `data/raw/`.
2. Build the targets in a clean copy: `python scripts/build_highres_targets.py`, which writes `data/occupancy_highres/`.
3. Build the model: `python scripts/build_ssm_v2.py`, which uses the 70 training subjects in `data/subject_split.csv` and writes `models/ssm_highres.npz`. It refuses to overwrite unless you pass `--force-overwrite`.

For research scripts, install `requirements-research.txt`. Colab works if you stage the data there yourself; licences still apply.

## Repository layout

```
serve.py                     start the web app
bonevision/network.py        learned-network path (wraps scripts/multiview_net_backend.py)
bonevision/pipeline.py       read, segment, standardize, fit, surface, overlay
bonevision/exports.py        STL / OBJ / PLY / PNG / JSON report / ZIP (bytes, no disk I/O)
bonevision/server.py         FastAPI backend (in-memory uploads and cases)
web/                         frontend (vanilla JS + vendored three.js and fonts; works offline)
scripts/reconstruct_single_xray.py   CLI over the package
scripts/evaluate_single_view.py      held-out synthetic evaluation (ideal / app modes)
scripts/build_ssm_v2.py, build_highres_targets.py   model training
scripts/make_demo_model.py           synthetic demo shape model for UI testing
scripts/make_demo_network.py         synthetic demo networks for UI testing
scripts/run_retraining.py, train_wsl.sh   one-command network retraining (docs/retraining-plan.md)
scripts/multiview_net*.py, train_multiview_net.py, build_biplanar_dataset.py   the network and its data
slicer/BoneVisionLoader/     3D Slicer loader for exported folders
scripts/*drr*, *retrieval*, *mesh*   CT/DRR/multiview research scripts; several are one-off
                                     Pat001 explorations with hard-coded paths
docs/multiview-reconstruction-audit.md
tests/                       python -m unittest discover -s tests
legacy/                      earlier YOLO experiments and the superseded v1 SSM
```

## Data and licences

Large CT volumes, meshes, DRRs, targets, model weights and generated results are excluded through `.gitignore`. Keep them out of Git history.

- **HFValid Collection** (CC BY 4.0): 101 CT femurs, segmentations and landmarks; the model's source.
- **xvr-data Femur subset** (CC BY-NC-ND 4.0, revision `a17273e3…`): five paired real X-ray/CT cases, stored unmodified under `data/external/xvr-femur/`. Not used for training or the reported metrics. Five cases cannot establish clinical performance. See the [dataset card](https://huggingface.co/datasets/eigenvivek/xvr-data/blob/main/README.md) and [xvr](https://github.com/eigenvivek/xvr).
- **VSD lower-extremity meshes** (CC BY-NC-SA 4.0, revision `85214b5a…`): 60 femur STLs under `data/external/vsd-femur-meshes/`. **Not** in the model, because they lack registered correspondence. Cite the [dataset](https://huggingface.co/datasets/BoneHub/vsd-lower-extremities-seg) (DOI `10.57967/hf/10471`) and [Fischer et al.](https://doi.org/10.1038/s41597-023-02669-z).

Vendored frontend assets: three.js (MIT) and the Outfit and Instrument Serif fonts (SIL OFL). Their licence files are in `web/vendor/`. The project itself has no licence file yet; choose one before publishing.
