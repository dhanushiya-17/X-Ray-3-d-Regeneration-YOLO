# Femur retraining plan

This plan answers three questions: why the current SSM gives weak results,
which approach to retrain with, and where more femur data can come from. The
new scripts it refers to are in `scripts/` and have been smoke-tested on
synthetic femur-like volumes only. None of them has been run on HFValid yet,
so no new accuracy numbers are claimed here.

## 1. Why the current model underperforms

| Issue | Where | Effect |
|---|---|---|
| PCA is run on **binary occupancy** voxels | `build_ssm_v2.py` | Each mode is a blend of 0/1 grids. Reconstructions are fuzzy and must be thresholded, and thin structures (neck, condyles) wash out. On held-out shapes the model's capacity is limited even before any X-ray is involved. |
| Fitting uses **one thresholded silhouette** | `reconstruct_single_xray.py` | The AP outline does not constrain anteversion, anterior bow or condyle depth, so the prior fills these in. The README's own test shows a 0.69 Dice ceiling with a perfect mask, and a worst case of 0.37. |
| Otsu thresholding on real radiographs | same | Overlapping pelvis, soft tissue and the other leg break the mask, and every later step inherits the error. |
| Mesh SSM correspondence is broken | `build_mesh_ssm.py` (CPD) | The audit measured about 12 mm mean error on a training subject, so the mesh SSM cannot be used yet. |
| 70 training shapes | split | This is small for PCA with 60 modes, and very small for a neural network. |
| Possible side mixing | landmark frame | The frame uses `z = x × y`. If HFValid contains both left and right femurs, anterior and posterior swap between sides, and the model averages mirror-image shapes. **Check this first**, for example by plotting the anterior-bow direction for each subject, and mirror one side if needed. |

The most important finding is that the method family is the limit, not the
training run. Retraining the same PCA-on-occupancy model with the same data
will give about the same result.

## 2. What has worked elsewhere

| Work | Input | Femur result (as reported) | Code |
|---|---|---|---|
| [xrayto3D benchmark, NeurIPS 2023](https://arxiv.org/abs/2309.13587) | Biplanar DRRs, TotalSegmentator femurs | AttentionUNet Dice 93.7%, HD95 3.1 mm, ASD 0.93 mm (SwinUNETR is similar) | Benchmark framework (naamiinepal) |
| [Swin-X2S, 2025](https://arxiv.org/abs/2501.05961) | Biplanar DRRs, 571 TotalSegmentator CTs with femur | Dice 92.3%, HD95 4.8 mm | [MIT, github.com/liukuan5625/Swin-X2S](https://github.com/liukuan5625/Swin-X2S) |
| [3DDX, MICCAI 2024](https://arxiv.org/abs/2409.16702) | Real X-rays (600 CTs, 2,651 paired radiographs) | 1.96 mm surface error | Loss code only; data and model not released |
| [Single-X-ray femur, IRBM 2024](https://doi.org/10.1016/j.irbm.2024.100822) | One standardized radiograph to SSM parameters | Study-specific | Not public |
| [Calibrated 2-view SSM fitting, MedIA 2011](https://doi.org/10.1016/j.media.2011.04.001) | Two or more calibrated X-rays | Classical baseline | Not public |

All of the learned results are on **synthetic DRRs** except 3DDX. The
xrayto3D femurs are 128³ crops that are often partial, so their numbers are
not directly comparable to full-length femurs.

Three findings from these papers apply directly here:

- **Two views beat one by a wide margin.** The lateral view is what fixes
  anteversion and bow. Keep the single-view mode, but treat AP plus lateral
  as the default.
- **The common pattern is a 2D encoder per view, a back-projection to 3D, and
  a 3D decoder.** Attention and global context help, and small models
  (AttentionUNet, about 1.5 M parameters) do as well as large ones.
- **View misalignment is the main failure mode.** A lateral view rotated 1–10°
  cost about 6 Dice points in xrayto3D, so calibration and pose matter as
  much as the architecture.

## 3. Recommended retraining path

### Stage A: SDF shape model (quick, same data, no raw CT needed)

`scripts/build_ssm_sdf.py` runs PCA on truncated signed-distance fields
computed from the existing `data/occupancy_highres/*.npy`, which avoids the
blurred-blend problem. It also writes a held-out oracle comparison against
occupancy PCA at the same number of modes, so you can see whether it helps
before changing the app.

```bash
python scripts/build_ssm_sdf.py                # writes models/ssm_sdf.npz + models/ssm_sdf_oracle.csv
```

This step runs on the CPU and takes a few minutes. Adopting it in the app
needs one change in the fitter: decode `occ = sigmoid(-sdf / 1.5mm)` instead
of clamping occupancy. That change is not part of this PR.

### Stage B: multi-view network (the main model)

1. Build grid-aligned training pairs. The full CT is resampled into the same
   landmark frame as the targets, so every AP and lateral pixel lines up with
   a column of the 3D target. This removes the camera-frame mismatch
   described in the audit by construction.

   ```bash
   python scripts/build_biplanar_dataset.py       # data/biplanar/<subject>.npz
   ```

2. Train with the saved 70/15/16 split. Validation picks the checkpoint; the
   test split is only used at the end.

   ```bash
   python scripts/train_multiview_net.py --views ap,lat --downsample 2 --epochs 300
   python scripts/train_multiview_net.py --views ap --downsample 2 --epochs 300   # single-view model
   python scripts/train_multiview_net.py --evaluate test --checkpoint models/multiview_ap_lat_d2.pt
   ```

   The model has 2.4 M parameters. On an 8 GB card, `--downsample 2`
   (128×48×48, about 4 mm voxels) is the fast setting. `--downsample 1 --batch-size 1`
   uses the full 2 mm grid with mixed precision. GPU memory and speed have not
   been measured here. For the RTX 5060 (Blackwell), install a PyTorch build
   for CUDA 12.8 or newer.

3. Compare the result with the README's single-view SSM numbers and the
   multi-view retrieval baseline, using the same `metrics()` function. Keep
   failures in the table.

4. Optional: fit the SDF SSM to the network's prediction. This gives a
   guaranteed anatomically plausible surface and a compact shape vector to
   show in the app.

### Stage C: more data (see section 4)

```bash
python scripts/import_external_femurs.py --totalseg-root /path/to/Totalsegmentator_dataset_v201
python scripts/import_external_femurs.py --meshes data/external/vsd-femur-meshes --mesh-source vsd --output data/external_occupancy_vsd
python scripts/build_biplanar_dataset.py --manifest data/external_occupancy/manifest.csv
python scripts/build_ssm_sdf.py --extra-train-dir data/external_occupancy --force-overwrite
python scripts/train_multiview_net.py --extra-train-dir data/biplanar_external
```

The import script rejects partial femurs, aligns each one rigidly to the mean
HFValid femur (mirroring left femurs automatically), and records the
residual and transform in `manifest.csv`. External cases are training-only,
so validation and test stay HFValid and remain comparable with earlier runs.
Spot-check overlays for a sample of imported cases before trusting the batch.

### Stage D: from synthetic DRRs to real radiographs

The network trained in Stage B has only seen parallel-beam DRRs. Before it is
given real images, it needs the following:

- **Realistic rendering and pose augmentation.** Re-render with cone-beam
  geometry (the existing DiffDRR set-up, or DeepDRR for scatter and noise),
  with random ±5–10° rotation, magnification 1.05–1.25 and detector offsets.
  The inputs then also need to be rectified back to the grid, or the network
  needs a pose input.
- **A real-image front end.** Train a femur segmentation and landmark model on
  real radiographs. The repo's earlier YOLO work is a good start, and a
  YOLO-seg or U-Net model is enough. Use it to crop the femur, set the
  orientation and estimate scale. Scale needs a calibration marker (the 25 mm
  ball used in orthopaedic templating) or DICOM pixel spacing together with a
  known magnification.
- **Pose estimation for two views.** Estimate the relative pose of AP and
  lateral, for example with [xvr](https://github.com/eigenvivek/xvr)-style
  2D/3D registration against the predicted shape, and iterate.
- **External real test.** Use the five xvr femur cases (CC BY-NC-ND,
  noncommercial) as a first real-image check. Five cases can show a failure;
  they cannot show accuracy.

## 4. Public femur data

| Source | What | Size | Licence and access | Use here |
|---|---|---|---|---|
| HFValid (in use) | CT, segmentations, landmarks | 101 | CC BY 4.0 | Train, validation, test |
| [TotalSegmentator v2 dataset](https://github.com/wasserth/TotalSegmentator) | CT with `femur_left` and `femur_right` labels | 1,228 CTs. Many are abdominal crops, so the full-length femur count is much lower. xrayto3D kept 465 usable (including partial) | CC BY 4.0 (Zenodo) | Extra training shapes and DRRs via `import_external_femurs.py` |
| [VSD lower body, Sci Data 2023](https://doi.org/10.1038/s41597-023-02669-z) | Cadaver CT and bone meshes | 30 subjects, 60 femurs | CC BY-NC-SA 4.0 ([Hugging Face](https://huggingface.co/datasets/BoneHub/vsd-lower-extremities-seg)) | Meshes already local. Download the CTs to make DRRs |
| [NMDID](https://nmdid.unm.edu/) | Whole-body CT of decedents, 1 mm slices | More than 15,000 | Free, but requires a research request, an institutional email and a data use agreement | The largest source of full-length femurs. Segment them with the TotalSegmentator tool, then import |
| [xvr-data Femur](https://huggingface.co/datasets/eigenvivek/xvr-data) | Real X-ray and CT pairs | 5 | CC BY-NC-ND 4.0 | External real test only |
| CTPelvic1K, Denver knee SSM | Proximal or distal femur only | — | Various | Not suitable for full-length femurs |

Expanding from 101 to several hundred CT femurs is realistic with
TotalSegmentator and NMDID. Shape diversity helps every stage. It does not
replace paired real X-ray data for measuring real-world accuracy.

## 5. About clinical use

Nothing in this repository is clinically validated, and these changes do not
make it so. A clinically usable system would need all of the following:

- A paired real cohort: radiographs and CT of the same patients, collected
  under ethics approval, from more than one site and scanner, and kept
  separate from training.
- Pre-defined accuracy criteria tied to the intended use, such as templating
  or implant sizing, with clinical measurements (head radius, neck-shaft
  angle, anteversion, length) as well as Dice and surface distance.
- Failure detection and an uncertainty indication for each case.
- A regulatory route for software as a medical device (for example FDA 510(k)
  or De Novo, or EU MDR) and a quality-management process.

Several of the datasets above are non-commercial (VSD, xvr, and NMDID's data
use agreement). Keep track of which sources each trained model has seen,
because that limits how the model can be used.
