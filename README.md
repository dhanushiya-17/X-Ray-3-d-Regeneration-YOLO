# BoneVision: X-Ray-Based 3D Bone Reconstruction

BoneVision is a research project exploring the reconstruction of
3D bone anatomy from medical imaging, with a current focus on the
femur. The repository also retains the earlier YOLO-based X-ray
experiments.

## Research objective

Develop a reproducible pipeline that uses CT-derived femur geometry
to generate synthetic X-ray projections and investigate methods for
reconstructing 3D anatomy from 2D images.

## Dataset

The current femur experiments use the HFValid Collection, which
contains 101 subjects with CT scans, femur segmentations, and
anatomical landmark data.

The dataset is not included in this repository. Obtain it from its
official source and follow its licence and usage conditions.

## Pipeline

1. Prepare and validate femur surface meshes.
2. Extract femur masks and femur-only CT volumes.
3. Generate synthetic digitally reconstructed radiographs (DRRs).
4. Organize experiments using subject-level training, validation,
   and test splits.
5. Evaluate geometry and investigate multiview and retrieval-based
   reconstruction baselines.

## Repository structure

- `scripts/prepare_meshes.py` — mesh preparation
- `scripts/check_femur_geometry.py` — geometry checks
- `scripts/clean_mesh_components.py` — mesh component cleanup
- `scripts/create_dataset_qc.py` — dataset quality checks
- `scripts/batch_generate_drr.py` — batch DRR generation
- `scripts/generate_drr.py` — DRR generation
- `scripts/generate_drr_diffdrr.py` — DiffDRR-based experiments
- `scripts/generate_drr_full_femur.py` — full-femur projection experiments
- `scripts/multiview_retrieval_baseline.py` — multiview retrieval baseline

The repository also contains earlier YOLO-related scripts and experiments.

## Environment

Use a dedicated Python environment. Install dependencies according to
the requirements of the selected scripts and the versions of the
imaging libraries used in your setup.

Before running an experiment, verify the input paths, image geometry,
voxel spacing, and output directory.

## Data management

Large CT volumes, source datasets, meshes, generated DRRs, and
reconstruction outputs are excluded through `.gitignore`. Keep
medical datasets and large generated files out of Git history.

## Evaluation

Reconstruction quality should be assessed against held-out ground-truth
geometry using appropriate 3D metrics, with subject-level separation
to prevent data leakage. Report the metrics, test protocol, and
limitations alongside any results.

## Current status

The repository contains femur mesh preparation, geometry quality
checks, CT mask processing, synthetic DRR generation, baseline
experiments, and a research single-X-ray reconstruction interface.
The single-view workflow has been evaluated on held-out synthetic
canonical silhouettes; it has not been validated on real radiographs.
See the later single-view application section for the protocol,
results, and limits.

## Responsible use

This is a research project, not a clinically validated diagnostic
system. It must not be used as a substitute for professional medical
assessment.

## Baseline evaluation results

The multiview template-retrieval baseline was evaluated on 15 held-out
validation subjects, comparing eight synthetic DRR views against 70
training candidates.

| Metric | Result |
|---|---:|
| Mean subject-level mean surface distance | 4.72 mm |
| Median subject-level mean distance | 5.40 mm |
| Standard deviation | 1.71 mm |
| Mean subject-level 95th-percentile distance | 14.72 mm |

These are preliminary template shape-comparison results, not measurements
of a newly reconstructed patient-specific femur. Coordinate-frame checks
passed for all 101 subjects, but full mesh-to-landmark registration
accuracy has not yet been established. The results do not establish
clinical validity.

Results are saved in
`reconstruction/retrieval_baseline/multiview_validation_matches.csv` and
`reconstruction/retrieval_baseline/multiview_validation_3d_metrics.csv`.

## Single-X-ray application and training

The one-image demonstration is `app.py`. It accepts a single AP-like femur PNG/JPEG/TIFF or DICOM, offers crop, contrast-polarity, and Otsu-threshold controls, fits the CT-trained voxel SSM to a normalized silhouette, and displays the resulting mesh in a rotatable shaded or wireframe 3D viewer. It exports an STL (open in 3D Slicer with **File → Add Data**), a silhouette comparison, and a JSON run report. The 2D comparison is an SSM line-integral silhouette, not a clinical DRR. The fit is a population-shape estimate from a thresholded contour; a one-view image cannot uniquely determine hidden depth or absolute scale.

The command-line path creates the same three outputs without the web UI:

```bash
./femur3d/bin/python scripts/reconstruct_single_xray.py /path/to/femur_xray.png --output /path/to/new_output_folder --device cuda
```

The command refuses to replace existing STL, JSON, or PNG outputs. Choose a new output folder, or pass `--force-overwrite` only when replacement is intended. In Slicer, use **File → Add Data** and select `femur_reconstruction.stl`; the STL is in the model's canonical millimetre frame and does not encode a patient-specific RAS/LPS orientation.

The current SSM is built from the 70 training subjects in the saved 70/15/16 split. The prepared local working copy has the generated artifact at `models/ssm_highres.npz`. Model files and CT-derived data are excluded from Git, so a fresh clone must obtain and prepare the required data and build a compatible model before running inference. To start the local interface, install the optional UI dependencies and launch Streamlit:

```bash
./femur3d/bin/python -m pip install -r requirements-app.txt
./femur3d/bin/python -m streamlit run app.py
```

For a new training run on a clean laptop/Colab working copy, stage the existing 101 subject-level `data/occupancy_highres/*.npy` CT-derived targets and `data/subject_split.csv`, then run:

```bash
python scripts/build_ssm_v2.py
```

This is PCA statistical-shape-model training, not a CNN. The higher-resolution target builder requires the raw HFValid meshes and landmarks; run `scripts/build_highres_targets.py` only in a clean training copy because it writes the occupancy target directory. The SSM builder refuses to replace existing model artifacts unless `--force-overwrite` is explicitly passed. The application uses a CUDA-enabled PyTorch build when available; CPU fallback is available.

Run the held-out single-view experiment with:

```bash
python scripts/evaluate_single_view.py --iterations 100
```

It compares the SSM fit with a one-view nearest-training-template baseline and writes per-subject metrics plus an aggregate row to `reconstruction/single_view_test_metrics.csv`. It keeps the saved test subjects held out from SSM fitting. If those output files already exist, the command stops before running; choose a new `--output` or deliberately pass `--force-overwrite`. This experiment uses the test CT occupancy's exact AP silhouette already in the SSM frame. It evaluates 3D shape fitting under ideal segmentation and known geometry; it does **not** validate automatic segmentation or accuracy on real X-rays. Surface distances are sampled from voxelized surfaces using each axis's millimetre spacing; Chamfer is the symmetric mean squared surface distance in mm². No ground-truth metrics are shown for an ordinary uploaded image.

### Single-view limits

The inference interface estimates hidden depth and physical size from the learned population prior. Plain PNG/JPEG images do not contain enough information to recover absolute scale, and the current DICOM reader uses pixel-value rescale metadata but does not infer source/object calibration. Projection overlap is a fit diagnostic, not a 3D-accuracy measurement. The available 101 HFValid CTs generate synthetic training/evaluation data. A separate public Femur subset in `eigenvivek/xvr-data` contains real X-ray/CT data for five cases and is licensed CC BY-NC-ND 4.0. An unmodified local copy is stored under `data/external/xvr-femur/` and excluded from Git. It has not been used to train the SSM or to produce the reported evaluation results. Its license restricts commercial use and redistribution of adapted material; follow the source terms and citation. Five cases do not replace a sufficiently powered clinical test cohort. Do not use these estimates for diagnosis or treatment planning.

### Additional femur data and provenance

The project has downloaded only the 60 femur STL meshes (left and right femurs for 30 subjects) from `BoneHub/vsd-lower-extremities-seg` at revision `85214b5a182b980a7869b57efba89a59a2445875` into `data/external/vsd-femur-meshes/`. Its source dataset contains 30 lower-extremity CT subjects and their labeled bone meshes; this project download is the mesh subset. The dataset license is CC BY-NC-SA 4.0. These meshes are kept out of Git and have **not** been added to the SSM: they do not provide the registered landmarks/canonical correspondence required by the current PCA builder. They must not be described as extra training subjects until a registration and evaluation pipeline is implemented. Cite [the dataset](https://huggingface.co/datasets/BoneHub/vsd-lower-extremities-seg) (DOI `10.57967/hf/10471`) and its source paper, Fischer et al., [Database of Segmentations and Surface Models of Bones of the Entire Lower Body Created from Cadaver CT Scans](https://doi.org/10.1038/s41597-023-02669-z).

The five-case Femur subset is listed in the [xvr-data card](https://huggingface.co/datasets/eigenvivek/xvr-data/blob/main/README.md) and used by the [xvr project](https://github.com/eigenvivek/xvr). The local copy is pinned to revision `a17273e3eadbd793bd861f3598a80ce4590c1124`. Its CC BY-NC-ND 4.0 terms restrict commercial use and redistribution of adapted material. The source files are kept unmodified outside Git.

### First held-out single-view result (synthetic ideal-mask protocol)

The 16 test subjects were evaluated with 100 maximum fitting iterations. Input was the exact AP silhouette projected from each held-out CT occupancy in the SSM anatomical frame; automatic X-ray segmentation, real detector geometry, and clinical-image domain shift are not included in this score.

| Metric | SSM silhouette fit | One-view template baseline |
|---|---:|---:|
| Mean voxel Dice | 0.687 | 0.613 |
| Mean voxel IoU | 0.543 | 0.479 |
| Mean symmetric surface distance | 4.73 mm | 6.23 mm |
| Median surface distance | 3.35 mm | 4.48 mm |
| Mean subject P95 surface distance | 14.43 mm | 17.61 mm |
| Symmetric mean-squared Chamfer | 53.77 mm² | 96.19 mm² |
| Soft silhouette projection Dice | 0.919 | 0.944 |

The SSM fit improves mean 3D geometry over the single-view template baseline in this idealized protocol, even though the template baseline has slightly higher 2D silhouette overlap. Results vary substantially: the worst SSM test case (Pat099) has Dice 0.373 and mean symmetric surface distance 10.09 mm. The low end is a real failure, not removed from the average. These synthetic results are suitable for a research demonstration, not a claim of high accuracy on arbitrary clinical X-rays.

Per-subject metrics are in `reconstruction/single_view_test_metrics.csv`; the corresponding summary figure is `reconstruction/single_view_test_metrics_summary.png`. The test split was not used to fit the SSM or choose the fitting settings.
