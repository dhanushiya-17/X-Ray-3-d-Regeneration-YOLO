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
checks, CT mask processing, synthetic DRR generation, and baseline
experiments. A validated end-to-end single-X-ray-to-3D reconstruction
system remains a research objective; results should be documented
only after reproducible evaluation.

## Responsible use

This is a research project, not a clinically validated diagnostic
system. It must not be used as a substitute for professional medical
assessment.
