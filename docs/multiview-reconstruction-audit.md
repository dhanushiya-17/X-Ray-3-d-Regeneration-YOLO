# Multi-view reconstruction audit

## Intended workflow

BoneVision will use multiple views of the same femur to fit a 3D shape model,
then export the fitted surface as an STL or PLY model for inspection in 3D
Slicer. The reproducible project pipeline currently uses synthetic DRRs and
CT-derived reference anatomy. A small public real-X-ray/CT femur cohort exists,
but it has not been integrated into this model or its evaluation.

## Existing synthetic projections

`scripts/batch_generate_drr.py` creates eight DRRs per subject at angles
0, 45, 90, 135, 180, 225, 270, and 315 degrees. The configured detector is
512 by 512 pixels, pixel spacing is 1.5 mm, source-to-detector distance is
1020 mm, and the configured translation is (0, 850, 0). The script masks and
crops the original CT around the femur, then saves each rendered projection as
a min-max normalized PNG.

The installed DiffDRR reader defaults to `orientation="AP"` and
`center_volume=True`: it converts the volume to RAS+, applies its AP frame
change, and shifts the volume isocenter to the world origin before rendering.
The generator passes the view angle as the first Euler angle with the `ZXY`
convention. These operations are part of the effective camera-to-volume
mapping and must be included when mapping the SSM's anatomical coordinates to
the DRR camera frame.

These images are useful for synthetic experiments. PNG values are normalized
for display, and the script does not save a per-image camera/volume transform
sidecar alongside each view. The render configuration is available in source,
but a reconstruction tool must reproduce the effective DiffDRR coordinate
conventions rather than assuming the PNG alone contains calibration.

## Coordinate compatibility risk

The occupancy SSM is built from meshes transformed by anatomical landmarks
into a femur-centered frame. The batch DRR generator instead renders a cropped
CT while preserving the CT spacing, origin, and direction. The repository has
`scripts/resample_anatomical_ct.py`, but it currently configures one subject
(Pat001), writes a separate anatomical CT, and is not the batch DRR generator
path.

Therefore the existing DRR PNGs and occupancy SSM targets must not yet be
assumed to share a known camera-to-anatomical transform. This must be resolved
and verified before using silhouette fitting against those images or reporting
their reconstruction accuracy. Existing V2 results that optimize silhouettes
generated directly from target occupancy are oracle capacity experiments,
not image-input reconstruction results.

## Evaluation boundary

Keep all views from one subject in that subject's train, validation, or test
split. Use training subjects to fit the SSM, validation subjects to select
settings, and test subjects only for final evaluation. Record failed fits.
Report surface distances in millimetres and voxel overlap only when prediction
and reference occupy the same physical grid. Do not describe synthetic DRR
results as validation on clinical radiographs.

## Held-out retrieval baseline

The existing multi-view retrieval baseline was run on the 16 test subjects,
using their eight synthetic views to select among the 70 training subjects.
It selected a training template mesh for each query and compared the meshes
after applying the existing landmark-based anatomical alignment. Symmetric
surface distances use area-weighted random surface samples (2,000 per mesh)
and VTK closest-cell queries. This is a sampled template-shape comparison;
it does not deform a template into a patient-specific surface.

| Test summary | Result |
|---|---:|
| Subjects | 16 |
| Mean subject-level symmetric mean surface distance | 3.48 mm |
| Median subject-level symmetric mean surface distance | 3.28 mm |
| Standard deviation across subject-level means | 0.97 mm |
| Mean subject-level 95th-percentile surface distance | 9.09 mm |
| Worst subject-level mean surface distance | 5.64 mm |

Per-subject matches and metrics are in
`reconstruction/retrieval_baseline/multiview_test_matches.csv` and
`reconstruction/retrieval_baseline/multiview_test_3d_metrics.csv`. These are
results on synthetic data with known generated views, not clinical accuracy.
The earlier validation summary used 10,000 samples per mesh; the test summary
uses 2,000, so treat the split summaries as approximate rather than as a
strictly identical sampling protocol.

## Next verification steps

1. Rework or replace the mesh CPD correspondence before trusting the mesh SSM.
   Gate registrations with 3D surface distances and projection silhouettes.
2. For real X-rays, obtain or estimate calibrated camera geometry before
   fitting SSM parameters. The current dataset has known synthetic poses;
   ordinary image files do not contain enough information to infer physical
   scale and view geometry reliably.
3. After geometry is validated, implement patient-shape fitting and evaluate
   it on held-out synthetic subjects before claiming clinical performance.

## Projection and coordinate check

The existing read-only check is `scripts/test_drr_views.py`. It regenerates
the eight Pat001 projections from its saved femur-only CT and compares them
with the stored PNGs. To keep memory use bounded it uses DiffDRR patch rendering
at size 64. It saves no images. All eight views pass the check with correlation
0.99999 and normalized mean absolute error of 0.00005–0.00006.

The physical-coordinate relationship for the SSM is now explicit. Let `p_LPS`
be a point in the source CT/STL frame, `O` the femoral-head landmark, and `R`
the landmark frame whose columns are the longitudinal, trochanter, and cross
product axes. The SSM point is `q = R.T @ (p_LPS - O)`, and maps back as
`p_LPS = O + R @ q`. DiffDRR reads the cropped NRRD in RAS coordinates,
centers the volume, then applies the AP detector reorientation and requested
camera pose. Thus the corresponding centered volume point is
`p_world = diag(-1, -1, 1) @ p_LPS - c_RAS`, where `c_RAS` is the cropped
volume's RAS isocenter. Projection regeneration verifies the saved CT/camera
path. The separate registered-mesh comparison below checks whether the SSM
geometry lands on the same projected anatomy.

That direct check was run on training subject Pat003. The original femur STL,
transformed from its own landmark frame back to LPS and through the CT crop
center into the DiffDRR camera, overlaps the stored thresholded DRR silhouette
at IoU 0.962, 0.954, and 0.958 for views 0°, 90°, and 270°. The registered
SSM STL for the same subject reaches only 0.688, 0.261, and 0.258 under the
same projection. Its symmetric nearest-vertex distances to the original
aligned mesh are about 12 mm mean in both directions (95th percentiles about
27–29 mm). These are diagnostic comparisons on synthetic imagery, not accuracy
claims. They localize the current failure to mesh CPD correspondence: the
anatomical-to-camera transform itself is consistent with the stored images.

## Additional public data reviewed

The local HFValid collection already contains 101 calibrated whole-femur CTs,
segmentations, and landmarks under CC BY 4.0. It is the best-aligned source for
the current pipeline and should not be downloaded a second time.

A small public paired cohort is available in the Femur subset of
[`eigenvivek/xvr-data`](https://huggingface.co/datasets/eigenvivek/xvr-data/blob/main/README.md): five cases with DICOM/NIfTI X-rays and CT-derived volumes/masks, as used by the
[xvr project](https://github.com/eigenvivek/xvr). The Femur subset retains a
CC BY-NC-ND 4.0 license. An unmodified local copy is stored at
`data/external/xvr-femur/` and excluded from Git. It has not been used to train
the SSM or generate reported metrics. Follow the original terms and citation;
five cases are not enough to establish general clinical accuracy. A published biplanar-reconstruction
study generated 525 radiograph pairs from CT scans of 175 patients, but its
page does not provide a downloadable dataset. The project's current eight-view
DRRs are synthetic; more images from those same 101 CT subjects would add
view variation, not independent anatomy.

Two possible sources of additional shapes are:

* [VSD lower-extremity segmentation dataset](https://huggingface.co/datasets/BoneHub/vsd-lower-extremities-seg): 30 additional cadaver CTs with femur
  label maps and meshes. Its dataset card reports 9.21 GB total size and a
  CC BY-NC-SA 4.0 license. The project is acquiring the mesh-only subset into
  `data/external/vsd-femur-meshes/`; the CT volumes are not part of this
  download. These STL shapes still need anatomical registration and checks
  before they can be used as training subjects or for DRR generation.
* [Xray-to-3D benchmark femur preparation notes](https://github.com/naamiinepal/xrayto3D-benchmark/blob/main/docs/dataset.md): the benchmark reports 451 femur
  segmentations derived from TotalSegmentator and scripts for making biplanar
  synthetic projections. The notes explicitly mention cropped bones; this is
  a candidate to inspect for complete-femur coverage, not 451 verified usable
  whole femurs or 451 real radiograph pairs.

The [University of Denver anatomical-variation dataset](https://digitalcommons.du.edu/anatomical_variation_knee/1/)
has meshes and SSM registrations for 102 subjects, but supplies proximal and
distal femur pieces rather than whole-femur radiograph pairs, and its license
is CC BY-NC 4.0. It is less directly useful for the current full-femur
reconstruction than the VSD CT collection.

Keep external source data out of the existing HFValid directories to avoid
mixing or duplicating it. The next data work is to register the VSD mesh-only
shapes without leakage and to assess whether the restricted five-case XVR
Femur subset can be used for a noncommercial external evaluation. Extra CT
subjects can improve shape diversity, but they do not by themselves establish
accuracy on clinical radiographs.

## Single-radiograph feasibility and data gap

Single-view femur reconstruction is a research problem with published approaches; it is not solved by the current repository. A 2024 paper describes predicting SSM deformation parameters from a single, designated-position femur X-ray, but its reported errors are study-specific and do not validate this project's model. A separate 2024 3DDX paper reports a dual-face depth network trained using 600 CTs and 2,651 paired X-rays (4–5 views per patient), with a reported 1.96 mm surface error. Its public GitHub repository currently provides loss code, while the README says the data/model release is limited by intellectual-property processes. This is not a downloadable training set for BoneVision.

Primary references:

* [Single-X-ray femur reconstruction using deep transfer learning (2024)](https://doi.org/10.1016/j.irbm.2024.100822)
* [3DDX paper (MICCAI 2024)](https://arxiv.org/abs/2409.16702) and its [public code repository](https://github.com/Kayaba-Akihiko/3DDX)
* [Distal femur reconstruction from two or more calibrated X-rays using an SSM (2011)](https://doi.org/10.1016/j.media.2011.04.001)

The practical options are:

1. For the requested single-image interface, train a model that predicts a complete 3D shape from a *standard-geometry, known-view* radiograph. HFValid can produce synthetic training pairs. A small five-case real-X-ray/CT Femur cohort exists in xvr-data, but it has not been used by BoneVision and is too small to establish generalization. Treat output as a research estimate until evaluated against unseen CTs and a sufficiently powered real paired cohort.
2. For the more constrained reconstruction path, use two calibrated views (e.g., AP and lateral) and fit an anatomically aligned shape model to both silhouettes/edges. Published SSM approaches use calibrated stereo views; the present mesh correspondence failure must be repaired before using this project's mesh SSM.

In either path, a screenshot or ordinary PNG does not reliably carry the projection geometry or physical scale. The inference interface must receive or enforce the view, scale/calibration, and bone region. The current `scripts/ssm_reconstruct.py` does not implement that contract: it is a synthetic split experiment with hard-coded view and target paths, and its image resizing is not a physical camera model. It must not be used as an arbitrary-X-ray-to-STL command.
