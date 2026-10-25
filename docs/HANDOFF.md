# BoneVision handoff

Last updated 2026-10-25. Repository: `dhanushiya-17/X-Ray-3-d-Regeneration-YOLO`, branch `main`.
Owner: Mahadev. A compressed version of this file for loading into a new AI chat is in
[`HANDOFF_COMPACT.md`](HANDOFF_COMPACT.md).

---

## 1. What the project is

BoneVision takes one or more plain X-rays of a **femur** and estimates its **3D shape**. The output can
be orbited in the web app and opened in **3D Slicer** (STL surface plus NRRD labelmap, both in RAS).

**End goal (owner's words, summarised):** a clinically usable 3D reconstruction. A doctor uploads
X-rays of a suspected fracture, including ones that are hard to see on the film, and gets a detailed 3D
view to inspect. Planned extras:

- synthetic fracture generation on 3D bones,
- a YOLO fracture detector trained on the resulting X-rays,
- explainable AI that analyses the fractured 3D bone,
- further clinical features.

**Status today:** a working research prototype. It reconstructs **intact** femurs from AP plus lateral
X-rays. It has been checked only on **simulated** X-rays, not on real patients. It does not model
fractures yet.

---

## 2. History (what was built, in order)

| Step | What | Where |
|---|---|---|
| Legacy | 2D YOLO fracture experiments (FracAtlas split, GRAZPEDWRI wrist counts), first low-res SSM | `legacy/yolo/`, `legacy/ssm_v1/` |
| PR #1 | 3D Slicer export: `femur_reconstruction.stl` (header `SPACE=RAS`), `femur_labelmap.nrrd`, Slicer loader module | `scripts/slicer_export.py`, `slicer/BoneVisionLoader/` |
| PR #2 | Multi-view SSM fitting backend (Streamlit era, superseded) | `scripts/reconstruct_multiview_xray.py`, `scripts/reconstruction_backends.py` |
| PR #3 | Retraining plan: SDF shape model plus a multi-view network | `docs/retraining-plan.md` |
| PR #5 | One-command retraining pipeline; right femurs mirrored onto the left | `scripts/run_retraining.py`, `scripts/train_wsl.sh`, `scripts/check_femur_sides.py`, `scripts/femur_grid.py` |
| Training run | AP+lateral net: test Dice **0.943**, mean surface distance 0.95 mm, HD95 3.45 mm. AP-only net: Dice 0.865, 2.19 mm. Old SSM: Dice 0.687, 4.73 mm. | Checkpoints exist only in the WSL clone |
| PR #6 | Network added as a reconstruction backend | `scripts/multiview_net_backend.py` |
| PR #7 | Owner's FastAPI + vanilla JS app (replaces Streamlit) wired to the network; drag/zoom fix; up to 3 AP + 3 lateral views averaged | `serve.py`, `bonevision/`, `web/` |
| Thread work | 4 synthetic "clinical-looking" test X-rays plus their generator | Project files `xray-samples/` (`synth_xray.py`, not in the repo) |

All PRs are merged into `main`, and CI (GitHub Actions, CPU torch) is green.

---

## 3. How it works now

### 3.1 Data

- **HFValid Collection** (CC BY 4.0): 101 CT femurs with segmentations and landmarks.
  - 64 are left and 37 are right. Right femurs are mirrored to the left (`femur_grid.CANONICAL_SIDE="L"`).
  - Split in `data/subject_split.csv`: 70 train, 15 validation, 16 test. **Keep validation and test fixed** so results stay comparable.
- **xvr-data femur subset** (CC BY-NC-ND): 5 real X-ray/CT pairs. Not used yet. This is the only real-X-ray test set available.
- **VSD lower-extremity meshes** (CC BY-NC-SA): 60 STLs. Can be imported as training-only shapes with `scripts/import_external_femurs.py`.
- **Where the data lives:** WSL Ubuntu 24.04 at `/home/dhanushiya/bonevision-femur/data`. That folder holds `raw/Subjects`, `occupancy_highres`, `occupancy_targets`, and the biplanar dataset built by the pipeline. The Windows clone has no data. Large files are gitignored, so never commit data or weights.

### 3.2 Canonical grid (`scripts/femur_grid.py`)

- `BOUNDS = (-470, 40, -60, 100, -70, 70)` mm (x = long axis, y, z) and `DIMENSIONS = (256, 96, 96)`.
- Voxels are about 2 × 1.68 × 1.47 mm at full resolution. The network runs at downsample 2, roughly 4 mm voxels in its output.
- Grid row 0 is the knee end. The bone is always a left femur.
- AP view = projection along z, `occ.max(axis=2)`. Lateral view = projection along y, `occ.max(axis=1)`.

### 3.3 Training pipeline (`scripts/run_retraining.py`)

The pipeline runs these steps in order: `check → ssm → dataset → smoke → train → evaluate`.

- Each step has a gate, and finished steps are skipped on a re-run.
- Logs go to `logs/` and the summary to `models/retraining_report.md`.
- The dataset step (`build_biplanar_dataset.py`) makes grid-aligned parallel-beam DRRs, i.e. line integrals of CT HU, for AP and lateral.
- The network (`multiview_net.py`) has three stages:
  1. a 2D U-Net per view,
  2. back-projection of the AP features along z and the lateral features along y into a 3D feature volume,
  3. a 3D U-Net that outputs occupancy logits.
- Loss is Dice + BCE (`dice_bce_loss`), with 300 epochs and batch size 2 by default. It fits in 8 GB.
- Run in WSL with `bash scripts/train_wsl.sh --also-ap`. Resume from a step with `--from train`.

### 3.4 Inference (`bonevision/network.py` → `scripts/multiview_net_backend.py`)

1. **Segment:** Otsu threshold plus morphology and component scoring (`bonevision/pipeline.segment_single_bone`). This is classical image processing and the weakest link on real films, because it can pick up soft tissue or the pelvis.
2. **Place on the grid** (`place_on_grid`):
   - match the mask's principal axis to the mean training silhouette,
   - anchor the proximal end,
   - scale the length to the training mean, since true size is unknown without calibration.
   It returns the affine `(M, t)` used for overlays.
3. **Orientation:** 4 hypotheses (head up/down × mirrored or not), picked by Dice against the reference silhouette. A small margin raises a warning.
4. **Network:** the AP+lateral checkpoint runs when a lateral view is given, otherwise the AP-only one. The largest component is kept and marching cubes produces the surface (`surface_from_occupancy`).
5. **Multiple views:** with up to 3 AP and 3 lateral views, the network runs once per AP×lateral pair. The occupancies are averaged and re-meshed. A view whose Dice is more than 0.05 below the median gets a `view_disagrees` warning.
6. **Exports** (`bonevision/exports.py`):
   - `femur_reconstruction.stl`, `femur_labelmap.nrrd`, `femur_estimate.obj`, `reconstruction.json`,
   - `view_fit.png` and `input_overlay_{ap1,lat1,…}.png`,
   - a ZIP with `BoneVisionLoader.py` for Slicer.

There is a second method, the SSM fit (`bonevision/pipeline.py`, `models/ssm_highres.npz`). It is AP-only and older, and is kept as a fallback.

### 3.5 Web app

- **Backend:** FastAPI (`bonevision/server.py`). Uploads and cases stay in memory and nothing is written to disk. The server binds to `127.0.0.1:8000`, and the API docs are at `/api/docs`.
- **Main endpoints:**
  - `GET /api/status`
  - `POST /api/uploads` (multipart file; PNG/JPG/DICOM)
  - `POST /api/uploads/{id}/segment` with `{crop, polarity, threshold_bias∈[-0.25,0.25]}`
  - `POST /api/cases` with `{method:"network"|"ssm", views:[{upload_id, kind:"ap"|"lat", crop, polarity, threshold_bias}]}`
  - `GET /api/cases/{id}`, `/mesh`, `/files/{name}`, `/bundle.zip`, `POST /cancel`, `GET /api/evaluation`
- **Frontend:** vanilla JS in `web/js/main.js` (pages: Case, Inferences, 3D model, Export, Method) and `web/js/viewer.js` (three.js, vendored, works offline). View slots are keyed `ap1, ap2, ap3, lat1, …`.
- **Model files** (in `models/`, gitignored):

| File | Purpose |
|---|---|
| `multiview_ap_lat_d2.pt` | Main network (AP + lateral) |
| `multiview_ap_d2.pt` | AP-only network |
| `ssm_sdf.npz` | Reference silhouettes for older checkpoints |
| `ssm_highres.npz` | SSM fit method |

---

## 4. How to run it

### 4.1 Windows (owner's laptop: RTX 5060 8 GB, needs CUDA 12.8+ torch)

```powershell
git clone https://github.com/dhanushiya-17/X-Ray-3-d-Regeneration-YOLO.git
cd X-Ray-3-d-Regeneration-YOLO
python -m venv .venv ; .\.venv\Scripts\activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-app.txt
# copy trained models from WSL:
mkdir models
copy \\wsl$\Ubuntu-24.04\home\dhanushiya\X-Ray-3-d-Regeneration-YOLO\models\multiview_*_d2.pt models\
copy \\wsl$\Ubuntu-24.04\home\dhanushiya\X-Ray-3-d-Regeneration-YOLO\models\ssm_sdf.npz models\
python serve.py            # real models  → http://127.0.0.1:8000
python serve.py --demo     # no models? trains tiny SYNTHETIC demo nets (UI testing only)
```

On Windows, deactivate conda before running. Mixing conda MKL with pip torch causes "OMP Error #15". Do not
work around it with `KMP_DUPLICATE_LIB_OK`.

### 4.2 WSL / Linux (where the data and trained models are)

```bash
cd ~/X-Ray-3-d-Regeneration-YOLO && git pull
source ~/bonevision-femur/femur3d/bin/activate      # python 3.12, torch cu130
pip install -r requirements-app.txt
python serve.py --device cuda                        # open http://127.0.0.1:8000 in the Windows browser
bash scripts/train_wsl.sh --also-ap                  # full retrain (data linked from ~/bonevision-femur/data)
```

### 4.3 Other commands

```bash
python -m unittest discover -s tests                 # 37 pass, 2 skip without licensed data
python serve.py --net-ap-lat other.pt --net-ap other_ap.pt   # try a different checkpoint, no code change
python scripts/reconstruct_single_xray.py femur.png --output runs/case01    # CLI, SSM path
```

### 4.4 Using the app

1. On the **Case** page, choose the Learned network method.
2. Upload an AP X-ray into the AP tab, and optionally a lateral one into the Lateral tab. Use **+ AP** or **+ Lateral** for more views (up to 3 each).
3. Check the coloured mask on each view. If it spills into soft tissue or the pelvis, crop the image or raise the threshold (about +0.05).
4. Run the case.
5. Look at **Inferences** for the per-view fit, **3D model** to orbit the result, and **Export** for Slicer files and the ZIP.

Test images: the project files folder `xray-samples/` has 2 AP and 2 lateral synthetic femur X-rays.

---

## 5. Known limits (be honest about these)

1. **No real-X-ray validation.** All metrics come from simulated, grid-aligned DRRs at the true length.
2. **Size is assumed.** Length is set to the training mean, so output dimensions are not patient measurements. Fixing this needs a calibration marker or DICOM pixel spacing plus a magnification factor.
3. **Fixed angles.** The network only accepts true AP (0°) and true lateral (90°). Oblique views are treated as noise. A decision on retraining for arbitrary angles is still pending.
4. **Classical segmentation** fails on real films with overlapping pelvis, implants, soft tissue or low contrast.
5. **Coarse output.** About 4 mm voxels and a smooth surface; no cortical thickness and no fine detail.
6. **Intact-shape prior.** Trained only on intact femurs. Given a fractured femur, the network will "heal" it and show a normal bone. ⚠️ This matters most for the clinical goal; see 6.0.
7. Parallel-beam training. Real X-rays are cone-beam, which brings magnification and distortion.
8. Not a medical device. There is no regulatory work, audit trail or authentication yet.

---

## 6. Future scope and execution plan

### 6.0 What reconstruction can and cannot do for hidden fractures

A 3D shape inferred from X-rays only contains information the X-rays carry, plus the model's prior. A
fracture that is **not visible on any of the input X-rays cannot be revealed by 3D reconstruction**. The
model would fill in a plausible intact bone, which is worse than useless because it looks reassuring. So
the clinical design should be:

- **Occult or subtle fractures:** a 2D detector on the X-ray (YOLO plus a classifier) that flags
  suspicion with heatmaps and recommends CT or MRI. Detectors can pick up subtle cues that humans miss,
  such as cortical buckling, trabecular disruption or a sclerotic line.
- **Visible or displaced fractures:** a fracture-aware 3D reconstruction showing fragments,
  displacement, angulation and shortening, for planning and communication.
- **Every 3D output** carries an uncertainty map and the "estimate, not measurement" label. Where CT
  exists, the app should load the CT segmentation instead of guessing.

### 6.1 Phase A: make the current pipeline robust on real X-rays (do first)

| # | Task | How | Done when |
|---|---|---|---|
| A1 | Learned femur segmenter | Fine-tune a U-Net/nnU-Net or YOLO-seg on DRR masks (free labels from CT) plus a few hundred hand-labelled real films. Replace `segment_single_bone` behind the same interface. | Mask Dice above 0.9 on held-out real films |
| A2 | Real-X-ray test set | Run the 5 xvr cases (real X-ray plus CT): register, then compute Dice and surface distance against CT. Add an `evaluate_real.py`. | A table in `models/retraining_report.md` |
| A3 | Realistic training inputs | Train on cone-beam DRRs (DiffDRR) with random source distance, pose jitter (±15°), soft tissue, noise, contrast and gamma, plus style transfer to real films. | Real-set Dice improves |
| A4 | Arbitrary angles | Condition the network on the view angle (angle embedding, or back-projection along the rotated ray), or add a 2D/3D pose-regression front end. | Works for 0–90° in 15° steps |
| A5 | Scale | Read DICOM PixelSpacing, detect a calibration ball, and store the scale in `reconstruction.json`. | Length error under 5 mm with a marker |
| A6 | Right-leg handling | Detect the side (UI toggle plus the L/R marker), mirror the input, then mirror the output back. | Right femurs exported as right |

### 6.2 Phase B: higher-fidelity 3D

- **B1 Resolution:**
  - train at `--downsample 1`, using gradient checkpointing or patches on 8 GB;
  - or predict an SDF and decode at 1 mm;
  - or **deform a high-res template mesh** (graph-conv / point offsets). This gives anatomically smooth surfaces with point correspondence, which helps measurements.
- **B2 More data:** TotalSegmentator CT (femur labels, about 1,200 scans), VSD meshes, NMDID, CTPelvic1K.
  - Import with `import_external_femurs.py` as **training-only** data.
  - Keep the HFValid validation and test sets fixed.
  - Check every licence before use (NC/ND licences limit commercial use).
- **B3 Internal structure:** predict cortical vs cancellous bone (multi-label), which is needed for fracture simulation and planning.
- **B4 Uncertainty:** deep ensembles or MC-dropout to give per-vertex uncertainty, coloured in the viewer (the viewer already supports a spread colour map).
- **B5 Measurements:** neck-shaft angle, femoral anteversion, offset, length, condylar width. Use landmarks on the template mesh and show them in a UI panel.

### 6.3 Phase C: synthetic fracture generator (3D → X-ray)

Goal: an unlimited labelled set of fractured femurs, with **3D fragment labels** plus **DRRs** plus
**2D boxes and masks**.

- **Input:** CT volumes plus femur segmentations (HFValid; later TotalSegmentator).
- **Fracture types (AO/OTA):**

| Region | AO/OTA code | Types to simulate |
|---|---|---|
| Proximal | 31A | intertrochanteric |
| Proximal | 31B | femoral neck: subcapital, transcervical, basicervical |
| Proximal | 31C | head |
| Shaft | 32A | simple: spiral, oblique, transverse |
| Shaft | 32B | wedge |
| Shaft | 32C | multifragmentary |
| Distal | 33A/B/C | extra-articular, partial articular, complete articular |
| Any | — | stress / hairline (non-displaced) |

- **Algorithm** (`scripts/fracture_synth.py`, new):
  1. Pick a region and type, and sample a **fracture surface**: a plane (transverse/oblique), a helicoid (spiral), or several surfaces (wedge/comminuted). Perturb it with low-frequency noise so edges look irregular.
  2. Split the femur label into fragments, keeping the components.
  3. Apply a **rigid displacement** per fragment: translation (0–20 mm), angulation (0–30°), rotation, shortening or overlap, impaction. Draw these from clinically plausible ranges per type. A **non-displaced** setting has a 0.2–1 mm gap.
  4. Resample the CT intensities with each fragment's transform. Set the gap voxels to soft-tissue HU, add a sclerotic or impacted band for some types, and optionally add hematoma (a soft-tissue HU bump).
  5. Save `ct_fractured.nii.gz`, `fragments_labelmap.nrrd`, `fracture_surface.ply`, and `meta.json` (AO class, displacement parameters).
  6. Render DRRs (AP, lateral, random obliques; cone-beam; augmentations) with DiffDRR. Project the fracture surface to get **2D YOLO boxes and polygons**, and store the per-view projection matrix.
- **QC:** spot-check renders against real fracture films; FID or KID against real images; have a clinician review a sample.
- **Volume:** about 100 CTs × 20 fractures × 6 views gives roughly 12,000 labelled images.

### 6.4 Phase D: YOLO fracture detection on X-rays

- **Model:** Ultralytics YOLO (v8 or 11) detect, then seg. ⚠️ Ultralytics is **AGPL-3.0**. For a closed clinical product, buy their licence or use an Apache-licensed detector (RT-DETR from PaddleDetection/HF, or YOLOX).
- **Data mix:**
  - synthetic fractures from phase C (pretraining);
  - real fractures: **FracAtlas** (CC BY 4.0, includes hip and leg; prep code in `legacy/yolo/prepare_dataset.py`), plus GRAZPEDWRI-DX for generic fracture features, plus hospital data under ethics approval.
  - Fine-tune on real data last.
- **Classes:** `fracture` (box) first. Then add the AO region or type, and `implant` (to suppress false positives).
- **Evaluation on real-only held-out data:**
  - sensitivity at fixed specificity, AUROC (image level), mAP50 (box level);
  - subgroups by fracture type and subtlety;
  - compare with radiologist reads;
  - **never report synthetic-only metrics as clinical performance.**
- **App integration:** the new endpoint `POST /api/cases/{id}/detect` returns boxes, scores and heatmaps. Draw them on the Inferences page and project them as 3D regions on the mesh (rays through a box intersect the bone).

### 6.5 Phase E: fracture-aware 3D reconstruction

- Retrain the multi-view network on phase-C pairs, with output = **multi-label occupancy** (background, fragment 1..K) or occupancy plus a fracture-surface probability.
- Give it the YOLO box as an extra input channel (attention to the region).
- Output fragments as separate STLs/labels in Slicer, and compute displacement metrics (translation mm, angulation °, shortening mm, rotation) between fragments.
- **Evaluate** on synthetic held-out data, then on real cases that have post-op or pre-op CT.

### 6.6 Phase F: explainable AI

- **2D:**
  - Grad-CAM / EigenCAM on the detector and classifier;
  - occlusion sensitivity;
  - show a "why" heatmap overlaid next to each box.
- **3D:**
  - per-vertex uncertainty (B4);
  - per-view agreement (already present);
  - per-fragment measurements with evidence: which view and which pixels drove the result, via back-projected saliency.
- **Counterfactual:** show the predicted intact femur next to the fractured reconstruction and highlight the difference.
- **Structured report:** AO class suggestion, displacement numbers, confidence, uncertainty, and recommended next imaging. Generate text from the structured fields with templates, or with an LLM (Claude API) constrained to those fields. Never let free text invent findings.
- **Faithfulness checks:** saliency sanity tests (model and label randomisation) and a clinician review of the explanations.

### 6.7 Phase G: clinical product

- **DICOM in/out:**
  - pydicom is already used for reading;
  - add DICOMweb/PACS connection (Orthanc for development);
  - export DICOM SEG and Secondary Capture and STL.
- **Accounts and storage:** case persistence, de-identification, audit log, role-based access. Today everything stays in memory and binds to localhost on purpose.
- **Viewer extras:**
  - fragment toggles and clipping planes;
  - measurement tools;
  - side-by-side X-ray with the 3D ray link;
  - 3D-print export;
  - 3D Slicer extension packaging.
- **Quality system and regulation:**
  - software as a medical device, under IEC 62304 (software lifecycle), ISO 14971 (risk) and ISO 13485 (QMS);
  - the regulator depends on the market (for example FDA, EU MDR, or CDSCO in India);
  - you need intended-use wording, a clinical validation study (retrospective, multi-site, CT ground truth, reader study), and ethics/IRB approval for patient data.
- **Deployment:** containerise (Docker), use a GPU inference server, and record model versions and data lineage per case (`reconstruction.json` already records pipeline and checkpoint info).

### 6.8 Suggested order

```
A1 segmenter → A2 real test set → A3 realistic DRRs → A4 angles → A5 scale/A6 side
→ C fracture generator → D YOLO (synthetic pretrain, FracAtlas fine-tune)
→ E fracture-aware 3D → F XAI → B1/B5 fidelity+measurements → G clinical product
```

A1–A2 come first because every later metric is meaningless until real X-rays work. C feeds both D and E.

---

## 7. Rules for whoever continues

- **Branch names:** generic and descriptive, such as `feature/fracture-synth` or `fix/segmentation`. No session or agent ids (owner's request).
- **Working style:** the owner trains on their own machine and wants to learn. Give one step at a time, explain why, and ask for logs. Do not run training on their device.
- **Evaluation:** keep the HFValid validation and test splits fixed. Report synthetic and real results separately and label them. Keep the worst case in averages.
- **Data and weights:** never commit them. Check the licence of every new dataset.
- **New reconstruction models:** add a module like `bonevision/network.py` that returns the same result keys, plus a branch in `server.py` (`create_case`, `_run_case`) and `exports.py`. Checkpoints swap via `--net-ap-lat` / `--net-ap` or `BONEVISION_NET_AP_LAT` / `BONEVISION_NET_AP`.
- **Exports:** keep the "research estimate, not a measurement" labelling and the RAS tags.
- **Tests:** `python -m unittest discover -s tests` must pass, and CI runs it on every push.

## 8. Open decisions

1. Retrain the network for arbitrary view angles (A4) now, or after real-X-ray validation (A1–A2)? The recommendation is A1–A2 first.
2. Detector licence: Ultralytics (AGPL or commercial) or an Apache-licensed alternative.
3. Where hospital data will come from, and under which ethics approval.
