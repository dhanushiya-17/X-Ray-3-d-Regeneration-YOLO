# BONEVISION::CTX v2026-10-25 (dense handoff for an AI agent; full prose = docs/HANDOFF.md)
# legend: →=leads to  |=or  ⊂=inside  ✓=done ✗=missing ⚠=risk  §=see section in HANDOFF.md  ~=approx

[ID]
repo=dhanushiya-17/X-Ray-3-d-Regeneration-YOLO br=main ci=.github/workflows/tests.yml(cpu torch,unittest) owner=Mahadev
goal=xray(s)→3D femur, clinically usable; +synthetic fractures on 3D→DRR→YOLO fracture det; +XAI on fractured 3D; +clinical product
state=research proto; intact femur only; validated on SYNTHETIC DRR only; real xray untested ⚠

[USER_RULES]
branch_names=generic,contextual(feature/x,fix/y); NO agent/session ids
style=teach step-by-step; owner runs training on own HW; ask for logs; never run training on their device
keep HFValid val/test fixed; never commit data/weights; label synthetic vs real metrics; keep worst case in means
exports keep "research estimate, not measurement" + SPACE=RAS
win: deactivate conda (OMP#15 from MKL+pip torch); never KMP_DUPLICATE_LIB_OK

[HW/ENV]
laptop=Windows,RTX5060 8GB(sm_120)→torch cu128+
wsl=Ubuntu24.04; data=/home/dhanushiya/bonevision-femur/data{raw/Subjects,occupancy_highres,occupancy_targets}
wsl_venv=~/bonevision-femur/femur3d(py3.12,torch cu130) wsl_clone=~/X-Ray-3-d-Regeneration-YOLO(models/ here ONLY)
win_clone has no data/models → copy \\wsl$\Ubuntu-24.04\home\dhanushiya\X-Ray-3-d-Regeneration-YOLO\models\*

[DATA]
HFValid(CC-BY4.0)=101 CT femurs+seg+landmarks; 64L/37R; R mirrored→L (femur_grid.CANONICAL_SIDE="L"); split data/subject_split.csv 70tr/15val/16test
xvr-femur(CC-BY-NC-ND)=5 real xray+CT pairs ✗unused → only real test set
VSD(CC-BY-NC-SA)=60 STL; import via scripts/import_external_femurs.py as train-only
legacy FracAtlas/GRAZPEDWRI prep ⊂ legacy/yolo/

[GRID] scripts/femur_grid.py
BOUNDS=(-470,40,-60,100,-70,70)mm(x=long,y,z) DIMS=(256,96,96) ~2x1.68x1.47mm; net ds2→~4mm
row0=knee; canonical LEFT; AP=proj z(occ.max(2)); LAT=proj y(occ.max(1))

[MODELS] models/ gitignored
multiview_ap_lat_d2.pt: test Dice .943, MSD .95mm, HD95 3.45, worst .917 Pat036
multiview_ap_d2.pt: Dice .865, 2.19mm, HD95 5.56, worst .760 Pat082
ssm_sdf.npz: ref silhouettes for old ckpts | ssm_highres.npz: SSM-fit method (Dice .687, 4.73mm, worst .373 Pat099)
⚠ metrics=parallel-beam DRR, grid-aligned, true length, 4mm voxels

[TRAIN] scripts/run_retraining.py steps=check→ssm→dataset→smoke→train→evaluate (gated, skip-done, logs/, models/retraining_report.md)
wsl: bash scripts/train_wsl.sh --also-ap [--from train] [--epochs 300 --downsample 2 --batch-size 2]
dataset=build_biplanar_dataset.py (HU line integrals AP/LAT, grid-aligned, --manifest for external)
net=scripts/multiview_net.py: per-view 2D UNet → backproject AP along z, LAT along y → 3D UNet → occ logits; loss dice+bce; fits 8GB
train=train_multiview_net.py(--views ap,lat|ap --extra-train-dir --evaluate validation|test)

[INFER] bonevision/network.py → scripts/multiview_net_backend.py
1 seg=pipeline.segment_single_bone(Otsu+morph+component score) ⚠weakest; leaks to soft tissue/pelvis
2 place_on_grid(img,mask,ref,kind,…,return_affine)→(M,t): PCA axis↔ref silhouette, proximal anchor, length:=train mean ⚠no true scale
3 orient: 4 hyp(flip×mirror) by Dice vs ref; small margin→warning
4 net: ap_lat ckpt if lat given else ap; largest comp; surface_from_occupancy(occ)→(occ,verts_mm,faces)
5 multi: MAX_VIEWS_PER_KIND=3; run per AP×LAT pair; mean occ; remesh; view Dice<median-.05→view_disagrees
warn codes: network_check view_disagrees view_ignored combined assumed_size
backend singleton: network.backend(); reset network._backend=None; env BONEVISION_NET_AP_LAT / BONEVISION_NET_AP

[APP] python serve.py [--demo] [--net-ap-lat P --net-ap P] [--device auto|cuda|cpu] [--host 127.0.0.1 --port 8000 --reload]
backend=FastAPI bonevision/server.py; in-memory uploads/cases(MAX_UPLOADS=8); /api/docs
GET /api/status{methods{network,ssm},default_method,max_views_per_kind}
POST /api/uploads(multipart png/jpg/dicom) | POST /api/uploads/{id}/segment{crop,polarity,threshold_bias∈[-.25,.25]}
POST /api/cases{method:"network"|"ssm",views:[{upload_id,kind:"ap"|"lat",crop,polarity,threshold_bias}]≤6}
GET /api/cases/{id} | /mesh | /files/{name} | /bundle.zip | POST /cancel | GET /api/evaluation
exports=bonevision/exports.py: femur_reconstruction.stl(RAS) femur_labelmap.nrrd(RAS) femur_estimate.obj reconstruction.json view_fit.png input_overlay_{ap1,lat1..}.png; zip=bonevision_femur/+BoneVisionLoader.py+README
frontend=web/js/main.js(pages Case,Inferences,3D model,Export,Method; slots ap1..ap3,lat1..lat3; blankSlot/defaultSlots/usedViews) web/js/viewer.js(three.js vendored) web/css/app.css(pointer-events:none on overlays → drag/zoom fix)
slicer=scripts/slicer_export.py + slicer/BoneVisionLoader/
demo=scripts/make_demo_network.py(synthetic, flagged synthetic_demo, UI banner) → models/demo/
tests: python -m unittest discover -s tests (37 pass, 2 skip w/o data)
test images: project files xray-samples/{femur_AP_1,femur_AP_2,femur_lateral_1,femur_lateral_2}.png + synth_xray.py (procedural leg, synthetic, labelled); Otsu mask leaks into hip → threshold +.05 or crop

[RUN_WIN]
python -m venv .venv; .\.venv\Scripts\activate; python -m pip install torch --index-url https://download.pytorch.org/whl/cu128; python -m pip install -r requirements-app.txt; copy models; python serve.py → http://127.0.0.1:8000
[RUN_WSL]
cd ~/X-Ray-3-d-Regeneration-YOLO; git pull; source ~/bonevision-femur/femur3d/bin/activate; pip install -r requirements-app.txt; python serve.py --device cuda

[HISTORY] PR1 slicer export | PR2 multiview SSM backend(streamlit, superseded) | PR3 retrain plan | PR5 pipeline+L/R mirror | PR6 net backend | PR7 FastAPI app+net, drag fix, 3AP+3LAT averaging | all merged, CI green

[LIMITS]
L1 no real-xray validation | L2 size assumed(mean length) | L3 only 0°/90° views | L4 classical seg | L5 ~4mm coarse, no cortex
L6 intact-shape prior → fracture gets "healed" ⚠⚠ | L7 parallel-beam train vs cone-beam real | L8 not a medical device, no auth/audit

[PRINCIPLE] occult fracture invisible on all input views ⇒ CANNOT be recovered by 3D recon (only prior hallucination).
design: occult→2D detector+heatmap+recommend CT/MRI | visible/displaced→fracture-aware 3D(fragments,displacement) | always uncertainty+estimate label | CT available→use CT seg instead

[ROADMAP] order: A1→A2→A3→A4→A5/A6→C→D→E→F→B1/B5→G
A1 learned femur segmenter(nnU-Net|YOLO-seg; free DRR masks + ~100s hand-labelled real) replace segment_single_bone same iface; DoD maskDice>.9 real
A2 evaluate_real.py on 5 xvr cases(register to CT; Dice,MSD) → retraining_report.md
A3 realistic DRR: DiffDRR cone-beam, SID rand, pose ±15°, soft tissue, noise/gamma, style transfer(CycleGAN/CUT)
A4 arbitrary angle: angle embedding | rotated-ray backprojection | pose-regression front end; DoD 0–90° step15
A5 scale: DICOM PixelSpacing + magnification | calibration ball detect; DoD len err<5mm
A6 side: UI toggle + L/R marker; mirror in, mirror out
B1 fidelity: ds1 + grad checkpoint/patches | SDF decode 1mm | template-mesh deformation(GCN/offsets; correspondence)
B2 data: TotalSegmentator(~1.2k CT femur), VSD, NMDID, CTPelvic1K → train-only; check licences(NC/ND)
B3 cortical/cancellous multilabel | B4 uncertainty(ensemble|MC-dropout)→viewer spread colormap | B5 measurements(NSA, anteversion, offset, length, condylar width) via template landmarks
C fracture synth (new scripts/fracture_synth.py): CT+seg → sample AO/OTA type{31A,31B neck sub/trans/basicervical,31C,32A spiral/oblique/transverse,32B wedge,32C multifrag,33A/B/C,stress hairline}
  → surface(plane|helicoid|multi)+lowfreq noise → split fragments → rigid disp(trans 0–20mm, ang 0–30°, rot, shortening, impaction; nondisplaced gap .2–1mm)
  → resample CT per fragment, gap=soft HU, sclerotic band opt, hematoma opt → save ct_fractured.nii.gz, fragments_labelmap.nrrd, fracture_surface.ply, meta.json
  → DiffDRR AP/LAT/oblique cone-beam+aug → project surface → YOLO boxes/polys + P matrix; QC FID/KID + clinician spot check; ~100CT×20×6≈12k imgs
D YOLO: ultralytics v8/11 det→seg ⚠AGPL-3.0 (buy licence | Apache alt RT-DETR/YOLOX); pretrain synth C → FracAtlas(CC-BY, hip/leg; legacy/yolo/prepare_dataset.py)+GRAZPEDWRI → hospital(ethics); classes fracture→+AO region,+implant
  eval REAL-only holdout: sens@spec, AUROC, mAP50, subgroups, vs radiologists; app POST /api/cases/{id}/detect → boxes,scores,heatmaps; draw on Inferences; rays∩mesh→3D region
E fracture-aware 3D: retrain net on C pairs; output multilabel occ(bg,frag1..K)|occ+fracture-surface prob; YOLO box as input channel; per-fragment STL; metrics trans mm/ang°/shortening/rot
F XAI: 2D Grad-CAM/EigenCAM/occlusion; 3D per-vertex uncertainty, per-view agreement, back-projected saliency; counterfactual intact vs fractured diff; structured report(AO, disp, conf, next imaging) templated|LLM(Claude API) constrained to fields; sanity tests(model/label randomisation)+clinician review
G clinical: DICOMweb/PACS(Orthanc dev), DICOM SEG/SC export; persistence, de-id, audit, RBAC; viewer fragments/clipping/measure/xray↔3D link/3D print; IEC62304, ISO14971, ISO13485; regulator per market(FDA|EU MDR|CDSCO); intended use; retrospective multi-site validation w/ CT GT + reader study; IRB; Docker+GPU server; model/data lineage in reconstruction.json

[EXTEND]
new recon model = module like bonevision/network.py (same result keys) + branch in server.py(create_case,_run_case) + exports.py
swap ckpt = --net-ap-lat/--net-ap or env; more data = import_external_femurs.py → train_wsl.sh

[OPEN]
O1 arbitrary-angle retrain now vs after A1–A2 (rec: after)
O2 detector licence choice
O3 hospital data source + ethics

[BOOT] new agent: read docs/HANDOFF.md + README.md → git pull main → python -m unittest discover -s tests → python serve.py --demo to see UI → ask owner which phase task to start (default A1) → branch feature/<task>
