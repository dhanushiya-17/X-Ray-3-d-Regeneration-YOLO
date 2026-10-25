"""BoneVision local X-ray to 3D femur interface (one or more views)."""
from __future__ import annotations
import csv
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from reconstruct_single_xray import read_image
from reconstruct_multiview_xray import ViewInput
from reconstruction_backends import get_backend
from slicer_export import MESH_NAME, slicer_bundle

st.set_page_config(page_title="BoneVision | X-ray to 3D femur", page_icon="🦴", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1500px; padding-top: 1.5rem;}
.bv-note {padding: .85rem 1rem; border: 1px solid #dbe5ef; border-radius: .5rem; background: #f5f9fc;}
</style>
""", unsafe_allow_html=True)
st.title("BoneVision")
st.caption("X-ray views of one femur → 3D femur estimate")
st.warning("Research demonstration, not validated for clinical use. The output is a shape-prior estimate fitted to bone silhouettes; view angles are declared by you, not calibrated, and absolute size comes from the shape model. Projection match is not proof of 3D accuracy.")

VIEW_CHOICES = {"AP (0°)": 0.0, "Lateral (90°)": 90.0, "Other angle": None}
MIRROR_CHOICES = {"Auto (try both)": "auto", "As uploaded": "no", "Mirrored": "yes"}


@st.cache_resource(show_spinner=False)
def load_backend():
    # Kept across reruns so the SSM and its per-angle projections load once.
    return get_backend()


def decode_upload(name: str, data: bytes) -> np.ndarray:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(name).suffix.lower() or ".png", delete=False) as tmp:
            tmp.write(data)
            temp_path = Path(tmp.name)
        return read_image(temp_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def render_view_fit(v: dict, src: np.ndarray, cinput, cproj) -> None:
    with cinput:
        overlay = np.repeat(src[..., None], 3, axis=2)
        overlay[v["mask"]] = overlay[v["mask"]] * 0.45 + np.array([0.0, 0.9, 0.2], dtype=np.float32) * 0.55
        st.image(np.clip(overlay, 0, 1), clamp=True, use_container_width=True)
        st.caption(f"{v['name']} · threshold-based component · {v['segmentation']['polarity']} contrast · {100*v['segmentation']['component_area_fraction']:.1f}% of image")
    with cproj:
        tgt = v["target_projection"]
        pred = v["projection"] >= 0.5
        rgb = np.zeros((*tgt.shape, 3), dtype=np.uint8)
        rgb[...,0] = (tgt & ~pred) * 255
        rgb[...,1] = (tgt & pred) * 220
        rgb[...,2] = (~tgt & pred) * 255
        st.image(rgb, use_container_width=True)
        st.caption(f"{v['angle_deg']:.0f}° silhouette · red = input only · green = overlap · blue = fitted SSM only. Not a full radiographic DRR.")


backend = load_backend()
backend_problem = backend.availability()

if "result" not in st.session_state:
    st.session_state.result = None
if "input_images" not in st.session_state:
    st.session_state.input_images = []

with st.sidebar:
    st.header("Inputs")
    uploads = st.file_uploader(
        "Upload X-rays of the same femur", type=["png", "jpg", "jpeg", "tif", "tiff", "dcm", "dicom"], accept_multiple_files=True,
        help=f"One AP view works; adding a lateral view constrains depth. Up to {backend.max_views} views.",
    )
    iterations = st.slider("Shape-fit iterations", 50, 400, 250, 25, help="Fast fit of every orientation hypothesis using projected shape modes.")
    refine_iterations = st.slider("3D solid refinement iterations", 0, 300, 100, 25, help="Refines the chosen fit so the exported mesh itself matches the silhouettes. Slower on CPU; 0 skips it.")
    st.caption(f"**Model:** {backend.label}")
    st.caption("Crop each view to a single femur and avoid implants if possible. DICOM pixel values are rescaled when metadata is present; physical scale is assumed from the SSM prior.")

uploads = list(uploads or [])
if len(uploads) > backend.max_views:
    st.error(f"{len(uploads)} files uploaded; only the first {backend.max_views} are used.")
    uploads = uploads[:backend.max_views]

views: list[ViewInput] = []
settings_key = [iterations, refine_iterations, backend.name]
if not uploads:
    st.info("Upload one or more X-rays of the same femur to begin. Label each one as AP, lateral, or its rotation about the femur's long axis. A fresh clone needs a compatible model built from the CT-derived training data.")
else:
    st.subheader("Views")
    for index, upload in enumerate(uploads):
        data = upload.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        key = f"{index}_{digest[:12]}"
        default_view = "AP (0°)" if index == 0 else "Lateral (90°)" if index == 1 else "Other angle"
        with st.expander(f"View {index + 1}: {upload.name}", expanded=True):
            try:
                image = decode_upload(upload.name, data)
            except Exception as exc:
                st.error(f"Could not read this X-ray: {exc}")
                continue
            cimg, cset = st.columns([1, 2.4])
            with cset:
                c1, c2 = st.columns(2)
                view_label = c1.selectbox("View", list(VIEW_CHOICES), index=list(VIEW_CHOICES).index(default_view), key=f"view_{key}")
                angle = VIEW_CHOICES[view_label]
                if angle is None:
                    angle = c2.number_input("Rotation about long axis (°)", -180.0, 360.0, 45.0, 5.0, key=f"angle_{key}", help="0 = AP, 90 = lateral. Rotation of the beam about the femur's long axis.")
                else:
                    c2.metric("Beam rotation", f"{angle:.0f}°")
                mirror = MIRROR_CHOICES[c1.selectbox("Left/right orientation", list(MIRROR_CHOICES), key=f"mirror_{key}", help="Auto fits both horizontal orientations and keeps the better one.")]
                polarity = c2.selectbox("Bone contrast", ["auto", "bright", "dark"], key=f"pol_{key}", help="Auto compares bright and dark connected components.")
                threshold_bias = st.slider("Threshold adjustment", -0.20, 0.20, 0.0, 0.01, key=f"thr_{key}", help="Shift the Otsu threshold when the mask misses bone or includes background.")
                k1, k2, k3, k4 = st.columns(4)
                crop_top = k1.number_input("Crop top %", 0, 40, 0, key=f"ct_{key}")
                crop_bottom = k2.number_input("Crop bottom %", 0, 40, 0, key=f"cb_{key}")
                crop_left = k3.number_input("Crop left %", 0, 40, 0, key=f"cl_{key}")
                crop_right = k4.number_input("Crop right %", 0, 40, 0, key=f"cr_{key}")
            h0, w0 = image.shape
            y0, y1 = int(h0*crop_top/100), int(h0*(100-crop_bottom)/100)
            x0, x1 = int(w0*crop_left/100), int(w0*(100-crop_right)/100)
            if y1-y0 < 128 or x1-x0 < 96:
                st.error("Crop leaves too little image. Reduce the crop margins.")
                continue
            image = image[y0:y1, x0:x1]
            with cimg:
                st.image(image, clamp=True, use_container_width=True)
                st.caption(f"{image.shape[1]} × {image.shape[0]} px · normalized grayscale")
                if image.shape[0] < 256 or image.shape[1] < 128:
                    st.warning("Resolution is low for a full-femur estimate.")
        views.append(ViewInput(image, float(angle), upload.name, polarity, float(threshold_bias), mirror))
        settings_key.append((digest, angle, mirror, polarity, threshold_bias, crop_top, crop_bottom, crop_left, crop_right))

request_key = tuple(settings_key)
if st.session_state.get("request_key") != request_key:
    st.session_state.result = None
    st.session_state.input_images = []
    st.session_state.request_key = request_key

if views:
    angles = sorted({round(v.angle_deg % 180) for v in views})
    if len(views) > 1 and len(angles) == 1:
        st.info("All views share the same beam angle, so the extra images add no depth information. Add a lateral or oblique view to constrain the 3D shape.")
    if backend_problem:
        st.error(backend_problem)
    run = st.button(f"Reconstruct 3D femur from {len(views)} view{'s' if len(views) > 1 else ''}", type="primary", disabled=bool(backend_problem), use_container_width=True)
    if run:
        try:
            with st.status(f"Reconstructing from {len(views)} view(s)…", expanded=True) as status:
                st.write("Extracting one bone-like component from each image")
                st.write("Fitting one CT-trained 3D shape to every silhouette at its declared angle")
                st.write("Refining the decoded 3D solid against each silhouette")
                result = backend.reconstruct(views, iterations=iterations, refine_iterations=refine_iterations)
                st.write("Extracting a millimetre-scale surface mesh")
                status.update(label="Reconstruction completed", state="complete", expanded=False)
            st.session_state.result = result
            st.session_state.input_images = [v.image for v in views]
        except Exception as exc:
            st.session_state.result = None
            st.error(f"Reconstruction failed: {exc}")

result = st.session_state.result
if result is not None:
    st.divider()
    st.subheader("Reconstruction workspace")
    c3d, cstat = st.columns([2.2, 1])
    with c3d:
        verts, faces = result["vertices_mm"], result["faces"]
        view_style = st.radio("Surface view", ["Shaded", "Wireframe"], horizontal=True)
        fig = go.Figure()
        if view_style == "Shaded":
            fig.add_trace(go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2], color="#d8c4a6", opacity=1.0, flatshading=False, lighting=dict(ambient=0.45, diffuse=0.75, specular=0.25, roughness=0.55), lightposition=dict(x=100, y=150, z=250), name="SSM estimate"))
        else:
            edges = np.unique(np.sort(np.concatenate((faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]), axis=0), axis=1), axis=0)
            if len(edges) > 12000:
                edges = edges[np.linspace(0, len(edges)-1, 12000, dtype=int)]
            wire = np.concatenate((verts[edges], np.full((len(edges),1,3), np.nan, dtype=np.float32)), axis=1).reshape(-1,3)
            fig.add_trace(go.Scatter3d(x=wire[:,0], y=wire[:,1], z=wire[:,2], mode="lines", line=dict(color="#66788a", width=1), name="SSM wireframe"))
        fig.update_layout(height=660, margin=dict(l=0,r=0,t=15,b=0), scene=dict(aspectmode="data", xaxis_title="Anatomical X (mm)", yaxis_title="Anatomical Y (mm)", zaxis_title="Anatomical Z (mm)"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False, "scrollZoom":True})
    with cstat:
        st.markdown("#### Fitting status")
        m1, m2 = st.columns(2)
        m1.metric("Mean silhouette Dice", f"{result['projection_dice']:.3f}")
        m2.metric("Worst view Dice", f"{result['min_view_dice']:.3f}")
        m1.metric("Runtime", f"{result['elapsed_seconds']:.1f} s")
        m2.metric("Compute", result["device"].upper())
        st.write(f"**Method:** {result['method']}")
        st.write(f"**Modes:** {result['n_components']} · **Training shapes:** {result['n_training_subjects']}")
        st.write(f"**Shape-prior weight:** {result['prior_weight']:.3f}")
        st.write(f"**Orientation hypotheses tested:** {result['hypotheses_tested']} (long axis flipped: {'yes' if result['flipped_long_axis'] else 'no'})")
        st.write(f"**3D solid refinement:** {result['refine_iterations']} iterations")
        st.write(f"**Voxel spacing:** {', '.join(f'{x:.2f}' for x in result['spacing_mm'])} mm")
        st.write("**Mesh dimensions (mm):** " + str(tuple(round(x,1) for x in result["dimensions_mm"])))
        st.line_chart(result["history"], height=150, use_container_width=True)
        st.write("**Ground-truth accuracy:** unavailable for uploaded X-rays")
        st.caption("A low Dice on one view usually means its declared angle, crop, or mask is wrong, or the views are not of the same bone.")

    st.markdown("#### Per-view fit")
    st.dataframe([{
        "view": v["name"], "beam angle (°)": v["angle_deg"], "silhouette Dice": round(v["soft_dice"], 3),
        "mirrored": v["mirrored"], "contrast": v["segmentation"]["polarity"],
        "pose (log scale, dy, dx)": tuple(round(x, 3) for x in v["fitted_pose_log_scale_shift"]),
    } for v in result["views"]], use_container_width=True, hide_index=True)
    pairs = list(zip(result["views"], st.session_state.input_images))
    for row in range(0, len(pairs), 2):
        columns = st.columns(4)
        for slot, (v, src) in enumerate(pairs[row:row + 2]):
            render_view_fit(v, src, columns[2 * slot], columns[2 * slot + 1])

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        backend.save(result, out)
        mesh_bytes = (out / MESH_NAME).read_bytes()
        report_bytes = (out / "reconstruction.json").read_bytes()
        projection_bytes = (out / "projection_fit.png").read_bytes()
        bundle_bytes = slicer_bundle(out)
    d0, d1, d2, d3 = st.columns(4)
    d0.download_button("Download 3D Slicer bundle (.zip)", bundle_bytes, "bonevision_femur_slicer.zip", "application/zip", type="primary", use_container_width=True)
    d1.download_button("Download STL only", mesh_bytes, MESH_NAME, "model/stl", use_container_width=True)
    d2.download_button("Download reconstruction report", report_bytes, "reconstruction.json", "application/json", use_container_width=True)
    d3.download_button("Download silhouette comparison", projection_bytes, "projection_fit.png", "image/png", use_container_width=True)
    st.caption("The Slicer bundle holds the surface (STL), a labelmap (NRRD), the report and a loader script. Unzip it and run "
               "`Slicer --python-script bonevision_femur/BoneVisionLoader.py`, or drag the STL into Slicer (File → Add Data). "
               "Coordinates are the SSM canonical frame in millimetres; the upright view is a display rotation, not a patient registration.")




# Dataset-level metrics are clearly separated from the uploaded case.
eval_file = ROOT / "reconstruction" / "single_view_test_metrics.csv"
if eval_file.exists():
    with st.expander("Held-out synthetic evaluation — separate from this X-ray", expanded=False):
        rows = list(csv.DictReader(eval_file.open(newline="")))
        aggregate = next((r for r in rows if r["subject"] == "MEAN"), None)
        st.caption("These are subject-level test results from synthetic canonical silhouettes with ground-truth CT masks. They are not accuracy metrics for the uploaded image or real clinical radiographs.")
        if aggregate:
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("SSM voxel Dice", f"{float(aggregate['ssm_dice']):.3f}")
            e2.metric("SSM voxel IoU", f"{float(aggregate['ssm_iou']):.3f}")
            e3.metric("SSM mean surface distance", f"{float(aggregate['ssm_mean_symmetric_surface_distance_mm']):.2f} mm")
            e4.metric("SSM surface P95", f"{float(aggregate['ssm_p95_surface_distance_mm']):.2f} mm")
            figure_path = eval_file.with_name(eval_file.stem + "_summary.png")
            if figure_path.exists():
                st.image(str(figure_path), caption="Synthetic held-out test summary", use_container_width=True)
            detail = [{"subject": r["subject"], "SSM Dice": float(r["ssm_dice"]), "SSM IoU": float(r["ssm_iou"]), "SSM mean surface (mm)": float(r["ssm_mean_symmetric_surface_distance_mm"]), "1-view template Dice": float(r["retrieval_dice"])} for r in rows if r["status"] == "ok"]
            st.dataframe(detail, use_container_width=True, hide_index=True)
