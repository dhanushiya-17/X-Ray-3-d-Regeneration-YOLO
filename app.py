"""BoneVision local single-X-ray demo interface."""
from __future__ import annotations
import csv
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from reconstruct_single_xray import MODEL_PATH, read_image, reconstruct, save_results

st.set_page_config(page_title="BoneVision | Single-view femur", page_icon="🦴", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1500px; padding-top: 1.5rem;}
.bv-note {padding: .85rem 1rem; border: 1px solid #dbe5ef; border-radius: .5rem; background: #f5f9fc;}
</style>
""", unsafe_allow_html=True)
st.title("BoneVision")
st.caption("Single-view X-ray → 3D femur estimate")
st.warning("Research demonstration. One X-ray does not determine hidden depth uniquely. This output is a shape-prior estimate, not a diagnostic or surgical-planning model. Projection match is not proof of 3D accuracy.")

if "result" not in st.session_state:
    st.session_state.result = None
if "input_image" not in st.session_state:
    st.session_state.input_image = None

with st.sidebar:
    st.header("Input and segmentation")
    uploaded = st.file_uploader("Upload one femur X-ray", type=["png", "jpg", "jpeg", "tif", "tiff", "dcm", "dicom"])
    polarity = st.selectbox("Bone contrast", ["auto", "bright", "dark"], help="Auto compares bright and dark connected components. Choose manually if it selects the wrong region.")
    threshold_bias = st.slider("Threshold adjustment", -0.20, 0.20, 0.0, 0.01, help="Adjust the Otsu threshold when the automatic mask misses the bone or includes too much background.")
    iterations = st.slider("Shape-fit iterations", 50, 400, 250, 25)
    crop_top = st.slider("Crop top edge (%)", 0, 40, 0)
    crop_bottom = st.slider("Crop bottom edge (%)", 0, 40, 0)
    crop_left = st.slider("Crop left edge (%)", 0, 40, 0)
    crop_right = st.slider("Crop right edge (%)", 0, 40, 0)
    st.caption("Use one AP-like view. Crop to a single femur and avoid implants if possible. DICOM pixel values are rescaled when metadata is present; PNG/JPEG physical scale is assumed from the SSM prior.")

uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
request_key = (
    uploaded.name if uploaded is not None else None,
    hashlib.sha256(uploaded_bytes).hexdigest() if uploaded_bytes is not None else None,
    polarity, threshold_bias, iterations, crop_top, crop_bottom, crop_left, crop_right,
)
if st.session_state.get("request_key") != request_key:
    st.session_state.result = None
    st.session_state.input_image = None
    st.session_state.request_key = request_key

if uploaded is None:
    st.info("Upload one femur X-ray to begin. In this prepared working copy the model is ready; a fresh clone needs a compatible model built from the required CT-derived training data.")
else:
    temp_path = None
    try:
        suffix = Path(uploaded.name).suffix.lower() or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_bytes)
            temp_path = Path(tmp.name)
        image = read_image(temp_path)
        h0, w0 = image.shape
        y0, y1 = int(h0*crop_top/100), int(h0*(100-crop_bottom)/100)
        x0, x1 = int(w0*crop_left/100), int(w0*(100-crop_right)/100)
        if y1-y0 < 128 or x1-x0 < 96:
            raise ValueError("Crop leaves too little image. Reduce the crop margins.")
        image = image[y0:y1, x0:x1]
        st.session_state.input_image = image
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Input X-ray")
            st.image(image, clamp=True, use_container_width=True)
            st.caption(f"{image.shape[1]} × {image.shape[0]} px · normalized grayscale")
        with right:
            st.subheader("Image checks")
            st.write(f"**Resolution:** {image.shape[1]} × {image.shape[0]} pixels")
            st.write(f"**Contrast range:** {image.max()-image.min():.3f}")
            st.write("**Projection assumption:** single AP-like view")
            if image.shape[0] < 256 or image.shape[1] < 128:
                st.error("Image resolution is low for a full-femur estimate.")
            if not MODEL_PATH.exists():
                st.error("The trained SSM model is missing. Build it from the HFValid data before reconstruction.")
            run = st.button("Reconstruct 3D Femur", type="primary", disabled=not MODEL_PATH.exists(), use_container_width=True)
        if run:
            try:
                with st.status("Reconstructing from one view…", expanded=True) as status:
                    st.write("Checking the image and extracting a single bone-like component")
                    st.write("Fitting the CT-trained 3D shape model to its normalized silhouette")
                    result = reconstruct(image, MODEL_PATH, iterations, polarity, threshold_bias)
                    st.write("Extracting a millimetre-scale surface mesh")
                    status.update(label="Reconstruction completed", state="complete", expanded=False)
                st.session_state.result = result
            except Exception as exc:
                st.session_state.result = None
                st.error(f"Reconstruction failed: {exc}")
    except Exception as exc:
        st.error(f"Could not read this X-ray: {exc}")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

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
        st.metric("Soft silhouette Dice", f"{result['projection_dice']:.3f}")
        st.metric("Runtime", f"{result['elapsed_seconds']:.1f} s")
        st.metric("Compute", result["device"].upper())
        st.write("**Model:** voxel-space PCA SSM")
        st.write(f"**Modes:** {result['n_components']}")
        st.write(f"**Training shapes:** {result['n_training_subjects']}")
        st.write(f"**Shape-prior weight:** {result['prior_weight']:.3f}")
        st.write(f"**Voxel spacing:** {', '.join(f'{x:.2f}' for x in result['spacing_mm'])} mm")
        st.write("**Mesh dimensions (mm):** " + str(tuple(round(x,1) for x in result["dimensions_mm"])))
        st.write("**2D projection nuisance pose (log scale, shifts):** " + str(tuple(round(x,3) for x in result["fitted_pose_log_scale_shift"])))
        st.line_chart(result["history"], height=150, use_container_width=True)
        st.write("**Ground-truth accuracy:** unavailable for this uploaded X-ray")
        st.caption("The learned SSM sets hidden depth and physical size. For ordinary PNG/JPEG images, absolute scale is not measured from the image.")

    cinput, cproj = st.columns(2)
    with cinput:
        st.markdown("#### Input and extracted contour")
        mask = result["mask"]
        src = st.session_state.input_image
        overlay = np.repeat(src[..., None], 3, axis=2)
        overlay[mask] = overlay[mask] * 0.45 + np.array([0.0, 0.9, 0.2], dtype=np.float32) * 0.55
        st.image(np.clip(overlay, 0, 1), clamp=True, use_container_width=True)
        st.caption(f"Threshold-based component · {result['segmentation']['polarity']} contrast · {100*result['segmentation']['component_area_fraction']:.1f}% of image")
    with cproj:
        st.markdown("#### Normalized silhouette fit")
        tgt = result["target_projection"]
        pred = result["projection"] >= 0.5
        rgb = np.zeros((*tgt.shape, 3), dtype=np.uint8)
        rgb[...,0] = (tgt & ~pred) * 255
        rgb[...,1] = (tgt & pred) * 220
        rgb[...,2] = (~tgt & pred) * 255
        st.image(rgb, use_container_width=True)
        st.caption("Red = input contour only · green = overlap · blue = fitted SSM only. This is a silhouette comparison, not a full radiographic DRR.")

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        save_results(result, out, st.session_state.input_image)
        mesh_bytes = (out / "femur_reconstruction.stl").read_bytes()
        report_bytes = (out / "reconstruction.json").read_bytes()
        projection_bytes = (out / "projection_fit.png").read_bytes()
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download STL for 3D Slicer", mesh_bytes, "femur_reconstruction.stl", "model/stl", use_container_width=True)
    d2.download_button("Download reconstruction report", report_bytes, "reconstruction.json", "application/json", use_container_width=True)
    d3.download_button("Download silhouette comparison", projection_bytes, "projection_fit.png", "image/png", use_container_width=True)
    st.caption("In 3D Slicer: drag the downloaded STL into the window or use File → Add Data. STL coordinates are the SSM canonical anatomical frame in millimetres.")




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
