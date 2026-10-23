# HANDOFF: BoneVision UI + 3D overhaul

Last updated: 2026-10-23 (branch `ui-overhaul`, nothing committed yet)

## Goal
Redesign the frontend, improve the 3D femur results, and build a proper backend, so the project can be shown to faculty. Design reference: VoltAgent/awesome-design-md.

## Progress so far (audit and planning only, no new code written yet)
- Cloned the repo and read the core code: `scripts/reconstruct_single_xray.py`, `scripts/ssm_reconstruct_v2.py`, `legacy/streamlit_app.py`, `tests/`, README.
- Read the Linear DESIGN.md from awesome-design-md.
- Saved the project goals to memory.
- The local checkout is on branch `ui-overhaul`. `app.py` was already moved to `legacy/streamlit_app.py`, with only its `ROOT` path fixed. Empty `bonevision/`, `server/`, `frontend/` and `samples/` folders are already there.

## Key findings
- The current 3D result is a PCA shape-model (SSM) fit to a thresholded 2D silhouette. One view cannot determine depth, so the output is limited by the method. Some of the "poor" look is likely mesh staircase artifacts, which can be fixed.
- `models/*.npz`, the CT data and `reconstruction/` are gitignored. They are not in the repo and not reachable from this sandbox: HuggingFace and CDNs are blocked, while npm and pip work. I cannot run or validate your real trained model here.
- The README's held-out numbers (about 4.7 mm mean surface distance, worst case Pat099 with Dice 0.37) were produced by `scripts/reconstruct_single_xray.py` and `evaluate_single_view.py`. I will leave those untouched so the numbers stay reproducible.
- The YOLO weights are not in the repo.
- `gh` is not authenticated, so I cannot push to your GitHub. I will hand over a zip or patch instead.

## Decisions made
1. **Design system:** Linear. Near-black canvas, surface ladder, hairline borders, one lavender accent (#5e6ad2). Inter and JetBrains Mono come from npm @fontsource, since CDNs are blocked.
2. **Layout:** the new core library goes in `bonevision/`, the FastAPI API in `server/`, a React + TypeScript + Vite + three.js app in `frontend/`, and demo X-rays in `samples/`. The built `frontend/dist` will be served by FastAPI.
3. **Honesty rules:**
   - A clearly labeled synthetic demo model (`model_kind: synthetic-demo`) is used only for development and tests, with a visible DEMO badge in the UI.
   - Your real `models/ssm_highres.npz` loads automatically when present.
   - No fake accuracy numbers. Held-out and synthetic metrics are shown separately from any uploaded case.
4. **3D quality work (validated on the synthetic model):**
   - Mesh: Gaussian-smoothed marching cubes, largest component only, Taubin smoothing, normals.
   - Fitting: multi-start, multi-scale Dice loss, in-plane rotation, optional lateral view.
   - Per-vertex uncertainty map.
5. **UI features:** drag-and-drop upload, live segmentation tuning, live fit progress, a 3D viewer with clipping, measurement and uncertainty overlay, an ROI/fracture marker mapped onto the mesh, a benchmarks page, exports (STL, OBJ, PLY, GLB, JSON), a printable report, turntable recording, and a command palette.

## Next 3 steps
1. Build `bonevision/` (io, segmentation, fit, mesh, uncertainty, measurements, synthetic generator and demo SSM) plus pytest tests.
2. Build the FastAPI `server/` (jobs, progress polling, mesh and export endpoints, run history) and test it with TestClient.
3. Build the `frontend/` in the Linear style, then screenshot it with Playwright to check, and package it as a zip.
