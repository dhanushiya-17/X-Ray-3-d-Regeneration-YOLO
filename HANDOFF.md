# HANDOFF: BoneVision UI + 3D overhaul

> **Standing rule (from the user):** update this file after EVERY step, started or finished, so a fresh chat can resume exactly where this one stopped. Keep "Step log" and "Remaining steps" current.

Last updated: 2026-11-18 13:20 IST
Repo (local): `/home/claude/work/repo`, branch `ui-overhaul`
Remote: https://github.com/dhanushiya-17/X-Ray-3-d-Regeneration-YOLO (push currently FAILS with 403, see Blockers)

## Goal
Redesign the frontend, improve the 3D femur results, and build a proper backend, so the project can be shown to faculty. Design reference: VoltAgent/awesome-design-md (cloned at `/home/claude/work/designmd`, Linear spec at `src/design-md/linear.app/DESIGN.md`).

## How to resume in a new chat
1. Read this file top to bottom, then `README.md` (esp. "Single-X-ray application" section).
2. Key existing code: `scripts/reconstruct_single_xray.py` (current fit + segmentation, DO NOT modify: it produced the README's published numbers), `scripts/evaluate_single_view.py`, `legacy/streamlit_app.py` (old UI), `tests/test_reconstruction_core.py`.
3. Work on branch `ui-overhaul`. Commit locally as steps finish (a stop hook asks for commit and push).
4. Environment: Python 3.13, torch 2.14 (CPU only, 2 cores, 7 GB RAM), fastapi, uvicorn, scipy, scikit-image, pydicom, matplotlib, Playwright + Chromium preinstalled (`/opt/pw-browsers`). Node 22. npm and pip work. HuggingFace, jsDelivr, unpkg, cdnjs and Google Fonts are BLOCKED, so all frontend assets must come from npm and be bundled.
5. `models/*.npz`, CT data and `reconstruction/` are gitignored and absent. Real SSM npz keys: `mean_shape` (flat), `components` (n_comp x N), `explained_variance`, `grid_shape`, optional `bounds` (default x -470..40, y -60..100, z -70..70 mm), `train_subjects`. Grid is (x long axis, y medio-lateral, z antero-posterior); the AP silhouette sums along z.

## Key findings
- The current 3D result is a PCA shape-model (SSM) fit to a thresholded 2D silhouette. One view cannot determine depth, so output quality is limited by the method. Part of the "poor" look is likely mesh staircase artifacts (raw marching cubes at 0.5 on ~2 mm voxels), which can be fixed.
- README held-out numbers: about 4.7 mm mean surface distance, Dice 0.687, worst case Pat099 with Dice 0.373. Never inflate these.
- YOLO weights and the real SSM are not in the repo or reachable from here.
- `gh` is not authenticated and the Claude GitHub App has no access to the repo.

## Decisions made
1. **Design system:** Linear. Near-black canvas `#010102`, surface ladder `#0f1011 / #141516 / #18191a / #191a1b`, hairlines `#23252a / #34343a`, text `#f7f8f8 / #d0d6e0 / #8a8f98 / #62666d`, single accent lavender `#5e6ad2` (hover `#828fff`), success `#27a644`. Radii: buttons/inputs 8px, cards 12px, panels 16px. No gradients, no second accent, dark only. Fonts: Inter + JetBrains Mono via npm `@fontsource`.
2. **Layout:** core library in `bonevision/`, FastAPI in `server/`, React + TypeScript + Vite + three.js in `frontend/`, demo images in `samples/`. Built `frontend/dist` is served by FastAPI at `/`.
3. **Honesty rules:**
   - A clearly labeled synthetic demo model (`model_kind: synthetic-demo`) is used only for development and tests, with a visible DEMO badge in the UI.
   - The real `models/ssm_highres.npz` loads automatically when present.
   - No fake accuracy numbers. Held-out and synthetic metrics are shown separately from any uploaded case.
4. **3D quality work (validated only on the synthetic model):**
   - Mesh: Gaussian-smoothed marching cubes, largest connected component, Taubin smoothing, vertex normals.
   - Fitting: multi-start, multi-scale Dice loss, in-plane rotation parameter, optional lateral view.
   - Per-vertex uncertainty map, from a linearized Gaussian posterior over PCA coefficients.
5. **UI features:** drag-and-drop upload (PNG/JPG/TIFF/DICOM), live segmentation tuning, live fit progress with a loss curve, 3D viewer (shaded / wireframe / x-ray, clipping plane, two-point measurement, uncertainty overlay), ROI/fracture marker mapped onto the mesh, optional YOLO hook (only if `ultralytics` and `models/yolo_fracture.pt` exist), benchmarks page reading the CSVs in `reconstruction/`, run history, exports (STL, OBJ, PLY, GLB, JSON), printable report, turntable recording, command palette.
6. **Delivery:** push is blocked, so the final handover is a zip/patch of the repo changes.

## Blockers
- `git push` returns 403: Claude has no GitHub access to the repo. Fix: install the Claude GitHub App at https://github.com/apps/claude/installations/select_target, or reconnect GitHub in claude.ai Settings, Connectors.

## Step log (newest last)
- [x] Cloned the repo and design-md collection; read the core code, README and Linear DESIGN.md.
- [x] Saved the project goals to memory (`/projects/.../areas/bonevision-ui-overhaul.md`).
- [x] Wrote the first HANDOFF.md and committed it as `e55e7ae` (local only, push failed with 403).
- [x] Restructured HANDOFF.md into a running log (this version).
- [x] 2026-11-18: user asked if she can learn Docker and Kubernetes through this project. Answer: yes. Decision: finish the app first (core, server, frontend), THEN add a containerization track (see "Docker/K8s learning track" below). State at this point: `bonevision/`, `server/`, `frontend/` are still empty; no core code written yet. Sandbox has the `docker` CLI only (daemon untested); no kubectl/kind.

## Docker/K8s learning track (added 2026-11-18, runs AFTER the app works)
User wants to learn Docker and Kubernetes through this project. She prefers to write things herself first and get reasoning explained, so deliver this as `LEARN-DEVOPS.md` with staged exercises and acceptance checks, and review her Dockerfiles rather than handing finished ones over.
- Stage 1, Docker: multi-stage build (node builds the frontend, python serves it), `.dockerignore`, CPU vs CUDA base image, non-root user, HEALTHCHECK, `models/` as a mounted volume (large and gitignored, never baked into the image), env-based config, layer caching.
- Stage 2, Compose: api plus optional YOLO service, volumes for `models/` and `runs/`.
- Stage 3, Kubernetes on kind or minikube: Deployment, Service, ConfigMap, Secret, PVC for models and runs, readiness/liveness plus a startupProbe (model load is slow), resource requests/limits (torch memory), Ingress, rolling update and rollback, Kustomize or Helm.
- Stage 4: Job and CronJob for `evaluate_single_view.py` and SSM building; initContainer to fetch the model; HPA.
- Stage 5 (optional): split the in-process job thread into an API Deployment plus a Redis-backed worker Deployment so workers scale independently. Prometheus `/metrics`.
- Stage 6: GitHub Actions to build and push to GHCR, then deploy to kind.
- Not covered by this project: cluster administration, control-plane internals, service mesh, multi-tenant security.

## Remaining steps (in order)
1. [ ] `bonevision/` core: `io.py`, `segmentation.py`, `preprocess.py` (returns the 2D transform), `ssm.py`, `fit.py`, `mesh.py`, `uncertainty.py`, `measure.py`, `fracture.py`, `synthetic.py`; plus `scripts/make_demo_ssm.py` and `samples/` generator.
2. [ ] `tests/test_bonevision_core.py`: mesh is closed and smooth, fit recovers a held-out synthetic femur, AP-only vs AP+lateral comparison on synthetic data.
3. [ ] `server/` FastAPI: upload, segmentation preview, job start and poll, mesh binary, exports, history, benchmarks, model info; tests with TestClient.
4. [ ] `frontend/`: Vite + React + TS + three.js in the Linear style; build to `frontend/dist`.
5. [ ] Playwright screenshots to verify the UI (headless Chromium with swiftshader for WebGL).
6. [ ] Update `README.md` (run instructions, honesty notes), `requirements`, `run.sh`.
7. [ ] Independent review by a separate agent, then zip/patch for the user.
8. [ ] Docker/K8s learning track: write `LEARN-DEVOPS.md` (stages above), then review the user's own Dockerfile/manifests as she writes them.
