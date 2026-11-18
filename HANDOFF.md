# HANDOFF: BoneVision UI + 3D overhaul

> **READ FIRST (2026-11-18 15:30 IST): THE PLAN BELOW IS PARTLY SUPERSEDED.**
> `origin/main` moved far ahead of this branch's base (`010e0bc`). Main (as of 2026-10-25) already has: `serve.py` + FastAPI `bonevision/server.py`, a vanilla-JS frontend in `web/` (three.js, fonts and libs vendored, works offline, NO node build step), a trained multi-view network (AP+lateral test Dice 0.943, 0.95 mm mean surface distance; AP-only Dice 0.865), `--demo` mode with synthetic nets, STL/NRRD/OBJ/Slicer exports, multi-view averaging (up to 3 AP + 3 lateral), YOLO fracture script on `feature/fracture-detect-threshold`, CI in `.github/workflows/tests.yml`, and its own `docs/HANDOFF.md` + `docs/HANDOFF_COMPACT.md` (read those: they are the project's source of truth).
> Consequences:
> - Branch `ui-overhaul` is STALE. Do not build on it or merge it. Its only content is the `app.py` move and this file. Start new work from a fresh branch off latest `origin/main` (generic contextual name, e.g. `feature/docker-k8s`, `feature/web-ui-redesign`; no session ids).
> - The React/Vite/Node rewrite, the new `bonevision/` core library, the synthetic femur SSM generator and the new FastAPI `server/` described below are NOT needed: they would duplicate main. Idea worth porting later, on top of main: mesh smoothing (Taubin), per-vertex uncertainty map (main's plan B4), clipping/measurement tools, Linear-style restyle of `web/css/app.css`.
> - Docker/K8s track is now the immediate focus (see that section), built against main's `serve.py`.
> - Other people have merged work into main (PRs #1-#7) and several feature branches exist, so branch from the latest main and keep changes small.

> **Standing rule (from the user):** update this file after EVERY step, started or finished, so a fresh chat can resume exactly where this one stopped. Keep "Step log" and "Remaining steps" current.

Last updated: 2026-11-18 15:40 IST
Repo (local): `/home/claude/work/repo`, branch `ui-overhaul`
Remote: https://github.com/dhanushiya-17/X-Ray-3-d-Regeneration-YOLO (push works as of 2026-11-18; branch `ui-overhaul` is on origin)

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
- None. (Earlier `git push` 403 resolved on 2026-11-18; GitHub access now works.) Item 6 under Decisions (zip/patch delivery) is now only a fallback.

## Step log (newest last)
- [x] Cloned the repo and design-md collection; read the core code, README and Linear DESIGN.md.
- [x] Saved the project goals to memory (`/projects/.../areas/bonevision-ui-overhaul.md`).
- [x] Wrote the first HANDOFF.md and committed it as `e55e7ae` (local only, push failed with 403).
- [x] Restructured HANDOFF.md into a running log (this version).
- [x] 2026-11-18: user asked if she can learn Docker and Kubernetes through this project. Answer: yes. Decision: finish the app first (core, server, frontend), THEN add a containerization track (see "Docker/K8s learning track" below). State at this point: `bonevision/`, `server/`, `frontend/` are still empty; no core code written yet. Sandbox has the `docker` CLI only (daemon untested); no kubectl/kind.

- [x] 2026-11-18: committed `e8e6e83` and pushed `ui-overhaul` to origin successfully.

- [x] 2026-11-18 15:30: user asked "what should I do first, the website runs on localhost, check the repo first". Fetched origin, found main far ahead (see READ FIRST). Re-planned: Docker/K8s track against main's `serve.py`. Nothing built on this branch beyond the handoff.
- [x] 2026-11-18 15:40: user says she is a COMPLETE beginner at Docker and wants to learn from scratch. Teaching mode: mental model first (image vs container, Dockerfile, layers, ports, volumes), then tiny hands-on steps; she runs commands on her own machine (Windows + WSL Ubuntu 24.04, Docker Desktop with WSL2 backend) and pastes output; Claude explains. Lesson 1 given: install Docker Desktop, `docker run hello-world`, `docker run -d -p 8080:80 nginx`, `docker ps`, `docker logs`, `docker stop`. Waiting for her output.

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

## Revised next steps (2026-11-18, supersede "Remaining steps" above)
1. [ ] User runs the app locally from latest main and confirms the baseline: `python serve.py --demo` (no models needed) then open http://127.0.0.1:8000 and check `GET /api/status`. This is the acceptance test for the container.
2. [ ] User writes her FIRST Dockerfile (CPU, `python:3.11-slim` to match CI, torch from the CPU index, `pip install -r requirements-app.txt`). Goal: `docker run -p 8000:8000 <image> ...` shows the same page. Lessons to discover: `serve.py` defaults to host 127.0.0.1 so the container must be started with `--host 0.0.0.0`; models (`models/*.pt`, `*.npz`) are gitignored and big, so mount them as a volume, never bake them in; `--demo` trains synthetic nets on first start (slow, do it at build time or in a volume).
3. [ ] Claude reviews her Dockerfile and explains each decision; then `.dockerignore`, non-root user, HEALTHCHECK on `/api/status`, layer caching.
4. [ ] Then Compose, then Kubernetes on kind. Key K8s lesson specific to this app: uploads and cases live in the server process memory (`MAX_UPLOADS=8`), so more than 1 replica breaks sessions; use 1 replica first, then discuss sticky sessions or moving state to Redis.
5. [ ] Separate, later: restyle `web/` with the Linear tokens listed under Decisions (user's original UI request), on a branch off main.

## Docker lesson ladder (beginner, one lesson per message, wait for her output before the next)
1. [ ] Lesson 1: install Docker Desktop (WSL2), `hello-world`, run nginx with `-p`, `ps`, `logs`, `stop`, `rm`.
2. [ ] Lesson 2: images vs containers, `docker images`, `docker pull python:3.11-slim`, `docker run -it python:3.11-slim python` (poke inside, show it is isolated), `docker exec`.
3. [ ] Lesson 3: write a tiny Dockerfile for a 5-line Python script, `docker build -t`, `docker run`; layers and cache (change a line, rebuild, watch what is reused).
4. [ ] Lesson 4: ports and `0.0.0.0` vs `127.0.0.1` using a tiny FastAPI hello app; then volumes (`-v`) with a file she edits on the host.
5. [ ] Lesson 5: BoneVision's first Dockerfile (her own attempt), `.dockerignore`, review. Then Compose, then Kubernetes.
