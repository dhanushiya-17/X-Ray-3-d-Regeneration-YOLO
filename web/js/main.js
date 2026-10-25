import { Stage } from "./viewer.js";
import { alerts, api, bytes, debounce, esc, fmt, icon, lineChart, modeBars, sparkline, toast } from "./ui.js";

const PAGES = [
  { id: "case", label: "Case", icon: icon.case, tag: "THE INPUT" },
  { id: "inferences", label: "Inferences", icon: icon.scan, tag: "THE READING" },
  { id: "model", label: "3D model", icon: icon.cube, tag: "THE SURFACE" },
  { id: "export", label: "Export", icon: icon.download, tag: "THE HANDOFF" },
  { id: "method", label: "Method", icon: icon.book, tag: "THE METHOD" },
];
const ITERATIONS = [{ v: 120, label: "Fast" }, { v: 250, label: "Standard" }, { v: 500, label: "Thorough" }];
const STAGES = { queued: "Queued", "loading model": "Loading shape model", segmenting: "Tracing the bone", fitting: "Fitting the shape prior", network: "Running the network", surface: "Extracting the surface", spread: "Sampling shape spread", "writing exports": "Writing exports" };
const VIEWS = { ap: "AP view", lat: "Lateral view" };
const METHODS = {
  network: { label: "Learned network", note: "AP + optional lateral" },
  ssm: { label: "Shape-model fit", note: "one AP view" },
};

// Each view slot holds one upload with its own frame and mask settings. The
// shape-model fit only uses the AP slot.
const blankSlot = () => ({ upload: null, seg: null, segError: null, segBusy: false, crop: { top: 0, bottom: 0, left: 0, right: 0 }, polarity: "auto", threshold_bias: 0 });

const S = {
  status: null, showOverlay: true, method: "ssm", active: "ap",
  slots: { ap: blankSlot(), lat: blankSlot() },
  settings: { iterations: 250, prior_weight: 0.02, use_dicom_spacing: false, magnification: 1.0, spread: true },
  caseId: null, case: null, mesh: null, meanMesh: null, poll: null, route: "case",
  view: { mode: "solid", colorBy: "bone", ghost: false },
};
const slot = (key = S.active) => S.slots[key];
const usedViews = () => (S.method === "network" ? ["ap", "lat"] : ["ap"]);
const anyUpload = () => usedViews().some((k) => S.slots[k].upload);
const methodOk = (m = S.method) => !!S.status?.methods?.[m]?.available;

const $ = (sel, root = document) => root.querySelector(sel);
const view = $("#view");
let stage = null;

// --------------------------------------------------------------------- boot
async function boot() {
  try { stage = new Stage($("#stage")); } catch (err) {
    console.error(err);
    showBanner("3D view unavailable: this browser could not start WebGL.", "err");
  }
  $("#nav").innerHTML = PAGES.map((p) => `<li><a href="#/${p.id}" data-page="${p.id}">${p.icon}<span>${p.label}</span></a></li>`).join("");
  $("#rail-ticks").innerHTML = PAGES.map(() => "<i></i>").join("");
  $("#new-case").innerHTML = `${icon.plus}<span>New case</span>`;
  $("#new-case").addEventListener("click", newCase);
  $("#toggle-rotate").addEventListener("click", () => {
    if (!stage) return;
    stage.setAutoRotate(!stage.autoRotate);
    syncDockToggles();
  });
  $("#toggle-wire").addEventListener("click", () => {
    setViewMode(S.view.mode === "wireframe" ? "solid" : "wireframe");
  });
  $("#dock-cta").addEventListener("click", () => dockAction?.());
  setupFileInput();
  window.addEventListener("hashchange", route);
  await loadStatus();
  const saved = sessionStorage.getItem("bv-case");
  if (saved) await restoreCase(saved);
  route();
}

async function loadStatus() {
  try {
    S.status = await api("/api/status");
  } catch (err) {
    showBanner(err.message, "err");
    setEngine("err", "Offline");
    return;
  }
  const st = S.status;
  S.method = st.default_method || "ssm";
  const net = st.methods?.network;
  const demo = (st.model_available && st.model?.synthetic_demo) || (net?.available && net.synthetic_demo);
  if (!st.model_available && !net?.available) {
    setEngine("err", "No model");
    showBanner(`No model found. Put the trained network at <code>models/multiview_ap_lat_d2.pt</code> (and <code>models/multiview_ap_d2.pt</code>), build the shape model with <code>scripts/build_ssm_v2.py</code>, or try the interface with <code>python serve.py --demo</code>.`, "err");
  } else {
    setEngine(demo ? "warn" : "ok", st.cuda_available ? `CUDA · ${st.cuda_device || "GPU"}` : "CPU · local");
    if (demo) showBanner("Demo models: procedural synthetic shapes, not anatomy. Use them to try the interface only.");
  }
  if (st.model_available) {
    try {
      S.meanMesh = await api("/api/model/mean-mesh");
      if (stage && !S.mesh) stage.setMesh(S.meanMesh, { kind: "mean", animate: false });
      stage?.setGhost(S.meanMesh);
    } catch (err) { console.warn(err); }
  }
}

function setEngine(state, text) {
  const el = $("#engine");
  el.dataset.state = state;
  el.innerHTML = `<i></i><span>${esc(text)}</span>`;
}

function showBanner(html, tone = "warn") {
  const b = $("#banner");
  b.innerHTML = html;
  b.dataset.tone = tone;
  b.hidden = false;
}

// ------------------------------------------------------------------- routing
let dockAction = null;

function route() {
  const id = (location.hash.replace(/^#\/?/, "") || "case").split("?")[0];
  S.route = PAGES.some((p) => p.id === id) ? id : "case";
  const index = PAGES.findIndex((p) => p.id === S.route);
  const page = PAGES[index];
  document.querySelectorAll("#nav a").forEach((a) => {
    a.toggleAttribute("aria-current", a.dataset.page === S.route);
    if (a.dataset.page === S.route) a.setAttribute("aria-current", "page");
    a.dataset.locked = String(["inferences", "export"].includes(a.dataset.page) && !hasResult());
  });
  $("#rail-num").textContent = String(index + 1).padStart(2, "0");
  $("#rail-label").textContent = page.tag.replace("THE ", "The ").toLowerCase().replace(/^the/, "The");
  document.querySelectorAll("#rail-ticks i").forEach((t, i) => t.classList.toggle("on", i === index));
  $("#tag-num").textContent = String(index + 1).padStart(2, "0");
  $("#tag-label").textContent = page.tag;
  document.title = `${page.label} · BoneVision`;
  renderPage();
}

function go(page) { location.hash = `#/${page}`; }
function hasResult() { return S.case?.status === "done" && !!S.case.result; }
function running() { return S.case && ["queued", "running"].includes(S.case.status); }

function renderPage() {
  const r = S.route;
  document.body.dataset.stage = r === "model" ? "full" : r === "case" ? "soft" : "dim";
  document.body.dataset.route = r;
  requestAnimationFrame(() => stage?._resize());
  document.body.dataset.hero = r === "case" && !anyUpload() && !running() ? "1" : "0";
  $(".dock .pill-group").hidden = !(r === "case" || r === "model");
  const renderers = { case: renderCase, inferences: renderInferences, model: renderModel, export: renderExport, method: renderMethod };
  renderers[r]();
  view.firstElementChild?.classList.add("page-enter");
  updateDock();
  syncDockToggles();
}

function updateDock() {
  const cta = $("#dock-cta");
  let label, ico = icon.arrow, disabled = false;
  if (S.route === "case") {
    if (running()) { label = "Cancel estimate"; ico = icon.stop; dockAction = cancelCase; }
    else if (!slot("ap").upload) { label = "Upload AP X-ray"; ico = icon.upload; dockAction = () => { S.active = "ap"; $("#file-input").click(); }; }
    else {
      label = "Estimate 3D femur"; dockAction = startCase;
      disabled = !methodOk() || usedViews().some((k) => { const v = S.slots[k]; return v.upload && (!v.seg || v.segBusy || !!v.segError); });
    }
  } else if (S.route === "inferences") { label = hasResult() ? "Open 3D model" : "Start a case"; dockAction = () => go(hasResult() ? "model" : "case"); }
  else if (S.route === "model") { label = hasResult() ? "Export files" : "Start a case"; ico = hasResult() ? icon.download : icon.arrow; dockAction = () => go(hasResult() ? "export" : "case"); }
  else if (S.route === "export") {
    label = hasResult() ? "Download bundle" : "Start a case"; ico = hasResult() ? icon.zip : icon.arrow;
    dockAction = hasResult() ? () => { location.href = `/api/cases/${S.caseId}/bundle.zip`; } : () => go("case");
  } else { label = "Start a case"; dockAction = () => go("case"); }
  cta.innerHTML = `${ico}<span>${label}</span>`;
  cta.disabled = disabled;
  cta.title = disabled && S.route === "case" && !methodOk() ? `${METHODS[S.method].label} is not available` : "";
}

function syncDockToggles() {
  const rot = $("#toggle-rotate"), wire = $("#toggle-wire");
  const on = !!stage?.autoRotate;
  rot.setAttribute("aria-pressed", String(on));
  rot.innerHTML = `${on ? icon.pause : icon.rotate}<span>${on ? "Pause" : "Rotate"}</span>`;
  const w = S.view.mode === "wireframe";
  wire.setAttribute("aria-pressed", String(w));
  wire.innerHTML = `${icon.grid}<span>${w ? "Solid" : "Wireframe"}</span>`;
}

function setViewMode(mode) {
  S.view.mode = mode;
  stage?.setMode(mode);
  syncDockToggles();
  document.querySelectorAll('input[name="surface"]').forEach((i) => { i.checked = i.value === mode; });
}

// --------------------------------------------------------------------- files
function setupFileInput() {
  const input = $("#file-input");
  input.addEventListener("change", () => { if (input.files[0]) loadFile(input.files[0]); input.value = ""; });
  const veil = $("#drop-veil");
  let depth = 0;
  window.addEventListener("dragenter", (e) => { if (e.dataTransfer?.types?.includes("Files")) { depth++; veil.hidden = false; } });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) veil.hidden = true; });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault(); depth = 0; veil.hidden = true;
    const f = e.dataTransfer?.files?.[0];
    if (f) loadFile(f);
  });
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") { veil.hidden = true; depth = 0; } });
}

async function loadFile(file) {
  if (running()) { toast("Wait for the current estimate to finish, or cancel it.", "err"); return; }
  if (S.method !== "network") S.active = "ap";
  const key = S.active;
  const fd = new FormData();
  fd.append("file", file);
  toast(`Reading ${file.name}…`);
  try {
    const up = await api("/api/uploads", { method: "POST", body: fd });
    const old = S.slots[key].upload;
    if (old) api(`/api/uploads/${old.id}`, { method: "DELETE" }).catch(() => {});
    S.slots[key] = { ...blankSlot(), upload: { ...up, name: file.name } };
    S.active = key; S.showOverlay = true;
    if (key === "ap") S.settings.use_dicom_spacing = false;
    clearCase();
    if (S.route !== "case") go("case"); else renderPage();
    runSegment(key);
  } catch (err) {
    toast(err.message, "err");
  }
}

function clearCase() {
  stopPolling();
  S.caseId = null; S.case = null; S.mesh = null;
  sessionStorage.removeItem("bv-case");
  if (stage && S.meanMesh) stage.setMesh(S.meanMesh, { kind: "mean" });
}

function newCase() {
  if (running()) cancelCase();
  Object.values(S.slots).forEach((v) => { if (v.upload) api(`/api/uploads/${v.upload.id}`, { method: "DELETE" }).catch(() => {}); });
  S.slots = { ap: blankSlot(), lat: blankSlot() }; S.active = "ap";
  clearCase();
  go("case");
  renderPage();
}

// ---------------------------------------------------------------- segmentation
async function runSegment(key = S.active) {
  const v = S.slots[key];
  if (!v.upload) return;
  v.segBusy = true; updateSegUI();
  const token = Symbol("seg"); v.token = token;
  try {
    const res = await api(`/api/uploads/${v.upload.id}/segment`, { json: { crop: v.crop, polarity: v.polarity, threshold_bias: v.threshold_bias } });
    if (v.token !== token) return;
    v.seg = res; v.segError = null;
  } catch (err) {
    if (v.token !== token) return;
    v.seg = null; v.segError = err.message;
    if (err.status === 404) { toast("This upload expired from server memory. Please load the image again.", "err"); S.slots[key] = blankSlot(); renderPage(); return; }
  } finally {
    if (v.token === token) { v.segBusy = false; updateSegUI(); updateDock(); updateViewTabs(); }
  }
}
const runSegmentSoon = debounce(() => runSegment(), 260);

// ------------------------------------------------------------------ case page
function methodStep() {
  const ms = S.status?.methods || {};
  const net = ms.network;
  const latOk = net?.lateral_supported;
  return `<div class="step">
        <h2 class="step-title"><b>00</b>Reconstruction model</h2>
        <div class="seg" role="radiogroup" aria-label="Reconstruction model">
          ${Object.entries(METHODS).map(([k, m]) => `<input type="radio" name="method" id="m-${k}" value="${k}" ${S.method === k ? "checked" : ""} ${ms[k]?.available ? "" : "disabled"}><label for="m-${k}" ${ms[k]?.available ? "" : 'style="opacity:.45" title="' + esc(ms[k]?.problem || "Not available") + '"'}>${m.label}</label>`).join("")}
        </div>
        <p class="tiny faint" style="margin:8px 0 0">${S.method === "network"
          ? `Trained on simulated X-rays of 101 CT femurs. Upload one AP view${latOk ? " and, for better depth, one lateral view" : " (no lateral checkpoint found, so AP only)"}.`
          : "Fits a statistical shape model to one AP outline. Depth comes from the population prior."}</p>
        ${S.method === "network" ? `<div class="seg" role="tablist" aria-label="Views" id="view-tabs" style="margin-top:12px">${viewTabs()}</div>` : ""}
      </div>`;
}

function viewTabs() {
  const latOk = S.status?.methods?.network?.lateral_supported;
  return ["ap", "lat"].map((k) => {
    const v = S.slots[k];
    const state = v.segError ? " ⚠" : v.upload ? (v.seg ? " ✓" : " …") : k === "lat" ? " (optional)" : "";
    const off = k === "lat" && !latOk;
    return `<input type="radio" name="vtab" id="vt-${k}" value="${k}" ${S.active === k ? "checked" : ""} ${off ? "disabled" : ""}><label for="vt-${k}" ${off ? 'style="opacity:.45"' : ""}>${VIEWS[k]}${state}</label>`;
  }).join("");
}

function updateViewTabs() {
  const el = $("#view-tabs");
  if (!el) return;
  el.innerHTML = viewTabs();
  bindViewTabs();
}

function bindViewTabs() {
  view.querySelectorAll('input[name="vtab"]').forEach((i) => i.addEventListener("change", () => { S.active = i.value; renderPage(); }));
}

function renderCase() {
  if (!anyUpload() && !running()) return renderHero();
  if (S.method !== "network") S.active = "ap";
  const v = slot();
  const up = v.upload;
  const meta = up?.meta || {};
  const spacing = meta.pixel_spacing_mm;
  const network = S.method === "network";
  view.innerHTML = `
  <section class="workspace" aria-label="Prepare the X-ray">
    <div class="panel scroll" id="prep">
      ${methodStep()}
      ${up ? `<div class="step">
        <div class="file-card">
          <div class="thumb" style="background-image:url('${up.preview || ""}')"></div>
          <div style="min-width:0;flex:1">
            <div class="name" title="${esc(up.name)}">${network ? `<span class="faint">${VIEWS[S.active]} · </span>` : ""}${esc(up.name || S.case?.name || "Case")}</div>
            <div class="small faint num">${esc(meta.format || "")} · ${meta.shape ? `${meta.shape[1]}×${meta.shape[0]} px` : ""}${spacing ? ` · ${fmt(spacing[0], 3)} mm/px` : ""}</div>
          </div>
          <button class="link-btn" type="button" id="replace">Replace</button>
          ${network && S.active === "lat" ? '<button class="link-btn" type="button" id="remove-view">Remove</button>' : ""}
        </div>
        ${alerts(up.warnings)}
      </div>` : `<div class="step"><p class="small muted" style="margin:0 0 12px">${network && S.active === "lat" ? "A lateral view of the same femur, taken at 90° to the AP, lets the network see depth instead of guessing it." : "Upload the AP view first."}</p>
        <button class="pill pill-ice" type="button" id="add-view">${icon.upload}<span>Upload ${VIEWS[S.active].toLowerCase()}</span></button></div>`}
      ${up ? `<div class="step">
        <h2 class="step-title"><b>01</b>Frame one femur</h2>
        <p class="small muted" style="margin:0">Drag the box on the image so it holds a single, full-length femur. Hardware and the other leg confuse the threshold.</p>
        <details class="more"><summary>Exact margins</summary>
          <div class="crop-grid">${["top", "bottom", "left", "right"].map((k) => `<label>${k[0].toUpperCase() + k.slice(1)} %<input class="num-input" type="number" min="0" max="45" step="1" data-crop="${k}" value="${Math.round(v.crop[k] * 100)}"></label>`).join("")}</div>
          <button class="link-btn" type="button" id="reset-crop" style="margin-top:8px">Reset frame</button>
        </details>
      </div>
      <div class="step">
        <h2 class="step-title"><b>02</b>Trace the bone</h2>
        <div class="field"><span class="label" id="pol-l">Bone contrast</span>
          <div class="seg" role="radiogroup" aria-labelledby="pol-l">
            ${[["auto", "Auto"], ["bright", "Bright bone"], ["dark", "Dark bone"]].map(([k, l]) => `<input type="radio" name="polarity" id="pol-${k}" value="${k}" ${v.polarity === k ? "checked" : ""}><label for="pol-${k}">${l}</label>`).join("")}
          </div>
        </div>
        <div class="field">
          <div class="field-row"><label for="thr">Threshold offset</label><span class="small num" id="thr-v">${fmt(v.threshold_bias, 2)}</span></div>
          <input id="thr" type="range" min="-0.2" max="0.2" step="0.01" value="${v.threshold_bias}">
        </div>
        <div id="seg-summary" class="small muted" style="margin-top:12px"></div>
        <div id="seg-alerts"></div>
      </div>` : ""}
      ${network ? `<div class="step">
        <h2 class="step-title"><b>03</b>What the network assumes</h2>
        <ul class="list small muted" style="margin:0">
          <li>Each view is a true AP or true lateral of the same femur.</li>
          <li>Femur length is set to the training mean; the image carries no scale.</li>
          <li>Right legs are mirrored, so the result is shown as a left femur.</li>
        </ul>
      </div>` : `<div class="step">
        <h2 class="step-title"><b>03</b>Fit settings</h2>
        <div class="field"><span class="label" id="it-l">Fit effort</span>
          <div class="seg" role="radiogroup" aria-labelledby="it-l">
            ${ITERATIONS.map((o) => `<input type="radio" name="iters" id="it-${o.v}" value="${o.v}" ${S.settings.iterations === o.v ? "checked" : ""}><label for="it-${o.v}">${o.label}</label>`).join("")}
          </div>
        </div>
        <label class="switch" style="margin-top:10px"><span>Estimate shape spread <span class="faint">(heuristic)</span></span><input type="checkbox" id="spread" ${S.settings.spread ? "checked" : ""}><span class="track"></span></label>
        <label class="switch"><span>Size from DICOM pixel spacing</span><input type="checkbox" id="dicom" ${S.settings.use_dicom_spacing ? "checked" : ""} ${spacing ? "" : "disabled"}><span class="track"></span></label>
        <p class="tiny faint" style="margin:2px 0 0">${spacing ? `${esc(meta.spacing_source)} is ${fmt(spacing[0], 3)} mm/px. Divide by the radiographic magnification if you know it; otherwise size is still uncertain.` : "No DICOM spacing in this file, so length is assumed from the population mean."}</p>
        <div class="field" id="mag-field" ${S.settings.use_dicom_spacing ? "" : "hidden"}><div class="field-row"><label for="mag">Magnification factor</label><input class="num-input" id="mag" type="number" min="0.6" max="1.9" step="0.01" value="${S.settings.magnification}"></div></div>
        <details class="more"><summary>Shape-prior weight</summary>
          <div class="field"><div class="field-row"><label for="prior">Prior weight</label><span class="small num" id="prior-v">${fmt(S.settings.prior_weight, 3)}</span></div>
          <input id="prior" type="range" min="0" max="0.2" step="0.005" value="${S.settings.prior_weight}">
          <p class="tiny faint" style="margin:0">0.020 is the evaluated default. Lower values follow the outline more closely and allow less typical shapes; nothing else has been validated.</p></div>
        </details>
      </div>`}
      <div class="step" id="case-error"></div>
    </div>
    <div class="spacer" aria-hidden="true"></div>
    <div class="panel" id="imgpanel" style="display:flex;flex-direction:column;align-self:start;max-height:100%" ${up ? "" : "hidden"}>
      <div class="panel-head"><span class="eyebrow muted">${network ? VIEWS[S.active] : "X-ray"}</span>
        <div class="seg" role="radiogroup" aria-label="Image layer" style="min-width:210px">
          <input type="radio" name="layer" id="ly-mask" value="mask" ${S.showOverlay ? "checked" : ""}><label for="ly-mask">Mask</label>
          <input type="radio" name="layer" id="ly-orig" value="orig" ${S.showOverlay ? "" : "checked"}><label for="ly-orig">Original</label>
        </div>
      </div>
      <div class="imgbox" id="imgbox"></div>
      <div class="legend"><span><i style="background:var(--warm)"></i>Selected bone region</span><span><i style="border:1.5px solid var(--ice)"></i>Frame</span><span><i style="background:#3a434c"></i>Outside frame</span></div>
    </div>
  </section>
  <div id="progress-host"></div>`;
  bindCase();
  updateSegUI();
  updateProgress();
}

function renderHero() {
  const st = S.status;
  const n = st?.model?.n_training_subjects;
  view.innerHTML = `
  <section class="hero" aria-label="Start">
    <div class="hero-left">
      <div class="eyebrow">${S.method === "network" ? "Femur reconstruction from X-rays" : "Single-view femur shape estimation"}</div>
      <h1 class="display">${S.method === "network" ? "Two views.<br>One femur." : "One view.<br>A shape within."}</h1>
      <p class="lede">${S.method === "network"
        ? "Upload an AP femur radiograph, and a lateral one if you have it. BoneVision traces the bone, runs a network trained on CT-derived X-rays and returns a surface you can orbit and open in 3D Slicer."
        : "Upload one AP femur radiograph. BoneVision traces the bone, fits a CT-trained shape prior and returns a surface you can orbit and open in 3D Slicer."} <span>A research estimate, not a measurement.</span></p>
      <div class="dropzone">
        <button class="pill pill-ice pill-lg" type="button" id="hero-upload">${icon.upload}<span>${S.method === "network" ? "Choose the AP X-ray" : "Choose an X-ray"}</span></button>
        <div class="formats" aria-label="Accepted formats"><span class="chip">PNG</span><span class="chip">JPEG</span><span class="chip">TIFF</span><span class="chip">DICOM</span></div>
      </div>
      <p class="small faint" style="margin-top:18px">or drop a file anywhere. Images stay in this computer's memory and are never written to disk.</p>
      <div id="recent"></div>
    </div>
    <div class="hero-right" aria-hidden="true">
      <span class="serif">Inferred,<br>not measured.</span>
      <span class="eyebrow">${S.method === "network" ? (st?.methods?.network?.synthetic_demo ? "Synthetic demo network." : "Network trained on 101 CT femurs.") : st?.model?.synthetic_demo ? `Depth from a ${n}-shape synthetic demo prior.` : n ? `Depth from a ${n}-shape CT prior.` : "Depth from a CT shape prior."}<br>Exported for 3D Slicer.</span>
    </div>
  </section>
  <p class="hero-caption">${S.mesh ? "Latest estimate" : st?.model_available ? "Population mean shape · drag to orbit" : ""}</p>`;
  $("#hero-upload").addEventListener("click", () => { S.active = "ap"; $("#file-input").click(); });
  api("/api/cases").then((list) => {
    const done = list.filter((c) => c.status === "done");
    if (!done.length) return;
    $("#recent").innerHTML = `<div class="eyebrow muted" style="margin:34px 0 10px">This session</div><div class="formats">${done.slice(0, 4).map((c) => `<button class="pill pill-sm" type="button" data-case="${c.id}">${esc(c.name)} · ${c.method === "network" ? "network" : "SSM"} · ${fmt(c.silhouette_soft_dice, 2)}</button>`).join("")}</div>`;
    $("#recent").querySelectorAll("[data-case]").forEach((b) => b.addEventListener("click", async () => { await restoreCase(b.dataset.case); go("model"); }));
  }).catch(() => {});
}

function bindCase() {
  view.querySelectorAll('input[name="method"]').forEach((i) => i.addEventListener("change", () => { S.method = i.value; clearCase(); renderPage(); }));
  bindViewTabs();
  $("#replace")?.addEventListener("click", () => $("#file-input").click());
  $("#add-view")?.addEventListener("click", () => $("#file-input").click());
  $("#remove-view")?.addEventListener("click", () => {
    const v = slot();
    if (v.upload) api(`/api/uploads/${v.upload.id}`, { method: "DELETE" }).catch(() => {});
    S.slots[S.active] = blankSlot(); clearCase(); renderPage();
  });
  view.querySelectorAll('input[name="polarity"]').forEach((i) => i.addEventListener("change", () => { slot().polarity = i.value; runSegment(); }));
  const thr = $("#thr");
  const paintRange = (el) => el.style.setProperty("--p", `${((el.value - el.min) / (el.max - el.min)) * 100}%`);
  if (thr) { paintRange(thr); thr.addEventListener("input", () => { slot().threshold_bias = Number(thr.value); $("#thr-v").textContent = fmt(thr.value, 2); paintRange(thr); runSegmentSoon(); }); }
  const prior = $("#prior");
  if (prior) { paintRange(prior); prior.addEventListener("input", () => { S.settings.prior_weight = Number(prior.value); $("#prior-v").textContent = fmt(prior.value, 3); paintRange(prior); }); }
  view.querySelectorAll('input[name="iters"]').forEach((i) => i.addEventListener("change", () => { S.settings.iterations = Number(i.value); }));
  $("#spread")?.addEventListener("change", (e) => { S.settings.spread = e.target.checked; });
  $("#dicom")?.addEventListener("change", (e) => { S.settings.use_dicom_spacing = e.target.checked; $("#mag-field").hidden = !e.target.checked; });
  $("#mag")?.addEventListener("change", (e) => { S.settings.magnification = Math.min(1.9, Math.max(0.6, Number(e.target.value) || 1)); e.target.value = S.settings.magnification; });
  view.querySelectorAll('input[name="layer"]').forEach((i) => i.addEventListener("change", () => { S.showOverlay = i.value === "mask"; updateSegUI(); }));
  view.querySelectorAll("[data-crop]").forEach((inp) => inp.addEventListener("change", () => {
    const k = inp.dataset.crop, crop = slot().crop;
    crop[k] = Math.min(45, Math.max(0, Number(inp.value) || 0)) / 100;
    inp.value = Math.round(crop[k] * 100);
    placeCropRect(); runSegment();
  }));
  $("#reset-crop")?.addEventListener("click", () => { slot().crop = { top: 0, bottom: 0, left: 0, right: 0 }; syncCropInputs(); placeCropRect(); runSegment(); });
}

function syncCropInputs() {
  view.querySelectorAll("[data-crop]").forEach((inp) => { inp.value = Math.round(slot().crop[inp.dataset.crop] * 100); });
}

function updateSegUI() {
  const box = $("#imgbox");
  const v = slot();
  if (!box || !v.upload) return;
  const [h, w] = v.upload.meta.shape;
  const src = S.showOverlay && v.seg ? v.seg.overlay : v.upload.preview;
  const maxH = Math.max(260, window.innerHeight - 330);
  box.innerHTML = `<div style="position:relative;width:min(100%, ${(maxH * w) / h}px);aspect-ratio:${w}/${h}">
      <img src="${src}" alt="${S.showOverlay ? "X-ray with the selected bone region highlighted" : "Uploaded X-ray"}" style="width:100%;height:100%;max-height:none">
      <div class="crop-layer" id="crop-layer" style="inset:0"><div class="crop-rect" id="crop-rect" tabindex="0" aria-label="Frame. Drag to move; drag corners to resize.">
        <span class="h" data-h="nw"></span><span class="h" data-h="ne"></span><span class="h" data-h="sw"></span><span class="h" data-h="se"></span></div></div>
    </div>${v.segBusy ? '<span class="busy spinner" role="status" aria-label="Updating mask"></span>' : ""}`;
  placeCropRect();
  bindCrop();
  const sum = $("#seg-summary"), al = $("#seg-alerts");
  if (v.segError) {
    sum.textContent = "";
    al.innerHTML = alerts([{ level: "error", message: v.segError }]);
  } else if (v.seg) {
    const g = v.seg.segmentation;
    sum.innerHTML = `Mask: <b>${esc(g.polarity)}</b> bone · ${fmt(g.component_area_fraction * 100, 1)}% of frame · elongation ${fmt(g.elongation, 1)} · Otsu ${fmt(g.threshold_used, 2)}`;
    al.innerHTML = alerts(v.seg.warnings);
  }
}

function placeCropRect() {
  const r = $("#crop-rect");
  if (!r) return;
  const c = slot().crop;
  Object.assign(r.style, { left: `${c.left * 100}%`, top: `${c.top * 100}%`, width: `${(1 - c.left - c.right) * 100}%`, height: `${(1 - c.top - c.bottom) * 100}%` });
}

function bindCrop() {
  const layer = $("#crop-layer"), rect = $("#crop-rect");
  if (!layer || running()) return;
  let drag = null;
  const clamp = (v) => Math.min(0.45, Math.max(0, v));
  rect.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    rect.setPointerCapture(e.pointerId);
    drag = { h: e.target.dataset.h || "move", x: e.clientX, y: e.clientY, start: { ...slot().crop }, box: layer.getBoundingClientRect() };
  });
  rect.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = (e.clientX - drag.x) / drag.box.width, dy = (e.clientY - drag.y) / drag.box.height;
    const s = drag.start, c = { ...s };
    if (drag.h === "move") {
      const mx = Math.min(Math.max(dx, -s.left), s.right), my = Math.min(Math.max(dy, -s.top), s.bottom);
      c.left = clamp(s.left + mx); c.right = clamp(s.right - mx); c.top = clamp(s.top + my); c.bottom = clamp(s.bottom - my);
    } else {
      if (drag.h.includes("w")) c.left = clamp(s.left + dx);
      if (drag.h.includes("e")) c.right = clamp(s.right - dx);
      if (drag.h.includes("n")) c.top = clamp(s.top + dy);
      if (drag.h.includes("s")) c.bottom = clamp(s.bottom - dy);
    }
    slot().crop = c; placeCropRect();
  });
  const end = () => { if (!drag) return; drag = null; syncCropInputs(); runSegment(); };
  rect.addEventListener("pointerup", end);
  rect.addEventListener("pointercancel", end);
  rect.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 0.05 : 0.01;
    const map = { ArrowUp: ["top", -step, "bottom", step], ArrowDown: ["top", step, "bottom", -step], ArrowLeft: ["left", -step, "right", step], ArrowRight: ["left", step, "right", -step] };
    const m = map[e.key];
    if (!m) return;
    e.preventDefault();
    const c = { ...slot().crop };
    if (c[m[0]] + m[1] < 0 || c[m[2]] + m[3] < 0) return;
    c[m[0]] = clamp(c[m[0]] + m[1]); c[m[2]] = clamp(c[m[2]] + m[3]);
    slot().crop = c; placeCropRect(); syncCropInputs(); runSegmentSoon();
  });
}

// ------------------------------------------------------------------ case runs
async function startCase() {
  const ap = slot("ap");
  if (!ap.upload || running()) return;
  const body = S.method === "network"
    ? { method: "network", views: usedViews().filter((k) => S.slots[k].upload).map((k) => { const v = S.slots[k]; return { upload_id: v.upload.id, kind: k, crop: v.crop, polarity: v.polarity, threshold_bias: v.threshold_bias }; }) }
    : { method: "ssm", upload_id: ap.upload.id, crop: ap.crop, polarity: ap.polarity, threshold_bias: ap.threshold_bias, ...S.settings };
  try {
    const { id } = await api("/api/cases", { json: body });
    S.caseId = id;
    S.case = { id, method: S.method, status: "queued", stage: "queued", iteration: 0, total: S.method === "network" ? 0 : S.settings.iterations, live_history: [] };
    sessionStorage.setItem("bv-case", id);
    renderPage();
    startPolling();
  } catch (err) {
    toast(err.message, "err");
  }
}

async function cancelCase() {
  if (!S.caseId) return;
  try { await api(`/api/cases/${S.caseId}/cancel`, { method: "POST" }); } catch (err) { toast(err.message, "err"); }
}

function startPolling() {
  stopPolling();
  S.poll = setInterval(pollCase, 400);
}
function stopPolling() { if (S.poll) clearInterval(S.poll); S.poll = null; }

async function pollCase() {
  if (!S.caseId) return stopPolling();
  let c;
  try { c = await api(`/api/cases/${S.caseId}`); } catch (err) {
    stopPolling(); toast(err.message, "err"); S.case = null; renderPage(); return;
  }
  S.case = c;
  if (running()) { updateProgress(); return; }
  stopPolling();
  if (c.status === "done") {
    try {
      S.mesh = await api(`/api/cases/${S.caseId}/mesh`);
      stage?.setMesh(S.mesh, { kind: "result" });
      if (S.view.colorBy === "spread" && !S.mesh.spread) S.view.colorBy = "bone";
      toast(`Estimate ready · ${c.result.method === "network" ? "view agreement" : "silhouette agreement"} ${fmt(c.result.silhouette_soft_dice, 2)}`);
      go("model");
      renderPage();
    } catch (err) { toast(err.message, "err"); }
  } else if (c.status === "failed") {
    toast("The estimate failed. Details are in the panel.", "err");
    renderPage();
    const el = $("#case-error");
    if (el) el.innerHTML = alerts([{ level: "error", message: c.error }]);
  } else {
    toast("Estimate cancelled.");
    renderPage();
  }
}

function updateProgress() {
  const host = $("#progress-host");
  if (!host) return;
  if (!running()) { host.innerHTML = ""; return; }
  const c = S.case;
  const pct = c.total ? Math.round((c.iteration / c.total) * 100) : 0;
  const hist = (c.live_history || []).map((p) => p[1]);
  if (c.method === "network") {
    host.innerHTML = `<div class="panel progress-card" role="status" aria-live="polite">
    <div class="eyebrow">${esc(STAGES[c.stage] || c.stage)}</div>
    <div class="small faint" style="margin-top:8px">Placing each view on the network grid, trying 4 orientations per view, then one network pass.</div>
    <div class="bar"><i style="width:${{ queued: 5, segmenting: 25, network: 55, surface: 80, "writing exports": 92 }[c.stage] || 10}%"></i></div>
  </div>`;
    view.querySelectorAll("#prep input, #prep button").forEach((el) => { el.disabled = true; });
    return;
  }
  host.innerHTML = `<div class="panel progress-card" role="status" aria-live="polite">
    <div class="eyebrow">${esc(STAGES[c.stage] || c.stage)}</div>
    <div class="count num">${c.iteration || 0}<small> / ${c.total}</small></div>
    <div class="small faint">iterations · 4 orientation hypotheses in parallel</div>
    <div class="bar"><i style="width:${pct}%"></i></div>
    ${sparkline(hist)}
    <div class="small faint num" style="margin-top:6px">best objective ${c.loss != null ? fmt(c.loss, 4) : "—"}</div>
  </div>`;
  view.querySelectorAll("#prep input, #prep button").forEach((el) => { el.disabled = true; });
}

async function restoreCase(id) {
  try {
    const c = await api(`/api/cases/${id}`);
    S.caseId = id; S.case = c;
    sessionStorage.setItem("bv-case", id);
    if (c.status === "done") {
      S.mesh = await api(`/api/cases/${id}/mesh`);
      stage?.setMesh(S.mesh, { kind: "result", animate: false });
    } else if (running()) startPolling();
  } catch {
    sessionStorage.removeItem("bv-case");
  }
}

// --------------------------------------------------------------- inferences
function emptyState(title, body) {
  view.innerHTML = `<section class="page"><div class="empty"><div><div class="eyebrow muted">Nothing here yet</div><h2 class="h2">${title}</h2><p class="muted">${body}</p><a class="pill pill-ice pill-lg" href="#/case">${icon.upload}<span>Start with an X-ray</span></a></div></div></section>`;
}

function renderNetworkInferences() {
  const r = S.case.result, id = S.caseId;
  const val = r.checkpoint_validation || {};
  view.innerHTML = `
  <section class="page" aria-labelledby="inf-h">
    <header class="page-head">
      <div><div class="eyebrow">02 · Inferences</div><h1 class="h2" id="inf-h">What the network saw, <em>and what it assumed.</em></h1></div>
      <p class="muted small">Every number here describes this upload. None of them is a 3D accuracy measurement: there is no ground truth for an uploaded X-ray.</p>
    </header>
    <div class="grid g4" style="margin-bottom:20px">
      <div class="card metric"><div class="eyebrow">View agreement</div><div class="value num">${fmt(r.silhouette_soft_dice, 3)}<small> Dice</small></div><p>Mean overlap, across views, of the 3D estimate's silhouette with your traced outline on the network grid. Lowest view ${fmt(r.min_view_dice, 3)}. 2D only.</p></div>
      <div class="card metric"><div class="eyebrow">Views used</div><div class="value" style="font-size:30px;margin:16px 0 12px">${r.model_views.map((k) => k.toUpperCase()).join(" + ")}</div><p>${r.model_views.includes("lat") ? "The lateral view constrains depth directly." : "AP only: depth comes from what the network learned."}${r.ignored_views.length ? ` Ignored: ${esc(r.ignored_views.join(", "))}.` : ""}</p></div>
      <div class="card metric"><div class="eyebrow">Checkpoint</div><div class="value num">${val.dice != null ? fmt(val.dice, 3) : "—"}<small> val Dice</small></div><p>${esc(r.checkpoint)}${val.epoch != null ? `, epoch ${val.epoch}` : ""}. Validation on synthetic X-rays at training time${val.synthetic_demo ? " (synthetic demo network)" : ""}.</p></div>
      <div class="card metric"><div class="eyebrow">Assumed length</div><div class="value num">${fmt(r.dimensions_mm[0], 0)}<small> mm</small></div><p>Not measured: every view is scaled so the femur has the training-mean length. Shown as a ${r.canonical_side === "L" ? "left" : "right"} femur.</p></div>
    </div>
    ${r.warnings.length ? `<div class="card flat" style="margin-bottom:20px"><div class="card-title"><h3>Flags for this case</h3><span class="small faint">${r.warnings.length}</span></div>${alerts(r.warnings)}</div>` : ""}
    <div class="grid g2" style="margin-bottom:20px">
      ${r.views.map((v) => `<figure class="card" style="margin:0"><div class="card-title"><h3>${esc(VIEWS[v.kind])}</h3><span class="small faint">your X-ray</span></div>
        <div class="figure"><img src="/api/cases/${id}/files/input_overlay_${v.kind}.png?inline=1" alt="${esc(VIEWS[v.kind])} with the traced mask and the estimate's outline"></div>
        <figcaption class="legend"><span><i style="background:var(--warm)"></i>Traced bone mask</span><span><i style="background:#8cd9ff"></i>Estimate's outline mapped back</span></figcaption></figure>`).join("")}
    </div>
    <div class="grid g2" style="margin-bottom:20px">
      <figure class="card" style="margin:0"><div class="card-title"><h3>On the network grid</h3><span class="small faint">${r.views.map((v) => VIEWS[v.kind]).join(" · ")}</span></div>
        <div class="figure" style="display:grid;place-items:center;background:#0e141b"><img class="pixel" style="max-height:520px;width:auto" src="/api/cases/${id}/files/view_fit.png?inline=1" alt="Each view's traced outline against the estimate's silhouette on the network grid"></div>
        <figcaption class="legend"><span><i style="background:#78c8a0"></i>Both</span><span><i style="background:#ed9e59"></i>Outline only</span><span><i style="background:#78bef5"></i>Estimate only</span></figcaption></figure>
      <div class="card"><div class="card-title"><h3>Per view</h3><span class="small faint">${r.hypotheses_tested} placements tried</span></div>
        <table class="data"><thead><tr><th>View</th><th class="num">Dice</th><th class="num">Orientation match</th><th>Placement</th></tr></thead><tbody>
        ${r.views.map((v) => `<tr><td>${esc(VIEWS[v.kind])}<div class="tiny faint">${esc(v.input.source_name || "")}</div></td><td class="num">${fmt(v.soft_dice, 3)}</td>
          <td class="num">${fmt(v.orientation_match, 2)} <span class="faint">(+${fmt(v.orientation_margin, 2)})</span></td>
          <td class="small">${v.flipped_long_axis ? "turned 180°" : "as oriented"}${v.mirrored ? ", mirrored" : ""} · ${fmt(v.assumed_mm_per_pixel, 3)} mm/px</td></tr>`).join("")}
        </tbody></table>
        <p class="tiny faint" style="margin:12px 0 0">Each view is placed on the grid in four ways (head up or down, mirrored or not); the placement whose outline best matches the mean training femur is used. A small margin means the choice was close. Mirrored usually means a right leg or a PA image.</p></div>
    </div>
    <div class="grid g3">
      ${r.views.map((v) => `<div class="card"><div class="card-title"><h3>${esc(VIEWS[v.kind])} mask</h3><span class="chip">classical</span></div><dl class="kv">
        <dt>Polarity</dt><dd>${esc(v.segmentation.polarity)}</dd><dt>Threshold</dt><dd>${fmt(v.segmentation.threshold_used, 3)}</dd>
        <dt>Region area</dt><dd>${fmt(v.segmentation.component_area_fraction * 100, 1)}%</dd><dt>Elongation</dt><dd>${fmt(v.segmentation.elongation, 1)}</dd>
        <dt>Touches edge</dt><dd>${v.segmentation.touches_border ? "yes" : "no"}</dd></dl></div>`).join("")}
      <div class="card"><div class="card-title"><h3>Model &amp; run</h3>${val.synthetic_demo ? '<span class="chip" data-tone="warn">synthetic demo</span>' : ""}</div><dl class="kv">
        <dt>Network</dt><dd>${esc(r.checkpoint)}</dd><dt>Views</dt><dd>${r.model_views.map((k) => k.toUpperCase()).join(" + ")}</dd>
        <dt>Grid</dt><dd>${r.grid_shape.join("×")} (÷${r.network_downsample} inside)</dd><dt>Device</dt><dd>${esc(r.device.toUpperCase())}</dd><dt>Run time</dt><dd>${fmt(r.elapsed_seconds, 1)} s</dd></dl></div>
    </div>
  </section>`;
}

function renderInferences() {
  if (!hasResult()) return emptyState("No estimate <em>yet.</em>", "Inferences appear after a run: the traced masks, the estimate's outline on your X-rays and the checks behind them.");
  if (S.case.result.method === "network") return renderNetworkInferences();
  const r = S.case.result, id = S.caseId;
  const tf = r.transform, spread = r.spread;
  const best = r.hypotheses[r.hypothesis.index];
  const hyp = r.hypotheses.map((h, i) => ({ ...h, i })).sort((a, b) => a.loss - b.loss);
  const thin = (arr, max = 500) => { if (arr.length <= max) return arr; const k = arr.length / max; return Array.from({ length: max }, (_, i) => arr[Math.floor(i * k)]); };
  const colors = ["#d8ecfa", "#ee9e59", "#7fcca5", "#c7a6f0"];
  view.innerHTML = `
  <section class="page" aria-labelledby="inf-h">
    <header class="page-head">
      <div><div class="eyebrow">02 · Inferences</div><h1 class="h2" id="inf-h">What the model read, <em>and what it assumed.</em></h1></div>
      <p class="muted small">Every number here describes this upload. None of them is a 3D accuracy measurement: there is no ground truth for an uploaded X-ray.</p>
    </header>
    <div class="grid g4" style="margin-bottom:20px">
      <div class="card metric"><div class="eyebrow">Silhouette agreement</div><div class="value num">${fmt(r.silhouette_soft_dice, 3)}<small> soft Dice</small></div><p>Overlap of the projected estimate with your traced outline, 0–1. 2D only; hidden depth is not scored.</p></div>
      <div class="card metric"><div class="eyebrow">Orientation</div><div class="value" style="font-size:30px;margin:16px 0 12px">${esc(best.label)}</div><p>${esc(best.meaning)} Margin to next: ${fmt(r.orientation_margin, 3)}${r.orientation_margin < 0.01 ? " (ambiguous)" : ""}.</p></div>
      <div class="card metric"><div class="eyebrow">Shape typicality</div><div class="value num">${fmt(r.coefficient_norm, 2)}<small> σ from mean</small></div><p>Length of the shape-coefficient vector. A typical training shape sits near ${fmt(r.coefficient_norm_expected, 1)}; the prior pulls estimates inward.</p></div>
      <div class="card metric"><div class="eyebrow">Assumed length</div><div class="value num">${fmt(tf.length_mm_assumed, 0)}<small> mm</small></div><p>${tf.calibrated ? "From DICOM pixel spacing (magnification as entered)." : "Not measured: an uncalibrated image is sized to the population mean."}</p></div>
    </div>
    ${r.warnings.length ? `<div class="card flat" style="margin-bottom:20px"><div class="card-title"><h3>Flags for this case</h3><span class="small faint">${r.warnings.length}</span></div>${alerts(r.warnings)}</div>` : ""}
    <div class="grid g2" style="margin-bottom:20px">
      <figure class="card" style="margin:0"><div class="card-title"><h3>On your X-ray</h3><span class="small faint">input pixel grid</span></div>
        <div class="figure"><img src="/api/cases/${id}/files/input_overlay.png?inline=1" alt="Uploaded X-ray with the selected mask and the fitted outline"></div>
        <figcaption class="legend"><span><i style="background:var(--warm)"></i>Traced bone mask</span><span><i style="background:#8cd9ff"></i>Fitted outline mapped back</span></figcaption></figure>
      <figure class="card" style="margin:0"><div class="card-title"><h3>In the model frame</h3><span class="small faint">canonical AP projection</span></div>
        <div class="figure" style="display:grid;place-items:center;background:#0e141b"><img class="pixel" style="max-height:520px;width:auto" src="/api/cases/${id}/files/silhouette_fit.png?inline=1" alt="Canonical-frame comparison of the traced outline and the fitted projection"></div>
        <figcaption class="legend"><span><i style="background:#78c8a0"></i>Both</span><span><i style="background:#ed9e59"></i>Outline only</span><span><i style="background:#78bef5"></i>Estimate only</span></figcaption></figure>
    </div>
    <div class="grid g2" style="margin-bottom:20px">
      <div class="card"><div class="card-title"><h3>Orientation hypotheses</h3><span class="small faint">fitted in parallel</span></div>
        <table class="data"><thead><tr><th>Hypothesis</th><th class="num">Soft Dice</th><th class="num">Objective</th><th style="width:30%"><span class="sr-only">Relative</span></th></tr></thead><tbody>
        ${hyp.map((h) => `<tr data-selected="${h.i === r.hypothesis.index}"><td>${esc(h.label)} ${h.i === r.hypothesis.index ? '<span class="chip" data-tone="info">selected</span>' : ""}<div class="tiny faint">${esc(h.meaning)}</div></td><td class="num">${fmt(h.soft_dice, 3)}</td><td class="num">${fmt(h.loss, 4)}</td><td><div class="meter"><i style="width:${Math.max(4, h.soft_dice * 100)}%"></i></div></td></tr>`).join("")}
        </tbody></table>
        <p class="tiny faint" style="margin:12px 0 0">The 2D fit cannot tell which leg a mirrored outline belongs to in 3D; it only finds the image orientation that matches the model convention.</p></div>
      <div class="card"><div class="card-title"><h3>Optimization trace</h3><span class="small faint">objective per iteration</span></div>
        ${lineChart(r.history_all.map((s, i) => ({ name: r.hypotheses[i].label, values: thin(s), color: colors[i], opacity: i === r.hypothesis.index ? 1 : 0.35, width: i === r.hypothesis.index ? 2.2 : 1.2 })), { xLabel: "iterations", yLabel: "objective" })}
        <div class="legend">${r.hypotheses.map((h, i) => `<span><i style="background:${colors[i]};opacity:${i === r.hypothesis.index ? 1 : 0.5}"></i>${esc(h.label)}</span>`).join("")}</div>
        <p class="tiny faint" style="margin:10px 0 0">Objective = (1 − soft Dice) + 0.08·BCE + prior·mean(z²) + pose penalty. Lower is better; the best state, not the last, is kept.</p></div>
    </div>
    <div class="grid g2" style="margin-bottom:20px">
      <div class="card"><div class="card-title"><h3>Shape modes</h3><span class="small faint">first ${Math.min(16, r.coefficients_z.length)} of ${r.coefficients_z.length}</span></div>
        ${modeBars(r.coefficients_z, r.explained_variance_ratio)}
        <p class="tiny faint" style="margin:10px 0 0">Each bar is how far the estimate moved along one principal shape mode of the training femurs. Amber bars hit the ±2.5σ clamp.</p></div>
      <div class="card"><div class="card-title"><h3>Shape spread</h3><span class="small faint">${spread.available ? `${spread.n_samples} samples` : "not computed"}</span></div>
        ${spread.available ? `<p class="small muted" style="margin-top:0">Shapes whose silhouettes fit within ${fmt(spread.tolerance, 2)} Dice of the estimate were sampled; the voxel spread shows where one view leaves geometry loose. A heuristic, not a confidence interval.</p>
          <table class="data"><tbody>${Object.entries(spread.by_region || {}).map(([k, v]) => `<tr><td style="text-transform:capitalize">${k}</td><td class="num">${fmt(v, 3)}</td><td style="width:45%"><div class="meter"><i style="width:${Math.min(100, (v / 0.5) * 100)}%;background:linear-gradient(90deg,#f2b56b,#e3572e)"></i></div></td></tr>`).join("")}
          <tr><td>Mean (all vertices)</td><td class="num">${fmt(spread.mean, 3)}</td><td></td></tr></tbody></table>
          <p class="tiny faint" style="margin:10px 0 0">Values are occupancy standard deviation (0 = every sample agrees, 0.5 = maximal disagreement). View them on the surface in the 3D model page.</p>`
          : `<p class="muted small">${esc(spread.reason || "Turn on shape spread in the fit settings to estimate it.")}</p>`}
      </div>
    </div>
    <div class="grid g3">
      <div class="card"><div class="card-title"><h3>Segmentation</h3><span class="chip">classical</span></div><dl class="kv">
        <dt>Method</dt><dd>Otsu + morphology</dd><dt>Polarity</dt><dd>${esc(r.segmentation.polarity)}</dd><dt>Threshold</dt><dd>${fmt(r.segmentation.threshold_used, 3)}</dd>
        <dt>Region area</dt><dd>${fmt(r.segmentation.component_area_fraction * 100, 1)}%</dd><dt>Elongation</dt><dd>${fmt(r.segmentation.elongation, 1)}</dd>
        <dt>Candidates</dt><dd>${r.segmentation.n_candidates}</dd><dt>Touches edge</dt><dd>${r.segmentation.touches_border ? "yes" : "no"}</dd></dl></div>
      <div class="card"><div class="card-title"><h3>Standardization</h3></div><dl class="kv">
        <dt>Rotation to long axis</dt><dd>${fmt(tf.rotation_deg, 1)}°</dd><dt>Scale</dt><dd>${fmt(tf.mm_per_pixel, 3)} mm/px</dd><dt>Calibrated</dt><dd>${tf.calibrated ? "DICOM spacing" : "no (prior length)"}</dd>
        <dt>Assumed width</dt><dd>${fmt(tf.width_mm_assumed, 0)} mm</dd><dt>Inside model grid</dt><dd>${fmt(tf.fraction_inside_grid * 100, 0)}%</dd>
        <dt>Fitted pose</dt><dd>×${fmt(Math.exp(r.pose_log_scale_shift[0]), 3)}, shift ${fmt(r.pose_log_scale_shift[1], 3)}/${fmt(r.pose_log_scale_shift[2], 3)}</dd></dl></div>
      <div class="card"><div class="card-title"><h3>Model &amp; run</h3>${r.model.synthetic_demo ? '<span class="chip" data-tone="warn">synthetic demo</span>' : ""}</div><dl class="kv">
        <dt>Shape model</dt><dd>${esc(r.model.file)}</dd><dt>SHA-256</dt><dd class="mono">${esc(r.model.sha256.slice(0, 12))}…</dd><dt>Modes</dt><dd>${r.model.n_components}</dd>
        <dt>Training shapes</dt><dd>${r.model.n_training_subjects ?? "—"}</dd><dt>Grid</dt><dd>${r.model.grid_shape.join("×")}</dd>
        <dt>Device</dt><dd>${esc(r.device.toUpperCase())}</dd><dt>Run time</dt><dd>${fmt(r.elapsed_seconds, 1)} s</dd></dl></div>
    </div>
  </section>`;
}

// ------------------------------------------------------------------ 3D model
function renderModel() {
  const r = hasResult() ? S.case.result : null;
  const hasSpread = !!S.mesh?.spread;
  if (S.view.colorBy === "spread" && !hasSpread) S.view.colorBy = "bone";
  stage?.setColorBy(S.view.colorBy);
  const dims = r ? r.dimensions_mm : S.meanMesh ? S.meanMesh.bounds_max.map((v, i) => v - S.meanMesh.bounds_min[i]) : null;
  view.innerHTML = `
  <section class="viewer-ui" aria-label="3D viewer controls">
    <div class="panel scroll">
      <div class="eyebrow muted">${r ? "Estimated surface" : "Population mean shape"}</div>
      <h1 class="h3" style="margin:8px 0 2px;font-size:22px;font-weight:400">${r ? esc(S.case.name) : "No estimate yet"}</h1>
      <p class="small faint" style="margin:0 0 6px">${r ? `${S.mesh.face_count.toLocaleString()} triangles · ${r.method === "network" ? "network estimate, left femur" : "canonical frame"}, mm` : "Upload an X-ray to fit it."}</p>
      <div class="field"><span class="label" id="sf-l">Surface</span><div class="seg" role="radiogroup" aria-labelledby="sf-l">
        ${[["solid", "Solid"], ["wireframe", "Wire"], ["xray", "X-ray"]].map(([v, l]) => `<input type="radio" name="surface" id="sf-${v}" value="${v}" ${S.view.mode === v ? "checked" : ""}><label for="sf-${v}">${l}</label>`).join("")}</div></div>
      <div class="field"><span class="label" id="co-l">Colour</span><div class="seg" role="radiogroup" aria-labelledby="co-l">
        <input type="radio" name="color" id="co-bone" value="bone" ${S.view.colorBy === "bone" ? "checked" : ""}><label for="co-bone">Bone</label>
        <input type="radio" name="color" id="co-spread" value="spread" ${S.view.colorBy === "spread" ? "checked" : ""} ${hasSpread ? "" : "disabled"}><label for="co-spread" ${hasSpread ? "" : 'style="opacity:.45"'}>Shape spread</label></div></div>
      <label class="switch" style="margin-top:10px"><span>Overlay population mean</span><input type="checkbox" id="ghost" ${S.view.ghost ? "checked" : ""} ${r && S.meanMesh ? "" : "disabled"}><span class="track"></span></label>
      <div class="field"><span class="label">Camera</span><div class="btn-grid">
        ${[["ap", "AP"], ["lateral", "Lateral"], ["axial", "Axial"], ["iso", "Oblique"]].map(([v, l]) => `<button class="pill" type="button" data-view="${v}">${l}</button>`).join("")}</div></div>
      <button class="pill" type="button" id="shot" style="width:100%;margin-top:14px">${icon.camera}<span>Save screenshot</span></button>
    </div>
    <div class="panel scroll">
      <div class="eyebrow muted">Extent</div>
      <dl class="kv" style="margin-top:12px">
        <dt>Long axis</dt><dd class="num">${dims ? fmt(dims[0], 1) : "—"} mm</dd>
        <dt>Mediolateral</dt><dd class="num">${dims ? fmt(dims[1], 1) : "—"} mm</dd>
        <dt>Anteroposterior</dt><dd class="num">${dims ? fmt(dims[2], 1) : "—"} mm</dd>
      </dl>
      <p class="tiny faint" style="margin:10px 0 0">${r?.transform?.calibrated ? "Sized from DICOM spacing; magnification as entered." : r?.method === "network" ? "Bounding extents of the estimate. Length is set to the training mean, so overall size is not measured from your images." : "Bounding extents of the estimate. Without calibration the overall size comes from the population prior, not your image."}</p>
      ${hasSpread ? `<div style="margin-top:22px"><div class="eyebrow muted">Shape spread</div><div class="legend-ramp" role="img" aria-label="Colour ramp from agreement to disagreement"></div>
        <div class="field-row tiny faint"><span>0 · agree</span><span>0.25</span><span>0.5 · disagree</span></div>
        <p class="tiny faint" style="margin:8px 0 0">Where silhouette-consistent samples disagree. Heuristic, not a confidence interval.</p></div>` : ""}
      <div style="margin-top:22px" class="tiny faint">Drag to orbit · scroll or pinch to zoom.</div>
    </div>
  </section>`;
  view.querySelectorAll('input[name="surface"]').forEach((i) => i.addEventListener("change", () => setViewMode(i.value)));
  view.querySelectorAll('input[name="color"]').forEach((i) => i.addEventListener("change", () => { S.view.colorBy = i.value; stage?.setColorBy(i.value); }));
  $("#ghost")?.addEventListener("change", (e) => { S.view.ghost = e.target.checked; stage?.setGhostVisible(e.target.checked); });
  view.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => stage?.frame(b.dataset.view)));
  $("#shot").addEventListener("click", () => {
    if (!stage) return;
    const a = document.createElement("a");
    a.href = stage.screenshot();
    a.download = r ? "bonevision_estimate.png" : "bonevision_mean_shape.png";
    a.click();
  });
  stage?.setGhostVisible(S.view.ghost && !!r);
}

// -------------------------------------------------------------------- export
function renderExport() {
  if (!hasResult()) return emptyState("Nothing to <em>export yet.</em>", "After a fit you can download the mesh as STL, OBJ or a colour PLY, plus the JSON report and the overlay images, one at a time or as a single bundle.");
  const r = S.case.result, id = S.caseId;
  const net = r.method === "network";
  const ico = (n) => (n.endsWith(".png") ? icon.image : n.endsWith(".json") ? icon.file : icon.cube);
  view.innerHTML = `
  <section class="page" aria-labelledby="ex-h">
    <header class="page-head"><div><div class="eyebrow">04 · Export</div><h1 class="h2" id="ex-h">Take it with you, <em>into 3D Slicer.</em></h1></div>
      <a class="pill pill-ice pill-lg" href="/api/cases/${id}/bundle.zip">${icon.zip}<span>Download everything (.zip)</span></a></header>
    <div class="grid g2" style="margin-bottom:20px">
      ${r.files.map((f) => `<div class="card file-tile"><div class="ico">${ico(f.name)}</div><div style="min-width:0"><div class="name">${esc(f.name)}</div><p>${esc(f.description)}</p><p class="tiny faint num">${bytes(f.bytes)}</p></div>
        <a class="pill pill-sm" href="/api/cases/${id}/files/${encodeURIComponent(f.name)}" download aria-label="Download ${esc(f.name)}">${icon.download}<span>Download</span></a></div>`).join("")}
    </div>
    <div class="grid g3" style="margin-bottom:20px">
      ${net ? `<div class="card steps4"><span class="n">01</span><h3>Open Slicer</h3><p>Unzip the bundle and run <span class="mono">Slicer --python-script BoneVisionLoader.py</span> in that folder, or use <b>File → Add Data</b> on <span class="mono">femur_reconstruction.stl</span> (tick LabelMap for the NRRD).</p></div>
      <div class="card steps4"><span class="n">02</span><h3>Know the frame</h3><p>Femur landmark frame in millimetres, always a left femur, tagged as RAS so Slicer loads it unchanged. It is not registered to the patient.</p></div>
      <div class="card steps4"><span class="n">03</span><h3>Keep the report</h3><p><span class="mono">reconstruction.json</span> records the inputs, checkpoint, per-view agreement, warnings and limitations for this estimate.</p></div>`
      : `<div class="card steps4"><span class="n">01</span><h3>Open Slicer</h3><p>Use <b>File → Add Data</b> or drag <span class="mono">femur_estimate.stl</span> into the window. It loads as a model node.</p></div>
      <div class="card steps4"><span class="n">02</span><h3>Know the frame</h3><p>Coordinates are the shape model's canonical frame in millimetres, not patient RAS/LPS. Slicer will not align it to a CT by itself.</p></div>
      <div class="card steps4"><span class="n">03</span><h3>Keep the report</h3><p><span class="mono">report.json</span> records the input hash, model hash, settings, metrics, warnings and limitations for this estimate.</p></div>`}
    </div>
    <div class="grid g2">
      <div class="card"><div class="card-title"><h3>Coordinate frame</h3></div><dl class="kv">
        <dt>Units</dt><dd>millimetres</dd><dt>Origin</dt><dd>femoral-head landmark</dd><dt>+X</dt><dd>long axis (distal negative)</dd><dt>+Y</dt><dd>toward greater trochanter</dd><dt>Z</dt><dd>X × Y</dd>
        ${net ? "<dt>Side</dt><dd>left femur (right legs mirrored)</dd><dt>Length</dt><dd>training mean</dd>" : "<dt>Pose applied</dt><dd>fitted in-plane scale/shift</dd>"}<dt>Patient space</dt><dd>not registered</dd></dl>
        <p class="tiny faint" style="margin:12px 0 0">Mesh dimensions ${r.dimensions_mm.map((d) => fmt(d, 1)).join(" × ")} mm · ${r.mesh.vertices.toLocaleString()} vertices · ${r.mesh.faces.toLocaleString()} faces.</p></div>
      <div class="card"><div class="card-title"><h3>Report preview</h3><a class="link-btn" href="/api/cases/${id}/files/${r.report_file}?inline=1" target="_blank" rel="noopener">Open raw</a></div><div class="code-block" id="report-pre">Loading…</div></div>
    </div>
  </section>`;
  api(`/api/cases/${id}/files/${r.report_file}?inline=1`).then((rep) => {
    const trimmed = { ...rep };
    for (const k of ["optimization_loss_history", "shape_coefficients_z"]) if (Array.isArray(rep[k])) trimmed[k] = `[${rep[k].length} values]`;
    const el = $("#report-pre");
    if (el) el.textContent = JSON.stringify(trimmed, null, 2);
  }).catch((err) => { const el = $("#report-pre"); if (el) el.textContent = err.message; });
}

// -------------------------------------------------------------------- method
function renderMethod() {
  const n = S.status?.model?.n_training_subjects;
  view.innerHTML = `
  <section class="page" aria-labelledby="me-h">
    <header class="page-head"><div><div class="eyebrow">05 · Method</div><h1 class="h2" id="me-h">How the estimate is made, <em>and where it stops.</em></h1></div>
      <p class="muted small">BoneVision is a research prototype with two methods: a network trained on simulated X-rays, and a statistical shape model fitted to one outline. Neither has been validated on real radiographs.</p></header>
    <div class="card" style="margin-bottom:20px"><div class="card-title"><h3>Learned network</h3><span class="chip" data-tone="info">AP + optional lateral</span></div>
      <p class="small muted" style="margin-top:0">A 2D encoder per view, back-projection into a 3D grid and a 3D decoder, trained on simulated X-rays (DRRs) of 101 HFValid CT femurs with right legs mirrored onto the left. For an upload, each view is traced, placed on the training grid (long axis matched to the mean femur, head at the grid's origin, length scaled to the training mean) in four orientations, and the best-matching placement is fed to the network once.</p>
      <div id="net-evidence"><p class="muted small">Loading…</p></div></div>
    <div class="eyebrow muted" style="margin:0 0 10px">Shape-model fit (one AP view)</div>
    <div class="grid g4 steps4" style="margin-bottom:20px">
      <div class="card"><span class="n">01</span><h3>Trace</h3><p>Otsu thresholding, morphology and an elongation/area score choose one bone-like region in the framed image.</p></div>
      <div class="card"><span class="n">02</span><h3>Standardize</h3><p>The outline is rotated to its long axis and converted to the model's millimetre grid, sized to the population mean unless DICOM spacing is used.</p></div>
      <div class="card"><span class="n">03</span><h3>Fit the prior</h3><p>A PCA shape model${n ? ` of ${n} CT femurs` : ""} is fitted to the outline under four orientation hypotheses, with a nuisance scale and shift.</p></div>
      <div class="card"><span class="n">04</span><h3>Surface</h3><p>Marching cubes turns the fitted occupancy into a millimetre mesh; nearby shapes that fit as well show where the result is loose.</p></div>
    </div>
    <div class="grid g2" style="margin-bottom:20px">
      <div class="card"><div class="card-title"><h3>What it is</h3></div><ul class="list">
        <li>A population-prior estimate of femur shape that agrees with one AP outline.</li><li>A navigable surface and STL for research visualization.</li>
        <li>An honest record of settings, orientation choice, flags and fit diagnostics.</li></ul></div>
      <div class="card"><div class="card-title"><h3>What it is not</h3></div><ul class="list">
        <li>Not a measurement: hidden depth and (without calibration) size come from the prior.</li><li>Not validated on real clinical radiographs.</li>
        <li>Not for diagnosis, treatment or surgical planning. Silhouette agreement is not 3D accuracy.</li></ul></div>
    </div>
    <div class="card" id="evidence" style="margin-bottom:20px"><div class="card-title"><h3>Evidence so far</h3><span class="small faint">synthetic only</span></div><p class="muted small">Loading…</p></div>
    <div class="card"><div class="card-title"><h3>Data &amp; licences</h3></div><ul class="list small">
      <li><b>HFValid Collection</b> (CC BY 4.0): 101 CT femurs with segmentations and landmarks; the shape model's source. Not redistributed here.</li>
      <li><b>xvr-data Femur subset</b> (CC BY-NC-ND 4.0): five paired real X-ray/CT cases; not yet used for training or evaluation.</li>
      <li><b>VSD lower-extremity meshes</b> (CC BY-NC-SA 4.0): 60 femur meshes; not in the model until they are registered and evaluated.</li></ul></div>
  </section>`;
  api("/api/evaluation").then((ev) => {
    const net = ev.network, ne = $("#net-evidence");
    if (net && ne) ne.innerHTML = `<p class="small muted" style="margin:0 0 10px">${esc(net.protocol)} <span class="faint">Source: ${esc(net.source)}.</span></p>
      <table class="data"><thead><tr>${net.columns.map((c, i) => `<th class="${i ? "num" : ""}">${esc(c)}</th>`).join("")}</tr></thead><tbody>
      ${net.rows.map((row) => `<tr>${row.map((c, i) => `<td class="${i ? "num" : ""}">${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>
      ${alerts([{ level: "warn", message: net.caveat }])}`;
    const rep = ev.reported;
    const el = $("#evidence");
    if (!el) return;
    el.innerHTML = `<div class="card-title"><h3>Evidence so far</h3><span class="chip" data-tone="warn">synthetic · ideal mask</span></div>
      <p class="small muted" style="margin-top:0">${esc(rep.protocol)} <span class="faint">Source: ${esc(rep.source)}.</span></p>
      <table class="data"><thead><tr>${rep.columns.map((c, i) => `<th class="${i ? "num" : ""}">${esc(c)}</th>`).join("")}</tr></thead><tbody>
      ${rep.rows.map((row) => `<tr>${row.map((c, i) => `<td class="${i ? "num" : ""}">${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>
      <p class="small muted" style="margin:14px 0 0">Worst case kept in the average: ${esc(rep.worst_case)}.</p>
      ${alerts([{ level: "warn", message: rep.caveat }])}
      ${ev.local_runs.length ? `<div class="eyebrow muted" style="margin:22px 0 8px">Runs found on this machine</div>${ev.local_runs.map((run) => `<p class="small" style="margin:4px 0"><span class="mono">${esc(run.file)}</span> · ${run.n_ok} subjects${run.n_failed ? `, ${run.n_failed} failed` : ""}${run.aggregate ? ` · SSM Dice ${fmt(Number(run.aggregate.ssm_dice), 3)} · mean surface ${fmt(Number(run.aggregate.ssm_mean_symmetric_surface_distance_mm), 2)} mm` : ""}</p>`).join("")}` : ""}`;
  }).catch(() => {});
}

boot();
