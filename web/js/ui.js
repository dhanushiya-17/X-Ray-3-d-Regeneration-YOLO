// Small shared helpers: icons, API client, DOM utilities, SVG charts.

const svg = (body, extra = "") =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" ${extra}>${body}</svg>`;

export const icon = {
  case: svg('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M9.5 16.5c1-1.2 1.4-3.6 2.5-5 1.1 1.4 1.5 3.8 2.5 5"/>'),
  scan: svg('<path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2"/><path d="M7 12h10"/><circle cx="12" cy="12" r="3"/>'),
  cube: svg('<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9z"/><path d="m4 7.5 8 4.5 8-4.5M12 12v9"/>'),
  download: svg('<path d="M12 4v11"/><path d="m7 10 5 5 5-5"/><path d="M5 20h14"/>'),
  book: svg('<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5z"/><path d="M4 20.5A2.5 2.5 0 0 0 6.5 23H20v-5"/>', 'style="transform:scale(.92)"'),
  upload: svg('<path d="M12 16V5"/><path d="m7 10 5-5 5 5"/><path d="M5 20h14"/>'),
  plus: svg('<path d="M12 5v14M5 12h14"/>'),
  arrow: svg('<path d="M7 17 17 7"/><path d="M8 7h9v9"/>'),
  rotate: svg('<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 4v5h-5"/>'),
  pause: svg('<path d="M9 5v14M15 5v14"/>'),
  grid: svg('<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9z"/><path d="m4 7.5 8 4.5 8-4.5M12 12v9M8 5.2l8 4.6M16 5.2 8 9.8"/>'),
  info: svg('<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/>'),
  warn: svg('<path d="M12 3 2 20h20z"/><path d="M12 10v4M12 17v.01"/>'),
  x: svg('<circle cx="12" cy="12" r="9"/><path d="m9 9 6 6M15 9l-6 6"/>'),
  camera: svg('<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>'),
  file: svg('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>'),
  zip: svg('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M11 6h1M11 9h1M11 12h1M10.5 15h2v3h-2z"/>'),
  image: svg('<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/>'),
  stop: svg('<rect x="6" y="6" width="12" height="12" rx="2"/>'),
};

export const levelIcon = { info: icon.info, warn: icon.warn, error: icon.x };

export function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export function fmt(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function bytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}

export function toast(message, tone = "ok") {
  const host = document.getElementById("toasts");
  const el = document.createElement("div");
  el.className = "toast";
  el.dataset.tone = tone;
  el.textContent = message;
  host.append(el);
  setTimeout(() => el.remove(), 4200);
}

export function alerts(list) {
  if (!list?.length) return "";
  return `<ul class="alerts">${list.map((w) => `<li data-level="${esc(w.level)}">${levelIcon[w.level] || icon.info}<span>${esc(w.message)}</span></li>`).join("")}</ul>`;
}

export async function api(path, options = {}) {
  const opts = { ...options };
  if (opts.json !== undefined) {
    opts.method = opts.method || "POST";
    opts.headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
    opts.body = JSON.stringify(opts.json);
    delete opts.json;
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error("Cannot reach the BoneVision server. Is serve.py still running?");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { const body = await res.json(); detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail); } catch { /* not json */ }
    const err = new Error(detail || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return res.headers.get("content-type")?.includes("application/json") ? res.json() : res;
}

export function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ------------------------------------------------------------------ charts (SVG)

export function lineChart(series, { width = 560, height = 200, yLabel = "", xLabel = "" } = {}) {
  const pad = { l: 44, r: 12, t: 12, b: 28 };
  const all = series.flatMap((s) => s.values).filter(Number.isFinite);
  if (!all.length) return "";
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; }
  const n = Math.max(...series.map((s) => s.values.length));
  const x = (i) => pad.l + (i / Math.max(n - 1, 1)) * (width - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (height - pad.t - pad.b);
  const ticks = [lo, (lo + hi) / 2, hi];
  const grid = ticks.map((t) => `<line x1="${pad.l}" x2="${width - pad.r}" y1="${y(t)}" y2="${y(t)}" stroke="rgba(255,255,255,.07)"/><text x="${pad.l - 8}" y="${y(t) + 4}" text-anchor="end">${t.toFixed(3)}</text>`).join("");
  const paths = series.map((s) => {
    const d = s.values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 1.6}" stroke-opacity="${s.opacity ?? 1}" vector-effect="non-scaling-stroke"><title>${esc(s.name)}</title></path>`;
  }).join("");
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(yLabel)} over ${esc(xLabel)}">${grid}
    <text x="${pad.l}" y="${height - 6}">0</text><text x="${width - pad.r}" y="${height - 6}" text-anchor="end">${n} ${esc(xLabel)}</text>${paths}</svg>`;
}

export function sparkline(values, { width = 320, height = 54 } = {}) {
  if (values.length < 2) return `<svg class="spark" viewBox="0 0 ${width} ${height}"></svg>`;
  const lo = Math.min(...values), hi = Math.max(...values) + 1e-9;
  const d = values.map((v, i) => `${i ? "L" : "M"}${((i / (values.length - 1)) * width).toFixed(1)},${(4 + (1 - (v - lo) / (hi - lo)) * (height - 8)).toFixed(1)}`).join("");
  return `<svg class="spark" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true"><path d="${d}" fill="none" stroke="#d8ecfa" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>`;
}

export function modeBars(z, explained, { width = 560, limit = 2.5, count = 16 } = {}) {
  const items = z.slice(0, count);
  const rowH = 22, pad = { l: 64, r: 64, t: 18, b: 6 };
  const height = pad.t + items.length * rowH + pad.b;
  const mid = pad.l + (width - pad.l - pad.r) / 2;
  const half = (width - pad.l - pad.r) / 2;
  const sx = (v) => (Math.max(-limit, Math.min(limit, v)) / limit) * half;
  const rows = items.map((v, i) => {
    const y = pad.t + i * rowH;
    const w = sx(v);
    const ev = explained?.[i];
    const atLimit = Math.abs(v) >= limit * 0.99;
    return `<text x="${pad.l - 10}" y="${y + 14}" text-anchor="end">Mode ${i + 1}</text>
      <rect x="${w < 0 ? mid + w : mid}" y="${y + 4}" width="${Math.max(Math.abs(w), 1)}" height="12" rx="3" fill="${atLimit ? "#f0b75e" : "#a9d4f2"}" fill-opacity="${0.35 + 0.65 * Math.min(Math.abs(v) / limit, 1)}"><title>Mode ${i + 1}: ${v.toFixed(2)} σ${ev !== undefined ? `, explains ${(ev * 100).toFixed(1)}% of training variance` : ""}</title></rect>
      <text x="${width - pad.r + 10}" y="${y + 14}">${v >= 0 ? "+" : ""}${v.toFixed(2)}σ</text>`;
  }).join("");
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Fitted shape-mode coefficients in standard deviations">
    <line x1="${mid}" x2="${mid}" y1="${pad.t - 6}" y2="${height}" stroke="rgba(255,255,255,.25)"/>
    <line x1="${mid - half}" x2="${mid - half}" y1="${pad.t - 6}" y2="${height}" stroke="rgba(240,183,94,.35)" stroke-dasharray="3 4"/>
    <line x1="${mid + half}" x2="${mid + half}" y1="${pad.t - 6}" y2="${height}" stroke="rgba(240,183,94,.35)" stroke-dasharray="3 4"/>
    <text x="${mid - half}" y="10" text-anchor="middle">−${limit}σ</text><text x="${mid}" y="10" text-anchor="middle">mean</text><text x="${mid + half}" y="10" text-anchor="middle">+${limit}σ</text>${rows}</svg>`;
}
