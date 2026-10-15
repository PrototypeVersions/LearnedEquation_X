// Small DOM, SVG and KaTeX helpers shared by every widget.
import { VOCAB } from "./model.js";

export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

const SVGNS = "http://www.w3.org/2000/svg";
export function s(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

// Numbers with a real minus sign.
export function fmt(v, d = 2) {
  if (!isFinite(v)) return "∞";
  let t = Number(v).toFixed(d);
  if (/^-0\.?0*$/.test(t)) t = t.slice(1);
  return t.replace("-", "−");
}
// A very small or very large number as "6.59 × 10⁻²⁰" (never the computer's "6.59e-20").
export function sci(x, digits = 2) {
  if (x === 0 || !isFinite(x)) return h("span", { class: "w-num" }, x === 0 ? "0" : "∞");
  let e = Math.floor(Math.log10(Math.abs(x)));
  let m = x / 10 ** e;
  if (Math.abs(+m.toFixed(digits)) >= 10) { m /= 10; e += 1; }
  return h("span", { class: "w-num", html: `${m.toFixed(digits).replace("-", "−")} × 10<sup>${String(e).replace("-", "−")}</sup>` });
}
// A probability in plain terms; tiny ones are written out in full where that's readable.
export function probText(p) {
  const fine = (q) => (q >= 0.01 && q <= 0.99 ? pct(q) : `${(q * 100).toFixed(2)}%`);
  if (p >= 0.0001) return h("span", { class: "w-num" }, `${fmt(p, 4)} (${fine(p)})`);
  const e = Math.floor(Math.log10(p));
  const out = h("span");
  if (e >= -30) out.append(h("span", { class: "w-num" }, p.toFixed(-e + 2)), " (that is, ", sci(p), ")");
  else out.append(sci(p));
  out.append(", essentially zero");
  return out;
}

export const pct = (p) => (p >= 0.9995 ? "100%" : p < 0.0005 ? "0%" : p < 0.01 || p > 0.99 ? (p * 100).toFixed(1) + "%" : Math.round(p * 100) + "%");
export const fmtTex = (v, d = 2) => {
  let t = Number(v).toFixed(d);
  if (/^-0\.?0*$/.test(t)) t = t.slice(1);
  return t;
};

export const MACROS = {
  "\\Xm": "\\htmlData{sym=X}{\\htmlClass{c-word}{X}}",
  "\\xn": "\\htmlData{sym=xn}{\\htmlClass{c-word}{x_n}}",
  "\\Ev": "\\htmlData{sym=E}{\\htmlClass{c-word}{E}}",
  "\\WQ": "\\htmlData{sym=WQ}{\\htmlClass{c-query}{W_Q}}",
  "\\WK": "\\htmlData{sym=WK}{\\htmlClass{c-key}{W_K}}",
  "\\WV": "\\htmlData{sym=WV}{\\htmlClass{c-value}{W_V}}",
  "\\WO": "\\htmlData{sym=WO}{\\htmlClass{c-out}{W_O}}",
  "\\sm": "\\htmlData{sym=softmax}{\\operatorname{softmax}}",
  "\\smA": "\\htmlData{sym=softmaxA}{\\htmlClass{c-attn}{\\operatorname{softmax}}}",
  "\\smO": "\\htmlData{sym=softmaxO}{\\htmlClass{c-out}{\\operatorname{softmax}}}",
  "\\T": "\\htmlData{sym=T}{\\top}",
  "\\rt": "\\htmlData{sym=sqrt}{\\htmlClass{c-attn}{\\sqrt{2}}}",
  "\\qv": "\\htmlData{sym=q}{\\htmlClass{c-query}{q}}",
  "\\kv": "\\htmlData{sym=k}{\\htmlClass{c-key}{k}}",
  "\\vv": "\\htmlData{sym=v}{\\htmlClass{c-value}{v}}",
  "\\av": "\\htmlData{sym=a}{\\htmlClass{c-attn}{a}}",
  "\\zv": "\\htmlData{sym=z}{\\htmlClass{c-value}{z}}",
};

export const KATEX_OPTS = {
  macros: MACROS,
  trust: (c) => c.command === "\\htmlClass" || c.command === "\\htmlData",
  strict: "ignore",
  throwOnError: false,
};

export function tex(el, src, display = false) {
  window.katex.render(src, el, { ...KATEX_OPTS, displayMode: display, macros: { ...MACROS } });
  return el;
}
export const texEl = (src, display = false, tag = "span", cls = "") => tex(h(tag, { class: cls }), src, display);

// LaTeX for a matrix of numbers.
export function matTex(M, d = 1, cls = null) {
  const body = M.map((r) => r.map((v) => fmtTex(v, d)).join(" & ")).join(" \\\\ ");
  const m = `\\begin{pmatrix} ${body} \\end{pmatrix}`;
  return cls ? `\\htmlClass{${cls}}{${m}}` : m;
}
export const vecTex = (v, d = 2) => `(${v.map((x) => fmtTex(x, d)).join(",\\ ")})`;

// HTML matrix with optional row labels.
export function matHTML(M, { d = 1, rowLabels = null, colLabels = null, cls = "", cellClass = null, sym = null } = {}) {
  const cols = M[0].length;
  const grid = h("div", { class: "mat " + cls + (colLabels ? " has-cl" : ""), "data-sym": sym, style: { gridTemplateColumns: `repeat(${cols}, auto)` } });
  if (colLabels) colLabels.forEach((l) => grid.append(h("span", { class: "clab" }, l)));
  M.forEach((row, i) => row.forEach((v, j) => grid.append(h("span", { class: "cell" + (cellClass ? " " + cellClass(i, j) : ""), "data-i": i, "data-j": j }, fmt(v, d)))));
  if (!rowLabels) return grid;
  const wrap = h("div", { class: "mat-wrap" });
  if (rowLabels) wrap.append(h("div", { class: "row-labels" }, rowLabels.map((l) => h("div", {}, l))));
  wrap.append(grid);
  return wrap;
}

export const tile = (id, extra = "") => h("span", { class: "tile " + extra }, VOCAB[id]);

export function bars(container, items, { cls = "", onPick = null } = {}) {
  container.innerHTML = "";
  container.classList.add("bars");
  for (const it of items) {
    const label = typeof it.label === "string" ? h("span", { class: "tile" }, it.label) : it.label;
    const bar = h("div", { class: "bar " + cls }, h("i", { style: { width: Math.max(0, Math.min(1, it.p)) * 100 + "%" } }));
    const p = h("span", { class: "pct" }, it.text ?? pct(it.p));
    if (onPick) {
      const row = [label, bar, p];
      row.forEach((e) => { e.classList.add("pick"); e.addEventListener("click", () => onPick(it)); });
    }
    container.append(label, bar, p);
  }
}

// A coordinate plane in SVG.
export function plane({ xmin, xmax, ymin, ymax, width = 420, grid = 1, ticks = true, label = "" }) {
  const pad = 26;
  const sx = (width - 2 * pad) / (xmax - xmin);
  const height = Math.round((ymax - ymin) * sx + 2 * pad);
  const X = (x) => pad + (x - xmin) * sx;
  const Y = (y) => height - pad - (y - ymin) * sx;
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, class: "plot", role: "img", "aria-label": label });
  const gGrid = s("g");
  for (let x = Math.ceil(xmin / grid) * grid; x <= xmax + 1e-9; x += grid) gGrid.append(s("line", { x1: X(x), x2: X(x), y1: Y(ymin), y2: Y(ymax), class: Math.abs(x) < 1e-9 ? "axis" : "gridline" }));
  for (let y = Math.ceil(ymin / grid) * grid; y <= ymax + 1e-9; y += grid) gGrid.append(s("line", { y1: Y(y), y2: Y(y), x1: X(xmin), x2: X(xmax), class: Math.abs(y) < 1e-9 ? "axis" : "gridline" }));
  svg.append(gGrid);
  if (ticks) {
    const step = niceStep((xmax - xmin) / 5);
    const gT = s("g");
    for (let x = Math.ceil(xmin / step) * step; x <= xmax; x += step) if (Math.abs(x) > 1e-9) gT.append(s("text", { x: X(x), y: Math.min(Y(0) + 14, height - 6), "text-anchor": "middle", class: "ticklabel" }, fmt(x, step < 1 ? 1 : 0)));
    for (let y = Math.ceil(ymin / step) * step; y <= ymax; y += step) if (Math.abs(y) > 1e-9) gT.append(s("text", { x: Math.max(X(0) - 6, 18), y: Y(y) + 4, "text-anchor": "end", class: "ticklabel" }, fmt(y, step < 1 ? 1 : 0)));
    svg.append(gT);
  }
  const layer = s("g");
  svg.append(layer);
  return { svg, X, Y, layer, width, height, sx, pad, gGrid, xmin, xmax, ymin, ymax };
}
export function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

export function arrow(P, x0, y0, x1, y1, color, { width = 2.4, head = 9, dash = null, opacity = 1 } = {}) {
  const ax = P.X(x0), ay = P.Y(y0), bx = P.X(x1), by = P.Y(y1);
  const ang = Math.atan2(by - ay, bx - ax);
  const len = Math.hypot(bx - ax, by - ay);
  const hd = Math.min(head, len * 0.5);
  const g = s("g", { opacity });
  g.append(s("line", { x1: ax, y1: ay, x2: bx - Math.cos(ang) * hd * 0.6, y2: by - Math.sin(ang) * hd * 0.6, stroke: color, "stroke-width": width, "stroke-linecap": "round", "stroke-dasharray": dash }));
  if (len > 2) g.append(s("path", { d: `M${bx},${by} L${bx - hd * Math.cos(ang - 0.42)},${by - hd * Math.sin(ang - 0.42)} L${bx - hd * Math.cos(ang + 0.42)},${by - hd * Math.sin(ang + 0.42)} Z`, fill: color }));
  return g;
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
export const COLORS = () => ({
  word: cssVar("--c-word"), query: cssVar("--c-query"), key: cssVar("--c-key"), value: cssVar("--c-value"),
  attn: cssVar("--c-attn"), out: cssVar("--c-out"), ink: cssVar("--ink"), ink2: cssVar("--ink-2"), ink3: cssVar("--ink-3"), paper: cssVar("--paper"), panel2: cssVar("--panel-2"),
});

// Re-render when the color scheme flips, so SVG colors follow.
const schemeListeners = [];
export function onScheme(fn) { schemeListeners.push(fn); }
window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", () => schemeListeners.forEach((f) => f()));

// Pointer dragging on an SVG plane.
export function draggable(P, el, onMove) {
  const toData = (ev) => {
    const r = P.svg.getBoundingClientRect();
    const vx = ((ev.clientX - r.left) / r.width) * P.width;
    const vy = ((ev.clientY - r.top) / r.height) * P.height;
    return [(vx - P.pad) / P.sx + P.xmin, (P.height - P.pad - vy) / P.sx + P.ymin];
  };
  el.addEventListener("pointerdown", (ev) => {
    ev.preventDefault();
    el.setPointerCapture(ev.pointerId);
    const move = (e) => onMove(...toData(e));
    const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
  });
}

// Simple publish/subscribe store.
export function store(initial) {
  let v = initial;
  const subs = new Set();
  return {
    get: () => v,
    set: (nv) => { v = nv; subs.forEach((f) => f(v)); },
    sub: (f) => { subs.add(f); f(v); return () => subs.delete(f); },
  };
}

// Phrase picker: three dropdowns ("—" for none) plus preset buttons.
export function phrasePicker(st, presets, { slots = 3 } = {}) {
  const wrap = h("div");
  const pre = h("div", { class: "w-row" });
  presets.forEach((p) => pre.append(h("button", { class: "ghost", onclick: () => st.set(p.slice()) }, p.map((i) => VOCAB[i]).join(" "))));
  const sels = h("div", { class: "w-row" }, h("span", { class: "w-label" }, "Or choose:"));
  const selects = [];
  for (let k = 0; k < slots; k++) {
    const sel = h("select", { "aria-label": `Word ${k + 1}` });
    if (k < slots - 1) sel.append(h("option", { value: "" }, "—"));
    VOCAB.forEach((w, i) => sel.append(h("option", { value: i }, w)));
    sel.addEventListener("change", () => {
      const ids = selects.map((x) => x.value).filter((x) => x !== "").map(Number);
      st.set(ids.length ? ids : [0]);
    });
    selects.push(sel);
    sels.append(sel);
  }
  st.sub((ids) => {
    const pad = slots - ids.length;
    selects.forEach((sel, k) => { sel.value = k < pad ? "" : String(ids[k - pad]); });
  });
  wrap.append(pre, sels);
  return wrap;
}
