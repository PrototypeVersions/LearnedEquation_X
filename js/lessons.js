// Widgets for the hero and chapters 1–9.
import * as M from "./model.js";
import { h, s, fmt, fmtTex, pct, tex, texEl, matTex, vecTex, matHTML, tile, bars, plane, arrow, COLORS, onScheme, draggable, store, phrasePicker } from "./ui.js";

const { VOCAB, TRAINED: W } = M;
const reduceMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
const I = M.INDEX;

// The phrase shared by the attention (ch 7) and prediction (ch 8) widgets.
export const lessonCtx = store([I.cat, I.chases, I.the]);
const PRESETS = [
  [I.cat, I.chases, I.the],
  [I.dog, I.chases, I.the],
  [I.chases, I.the, I.cat],
  [I["."], I.the, I.cat],
  [I.the, I.cat],
];

// ---------------------------------------------------------------- hero
export function hero(root) {
  const prompt = [I.the, I.dog, I.chases, I.the];
  const line = h("div", { class: "demo-line" });
  const meta = h("p", { class: "demo-meta" });
  const text = h("p", { class: "demo-text", "aria-live": "polite" });
  const btn = h("button", { class: "primary" }, "Write a sentence");
  const again = h("button", { class: "ghost" }, "Start over");
  root.append(line, meta, h("div", { class: "w-row" }, btn, again), text);

  const p = M.forward(W, prompt).probs;
  const top = M.argmax(p);
  line.append(...prompt.map((t) => VOCAB[t] + " "), h("span", { class: "blank" }, VOCAB[top]));
  meta.textContent = `The equation gives “${VOCAB[top]}” a probability of ${pct(p[top])}. Every other word gets almost nothing.`;

  let seq = [];
  let busy = false;
  const render = (fresh = 0) => {
    text.innerHTML = "";
    const words = M.detok(seq).split(" ");
    const keepFrom = Math.max(0, words.length - 60);
    words.slice(keepFrom).forEach((w, i, arr) => text.append(h("span", { class: i >= arr.length - fresh ? "w-new" : "" }, w + " ")));
  };
  btn.addEventListener("click", async () => {
    if (busy) return;
    busy = true; btn.disabled = true;
    const startLen = seq.length;
    if (seq.length === 0) seq.push(I.the);
    let guard = 0;
    while (guard++ < 12) {
      const pr = M.forward(W, seq).probs;
      const t = M.sampleFrom(pr, Math.random);
      seq.push(t);
      render(seq.length - startLen);
      if (!reduceMotion()) await new Promise((r) => setTimeout(r, 140));
      if (t === I["."]) break;
    }
    busy = false; btn.disabled = false;
    btn.textContent = "Write another";
  });
  again.addEventListener("click", () => { seq = []; text.innerHTML = ""; btn.textContent = "Write a sentence"; });
}

export function heroWeights(root) {
  const row = h("div", { class: "weights-row" });
  const fig = (el, cap) => h("figure", {}, el, h("figcaption", { html: cap }));
  row.append(
    fig(matHTML(W.E, { rowLabels: VOCAB, cls: "c-word" }), "E: each word's two numbers"),
    fig(matHTML(W.WQ, { cls: "c-query" }), "W<sub>Q</sub>: queries"),
    fig(matHTML(W.WK, { cls: "c-key" }), "W<sub>K</sub>: keys"),
    fig(matHTML(W.WV, { cls: "c-value" }), "W<sub>V</sub>: values"),
    fig(matHTML(W.WO, { colLabels: VOCAB, cls: "c-out" }), "W<sub>O</sub>: one column per word"),
  );
  root.append(h("p", { class: "w-label", style: { fontFamily: "var(--font-ui)", margin: "0 0 4px" } }, "The 44 numbers that make it this particular model:"), row);
}

export function vocab(root) {
  VOCAB.forEach((w, i) => root.append(tile(i)));
}

// ---------------------------------------------------------------- 1 fitting
export function fit(root) {
  const data = [[10, 28], [12, 34], [14, 41], [16, 49], [18, 54]];
  const best = { a: -5.7, b: 3.35 };
  const st = { a: 5, b: 2 };
  const Wd = 520, Ht = 340, pl = 44, pr = 14, pt = 14, pb = 40;
  const xmin = 8, xmax = 20, ymin = 10, ymax = 70;
  const X = (x) => pl + ((x - xmin) / (xmax - xmin)) * (Wd - pl - pr);
  const Y = (y) => Ht - pb - ((y - ymin) / (ymax - ymin)) * (Ht - pt - pb);
  const svg = s("svg", { viewBox: `0 0 ${Wd} ${Ht}`, class: "plot", role: "img", "aria-label": "Five data points with a line through them" });
  const axes = s("g");
  for (let x = 8; x <= 20; x += 2) {
    axes.append(s("line", { x1: X(x), x2: X(x), y1: Y(ymin), y2: Y(ymax), class: "gridline" }));
    axes.append(s("text", { x: X(x), y: Ht - pb + 18, "text-anchor": "middle", class: "ticklabel" }, x));
  }
  for (let y = 10; y <= 70; y += 10) {
    axes.append(s("line", { y1: Y(y), y2: Y(y), x1: X(xmin), x2: X(xmax), class: "gridline" }));
    axes.append(s("text", { x: pl - 8, y: Y(y) + 4, "text-anchor": "end", class: "ticklabel" }, y));
  }
  axes.append(s("text", { x: X(14), y: Ht - 4, "text-anchor": "middle", class: "ticklabel" }, "years of education, x"));
  axes.append(s("text", { x: 12, y: Y(40), "text-anchor": "middle", class: "ticklabel", transform: `rotate(-90 12 ${Y(40)})` }, "income, y"));
  const sq = s("g"), ln = s("g"), pts = s("g");
  svg.append(axes, sq, ln, pts);

  const eq = h("div", { class: "out-line" });
  const err = h("div", { class: "out-line" });
  const sa = h("input", { type: "range", min: -30, max: 30, step: 0.1, value: st.a, "aria-label": "a, where the line starts" });
  const sb = h("input", { type: "range", min: 0, max: 6, step: 0.05, value: st.b, "aria-label": "b, how steeply it climbs" });
  const va = h("span", { class: "w-num" }), vb = h("span", { class: "w-num" });
  const bestBtn = h("button", { class: "primary" }, "Find the best line");
  const left = h("div", {}, svg);
  const right = h("div", {},
    h("p", { class: "w-title" }, "Fit a line by hand"),
    h("p", { class: "w-help" }, "Make the total area of the squares as small as you can."),
    h("div", { class: "slider" }, h("span", { class: "w-label" }, "a (start)"), sa, va),
    h("div", { class: "slider" }, h("span", { class: "w-label" }, "b (steepness)"), sb, vb),
    eq, err, h("div", { class: "w-row" }, bestBtn));
  root.append(h("div", { class: "w-grid two" }, left, right));

  const draw = () => {
    const c = COLORS();
    sq.innerHTML = ""; ln.innerHTML = ""; pts.innerHTML = "";
    let total = 0;
    const pxPerY = (Ht - pt - pb) / (ymax - ymin);
    for (const [x, y] of data) {
      const yh = st.a + st.b * x;
      const e = y - yh;
      total += e * e;
      const side = Math.abs(e) * pxPerY;
      const top = Math.min(Y(y), Y(yh));
      const leftEdge = X(x) + side < Wd - pr ? X(x) : X(x) - side;
      sq.append(s("rect", { x: leftEdge, y: Math.max(pt, top), width: side, height: Math.min(side, Ht - pb - Math.max(pt, top)), fill: c.attn, "fill-opacity": 0.16, stroke: c.attn, "stroke-opacity": 0.7 }));
    }
    ln.append(s("line", { x1: X(xmin), y1: Y(st.a + st.b * xmin), x2: X(xmax), y2: Y(st.a + st.b * xmax), stroke: c.word, "stroke-width": 2.5 }));
    for (const [x, y] of data) pts.append(s("circle", { cx: X(x), cy: Y(y), r: 5.5, fill: c.ink }));
    va.textContent = fmt(st.a, 1); vb.textContent = fmt(st.b, 2);
    tex(eq, `\\hat y = ${fmtTex(st.a, 1)} + ${fmtTex(st.b, 2)}\\,x`);
    err.innerHTML = "";
    err.append("Total squared error: ", h("strong", { class: "w-num" }, fmt(total, 1)), Math.abs(st.a - best.a) < 0.05 && Math.abs(st.b - best.b) < 0.01 ? h("span", { class: "good" }, ". That's the smallest possible.") : "");
  };
  sa.addEventListener("input", () => { st.a = +sa.value; draw(); });
  sb.addEventListener("input", () => { st.b = +sb.value; draw(); });
  bestBtn.addEventListener("click", () => {
    const a0 = st.a, b0 = st.b, t0 = performance.now(), dur = reduceMotion() ? 0 : 900;
    const tick = (t) => {
      const u = dur ? Math.min(1, (t - t0) / dur) : 1;
      const e = u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2;
      st.a = a0 + (best.a - a0) * e; st.b = b0 + (best.b - b0) * e;
      sa.value = st.a; sb.value = st.b; draw();
      if (u < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
  draw();
  onScheme(draw);
}

// ---------------------------------------------------------------- 3 words
const LABEL_POS = { the: [0, -12, "middle"], cat: [10, 4, "start"], dog: [-10, 4, "end"], runs: [10, 4, "start"], sleeps: [10, 4, "start"], chases: [-10, 18, "end"], ".": [10, 14, "start"] };

export function embed(root) {
  const P = plane({ xmin: -5, xmax: 5, ymin: -3.5, ymax: 6.5, width: 460, label: "The eight words as points on a plane" });
  const right = h("div");
  const table = matHTML(W.E, { rowLabels: VOCAB, cls: "c-word" });
  const read = h("p", { class: "out-line", style: { minHeight: "2.6em" } });
  right.append(h("p", { class: "w-title" }, "Every word is a point"), h("p", { class: "w-help" }, "Point at a word, or a row of the table, to link the two."), table, read);
  root.append(h("div", { class: "w-grid two" }, h("div", {}, P.svg), right));
  let hot = null;
  const draw = () => {
    const c = COLORS();
    P.layer.innerHTML = "";
    VOCAB.forEach((w, i) => {
      if (w === "sees") return; // drawn together with "chases"
      const [x, y] = W.E[i];
      const on = hot === i || (w === "chases" && hot === I.sees);
      const g = s("g", { style: "cursor:pointer", tabindex: 0, "aria-label": `${w}: ${fmt(x, 1)}, ${fmt(y, 1)}` });
      g.append(s("circle", { cx: P.X(x), cy: P.Y(y), r: on ? 8 : 6, fill: c.word, "fill-opacity": on ? 1 : 0.85 }));
      const [dx, dy, anchor] = LABEL_POS[w];
      g.append(s("text", { x: P.X(x) + dx, y: P.Y(y) + dy, "text-anchor": anchor, class: "lbl", "font-weight": on ? 700 : 400 }, w === "chases" ? "chases = sees" : w === "." ? ". (period)" : w));
      g.addEventListener("pointerenter", () => setHot(i));
      g.addEventListener("focus", () => setHot(i));
      P.layer.append(g);
    });
  };
  const setHot = (i) => {
    hot = i;
    table.querySelectorAll(".cell").forEach((cEl) => cEl.classList.toggle("hl", +cEl.dataset.i === i || (i === I.chases && +cEl.dataset.i === I.sees) || (i === I.sees && +cEl.dataset.i === I.chases)));
    read.innerHTML = "";
    if (i != null) read.append(tile(i), `  is the point (${fmt(W.E[i][0], 1)}, ${fmt(W.E[i][1], 1)}): ${fmt(W.E[i][0], 1)} across, ${fmt(W.E[i][1], 1)} up.`);
    draw();
  };
  table.querySelectorAll(".row-labels > div").forEach((d, i) => d.addEventListener("pointerenter", () => setHot(i)));
  table.querySelectorAll(".cell").forEach((cEl) => cEl.addEventListener("pointerenter", () => setHot(+cEl.dataset.i)));
  setHot(I.cat);
  onScheme(draw);
}

export function Etex(root) {
  const body = W.E.map((r) => r.map((v) => fmtTex(v, 1)).join(" & ")).join(" \\\\ ");
  const labels = VOCAB.map((w) => `\\leftarrow\\ \\text{${w === "." ? "." : w}}`).join(" \\\\ ");
  tex(root, `\\Ev = \\htmlClass{c-word}{\\begin{pmatrix} ${body} \\end{pmatrix}} \\begin{matrix} ${labels} \\end{matrix}`, true);
}

export function phrase(root) {
  let ids = [I.the, I.cat, I.sleeps];
  const strip = h("div", { class: "tok-strip" });
  const out = h("div", { class: "scroll-x" });
  const keys = h("div", { class: "tiles" });
  VOCAB.forEach((w, i) => keys.append(h("button", { class: "tile", onclick: () => { ids.push(i); draw(); } }, w)));
  root.append(
    h("p", { class: "w-title" }, "Build a phrase"),
    h("p", { class: "w-help" }, "Add words. The table holds only the last three: that's all the model ever sees."),
    keys,
    h("div", { class: "w-row" }, h("button", { class: "ghost", onclick: () => { ids.pop(); if (!ids.length) ids = [I.the]; draw(); } }, "Remove last word"), h("button", { class: "ghost", onclick: () => { ids = [I.the]; draw(); } }, "Clear")),
    strip, out);
  const draw = () => {
    if (ids.length > 9) ids = ids.slice(-9);
    strip.innerHTML = "";
    ids.forEach((t, k) => strip.append(tile(t, k >= ids.length - 3 ? "window-mark" : "dim")));
    const ctx = ids.slice(-3);
    const rows = ctx.map((t) => W.E[t].map((v) => fmtTex(v, 1)).join(" & ")).join(" \\\\ ");
    const labs = ctx.map((t, k) => `\\leftarrow\\ x_${k + 1}${k === ctx.length - 1 ? " = x_n" : ""}\\ \\text{(${VOCAB[t]})}`).join(" \\\\ ");
    tex(out, `\\Xm = \\htmlClass{c-word}{\\begin{pmatrix} ${rows} \\end{pmatrix}} \\begin{matrix} ${labs} \\end{matrix}`, true);
  };
  draw();
}

// ---------------------------------------------------------------- 4 matrices
const MATS = { WQ: ["the query matrix", "query", "c-query"], WK: ["the key matrix", "key", "c-key"], WV: ["the value matrix", "value", "c-value"] };

export function matmul(root) {
  let word = I.cat, mat = "WQ", step = 0;
  const wsel = h("select", { "aria-label": "Word" }); VOCAB.forEach((w, i) => wsel.append(h("option", { value: i }, w))); wsel.value = word;
  const msel = h("select", { "aria-label": "Matrix" }); Object.entries(MATS).forEach(([k, [n, r]]) => msel.append(h("option", { value: k }, `${r[0].toUpperCase() + r.slice(1)} matrix`)));
  const disp = h("div", { class: "scroll-x", style: { padding: "6px 0" } });
  const lines = h("div");
  const next = h("button", { class: "primary" }, "Next step");
  const reset = h("button", { class: "ghost" }, "Start over");
  root.append(h("p", { class: "w-title" }, "The multiplication machine"),
    h("div", { class: "w-row" }, h("span", { class: "w-label" }, "Word"), wsel, h("span", { class: "w-label" }, "Matrix"), msel),
    disp, lines, h("div", { class: "w-row" }, next, reset));
  const draw = () => {
    const x = W.E[word], A = W[mat], y = M.vecMat(x, A), [nm, , cl] = MATS[mat];
    const hl = (j) => (step === j + 1 ? "\\htmlClass{hlcol}{" : "{");
    const colTex = `\\begin{pmatrix} ${hl(0)}${fmtTex(A[0][0], 1)}} & ${hl(1)}${fmtTex(A[0][1], 1)}} \\\\ ${hl(0)}${fmtTex(A[1][0], 1)}} & ${hl(1)}${fmtTex(A[1][1], 1)}} \\end{pmatrix}`;
    const res = [step >= 1 ? fmtTex(y[0], 2) : "\\,?\\,", step >= 2 ? fmtTex(y[1], 2) : "\\,?\\,"];
    tex(disp, `\\underbrace{\\htmlClass{c-word}{(${fmtTex(x[0], 1)},\\ ${fmtTex(x[1], 1)})}}_{\\text{${VOCAB[word]}}} \\htmlClass{${cl}}{${colTex}} = (${res[0]},\\ ${res[1]})`, true);
    lines.innerHTML = "";
    const L = (j) => {
      const a = A[0][j], c = A[1][j];
      return h("p", { class: "out-line" }, h("strong", {}, j === 0 ? "First number: " : "Second number: "),
        texEl(`${fmtTex(x[0], 1)}\\times(${fmtTex(a, 1)}) + (${fmtTex(x[1], 1)})\\times(${fmtTex(c, 1)}) = ${fmtTex(x[0] * a, 2)} + (${fmtTex(x[1] * c, 2)}) = ${fmtTex(y[j], 2)}`));
    };
    if (step === 0) lines.append(h("p", { class: "w-help" }, `Press “Next step” to compute the first number, using the first column of ${nm}.`));
    if (step >= 1) lines.append(L(0));
    if (step >= 2) lines.append(L(1), h("p", { class: "w-help" }, `So ${VOCAB[word]}'s ${MATS[mat][1]} vector is (${fmt(y[0])}, ${fmt(y[1])}).`));
    next.disabled = step >= 2;
  };
  wsel.addEventListener("change", () => { word = +wsel.value; step = 0; draw(); });
  msel.addEventListener("change", () => { mat = msel.value; step = 0; draw(); });
  next.addEventListener("click", () => { step = Math.min(2, step + 1); draw(); });
  reset.addEventListener("click", () => { step = 0; draw(); });
  draw();
}

export function transform(root) {
  let mat = "WQ", t = 0, raf = null;
  const msel = h("select", { "aria-label": "Matrix" }); Object.entries(MATS).forEach(([k, [n, r]]) => msel.append(h("option", { value: k }, `${r[0].toUpperCase() + r.slice(1)} matrix`)));
  const slider = h("input", { type: "range", min: 0, max: 1, step: 0.01, value: 0, "aria-label": "Before to after" });
  const play = h("button", { class: "primary" }, "Play");
  const holder = h("div", { style: { maxWidth: "560px" } });
  const note = h("p", { class: "w-help", style: { marginTop: "10px" } });
  root.append(h("p", { class: "w-title" }, "Multiply the whole plane"),
    h("div", { class: "w-row" }, h("span", { class: "w-label" }, "Matrix"), msel, play),
    h("div", { class: "slider" }, h("span", { class: "w-label" }, "Before → after"), slider, h("span")),
    holder, note);
  let P;
  const build = () => {
    const A = W[mat];
    let m = 4;
    W.E.forEach((x) => { const y = M.vecMat(x, A); m = Math.max(m, Math.abs(x[0]), Math.abs(x[1]), Math.abs(y[0]), Math.abs(y[1])); });
    m = Math.ceil(m + 1.5);
    P = plane({ xmin: -m, xmax: m, ymin: -m, ymax: m, width: 480, grid: 1e9, ticks: true, label: "Words moving under a matrix" });
    P.m = m;
    P.svg.prepend(s("defs", {}, s("clipPath", { id: "clipT" }, s("rect", { x: P.X(-m), y: P.Y(m), width: P.X(m) - P.X(-m), height: P.Y(-m) - P.Y(m) }))));
    holder.innerHTML = ""; holder.append(P.svg);
    note.textContent = mat === "WQ" ? "The query matrix stretches the plane, turns it, and flips it over. Every word's query is where its point ends up." : mat === "WK" ? "The key matrix flips and stretches the plane differently. Keys and queries live in the same plane so their dot products can be compared." : "The value matrix flips the plane and flattens it toward a narrow band, so the values mostly differ along one direction.";
    draw();
  };
  const draw = () => {
    const c = COLORS(), A = W[mat];
    const Mt = [[1 - t + t * A[0][0], t * A[0][1]], [t * A[1][0], 1 - t + t * A[1][1]]];
    const map = (x, y) => [x * Mt[0][0] + y * Mt[1][0], x * Mt[0][1] + y * Mt[1][1]];
    P.layer.innerHTML = "";
    const g = s("g", { "clip-path": "url(#clipT)" });
    const R = P.m * 2, stepG = P.m > 10 ? 2 : 1;
    for (let k = -R; k <= R; k += stepG) {
      for (const [a, b] of [[[k, -R], [k, R]], [[-R, k], [R, k]]]) {
        const [x0, y0] = map(...a), [x1, y1] = map(...b);
        g.append(s("line", { x1: P.X(x0), y1: P.Y(y0), x2: P.X(x1), y2: P.Y(y1), stroke: k === 0 ? c.ink3 : c.ink3, "stroke-opacity": k === 0 ? 0.6 : 0.22, "stroke-width": k === 0 ? 1.4 : 1 }));
      }
    }
    P.layer.append(g);
    const col = c[MATS[mat][1]];
    VOCAB.forEach((w, i) => {
      if (w === "sees") return;
      const [x, y] = map(...W.E[i]);
      P.layer.append(s("circle", { cx: P.X(x), cy: P.Y(y), r: 5.5, fill: t > 0.5 ? col : c.word }));
      const [dx, dy, anchor] = LABEL_POS[w];
      P.layer.append(s("text", { x: P.X(x) + dx, y: P.Y(y) + dy, "text-anchor": anchor, class: "lbl" }, w === "chases" ? "chases, sees" : w === "." ? ". (period)" : w));
    });
  };
  slider.addEventListener("input", () => { t = +slider.value; draw(); });
  msel.addEventListener("change", () => { mat = msel.value; t = 0; slider.value = 0; build(); });
  play.addEventListener("click", () => {
    cancelAnimationFrame(raf);
    const from = t > 0.5 ? 1 : 0, to = 1 - from, t0 = performance.now(), dur = reduceMotion() ? 0 : 1400;
    const tick = (now) => {
      const u = dur ? Math.min(1, (now - t0) / dur) : 1;
      const e = u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2;
      t = from + (to - from) * e; slider.value = t; draw();
      if (u < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  });
  build();
  onScheme(draw);
}

// ---------------------------------------------------------------- 5 dot product
export function dot(root) {
  let a = [2.5, 1], b = [1, 2.5];
  const P = plane({ xmin: -4, xmax: 4, ymin: -4, ymax: 4, width: 420, label: "Two draggable arrows" });
  const out = h("div");
  root.append(h("div", { class: "w-grid two" }, h("div", {}, P.svg), h("div", {}, h("p", { class: "w-title" }, "Drag the two arrow tips"), out)));
  const hA = s("circle", { r: 14, fill: "transparent", style: "cursor:grab" });
  const hB = s("circle", { r: 14, fill: "transparent", style: "cursor:grab" });
  const arrows = s("g");
  P.layer.append(arrows, hA, hB);
  const draw = () => {
    const c = COLORS();
    arrows.innerHTML = "";
    arrows.append(arrow(P, 0, 0, a[0], a[1], c.query, { width: 3 }), arrow(P, 0, 0, b[0], b[1], c.key, { width: 3 }));
    arrows.append(s("text", { x: P.X(a[0]) + 10, y: P.Y(a[1]) - 8, class: "lbl", fill: c.query }, "a"), s("text", { x: P.X(b[0]) + 10, y: P.Y(b[1]) - 8, class: "lbl", fill: c.key }, "b"));
    arrows.append(s("circle", { cx: P.X(a[0]), cy: P.Y(a[1]), r: 6, fill: c.query }), s("circle", { cx: P.X(b[0]), cy: P.Y(b[1]), r: 6, fill: c.key }));
    hA.setAttribute("cx", P.X(a[0])); hA.setAttribute("cy", P.Y(a[1]));
    hB.setAttribute("cx", P.X(b[0])); hB.setAttribute("cy", P.Y(b[1]));
    const d = a[0] * b[0] + a[1] * b[1];
    const la = Math.hypot(...a), lb = Math.hypot(...b);
    const zero = la < 1e-9 || lb < 1e-9;
    const cos = zero ? 0 : d / (la * lb);
    const verdict = zero ? "" : cos > 0.92 ? "pointing nearly the same way" : cos > 0.3 ? "pointing roughly the same way" : cos > -0.3 ? "close to right angles" : cos > -0.92 ? "pointing roughly apart" : "pointing in opposite directions";
    out.innerHTML = "";
    out.append(
      h("p", { class: "out-line" }, texEl(`\\htmlClass{c-query}{a} = ${vecTex(a, 1)},\\quad \\htmlClass{c-key}{b} = ${vecTex(b, 1)}`)),
      h("p", { class: "out-line" }, texEl(`\\htmlClass{c-query}{a}\\cdot \\htmlClass{c-key}{b} = ${fmtTex(a[0], 1)}\\times${fmtTex(b[0], 1)} + ${fmtTex(a[1], 1)}\\times${fmtTex(b[1], 1)}`)),
      h("p", {}, h("span", { class: "big-num " + (d > 0.05 ? "good" : d < -0.05 ? "bad" : "") }, fmt(d, 2))),
      zero ? h("p", { class: "w-help" }, "One arrow has zero length, so the dot product is 0 whatever the direction.")
        : h("p", { class: "out-line" }, texEl(`= \\underbrace{${fmtTex(la, 2)}}_{\\text{length of } a} \\times \\underbrace{${fmtTex(lb, 2)}}_{\\text{length of } b} \\times \\underbrace{${fmtTex(cos, 2)}}_{\\text{alignment}}`)),
      zero ? "" : h("p", { class: "w-help" }, `The arrows are ${verdict}.`));
  };
  const snap = (v) => Math.max(-3.8, Math.min(3.8, Math.round(v * 2) / 2));
  draggable(P, hA, (x, y) => { a = [snap(x), snap(y)]; draw(); });
  draggable(P, hB, (x, y) => { b = [snap(x), snap(y)]; draw(); });
  draw();
  onScheme(draw);
}

// ---------------------------------------------------------------- 6 softmax
export function softmax(root) {
  const words = [I.dog, I.cat, I.runs, I["."]];
  const ZMIN = -10, ZMAX = 15;
  let z = [3, 2, 0, -1];
  const sl = h("div"), tbl = h("div", { class: "scroll-x" }), barBox = h("div");
  const shift = h("button", { class: "ghost" }, "Add 5 to every score");
  const unshift = h("button", { class: "ghost" }, "Subtract 5 from every score");
  const resetB = h("button", { class: "ghost" }, "Reset");
  root.append(h("div", { class: "w-grid two" },
    h("div", {}, h("p", { class: "w-title" }, "Scores"), h("p", { class: "w-help" }, "Any numbers, positive or negative."), sl, h("div", { class: "w-row" }, shift, unshift, resetB)),
    h("div", {}, h("p", { class: "w-title" }, "Probabilities"), barBox)), tbl);
  const inputs = words.map((t, k) => {
    const inp = h("input", { type: "range", min: ZMIN, max: ZMAX, step: 0.5, value: z[k], "aria-label": `Score for ${VOCAB[t]}` });
    const v = h("span", { class: "w-num" });
    inp.addEventListener("input", () => { z[k] = +inp.value; draw(); });
    sl.append(h("div", { class: "slider" }, h("span", {}, tile(t)), inp, v));
    return [inp, v];
  });
  const draw = () => {
    const p = M.softmax(z);
    const e = z.map(Math.exp), tot = e.reduce((x, y) => x + y, 0);
    inputs.forEach(([inp, v], k) => { inp.value = z[k]; v.textContent = fmt(z[k], 1); });
    shift.disabled = Math.max(...z) + 5 > ZMAX;
    unshift.disabled = Math.min(...z) - 5 < ZMIN;
    bars(barBox, words.map((t, k) => ({ label: VOCAB[t], p: p[k] })));
    tbl.innerHTML = "";
    const big = (x) => (x >= 1e4 ? x.toExponential(2).replace("e+", "×10^") : fmt(x, x < 1 ? 3 : 2));
    tbl.append(h("table", { class: "dtable", style: { marginTop: "14px" } },
      h("thead", {}, h("tr", {}, h("th", {}, "word"), h("th", { class: "r" }, "score z"), h("th", { class: "r" }, "step 1: e^z"), h("th", { class: "r" }, "step 2: e^z ÷ total"))),
      h("tbody", {}, words.map((t, k) => h("tr", {}, h("td", {}, tile(t)), h("td", { class: "r num" }, fmt(z[k], 1)), h("td", { class: "r num" }, big(e[k])), h("td", { class: "r num" }, fmt(p[k], 3)))),
        h("tr", {}, h("td", {}, "total"), h("td"), h("td", { class: "r num" }, big(tot)), h("td", { class: "r num" }, "1.000")))));
  };
  shift.addEventListener("click", () => { z = z.map((v) => v + 5); draw(); });
  unshift.addEventListener("click", () => { z = z.map((v) => v - 5); draw(); });
  resetB.addEventListener("click", () => { z = [3, 2, 0, -1]; draw(); });
  draw();
}

// ---------------------------------------------------------------- 7 attention
export function attention(root) {
  const picker = phrasePicker(lessonCtx, PRESETS);
  const qBox = h("div", { class: "scroll-x" });
  const tblBox = h("div", { class: "scroll-x" });
  const arcBox = h("div");
  const zBox = h("div", { class: "scroll-x" });
  root.append(h("p", { class: "w-title" }, "Attention, with every number"), h("p", { class: "w-help" }, "Pick a phrase. The last word is the one predicting what comes next."), picker,
    h("h4", { style: { margin: "16px 0 4px" } }, "1. The last word's query"), qBox,
    h("h4", { style: { margin: "16px 0 4px" } }, "2–3. Keys, scores and attention weights"), tblBox, arcBox,
    h("h4", { style: { margin: "16px 0 4px" } }, "4. Blend the values"), zBox);
  const draw = (ids) => {
    const f = M.forward(W, ids), n = f.ctx.length - 1, c = COLORS();
    tex(qBox, `\\qv = \\xn\\WQ = \\htmlClass{c-word}{${vecTex(f.X[n], 1)}}\\ \\htmlClass{c-query}{${matTex(W.WQ)}} = \\htmlClass{c-query}{${vecTex(f.q)}} \\qquad (\\text{query of “${VOCAB[f.ctx[n]]}”})`, true);
    tblBox.innerHTML = "";
    tblBox.append(h("table", { class: "dtable" },
      h("thead", {}, h("tr", {}, h("th", {}, "word"), h("th", { class: "r" }, "numbers x"), h("th", { class: "r c-key", html: "key k = xW<sub>K</sub>" }), h("th", { class: "r" }, "score q·k"), h("th", { class: "r" }, "÷ √2"), h("th", { class: "c-attn" }, "attention weight"))),
      h("tbody", {}, f.ctx.map((t, j) => {
        const bar = h("div", { class: "bars", style: { gridTemplateColumns: "1fr 3.4rem" } }, h("div", { class: "bar attn" }, h("i", { style: { width: f.attn[j] * 100 + "%" } })), h("span", { class: "pct" }, pct(f.attn[j])));
        return h("tr", {}, h("td", {}, tile(t)), h("td", { class: "r num" }, `(${fmt(f.X[j][0], 1)}, ${fmt(f.X[j][1], 1)})`), h("td", { class: "r num c-key" }, `(${fmt(f.K[j][0])}, ${fmt(f.K[j][1])})`),
          h("td", { class: "r num" }, fmt(f.scores[j])), h("td", { class: "r num" }, fmt(f.scaled[j])), h("td", { style: { minWidth: "9rem" } }, bar));
      }))));
    // arcs
    arcBox.innerHTML = "";
    const Wd = 140 * f.ctx.length + 60, Ht = 140, gap = Wd / (f.ctx.length + 0.5);
    const svg = s("svg", { viewBox: `0 0 ${Wd} ${Ht}`, class: "plot", style: `max-width:${Wd + 40}px;margin-top:12px`, role: "img", "aria-label": "Arcs from the last word back to each word, thicker for more attention" });
    const xs = f.ctx.map((_, j) => gap * (j + 0.75));
    f.ctx.forEach((t, j) => {
      const w = f.attn[j];
      if (w > 0.004) {
        const x0 = xs[n], x1 = xs[j];
        const hgt = j === n ? 30 : 26 + Math.abs(x0 - x1) * 0.22;
        const d = j === n ? `M${x0 - 14},${Ht - 40} C${x0 - 30},${Ht - 40 - hgt} ${x0 + 30},${Ht - 40 - hgt} ${x0 + 14},${Ht - 40}` : `M${x0},${Ht - 40} C${x0},${Ht - 40 - hgt} ${x1},${Ht - 40 - hgt} ${x1},${Ht - 40}`;
        svg.append(s("path", { d, fill: "none", stroke: c.attn, "stroke-width": 1 + 9 * w, "stroke-opacity": 0.35 + 0.6 * w, "stroke-linecap": "round" }));
        svg.append(s("text", { x: j === n ? x0 : (x0 + x1) / 2, y: Ht - 40 - hgt * 0.75 - (1 + 9 * w) / 2 - 6, "text-anchor": "middle", class: "lbl", fill: c.attn }, pct(w)));
      }
      svg.append(s("text", { x: xs[j], y: Ht - 16, "text-anchor": "middle", class: "lbl", "font-size": 16, "font-weight": j === n ? 700 : 400 }, VOCAB[t]));
    });
    arcBox.append(svg);
    const terms = f.ctx.map((t, j) => `\\htmlClass{c-attn}{${fmtTex(f.attn[j], 3)}}\\,\\htmlClass{c-value}{${vecTex(f.V[j])}}`).join(" + ");
    tex(zBox, `\\zv = \\textstyle\\sum_j \\av_j\\,\\vv_j = ${terms} = \\htmlClass{c-value}{${vecTex(f.z)}}`, true);
  };
  lessonCtx.sub(draw);
  onScheme(() => draw(lessonCtx.get()));
}

// ---------------------------------------------------------------- 8 output
export function output(root) {
  const picker = phrasePicker(lessonCtx, PRESETS);
  const rBox = h("div", { class: "scroll-x" });
  const plotBox = h("div");
  const right = h("div");
  root.append(h("p", { class: "w-title" }, "From r to a prediction"), h("p", { class: "w-help", html: "Same phrase as above. Each green arrow is one column of W<sub>O</sub>, belonging to one word. The thick arrow is r." }), picker, rBox,
    h("div", { class: "w-grid two", style: { marginTop: "12px" } }, plotBox, right));
  const draw = (ids) => {
    const f = M.forward(W, ids), n = f.ctx.length - 1, c = COLORS();
    tex(rBox, `r = \\xn + \\zv = \\htmlClass{c-word}{${vecTex(f.X[n], 1)}} + \\htmlClass{c-value}{${vecTex(f.z)}} = ${vecTex(f.r)}`, true);
    const cols = VOCAB.map((_, t) => [W.WO[0][t], W.WO[1][t]]);
    let m = 5;
    cols.forEach((v) => { m = Math.max(m, Math.abs(v[0]), Math.abs(v[1])); });
    m = Math.max(m, Math.abs(f.r[0]), Math.abs(f.r[1]));
    m = Math.ceil(m + 1);
    const P = plane({ xmin: -m, xmax: m, ymin: -m, ymax: m, width: 440, grid: niceGrid(m), label: "Output arrows for each word and the vector r" });
    const top = M.argmax(f.probs);
    const labels = [];
    cols.forEach((v, t) => {
      P.layer.append(arrow(P, 0, 0, v[0], v[1], c.out, { width: t === top ? 3 : 1.8, opacity: t === top ? 1 : 0.55 }));
      const len = Math.hypot(...v) || 1;
      labels.push({ x: P.X(v[0]) + (v[0] / len) * 14, y: P.Y(v[1]) - (v[1] / len) * 14 + 4, text: VOCAB[t] === "." ? ". (period)" : VOCAB[t], bold: t === top });
    });
    P.layer.append(arrow(P, 0, 0, f.r[0], f.r[1], c.ink, { width: 3.4, head: 12 }));
    const rl = Math.hypot(...f.r) || 1;
    labels.push({ x: P.X(f.r[0]) + (f.r[0] / rl) * 16, y: P.Y(f.r[1]) - (f.r[1] / rl) * 16 + 4, text: "r", bold: true, ink: true });
    spreadLabels(labels);
    labels.forEach((L) => P.layer.append(s("text", { x: L.x, y: L.y, "text-anchor": "middle", class: "lbl", "font-weight": L.bold ? 700 : 400, fill: L.ink ? c.ink : c.out }, L.text)));
    plotBox.innerHTML = ""; plotBox.append(P.svg);
    right.innerHTML = "";
    const order = [...f.probs.keys()].sort((a, b) => f.probs[b] - f.probs[a]);
    const tb = h("table", { class: "dtable" }, h("thead", {}, h("tr", {}, h("th", {}, "word"), h("th", { class: "r c-out" }, "score r · column"), h("th", { class: "r" }, "probability"))),
      h("tbody", {}, order.map((t) => h("tr", {}, h("td", {}, tile(t)), h("td", { class: "r num" }, fmt(f.logits[t])), h("td", { class: "r num" }, pct(f.probs[t]))))));
    // A dot product rewards length as well as direction: say so when they disagree.
    const ang = (v) => Math.atan2(v[1], v[0]);
    const off = cols.map((v) => { let d = Math.abs(ang(v) - ang(f.r)) * 180 / Math.PI; return d > 180 ? 360 - d : d; });
    const lens = cols.map((v) => Math.hypot(...v));
    const closest = off.indexOf(Math.min(...off));
    const note = h("p", { class: "w-help", style: { marginTop: "10px" } }, `Top prediction after “${f.ctx.map((t) => VOCAB[t]).join(" ")}”: “${VOCAB[top]}”. `);
    if (closest !== top) {
      note.append(`Its arrow is ${fmt(off[top], 0)}° away from r's direction, while the arrow for “${VOCAB[closest]}” is only ${fmt(off[closest], 0)}° away. But the “${VOCAB[top]}” arrow is longer (${fmt(lens[top], 1)} vs ${fmt(lens[closest], 1)}), and a dot product rewards length as well as direction: score = length of r × length of arrow × how well they line up.`);
    } else {
      note.append(`Its arrow points closer to r's direction than any other word's (${fmt(off[top], 0)}° away).`);
    }
    right.append(tb, note);
  };
  lessonCtx.sub(draw);
  onScheme(() => draw(lessonCtx.get()));
}
function niceGrid(m) { return m > 16 ? 5 : m > 8 ? 2 : 1; }
function spreadLabels(L) {
  for (let it = 0; it < 40; it++) {
    let moved = false;
    for (let i = 0; i < L.length; i++) for (let j = i + 1; j < L.length; j++) {
      const a = L[i], b = L[j];
      const wa = a.text.length * 7.4, wb = b.text.length * 7.4;
      if (Math.abs(a.x - b.x) < (wa + wb) / 2 && Math.abs(a.y - b.y) < 15) {
        const dy = a.y <= b.y ? -1 : 1;
        a.y += dy * 2; b.y -= dy * 2; moved = true;
      }
    }
    if (!moved) break;
  }
}

// ---------------------------------------------------------------- 9 the equation
export function fullNumeric(root) {
  const q = `\\htmlClass{c-query}{${matTex(W.WQ)}}`;
  const k = `\\htmlClass{c-key}{${matTex(W.WK)}}`;
  const v = `\\htmlClass{c-value}{${matTex(W.WV)}}`;
  const o = `\\htmlClass{c-out}{${matTex(W.WO)}}`;
  tex(root, `\\begin{aligned} P(\\,\\text{next word}\\mid x_1,\\dots,x_n) = \\smO\\!\\Bigg( &\\Big( \\xn + \\smA\\!\\Big( \\xn ${q} \\Big(\\Xm ${k}\\Big)^{\\T}\\! / \\rt \\Big) \\\\[4pt] & \\qquad \\Xm ${v} \\Big)\\, ${o} \\Bigg) \\end{aligned}`, true);
}

export function expanded(root) {
  const st = store([I.cat, I.chases, I.the]);
  let target = I.dog, mode = "num";
  const picker = phrasePicker(st, PRESETS);
  const tsel = h("select", { "aria-label": "Which next word" }); VOCAB.forEach((w, i) => tsel.append(h("option", { value: i }, w))); tsel.value = target;
  const modeSel = h("select", { "aria-label": "Show input as" }, h("option", { value: "num" }, "this phrase's numbers"), h("option", { value: "sym" }, "symbols x₁₁, x₁₂, …"));
  modeSel.value = mode;
  const info = h("p", { class: "status" });
  const box = h("div", { class: "expanded", tabindex: 0, "aria-label": "The fully expanded formula" });
  root.append(h("p", { class: "w-title" }, "One word's probability, fully written out"), picker,
    h("div", { class: "w-row" }, h("span", { class: "w-label" }, "Probability of"), tsel, h("span", { class: "w-label" }, "Write the input as"), modeSel), info, box);
  const draw = () => {
    const ids = st.get().slice(-3);
    const str = M.expandedFormula(W, ids.length, target, { symbolic: mode === "sym", ids });
    box.textContent = str;
    const p = M.forward(W, ids).probs[target];
    info.innerHTML = "";
    info.append(`${str.length.toLocaleString()} characters for a ${ids.length}-word input. `,
      mode === "num" ? `Worked out, it equals ${p < 0.0001 ? p.toExponential(2) : fmt(p, 4)}: the same answer the matrix version gives.` : "Swap in the numbers of any phrase and it gives that phrase's answer.");
  };
  st.sub(draw);
  tsel.addEventListener("change", () => { target = +tsel.value; draw(); });
  modeSel.addEventListener("change", () => { mode = modeSel.value; draw(); });
}
