// Widgets for the hero and chapters 1–9.
import * as M from "./model.js";
import { h, s, fmt, fmtTex, pct, sci, probText, tex, texEl, matTex, vecTex, matHTML, tile, bars, plane, arrow, COLORS, onScheme, draggable, store, phrasePicker } from "./ui.js";

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
    fig(matHTML(W.E, { rowLabels: VOCAB, cls: "c-word", sym: "E" }), "E: each word's two numbers"),
    fig(matHTML(W.WQ, { cls: "c-query", sym: "WQ" }), "W<sub>Q</sub>: queries"),
    fig(matHTML(W.WK, { cls: "c-key", sym: "WK" }), "W<sub>K</sub>: keys"),
    fig(matHTML(W.WV, { cls: "c-value", sym: "WV" }), "W<sub>V</sub>: values"),
    fig(matHTML(W.WO, { colLabels: VOCAB, cls: "c-out", sym: "WO" }), "W<sub>O</sub>: one column per word"),
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
  // The chart starts at x = 0 so the line's starting height a is visible,
  // and changing the steepness b visibly swings the line around that point.
  const Wd = 520, Ht = 360, pl = 44, pr = 14, pt = 14, pb = 40;
  const xmin = 0, xmax = 20, ymin = -10, ymax = 70;
  const X = (x) => pl + ((x - xmin) / (xmax - xmin)) * (Wd - pl - pr);
  const Y = (y) => Ht - pb - ((y - ymin) / (ymax - ymin)) * (Ht - pt - pb);
  const svg = s("svg", { viewBox: `0 0 ${Wd} ${Ht}`, class: "plot", role: "img", "aria-label": "Five data points with a line through them" });
  svg.append(s("defs", {}, s("clipPath", { id: "fitClip" }, s("rect", { x: X(xmin), y: Y(ymax), width: X(xmax) - X(xmin), height: Y(ymin) - Y(ymax) }))));
  const axes = s("g");
  for (let x = 0; x <= 20; x += 2) {
    axes.append(s("line", { x1: X(x), x2: X(x), y1: Y(ymin), y2: Y(ymax), class: x === 0 ? "axis" : "gridline" }));
    if (x % 4 === 0 || x >= 10) axes.append(s("text", { x: X(x), y: Ht - pb + 18, "text-anchor": "middle", class: "ticklabel" }, x));
  }
  for (let y = -10; y <= 70; y += 10) {
    axes.append(s("line", { y1: Y(y), y2: Y(y), x1: X(xmin), x2: X(xmax), class: y === 0 ? "axis" : "gridline" }));
    axes.append(s("text", { x: pl - 8, y: Y(y) + 4, "text-anchor": "end", class: "ticklabel" }, fmt(y, 0)));
  }
  axes.append(s("text", { x: X(10), y: Ht - 4, "text-anchor": "middle", class: "ticklabel" }, "years of education, x"));
  axes.append(s("text", { x: 12, y: Y(30), "text-anchor": "middle", class: "ticklabel", transform: `rotate(-90 12 ${Y(30)})` }, "income, y"));
  const sq = s("g", { "clip-path": "url(#fitClip)" }), ln = s("g"), pts = s("g");
  svg.append(axes, sq, ln, pts);

  const eq = h("div", { class: "out-line" });
  const err = h("div", { class: "out-line" });
  const sa = h("input", { type: "range", min: -10, max: 50, step: 0.1, value: st.a, "aria-label": "a, the line's height at x = 0" });
  const sb = h("input", { type: "range", min: 0, max: 6, step: 0.05, value: st.b, "aria-label": "b, how much the line climbs per year" });
  const va = h("span", { class: "w-num" }), vb = h("span", { class: "w-num" });
  const bestBtn = h("button", { class: "primary" }, "Find the best line");
  const left = h("div", {}, svg);
  const right = h("div", {},
    h("p", { class: "w-title" }, "Fit a line by hand"),
    h("p", { class: "w-help" }, "Make the total area of the squares as small as you can. The hollow dot is where the line starts, at x = 0; changing the steepness swings the line around it."),
    h("div", { class: "slider" }, h("span", { class: "w-label" }, "a (start, at x = 0)"), sa, va),
    h("div", { class: "slider" }, h("span", { class: "w-label" }, "b (climb per year)"), sb, vb),
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
      sq.append(s("rect", { x: leftEdge, y: top, width: side, height: side, fill: c.attn, "fill-opacity": 0.16, stroke: c.attn, "stroke-opacity": 0.7 }));
    }
    const g = s("g", { "clip-path": "url(#fitClip)" });
    g.append(s("line", { x1: X(xmin), y1: Y(st.a + st.b * xmin), x2: X(xmax), y2: Y(st.a + st.b * xmax), stroke: c.word, "stroke-width": 2.5 }));
    ln.append(g);
    // the starting point (0, a)
    if (st.a >= ymin && st.a <= ymax) {
      ln.append(s("circle", { cx: X(0), cy: Y(st.a), r: 6, fill: c.paper, stroke: c.word, "stroke-width": 2.5 }));
      ln.append(s("text", { x: X(0) + 10, y: Y(st.a) + (st.b > 1.5 ? 16 : -10), class: "lbl", fill: c.word }, "a"));
    }
    for (const [x, y] of data) pts.append(s("circle", { cx: X(x), cy: Y(y), r: 5, fill: c.ink }));
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
  const table = matHTML(W.E, { rowLabels: VOCAB, cls: "c-word", sym: "E" });
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
  tex(root, `\\Ev = \\htmlData{sym=E}{\\htmlClass{c-word}{\\begin{pmatrix} ${body} \\end{pmatrix}}} \\begin{matrix} ${labels} \\end{matrix}`, true);
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
  let a = [2.5, 1], b = [1, 2.5], unit = false;
  const plotBox = h("div");
  let P, hA, hB, arrows, dragging = null;
  const out = h("div");
  const unitBox = h("input", { type: "checkbox", id: "dot-unit" });
  root.append(h("div", { class: "w-grid two" }, plotBox,
    h("div", {}, h("p", { class: "w-title" }, "Drag the two arrow tips"), h("p", { class: "w-help" }, "Press anywhere near a tip; the closer one follows you."),
      h("label", { class: "w-row", style: { gap: "8px", cursor: "pointer" }, for: "dot-unit" }, unitBox, "Keep both arrows at length 1"),
      out)));
  // the plane zooms in when both arrows are kept at length 1
  const build = () => {
    const m = unit ? 1.4 : 4;
    P = plane({ xmin: -m, xmax: m, ymin: -m, ymax: m, width: 420, grid: unit ? 0.5 : 1, label: "Two draggable arrows" });
    arrows = s("g");
    P.layer.append(arrows);
    // Press anywhere near a tip (not just exactly on it) and the closer tip follows the pointer.
    const toData = (ev) => {
      const r = P.svg.getBoundingClientRect();
      const vx = ((ev.clientX - r.left) / r.width) * P.width, vy = ((ev.clientY - r.top) / r.height) * P.height;
      return [vx, vy, (vx - P.pad) / P.sx + P.xmin, (P.height - P.pad - vy) / P.sx + P.ymin];
    };
    P.svg.style.cursor = "grab";
    P.svg.addEventListener("pointerdown", (ev) => {
      const [vx, vy, x, y] = toData(ev);
      const dA = Math.hypot(vx - P.X(a[0]), vy - P.Y(a[1])), dB = Math.hypot(vx - P.X(b[0]), vy - P.Y(b[1]));
      if (Math.min(dA, dB) > 90) return; // too far from both tips
      ev.preventDefault();
      dragging = dA <= dB ? "a" : "b";
      P.svg.dataset.dragging = "1";
      document.getElementById("tip")?.classList.remove("on");
      try { P.svg.setPointerCapture(ev.pointerId); } catch {}
      P.svg.style.cursor = "grabbing";
      if (dragging === "a") a = place(x, y); else b = place(x, y);
      draw();
    });
    P.svg.addEventListener("pointermove", (ev) => {
      if (!dragging) return;
      const [, , x, y] = toData(ev);
      if (dragging === "a") a = place(x, y); else b = place(x, y);
      draw();
    });
    const stop = () => { dragging = null; delete P.svg.dataset.dragging; P.svg.style.cursor = "grab"; draw(); };
    P.svg.addEventListener("pointerup", stop);
    P.svg.addEventListener("pointercancel", stop);
    plotBox.innerHTML = ""; plotBox.append(P.svg);
  };
  const draw = () => {
    const c = COLORS();
    arrows.innerHTML = "";
    if (unit) arrows.append(s("circle", { cx: P.X(0), cy: P.Y(0), r: P.X(1) - P.X(0), fill: "none", stroke: c.ink3, "stroke-dasharray": "4 4", "stroke-opacity": 0.6 }));
    arrows.append(arrow(P, 0, 0, a[0], a[1], c.query, { width: 3 }), arrow(P, 0, 0, b[0], b[1], c.key, { width: 3 }));
        arrows.append(
      s("circle", { cx: P.X(a[0]), cy: P.Y(a[1]), r: dragging === "a" ? 22 : 16, fill: c.query, "fill-opacity": dragging === "a" ? 0.28 : 0.14 }),
      s("circle", { cx: P.X(b[0]), cy: P.Y(b[1]), r: dragging === "b" ? 22 : 16, fill: c.key, "fill-opacity": dragging === "b" ? 0.28 : 0.14 }),
      s("circle", { cx: P.X(a[0]), cy: P.Y(a[1]), r: 7, fill: c.query }), s("circle", { cx: P.X(b[0]), cy: P.Y(b[1]), r: 7, fill: c.key }));
    arrows.append(s("text", { x: P.X(a[0]) + 14, y: P.Y(a[1]) - 12, class: "lbl", fill: c.query, "data-sym": "dotArrow" }, "a"), s("text", { x: P.X(b[0]) + 14, y: P.Y(b[1]) - 12, class: "lbl", fill: c.key, "data-sym": "dotArrow" }, "b"));
    const d = a[0] * b[0] + a[1] * b[1];
    const la = Math.hypot(...a), lb = Math.hypot(...b);
    const zero = la < 1e-9 || lb < 1e-9;
    const cos = zero ? 0 : Math.max(-1, Math.min(1, d / (la * lb)));
    const deg = Math.round((Math.acos(cos) * 180) / Math.PI);
    const verdict = zero ? "" : deg <= 3 ? "pointing the same way" : deg < 25 ? "pointing nearly the same way" : deg < 72 ? "pointing roughly the same way" : deg <= 87 ? "close to right angles" : deg <= 93 ? "at right angles" : deg <= 108 ? "close to right angles" : deg < 155 ? "pointing roughly apart" : deg < 177 ? "pointing nearly opposite ways" : "pointing in opposite directions";
    const dd = unit ? 2 : 1;
    const pn = (v) => (v < 0 ? `(${fmtTex(v, dd)})` : fmtTex(v, dd)); // brackets around negatives
    out.innerHTML = "";
    out.append(
      h("p", { class: "out-line" }, texEl(`\\htmlClass{c-query}{a} = ${vecTex(a, dd)},\\quad \\htmlClass{c-key}{b} = ${vecTex(b, dd)}`)),
      h("p", { class: "out-line" }, texEl(`\\htmlClass{c-query}{a}\\cdot \\htmlClass{c-key}{b} = ${pn(a[0])}\\times${pn(b[0])} + ${pn(a[1])}\\times${pn(b[1])}`)),
      h("p", {}, h("span", { class: "big-num " + (d > 0.05 ? "good" : d < -0.05 ? "bad" : "") }, fmt(d, 2))),
      zero ? h("p", { class: "w-help" }, "One arrow has zero length, so the dot product is 0 whatever the direction.")
        : h("p", { class: "out-line" }, texEl(`= \\underbrace{${fmtTex(la, 2)}}_{\\text{length of } a} \\times \\underbrace{${fmtTex(lb, 2)}}_{\\text{length of } b} \\times \\underbrace{${fmtTex(cos, 2)}}_{\\text{alignment}}`)),
      zero ? "" : h("p", { class: "w-help" }, `The arrows are ${deg}° apart: ${verdict}, so the alignment is ${fmt(cos, 2)}.` +
        (unit ? " With both lengths 1, the dot product is the alignment." : " The dot product is that alignment scaled up by both lengths, so it can be much bigger than 1.")));
  };
  const snapGrid = (v) => Math.max(-3.5, Math.min(3.5, Math.round(v * 2) / 2));
  const place = (x, y) => {
    if (unit) {
      const ang = Math.round((Math.atan2(y, x) * 180) / Math.PI / 5) * 5 * (Math.PI / 180); // 5° steps
      return [Math.cos(ang), Math.sin(ang)].map((v) => (Math.abs(v) < 1e-12 ? 0 : v));
    }
    return [snapGrid(x), snapGrid(y)];
  };
  unitBox.addEventListener("change", () => {
    unit = unitBox.checked;
    if (unit) {
      a = place(...(Math.hypot(...a) > 1e-9 ? a : [1, 0]));
      b = place(...(Math.hypot(...b) > 1e-9 ? b : [0, 1]));
    } else { a = [2.5, 1]; b = [1, 2.5]; }
    build(); draw();
  });
  build(); draw();
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
    const big = (x) => (x >= 1e4 ? sci(x, 2) : fmt(x, x < 1 ? 3 : 2));
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
      h("thead", {}, h("tr", {}, h("th", {}, "word"), h("th", { class: "r" }, "numbers x"), h("th", { class: "r c-key", "data-sym": "k", html: "key k = xW<sub>K</sub>" }), h("th", { class: "r" }, "score q·k"), h("th", { class: "r" }, "÷ √2"), h("th", { class: "c-attn", "data-sym": "a" }, "attention weight"))),
      h("tbody", {}, f.ctx.map((t, j) => {
        const bar = h("div", { class: "bars", style: { gridTemplateColumns: "1fr 3.4rem" } }, h("div", { class: "bar attn" }, h("i", { style: { width: f.attn[j] * 100 + "%" } })), h("span", { class: "pct" }, pct(f.attn[j])));
        return h("tr", {}, h("td", {}, tile(t)), h("td", { class: "r num" }, `(${fmt(f.X[j][0], 1)}, ${fmt(f.X[j][1], 1)})`), h("td", { class: "r num c-key", "data-sym": "k" }, `(${fmt(f.K[j][0])}, ${fmt(f.K[j][1])})`),
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
        svg.append(s("text", { x: j === n ? x0 : (x0 + x1) / 2, y: Ht - 40 - hgt * 0.75 - (1 + 9 * w) / 2 - 6, "text-anchor": "middle", class: "lbl", fill: c.attn, "data-sym": "a" }, pct(w)));
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
    labels.forEach((L) => P.layer.append(s("text", { x: L.x, y: L.y, "text-anchor": "middle", class: "lbl", "font-weight": L.bold ? 700 : 400, fill: L.ink ? c.ink : c.out, "data-sym": L.ink ? "r" : "WOarrow" }, L.text)));
    plotBox.innerHTML = ""; plotBox.append(P.svg);
    right.innerHTML = "";
    const order = [...f.probs.keys()].sort((a, b) => f.probs[b] - f.probs[a]);
    const tb = h("table", { class: "dtable" }, h("thead", {}, h("tr", {}, h("th", {}, "word"), h("th", { class: "r c-out", "data-sym": "score" }, "score r · column"), h("th", { class: "r" }, "probability"))),
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
  const q = `\\htmlData{sym=WQ}{\\htmlClass{c-query}{${matTex(W.WQ)}}}`;
  const k = `\\htmlData{sym=WK}{\\htmlClass{c-key}{${matTex(W.WK)}}}`;
  const v = `\\htmlData{sym=WV}{\\htmlClass{c-value}{${matTex(W.WV)}}}`;
  const o = `\\htmlData{sym=WO}{\\htmlClass{c-out}{${matTex(W.WO)}}}`;
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
  const value = h("p", { class: "out-line" });
  const box = h("div", { class: "expanded", tabindex: 0, "aria-label": "The fully expanded formula" });
  root.append(h("p", { class: "w-title" }, "One word's probability, fully written out"), picker,
    h("div", { class: "w-row" }, h("span", { class: "w-label" }, "Probability of"), tsel, h("span", { class: "w-label" }, "Write the input as"), modeSel), value, info, box);
  const draw = () => {
    const ids = st.get().slice(-3);
    const str = M.expandedFormula(W, ids.length, target, { symbolic: mode === "sym", ids });
    box.textContent = str;
    const probs = M.forward(W, ids).probs, p = probs[target], top = M.argmax(probs);
    const phrase = ids.map((t) => VOCAB[t]).join(" "), q = (t) => `“${VOCAB[t]}”`;
    value.innerHTML = "";
    if (mode === "num") {
      value.append(`Worked out, it equals `, probText(p), `: the same answer the matrix version gives. `,
        top === target ? `After “${phrase}”, ${q(target)} is the model's top choice.` : `After “${phrase}”, the model's top choice is ${q(top)} (${pct(probs[top])}), so ${q(target)} gets ${p < 0.01 ? "almost nothing" : "less"}.`);
    } else {
      value.append("Put in the two numbers of each word of any phrase, and it gives that phrase's answer.");
    }
    info.textContent = `${str.length.toLocaleString()} characters for a ${ids.length}-word input; scroll the box to see all of it. The whole thing has the shape exp(score of ${q(target)}) ÷ (exp(score of “the”) + exp(score of “cat”) + …, one term for each of the eight words), with every score spelled out in full.`;
  };
  st.sub(draw);
  tsel.addEventListener("change", () => { target = +tsel.value; draw(); });
  modeSel.addEventListener("change", () => { mode = modeSel.value; draw(); });
}
