// Chapter 12: how training works, step by step.
import * as M from "./model.js";
import { h, s, fmt, fmtTex, pct, sci, tex, texEl, vecTex, tile, COLORS, onScheme, store } from "./ui.js";
import { corpus } from "./training.js";

const { VOCAB, INDEX: I } = M;
const reduceMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

// ------------------------------------------------------------ walking downhill on the line
// The chapter 1 line, written as y = c + b(x − 14): c is the income predicted at 14 years.
const DATA = [[10, 28], [12, 34], [14, 41], [16, 49], [18, 54]];
const err = (c, b) => DATA.reduce((sum, [x, y]) => sum + (y - c - b * (x - 14)) ** 2, 0);
const slopeC = (c, b) => DATA.reduce((sum, [x, y]) => sum - 2 * (y - c - b * (x - 14)), 0);
const slopeB = (c, b) => DATA.reduce((sum, [x, y]) => sum - 2 * (y - c - b * (x - 14)) * (x - 14), 0);
const BEST = { c: 41.2, b: 3.35 };

export function descent(root) {
  const START = { c: 25, b: 6.5 };
  let st = { ...START }, lr = 0.01, path = [[st.c, st.b]], last = null;
  const cmin = 18, cmax = 64, bmin = -1.5, bmax = 8.5;
  const Wd = 480, Ht = 360, pl = 44, pr = 12, pt = 12, pb = 38;
  const X = (c) => pl + ((c - cmin) / (cmax - cmin)) * (Wd - pl - pr);
  const Y = (b) => Ht - pb - ((b - bmin) / (bmax - bmin)) * (Ht - pt - pb);
  const inv = (px, py) => [cmin + ((px - pl) / (Wd - pl - pr)) * (cmax - cmin), bmin + ((Ht - pb - py) / (Ht - pt - pb)) * (bmax - bmin)];
  const svg = s("svg", { viewBox: `0 0 ${Wd} ${Ht}`, class: "plot", role: "img", "aria-label": "A map of every possible line, shaded by its error, with a path walking downhill", style: "cursor:crosshair" });
  const mini = s("svg", { viewBox: "0 0 300 180", class: "plot", role: "img", "aria-label": "The current line through the data" });
  const slider = h("input", { type: "range", min: 0.002, max: 0.03, step: 0.001, value: lr, "aria-label": "Step size" });
  const lrOut = h("span", { class: "w-num" });
  const b1 = h("button", { class: "primary" }, "Take one step");
  const b10 = h("button", {}, "Take 10 steps");
  const bReset = h("button", { class: "ghost" }, "Start over");
  const read = h("div");
  root.append(
    h("p", { class: "w-title" }, "Every possible line, as a map"),
    h("p", { class: "w-help" }, "Darker means lower error. Click anywhere on the map to start from that line, then take steps."),
    h("div", { class: "w-grid two" },
      h("div", {}, svg, h("div", { class: "slider", style: { marginTop: "10px" } }, h("span", { class: "w-label" }, "Step size"), slider, lrOut), h("div", { class: "w-row" }, b1, b10, bReset)),
      h("div", {}, mini, read)));

  const draw = () => {
    const c = COLORS();
    svg.innerHTML = "";
    const levels = [3000, 2000, 1200, 700, 400, 200, 100, 50, 20, 8];
    svg.append(s("defs", {}, s("clipPath", { id: "descentClip" }, s("rect", { x: pl, y: pt, width: Wd - pl - pr, height: Ht - pt - pb }))));
    const g = s("g", { "clip-path": "url(#descentClip)" });
    for (const L of levels) {
      const rx = Math.sqrt((L - 1.9) / 5), ry = Math.sqrt((L - 1.9) / 40);
      g.append(s("ellipse", { cx: X(BEST.c), cy: Y(BEST.b), rx: X(BEST.c + rx) - X(BEST.c), ry: Y(BEST.b) - Y(BEST.b + ry), fill: c.word, "fill-opacity": 0.075, stroke: c.word, "stroke-opacity": 0.28 }));
    }
    svg.append(g);
    for (let v = 20; v <= 60; v += 10) svg.append(s("text", { x: X(v), y: Ht - pb + 16, "text-anchor": "middle", class: "ticklabel" }, v));
    for (let v = 0; v <= 8; v += 2) svg.append(s("text", { x: pl - 6, y: Y(v) + 4, "text-anchor": "end", class: "ticklabel" }, v));
    svg.append(s("text", { x: (pl + Wd - pr) / 2, y: Ht - 4, "text-anchor": "middle", class: "ticklabel" }, "c: income the line predicts at 14 years"));
    svg.append(s("text", { x: 12, y: Y(3.5), "text-anchor": "middle", class: "ticklabel", transform: `rotate(-90 12 ${Y(3.5)})` }, "b: slope"));
    svg.append(s("circle", { cx: X(BEST.c), cy: Y(BEST.b), r: 4, fill: "none", stroke: c.out, "stroke-width": 2 }));
    const clip = (v, a, b) => Math.max(a, Math.min(b, v));
    if (path.length > 1) svg.append(s("polyline", { "clip-path": "url(#descentClip)", points: path.map(([cc, bb]) => `${X(clip(cc, cmin - 50, cmax + 50))},${Y(clip(bb, bmin - 20, bmax + 20))}`).join(" "), fill: "none", stroke: c.attn, "stroke-width": 2, "stroke-linejoin": "round" }));
    path.forEach(([cc, bb], k) => { if (cc >= cmin && cc <= cmax && bb >= bmin && bb <= bmax) svg.append(s("circle", { cx: X(cc), cy: Y(bb), r: k === path.length - 1 ? 6 : 2.5, fill: c.attn })); });

    // the current line through the data
    mini.innerHTML = "";
    const mx = (x) => 24 + ((x - 8) / 12) * 266, my = (y) => 166 - ((y - 10) / 60) * 156;
    for (let x = 8; x <= 20; x += 4) mini.append(s("line", { x1: mx(x), x2: mx(x), y1: 10, y2: 166, class: "gridline" }));
    for (let y = 10; y <= 70; y += 20) mini.append(s("line", { y1: my(y), y2: my(y), x1: 24, x2: 290, class: "gridline" }));
    const yAt = (x) => st.c + st.b * (x - 14);
    mini.append(s("line", { x1: mx(8), y1: my(yAt(8)), x2: mx(20), y2: my(yAt(20)), stroke: c.word, "stroke-width": 2.2 }));
    DATA.forEach(([x, y]) => mini.append(s("circle", { cx: mx(x), cy: my(y), r: 4, fill: c.ink })));

    lrOut.textContent = fmt(lr, 3);
    read.innerHTML = "";
    const E = err(st.c, st.b);
    const blown = !isFinite(E) || E > 1e7;
    read.append(h("p", { class: "out-line" }, `Step ${path.length - 1}. Total squared error: `, h("strong", { class: "w-num" }, blown ? "enormous" : fmt(E, 1)), blown ? h("span", { class: "bad" }, " — the steps are too big and it has flown off the map.") : ""));
    if (last && !blown) {
      const { c0, b0, sc, sb, E0 } = last;
      read.append(
        h("p", { class: "w-help", style: { margin: "8px 0 2px" } }, "How the last step was worked out:"),
        h("p", { class: "out-line" }, texEl(`\\text{slope in } c:\\ \\frac{\\text{error}(c{+}0.001) - \\text{error}(c)}{0.001} = \\frac{${fmtTex(err(c0 + 0.001, b0), 3)} - ${fmtTex(E0, 3)}}{0.001} \\approx ${fmtTex(sc, 1)}`)),
        h("p", { class: "out-line" }, texEl(`c \\leftarrow ${fmtTex(c0, 2)} - ${fmtTex(lr, 3)} \\times (${fmtTex(sc, 1)}) = ${fmtTex(st.c, 2)}`)),
        h("p", { class: "out-line" }, texEl(`\\text{slope in } b \\approx ${fmtTex(sb, 1)},\\qquad b \\leftarrow ${fmtTex(b0, 2)} - ${fmtTex(lr, 3)} \\times (${fmtTex(sb, 1)}) = ${fmtTex(st.b, 2)}`)));
    } else if (!last) {
      read.append(h("p", { class: "w-help" }, "Press “Take one step” to see the arithmetic."));
    }
    b1.disabled = b10.disabled = blown;
  };
  const step = () => {
    const c0 = st.c, b0 = st.b, sc = slopeC(c0, b0), sb = slopeB(c0, b0);
    last = { c0, b0, sc, sb, E0: err(c0, b0) };
    st = { c: c0 - lr * sc, b: b0 - lr * sb };
    path.push([st.c, st.b]);
  };
  b1.addEventListener("click", () => { step(); draw(); });
  b10.addEventListener("click", async () => {
    for (let k = 0; k < 10; k++) {
      step(); draw();
      if (!isFinite(err(st.c, st.b)) || err(st.c, st.b) > 1e7) break;
      if (!reduceMotion()) await new Promise((r) => setTimeout(r, 90));
    }
  });
  const restart = (c0, b0) => { st = { c: c0, b: b0 }; path = [[c0, b0]]; last = null; draw(); };
  bReset.addEventListener("click", () => restart(START.c, START.b));
  slider.addEventListener("input", () => { lr = +slider.value; draw(); });
  svg.addEventListener("click", (ev) => {
    const r = svg.getBoundingClientRect();
    const [c0, b0] = inv(((ev.clientX - r.left) / r.width) * Wd, ((ev.clientY - r.top) / r.height) * Ht);
    if (c0 >= cmin && c0 <= cmax && b0 >= bmin && b0 <= bmax) restart(Math.round(c0 * 10) / 10, Math.round(b0 * 100) / 100);
  });
  draw();
  onScheme(draw);
}

// ------------------------------------------------------------ one real training step
const EXAMPLES = [
  ["cat chases the", "dog"],
  ["the cat", "sleeps"],
  ["the", "cat"],
  ["chases the cat", "."],
];
const START_SEED = 3; // "random start 1" in chapter 11
export const stepState = store({ ex: 0, w: M.randomWeights(START_SEED), steps: 0, note: "" });

const wordQ = (t) => `“${VOCAB[t]}”`;
export function paramLabel(k, i, j) {
  const nth = ["first", "second"];
  if (k === "E") return `E: ${wordQ(i)}, ${nth[j]} number`;
  if (k === "WO") return `W_O: ${wordQ(j)} arrow, ${nth[i]} number`;
  return `${k.replace("W", "W_")}: row ${i + 1}, column ${j + 1}`;
}
const exIds = (e) => M.tokenize(EXAMPLES[e][0]);
const exTarget = (e) => I[EXAMPLES[e][1]];

export function onestep(root) {
  const C = corpus();
  const exRow = h("div", { class: "w-row" });
  EXAMPLES.forEach(([ph, nx], k) => exRow.append(h("button", { class: "ghost", "data-k": k, onclick: () => stepState.set({ ...stepState.get(), ex: k }) }, `${ph} → ${nx}`)));
  const now = h("div");
  const psel = h("select", { "aria-label": "Which of the 44 numbers to nudge" });
  for (const k of M.PARAM_NAMES) {
    const grp = h("optgroup", { label: k === "E" ? "E (word numbers)" : k.replace("W", "W_") });
    const [r, c] = M.SHAPES[k];
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) grp.append(h("option", { value: `${k},${i},${j}` }, paramLabel(k, i, j)));
    psel.append(grp);
  }
  const nudgeBox = h("div");
  const gridBox = h("div", { class: "heat", style: { marginTop: "8px" } });
  const lrSel = h("select", { "aria-label": "Step size" }, [0.5, 1, 2].map((v) => h("option", { value: v }, String(v))));
  lrSel.value = "1";
  const bEx = h("button", { class: "primary" }, "Take one step on this example only");
  const bAll = h("button", {}, "Take one step on the whole text");
  const bReset = h("button", { class: "ghost" }, "Back to the random start");
  const result = h("p", { class: "status" });
  root.append(
    h("p", { class: "w-title" }, "One prediction, one step"),
    h("p", { class: "w-help" }, "The model starts at random start 1 from chapter 11, before any training. Pick a situation from the training text."),
    exRow, now,
    h("h4", { style: { margin: "16px 0 4px" } }, "1. Nudge one number and measure the slope"),
    h("div", { class: "w-row" }, h("span", { class: "w-label" }, "Number to nudge"), psel), nudgeBox,
    h("h4", { style: { margin: "16px 0 4px" } }, "2. The slopes for all 44 numbers"),
    h("p", { class: "w-help" }, "Blue: nudging the number up raises the surprise, so the step will lower it. Red: the reverse. The outlined cell is the one you nudged."),
    gridBox,
    h("h4", { style: { margin: "16px 0 4px" } }, "3. Take the step"),
    h("div", { class: "w-row" }, h("span", { class: "w-label" }, "Step size"), lrSel, bEx, bAll, bReset), result);

  const draw = () => {
    const { ex, w, steps, note } = stepState.get();
    const ids = exIds(ex), target = exTarget(ex);
    exRow.querySelectorAll("button").forEach((b) => b.classList.toggle("on", +b.dataset.k === ex));
    const bt = M.backwardTrace(w, ids, target);
    const p = bt.f.probs[target];
    now.innerHTML = "";
    now.append(h("p", { class: "out-line" }, `After “${EXAMPLES[ex][0]}”, the text continued with “${EXAMPLES[ex][1]}”. The model ${steps ? `(after ${steps} step${steps === 1 ? "" : "s"})` : "at its random start"} gave that word a probability of `, h("strong", { class: "w-num" }, pct(p)), ". Surprise: ", texEl(`-\\log ${fmtTex(p, 3)} = ${fmtTex(bt.loss, 3)}`)));

    // nudge
    const [k, i, j] = psel.value.split(",").map((v, n) => (n === 0 ? v : +v));
    const w2 = M.cloneWeights(w); w2[k][i][j] += 0.01;
    const f2 = M.forward(w2, ids), p2 = f2.probs[target], L2 = -Math.log(p2);
    const slope = (L2 - bt.loss) / 0.01;
    nudgeBox.innerHTML = "";
    nudgeBox.append(
      h("p", { class: "out-line" }, `Nudge it from ${fmt(w[k][i][j], 3)} to ${fmt(w2[k][i][j], 3)}: the probability goes from ${fmt(p * 100, 3)}% to ${fmt(p2 * 100, 3)}%, and the surprise from ${fmt(bt.loss, 4)} to ${fmt(L2, 4)}.`),
      h("p", { class: "out-line" }, texEl(`\\text{slope} \\approx \\frac{${fmtTex(L2, 4)} - ${fmtTex(bt.loss, 4)}}{0.01} = ${fmtTex(slope, 3)}`), Math.abs(slope) < 1e-9 ? h("span", { class: "w-help" }, "  Zero: this number isn't used for this phrase at all.") : ""));

    // grid of all slopes
    gridBox.innerHTML = "";
    const c = COLORS();
    const G = bt.grad;
    let mx = 1e-9;
    for (const kk of M.PARAM_NAMES) G[kk].forEach((r) => r.forEach((v) => { mx = Math.max(mx, Math.abs(v)); }));
    for (const kk of M.PARAM_NAMES) {
      const [r, cc] = M.SHAPES[kk];
      const lab = kk === "E" ? "auto " : "";
      const cells = h("div", { class: "cells", style: { gridTemplateColumns: `${lab}repeat(${cc}, 40px)` } });
      if (kk === "WO") VOCAB.forEach((wd) => cells.append(h("span", { class: "glab" }, wd)));
      for (let a = 0; a < r; a++) for (let b = 0; b < cc; b++) {
        if (kk === "E" && b === 0) cells.append(h("span", { class: "glab rl" }, VOCAB[a]));
        const v = G[kk][a][b];
        const hit = kk === k && a === i && b === j;
        cells.append(h("span", { title: `${paramLabel(kk, a, b)}: slope ${fmt(v, 4)}`, style: { width: "40px", background: `color-mix(in srgb, ${v >= 0 ? c.word : c.attn} ${Math.round(Math.min(1, Math.abs(v) / mx) * 70)}%, transparent)`, outline: hit ? `2px solid ${c.ink}` : "none" } }, Math.abs(v) < 0.0005 ? "0" : fmt(v, 2)));
      }
      gridBox.append(h("div", { class: "grp" }, h("span", { html: kk === "E" ? "E" : `W<sub>${kk[1]}</sub>` }), cells));
    }
    result.textContent = note;
  };

  const doStep = (wholeText) => {
    const st = stepState.get();
    const lr = +lrSel.value;
    const ids = exIds(st.ex), target = exTarget(st.ex);
    const grad = wholeText ? M.gradient(st.w, C.data).grad : M.backwardTrace(st.w, ids, target).grad;
    const w2 = M.plainStep(st.w, grad, lr);
    const p0 = M.forward(st.w, ids).probs[target], p1 = M.forward(w2, ids).probs[target];
    const L0 = M.loss(st.w, C.data), L1 = M.loss(w2, C.data);
    const note = `${wholeText ? "Stepped using the average slope over the whole text." : "Stepped using this one example's slopes."} This example: ${pct(p0)} → ${pct(p1)}. Loss over the whole text: ${fmt(L0, 3)} → ${fmt(L1, 3)}${!wholeText && L1 > L0 ? " (worse: helping one prediction hurt others)." : "."}`;
    stepState.set({ ...st, w: w2, steps: st.steps + 1, note });
  };
  bEx.addEventListener("click", () => doStep(false));
  bAll.addEventListener("click", () => doStep(true));
  bReset.addEventListener("click", () => stepState.set({ ...stepState.get(), w: M.randomWeights(START_SEED), steps: 0, note: "" }));
  psel.addEventListener("change", draw);
  psel.value = `WO,0,${exTarget(0)}`;
  stepState.sub(draw);
}

// ------------------------------------------------------------ backpropagation with real numbers
export function blame(root) {
  const head = h("div");
  const barBox = h("div", { class: "scroll-x" });
  const chain = h("ol", { class: "steps", style: { fontFamily: "var(--font-ui)", fontSize: "16px" } });
  const check = h("p", { class: "status" });
  root.append(h("p", { class: "w-title" }, "Where the blame goes"), head, barBox,
    h("p", { class: "w-title", style: { marginTop: "16px" } }, "Passing the blame backwards"), chain, check);
  const draw = () => {
    const { ex, w, steps } = stepState.get();
    const ids = exIds(ex), target = exTarget(ex);
    const bt = M.backwardTrace(w, ids, target);
    const f = bt.f, n = f.ctx.length - 1;
    head.innerHTML = "";
    head.append(h("p", { class: "w-help" }, `Same example and model as above: “${EXAMPLES[ex][0]}” followed by “${EXAMPLES[ex][1]}”, ${steps ? `after ${steps} step${steps === 1 ? "" : "s"}` : "at the random start"}.`));
    // predicted vs actual vs blame
    const c = COLORS();
    const tb = h("tbody");
    const mx = Math.max(...bt.dl.map(Math.abs), 1e-9);
    VOCAB.forEach((_, t) => {
      const d = bt.dl[t];
      const bar = h("div", { style: { position: "relative", height: "14px", width: "160px", background: c.panel2, borderRadius: "3px" } },
        h("i", { style: { position: "absolute", top: 0, bottom: 0, left: d < 0 ? `${80 - (Math.abs(d) / mx) * 80}px` : "80px", width: `${(Math.abs(d) / mx) * 80}px`, background: d < 0 ? c.out : c.attn, borderRadius: "3px" } }),
        h("i", { style: { position: "absolute", top: "-2px", bottom: "-2px", left: "79.5px", width: "1px", background: c.ink3 } }));
      tb.append(h("tr", {}, h("td", {}, tile(t, t === target ? "on" : "")), h("td", { class: "r num" }, fmt(f.probs[t], 3)), h("td", { class: "r num" }, String(bt.y[t])), h("td", { class: "r num" }, fmt(d, 3)), h("td", {}, bar)));
    });
    barBox.innerHTML = "";
    barBox.append(h("table", { class: "dtable" }, h("thead", {}, h("tr", {}, h("th", {}, "word"), h("th", { class: "r" }, "model said"), h("th", { class: "r" }, "actually"), h("th", { class: "r" }, "blame = said − actual"), h("th", {}, ""))), tb));

    const L = (title, src, note = "") => {
      const m = h("div", { class: "scroll-x" });
      tex(m, src, false);
      chain.append(h("li", {}, h("strong", { html: title }), " ", h("span", { html: note }), m));
    };
    chain.innerHTML = "";
    const v2 = (v, d = 3) => vecTex(v, d);
    L("Scores.", `\\text{blame} = ${v2(bt.dl.map((x) => x), 2)}`, "Start with the blame on the eight scores, from the table.");
    L("Back through W<sub>O</sub>.", `\\text{blame on } r = \\WO \\times \\text{blame} = ${v2(bt.dr)}`, "Each word's arrow passes back its share. The slope for each number in W<sub>O</sub> is the matching number of r times that word's blame.");
    L("Split at the plus sign.", `r = \\xn + \\zv \\;\\Rightarrow\\; \\text{blame on } \\zv = \\text{blame on } \\xn\\text{ (direct path)} = ${v2(bt.dr)}`, "Adding passes the same blame to both parts.");
    L("Into the blend.", `\\text{blame on each value} = \\av_j \\times ${v2(bt.dz)}:\\ ${bt.dv.map((v) => v2(v)).join(",\\ ")}`, "Each word's value is blamed in proportion to how much attention it got.");
    L("Into the attention weights.", `\\text{blame on each weight} = \\vv_j \\cdot \\text{blame on } \\zv:\\ ${v2(bt.da)}`, "A weight is blamed by how much its word's value pointed along the blame.");
    L("Back through softmax.", `\\text{blame on each score} = ${v2(bt.ds, 4)}`, "Softmax shares blame out so that the scores' blame always adds to zero.");
    L("Into the query and keys.", `\\text{blame on } \\qv = ${v2(bt.dq)},\\quad \\text{on the keys: } ${bt.dk.map((v) => v2(v)).join(",\\ ")}`, "Each score was query · key ÷ √2, so each side is blamed by the other side's numbers.");
    L("Into the words' numbers.", `${f.ctx.map((t, i) => `\\text{${VOCAB[t]}}: ${v2(bt.dX[i])}`).join(",\\quad ")}`, "Each word's embedding collects blame from its key, its value and, for the last word, its query and the direct path. Words that aren't in the phrase get none.");

    // check against nudging
    let worst = 0;
    const base = bt.loss;
    for (const k of M.PARAM_NAMES) {
      const [r, cc] = M.SHAPES[k];
      for (let i = 0; i < r; i++) for (let j = 0; j < cc; j++) {
        const w2 = M.cloneWeights(w); w2[k][i][j] += 1e-6; const w3 = M.cloneWeights(w); w3[k][i][j] -= 1e-6;
        const num = (-Math.log(M.forward(w2, ids).probs[target]) + Math.log(M.forward(w3, ids).probs[target])) / 2e-6;
        worst = Math.max(worst, Math.abs(num - bt.grad[k][i][j]));
      }
    }
    check.innerHTML = "";
    check.append("Collected up, these give all 44 slopes. Checked against nudging every number one at a time (88 extra runs of the model), the largest difference is ", sci(worst, 1), ": the same answers, from one backward pass.");
  };
  stepState.sub(draw);
}

// ------------------------------------------------------------ plain steps vs Adam
export function compare(root) {
  const C = corpus();
  const lrSel = h("select", { "aria-label": "Plain step size" }, [0.3, 1, 3].map((v) => h("option", { value: v }, String(v))));
  lrSel.value = "1";
  const run = h("button", { class: "primary" }, "Train both from random start 1");
  const chart = h("div");
  const out = h("p", { class: "status" });
  root.append(h("p", { class: "w-title" }, "Plain steps against Adam, 3,000 steps each"),
    h("div", { class: "w-row" }, h("span", { class: "w-label" }, "Plain step size"), lrSel, run), chart, out);
  const Wc = 640, Hc = 220, pl = 44, pr = 12, pt = 12, pb = 30, STEPS = 3000, ymin = 0.3, ymax = 2.3;
  const X = (k) => pl + (k / STEPS) * (Wc - pl - pr);
  const Y = (v) => pt + (1 - (Math.min(ymax, Math.max(ymin, v)) - ymin) / (ymax - ymin)) * (Hc - pt - pb);
  let curves = null;
  const draw = () => {
    const c = COLORS();
    const svg = s("svg", { viewBox: `0 0 ${Wc} ${Hc}`, class: "plot", role: "img", "aria-label": "Loss curves for plain steps and Adam" });
    for (let v = 0.5; v <= 2.0; v += 0.5) svg.append(s("line", { x1: pl, x2: Wc - pr, y1: Y(v), y2: Y(v), class: "gridline" }), s("text", { x: pl - 6, y: Y(v) + 4, "text-anchor": "end", class: "ticklabel" }, fmt(v, 1)));
    for (let k = 0; k <= STEPS; k += 1000) svg.append(s("text", { x: Math.min(X(k), Wc - pr - 24), y: Hc - 10, "text-anchor": "middle", class: "ticklabel" }, k === 0 ? "0 steps" : k.toLocaleString()));
    svg.append(s("line", { x1: pl, x2: Wc - pr, y1: Y(C.floor), y2: Y(C.floor), stroke: c.out, "stroke-dasharray": "5 4", "stroke-width": 1.5 }));
    if (curves) {
      const path = (hist) => hist.map((v, k) => (isFinite(v) ? `${k && isFinite(hist[k - 1]) ? "L" : "M"}${X(k + 1).toFixed(1)},${Y(v).toFixed(1)}` : "")).join("");
      svg.append(s("path", { d: path(curves.plain), fill: "none", stroke: c.attn, "stroke-width": 2 }));
      svg.append(s("path", { d: path(curves.adam), fill: "none", stroke: c.word, "stroke-width": 2.4 }));
      svg.append(s("text", { x: Wc - pr - 4, y: Y(curves.adam.at(-1)) - 8, "text-anchor": "end", class: "lbl", fill: c.word }, "Adam"));
      const lastPlain = curves.plain.filter(isFinite).at(-1);
      if (lastPlain != null) svg.append(s("text", { x: Wc - pr - 4, y: Y(lastPlain) - 8, "text-anchor": "end", class: "lbl", fill: c.attn }, "plain steps"));
    }
    chart.innerHTML = ""; chart.append(svg);
  };
  run.addEventListener("click", async () => {
    run.disabled = true; out.textContent = "Training…";
    await new Promise((r) => setTimeout(r, 30));
    const A = new M.Trainer(M.randomWeights(START_SEED), C.data);
    const P = new M.Trainer(M.randomWeights(START_SEED), C.data, +lrSel.value, "plain");
    for (let k = 0; k < STEPS; k += 300) { A.step(300); P.step(300); curves = { adam: A.history, plain: P.history }; draw(); await new Promise((r) => setTimeout(r, 0)); }
    const pl_ = P.history.at(-1);
    out.textContent = `Adam: ${fmt(A.history.at(-1), 3)} (the best possible is ${fmt(C.floor, 3)}). Plain steps of size ${lrSel.value}: ${isFinite(pl_) && pl_ < 50 ? fmt(pl_, 3) : "blew up: the steps were so large the numbers ran off to infinity"}.`;
    run.disabled = false;
  });
  draw();
  onScheme(draw);
}
