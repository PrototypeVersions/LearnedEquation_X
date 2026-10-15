// Chapter 11: watch data choose the 44 numbers.
import * as M from "./model.js";
import { h, s, fmt, pct, COLORS, onScheme } from "./ui.js";
import { playWeights, playSource } from "./playground.js";

const { VOCAB, INDEX: I } = M;
let CORPUS = null;
export function corpus() { if (!CORPUS) CORPUS = M.buildCorpus(); return CORPUS; }

export function corpusWidget(root) {
  const C = corpus();
  const text = C.docs.slice(0, 7).map((d) => M.detok(d.map((w) => I[w]))).join(" ") + " …";
  root.append(h("p", { class: "aside", style: { fontStyle: "italic" } }, text));
}

const SEEDS = [3, 7, 1, 5, 9, 2, 11, 4, 6, 8, 10, 12];
const STEPS = 3000;

export function trainer(root) {
  const C = corpus();
  let seedIdx = 0, T = null, running = false, raf = null;
  const runs = []; // finished curves for comparison

  const bStart = h("button", { class: "primary" }, "Start training");
  const bNew = h("button", {}, "New random start");
  const bUse = h("button", { class: "ghost", disabled: true }, "Use these numbers in the playground");
  const readout = h("p", { class: "out-line" });
  const verdict = h("p", { class: "status" });
  const sample = h("p", { class: "aside", style: { minHeight: "3.2em", margin: "6px 0 0" } });
  const chartBox = h("div");
  const heat = h("div", { class: "heat" });

  root.append(
    h("p", { class: "w-title" }, "Training, live"),
    h("p", { class: "w-help" }, `Start from random numbers and take ${STEPS.toLocaleString()} small downhill steps. The dashed line is the best any model could do on this text.`),
    h("div", { class: "w-row" }, bStart, bNew, bUse),
    readout, chartBox,
    h("p", { class: "w-title", style: { marginTop: "14px" } }, "What the model writes with its current numbers"), sample,
    h("p", { class: "w-title", style: { marginTop: "14px" } }, "The 44 numbers"), heat, verdict);

  const Wc = 640, Hc = 230, pl = 44, pr = 12, pt = 12, pb = 30;
  const ymin = 0.3, ymax = 2.2;
  const X = (k) => pl + (k / STEPS) * (Wc - pl - pr);
  const Y = (v) => pt + (1 - (Math.min(ymax, Math.max(ymin, v)) - ymin) / (ymax - ymin)) * (Hc - pt - pb);

  const drawChart = () => {
    const c = COLORS();
    const svg = s("svg", { viewBox: `0 0 ${Wc} ${Hc}`, class: "plot", role: "img", "aria-label": "Loss going down during training" });
    for (let v = 0.5; v <= 2.0; v += 0.5) {
      svg.append(s("line", { x1: pl, x2: Wc - pr, y1: Y(v), y2: Y(v), class: "gridline" }), s("text", { x: pl - 6, y: Y(v) + 4, "text-anchor": "end", class: "ticklabel" }, fmt(v, 1)));
    }
    for (let k = 0; k <= STEPS; k += 1000) svg.append(s("text", { x: Math.min(X(k), Wc - pr - 24), y: Hc - 10, "text-anchor": "middle", class: "ticklabel" }, k === 0 ? "0 steps" : k.toLocaleString()));
    svg.append(s("line", { x1: pl, x2: Wc - pr, y1: Y(C.floor), y2: Y(C.floor), stroke: c.out, "stroke-dasharray": "5 4", "stroke-width": 1.5 }));
    svg.append(s("text", { x: Wc - pr - 4, y: Y(C.floor) + 15, "text-anchor": "end", class: "ticklabel", fill: c.out }, `best possible ${fmt(C.floor, 3)}`));
    const path = (hist) => hist.map((v, k) => `${k ? "L" : "M"}${X(k + 1).toFixed(1)},${Y(v).toFixed(1)}`).join("");
    runs.forEach((r) => svg.append(s("path", { d: path(r.hist), fill: "none", stroke: c.ink3, "stroke-width": 1.2, "stroke-opacity": 0.6 })));
    if (T && T.history.length) svg.append(s("path", { d: path(T.history), fill: "none", stroke: c.word, "stroke-width": 2.4 }));
    chartBox.innerHTML = ""; chartBox.append(svg);
  };

  const drawHeat = (w) => {
    heat.innerHTML = "";
    const c = COLORS();
    const names = { E: "E", WQ: "W<sub>Q</sub>", WK: "W<sub>K</sub>", WV: "W<sub>V</sub>", WO: "W<sub>O</sub>" };
    for (const k of M.PARAM_NAMES) {
      const [r, cc] = M.SHAPES[k];
      const cells = h("div", { class: "cells", style: { gridTemplateColumns: `repeat(${cc}, 30px)` } });
      for (let i = 0; i < r; i++) for (let j = 0; j < cc; j++) {
        const v = w[k][i][j];
        const a = Math.min(1, Math.abs(v) / 6);
        cells.append(h("span", { title: fmt(v, 2), style: { background: `color-mix(in srgb, ${v >= 0 ? c.word : c.attn} ${Math.round(a * 70)}%, transparent)` } }, fmt(v, 1)));
      }
      heat.append(h("div", { class: "grp" }, h("span", { html: names[k] }), cells));
    }
  };

  const drawSample = (w) => {
    const g = M.generate(w, [I.the], 22, "sample", M.rng(11));
    sample.textContent = M.detok(g.seq) + " …";
  };

  const status = () => {
    const step = T ? T.t : 0;
    const L = T && T.history.length ? T.history.at(-1) : M.loss(T ? T.w : M.randomWeights(SEEDS[seedIdx]), C.data);
    readout.innerHTML = "";
    readout.append(`Random start ${seedIdx + 1}, step ${step.toLocaleString()} of ${STEPS.toLocaleString()}. Loss: `, h("strong", { class: "w-num" }, fmt(L, 3)));
  };

  const diagnose = () => {
    const L = T.history.at(-1);
    const pDog = M.forward(T.w, [I.cat, I.chases, I.the]).probs[I.dog];
    const pCat = M.forward(T.w, [I.dog, I.chases, I.the]).probs[I.cat];
    const gap = L - C.floor;
    let ok = 0, tot = 0, okLoose = 0;
    const r = M.rng(5);
    for (let k = 0; k < 40; k++) {
      const seq = M.generate(T.w, [I.the], 40, "sample", r).seq;
      const [a, b] = M.grammarCheck(seq); ok += a; tot += b;
      okLoose += M.grammarCheck(seq, true)[0];
    }
    verdict.innerHTML = "";
    const chase = `after “cat chases the” it gives dog ${pct(pDog)}, and after “dog chases the” it gives cat ${pct(pCat)}`;
    if (gap < 0.005) {
      verdict.append(h("span", { class: "good" }, `Reached ${fmt(L, 4)}, just ${fmt(Math.max(gap, 0.0001), 4)} above the best possible ${fmt(C.floor, 4)}: essentially as good as any model can do on this text.`), ` ${ok} of ${tot} sampled sentences follow every rule, and it knows who gets chased (${chase}).`);
    } else if (okLoose >= 0.95 * tot && (pDog < 0.9 || pCat < 0.9)) {
      verdict.append(h("span", { class: "bad" }, `Stuck at ${fmt(L, 3)}.`), ` It learned what kind of word comes next, but not who gets chased: ${chase}. That's the one rule that needs attention to look back at the subject. Try another start.`);
    } else {
      verdict.append(h("span", { class: "bad" }, `Stuck at ${fmt(L, 3)}.`), ` Only ${okLoose} of ${tot} sampled sentences even have the right shape (the + animal + action …), so it hasn't learned the grammar either; ${chase}. Try another start.`);
    }
  };

  const reset = () => {
    cancelAnimationFrame(raf); running = false;
    T = new M.Trainer(M.randomWeights(SEEDS[seedIdx]), C.data);
    bStart.textContent = "Start training"; bStart.disabled = false; bUse.disabled = true;
    verdict.textContent = "";
    drawChart(); drawHeat(T.w); drawSample(T.w); status();
  };

  const loop = () => {
    if (!running) return;
    T.step(15);
    status(); drawChart();
    if (T.t % 150 === 0 || T.t >= STEPS) { drawHeat(T.w); drawSample(T.w); }
    if (T.t >= STEPS) {
      running = false;
      runs.push({ hist: T.history.slice(), seed: SEEDS[seedIdx] });
      if (runs.length > 5) runs.shift();
      bStart.textContent = "Finished"; bStart.disabled = true; bUse.disabled = false;
      drawHeat(T.w); drawSample(T.w); diagnose();
      return;
    }
    raf = requestAnimationFrame(loop);
  };

  bStart.addEventListener("click", () => {
    if (T.t >= STEPS) return;
    running = !running;
    bStart.textContent = running ? "Pause" : "Continue training";
    if (running) raf = requestAnimationFrame(loop);
  });
  bNew.addEventListener("click", () => {
    if (T && T.t > 0 && T.t < STEPS) { runs.push({ hist: T.history.slice(), seed: SEEDS[seedIdx] }); if (runs.length > 5) runs.shift(); }
    seedIdx = (seedIdx + 1) % SEEDS.length;
    reset();
  });
  bUse.addEventListener("click", () => {
    playWeights.set(M.cloneWeights(T.w));
    playSource.set("fromTraining");
    document.getElementById("playground")?.scrollIntoView({ behavior: "smooth" });
  });
  reset();
  onScheme(() => { drawChart(); drawHeat(T.w); });
}
