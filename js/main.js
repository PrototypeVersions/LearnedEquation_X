// Page start-up: render math, mount widgets, run the equation bar and tooltips.
import { KATEX_OPTS, MACROS, tex } from "./ui.js";
import * as L from "./lessons.js";
import { playground } from "./playground.js";
import { trainer, corpusWidget, corpus } from "./training.js";
import { rules, strange, same } from "./deep.js";

const WIDGETS = {
  hero: L.hero, heroWeights: L.heroWeights, vocab: L.vocab, fit: L.fit, embed: L.embed, Etex: L.Etex, phrase: L.phrase,
  matmul: L.matmul, transform: L.transform, dot: L.dot, softmax: L.softmax, attention: L.attention, output: L.output,
  fullNumeric: L.fullNumeric, expanded: L.expanded, playground, corpus: corpusWidget, trainer, rules, strange, same,
};

const SYM = {
  X: ["X", "The last three words, one row of two numbers per word."],
  xn: ["x<sub>n</sub>", "The last word's two numbers. This is the word that is predicting what comes next."],
  E: ["E", "The embedding table: two numbers for each of the 8 words."],
  WQ: ["W<sub>Q</sub>", "The query matrix. Turns a word's numbers into what it is looking for."],
  WK: ["W<sub>K</sub>", "The key matrix. Turns a word's numbers into what it offers to be found by."],
  WV: ["W<sub>V</sub>", "The value matrix. Turns a word's numbers into what it hands over when chosen."],
  WO: ["W<sub>O</sub>", "The output matrix. One column per word; scores every possible next word."],
  softmax: ["softmax", "Turns any list of scores into probabilities that add up to 1."],
  softmaxA: ["softmax", "Here it turns relevance scores into attention weights."],
  softmaxO: ["softmax", "Here it turns the 8 word scores into the final probabilities."],
  T: ["⊤", "Transpose: flip rows into columns, so one multiplication gives every dot product at once."],
  sqrt: ["√2", "About 1.414. A fixed shrink of the attention scores."],
  q: ["q", "The query: what the last word is looking for."],
  k: ["k", "A key: what a word offers to be found by."],
  v: ["v", "A value: what a word hands over if attention picks it."],
  a: ["a", "Attention weights: how much the last word listens to each word. They add up to 1."],
  z: ["z", "The blend of values that attention hands back."],
};

function whenKatex() {
  return new Promise((res) => {
    const ok = () => window.katex && window.renderMathInElement;
    if (ok()) return res();
    const t = setInterval(() => { if (ok()) { clearInterval(t); res(); } }, 30);
  });
}

function tooltips() {
  const tip = document.getElementById("tip");
  let cur = null;
  const show = (el) => {
    const d = SYM[el.dataset.sym];
    if (!d) return;
    cur = el;
    tip.innerHTML = `<b>${d[0]}</b>&nbsp; ${d[1]}`;
    tip.classList.add("on");
    const r = el.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let x = r.left + r.width / 2 - tw / 2 + window.scrollX;
    x = Math.max(8 + window.scrollX, Math.min(x, window.scrollX + document.documentElement.clientWidth - tw - 8));
    let y = r.top + window.scrollY - th - 10;
    if (r.top < th + 70) y = r.bottom + window.scrollY + 10;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  };
  const hide = () => { cur = null; tip.classList.remove("on"); };
  document.addEventListener("pointerover", (e) => { const el = e.target.closest?.("[data-sym]"); if (el && e.pointerType !== "touch") show(el); else if (!el && cur) hide(); });
  document.addEventListener("click", (e) => { const el = e.target.closest?.("[data-sym]"); if (el && !el.closest(".eqbar")) { if (cur === el) hide(); else show(el); } else if (!e.target.closest?.(".eqbar")) hide(); });
  window.addEventListener("scroll", hide, { passive: true });
}

function eqbar() {
  const bar = document.getElementById("eqbar");
  const target = document.getElementById("eqbar-eq");
  const p = (cls, x) => `\\htmlClass{eqp ${cls}}{${x}}`;
  const src = `P(\\text{next}) = ${p("p-out", "\\operatorname{softmax}")}\\Big(\\big(${p("p-word", "x_n")} ${p("p-res", "+")} ${p("p-attn", "\\operatorname{softmax}")}\\big(${p("p-word", "x_n")} ${p("p-query", "W_Q")} (${p("p-word", "X")} ${p("p-key", "W_K")})^{${p("p-key", "\\top")}} ${p("p-attn", "/\\sqrt{2}")}\\big) ${p("p-word", "X")} ${p("p-value", "W_V")}\\big) ${p("p-out", "W_O")}\\Big)`;
  window.katex.render(src, target, { ...KATEX_OPTS, macros: { ...MACROS } });
  const JUMP = { "p-word": "words", "p-query": "attention", "p-key": "attention", "p-value": "attention", "p-attn": "attention", "p-res": "predict", "p-out": "predict" };
  target.addEventListener("click", (e) => {
    const part = e.target.closest(".eqp");
    if (!part) return;
    const cls = [...part.classList].find((c) => c.startsWith("p-"));
    document.getElementById(JUMP[cls])?.scrollIntoView({ behavior: "smooth" });
  });
  const sections = [...document.querySelectorAll("[data-learn]")];
  const heroEq = document.getElementById("hero-eq");
  let ticking = false;
  const update = () => {
    ticking = false;
    const vh = window.innerHeight;
    bar.classList.toggle("shown", heroEq.getBoundingClientRect().bottom < 0);
    for (const sct of sections) {
      if (sct.getBoundingClientRect().top < vh * 0.55) sct.dataset.learn.split(" ").forEach((k) => bar.classList.add("k-" + k));
    }
  };
  window.addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
  update();
}

async function start() {
  await whenKatex();
  window.renderMathInElement(document.body, {
    ...KATEX_OPTS,
    macros: { ...MACROS },
    delimiters: [
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
    ],
    ignoredClasses: ["widget", "eqbar"],
  });
  const C = corpus();
  const fills = { nSent: C.nSent.toLocaleString(), nTokens: C.nTokens.toLocaleString(), floor: C.floor.toFixed(3) };
  document.querySelectorAll("[data-fill]").forEach((el) => { el.textContent = fills[el.dataset.fill] ?? el.textContent; });
  for (const el of document.querySelectorAll("[data-widget]")) {
    const fn = WIDGETS[el.dataset.widget];
    try { fn?.(el); } catch (err) { console.error("widget", el.dataset.widget, err); el.append(document.createTextNode("This part of the page failed to load.")); }
  }
  eqbar();
  tooltips();
}
start();
