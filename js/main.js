// Page start-up: render math, mount widgets, run the equation bar and tooltips.
import { KATEX_OPTS, MACROS, tex } from "./ui.js";
import * as L from "./lessons.js";
import { playground } from "./playground.js";
import { trainer, corpusWidget, corpus } from "./training.js";
import { rules, strange, same } from "./deep.js";
import { descent, onestep, errorSignal, compare } from "./trainsteps.js";

const WIDGETS = {
  hero: L.hero, heroWeights: L.heroWeights, vocab: L.vocab, fit: L.fit, embed: L.embed, Etex: L.Etex, phrase: L.phrase,
  matmul: L.matmul, transform: L.transform, dot: L.dot, softmax: L.softmax, attention: L.attention, output: L.output,
  fullNumeric: L.fullNumeric, expanded: L.expanded, playground, corpus: corpusWidget, trainer, descent, onestep, errorSignal, compare, rules, strange, same,
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
  res: ["+", "Adds the last word's own numbers back in, so attention's findings sit on top of what the word already was (a residual connection)."],
  r: ["r", "The last word's numbers plus what attention found (r = x<sub>n</sub> + z). The output step scores every word against it."],
  score: ["Scores", "One score per word: the dot product of r with that word's arrow (its column of W<sub>O</sub>). Softmax turns the scores into probabilities."],
  WOarrow: ["A word's arrow", "This word's column of W<sub>O</sub>, drawn as an arrow. The word's score is the dot product of r with this arrow."],
  dotArrow: ["Arrow", "One of the two arrows you can drag. Its two numbers are listed beside the chart."],
  ch3: ["Chapter 3", "Its number is blue because this chapter explains the blue parts of the equation: the words' own numbers, X and x<sub>n</sub>."],
  ch7: ["Chapter 7", "Its number is red because this chapter explains attention, the red parts of the equation (with the teal, orange and purple pieces that feed it)."],
  ch8: ["Chapter 8", "Its number is green because this chapter explains the green parts of the equation: W<sub>O</sub> and the final softmax."],
};

// Which chapter explains each symbol (used by the pinned equation).
const CHAPTER = { X: 3, xn: 3, E: 3, WQ: 7, WK: 7, T: 7, sqrt: 7, softmaxA: 7, WV: 7, res: 8, WO: 8, softmaxO: 8 };
const CHAPTER_ID = { 3: "words", 7: "attention", 8: "predict" };

// Colored numbers and matrices without a named symbol explain their color instead.
const COLOR_TIP = {
  "c-word": ["Word numbers", "Blue marks the words' own numbers: their rows of E, stacked into X, with x<sub>n</sub> the last word."],
  "c-query": ["Query", "Teal marks the query part: the matrix W<sub>Q</sub>, and the query q = x<sub>n</sub>W<sub>Q</sub> it makes, what the last word is looking for."],
  "c-key": ["Keys", "Orange marks the key part: the matrix W<sub>K</sub>, and each word's key k = xW<sub>K</sub>, what it offers to be found by."],
  "c-value": ["Values", "Purple marks the value part: the matrix W<sub>V</sub>, each word's value v = xW<sub>V</sub>, and their blend z."],
  "c-attn": ["Attention", "Red marks attention: the attention weights, the softmax that makes them, and the ÷√2 scaling."],
  "c-out": ["Output", "Green marks the output part: the matrix W<sub>O</sub> (one arrow per word), the eight word scores, and the softmax that turns them into probabilities."],
};
const COLOR_SEL = Object.keys(COLOR_TIP).map((c) => `.katex .${c}, .mat.${c}`).join(", ");

function whenKatex() {
  return new Promise((res) => {
    const ok = () => window.katex && window.renderMathInElement;
    if (ok()) return res();
    const t = setInterval(() => { if (ok()) { clearInterval(t); res(); } }, 30);
  });
}

function tooltips() {
  const tip = document.getElementById("tip");
  let cur = null, pinned = false, lastPtr = "mouse";
  // The thing a pointer is on: a named symbol first, otherwise a colored number or matrix.
  // Typeset math leaves gaps (between W and its subscript, between a matrix's numbers)
  // that belong to invisible layout boxes, so inside math we also check by position:
  // the smallest explained item whose drawn area contains the pointer.
  const range = document.createRange();
  const CAND = "[data-sym], " + Object.keys(COLOR_TIP).map((c) => "." + c).join(", ");
  const find = (e) => {
    const t = e.target;
    const direct = t?.closest?.("[data-sym]") || t?.closest?.(COLOR_SEL);
    if (direct) return direct;
    const root = t?.closest?.(".katex");
    if (!root) return null;
    let best = null, bestA = Infinity;
    for (const c of root.querySelectorAll(CAND)) {
      range.selectNodeContents(c);
      const r = range.getBoundingClientRect();
      if (e.clientX >= r.left - 1 && e.clientX <= r.right + 1 && e.clientY >= r.top - 1 && e.clientY <= r.bottom + 1) {
        const a = r.width * r.height;
        if (a < bestA) { best = c; bestA = a; }
      }
    }
    // a colored piece inside a named symbol gets the symbol's own explanation
    return best && !best.dataset.sym ? best.closest("[data-sym]") || best : best;
  };
  const show = (el, pin = false) => {
    let d;
    if (el.dataset.sym) d = SYM[el.dataset.sym];
    else d = COLOR_TIP[Object.keys(COLOR_TIP).find((c) => el.classList.contains(c))];
    if (!d) return;
    cur = el; pinned = pin;
    tip.innerHTML = `<b>${d[0]}</b>&nbsp; ${d[1]}`;
    const ch = CHAPTER[el.dataset.sym];
    if (el.closest(".eqbar") && ch) {
      tip.insertAdjacentHTML("beforeend", `<br><button class="tip-go" data-go="${CHAPTER_ID[ch]}">Go to chapter ${ch}</button>`);
      tip.style.pointerEvents = "auto";
    } else tip.style.pointerEvents = "none";
    tip.classList.add("on");
    const r = el.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let x = r.left + r.width / 2 - tw / 2 + window.scrollX;
    x = Math.max(8 + window.scrollX, Math.min(x, window.scrollX + document.documentElement.clientWidth - tw - 8));
    let y = r.top + window.scrollY - th - 10;
    if (r.top < th + 70) y = r.bottom + window.scrollY + 10;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  };
  const hide = () => { cur = null; pinned = false; tip.classList.remove("on"); tip.style.pointerEvents = "none"; };
  document.addEventListener("pointerdown", (e) => { lastPtr = e.pointerType || "mouse"; }, true);
  let moveQueued = null;
  document.addEventListener("pointermove", (e) => {
    if (e.pointerType === "touch") return;
    if (moveQueued) { moveQueued = e; return; }
    moveQueued = e;
    requestAnimationFrame(() => {
      const ev = moveQueued; moveQueued = null;
      if (ev.target.closest?.("#tip") || ev.target.closest?.("[data-dragging]")) return;
      const el = find(ev);
      if (el) { if (el !== cur) show(el); }
      else if (cur && !pinned) hide();
    });
  }, { passive: true });
  document.addEventListener("click", (e) => {
    const go = e.target.closest?.(".tip-go");
    if (go) { document.getElementById(go.dataset.go)?.scrollIntoView({ behavior: "smooth" }); hide(); return; }
    if (e.target.closest?.("#tip")) return;
    const el = find(e);
    if (!el) { hide(); return; }
    // A click keeps the explanation open; a second tap on a touch screen closes it.
    if (cur === el && pinned && lastPtr === "touch") hide();
    else show(el, true);
  });
  window.addEventListener("scroll", hide, { passive: true });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hide(); });
}

function eqbar() {
  const bar = document.getElementById("eqbar");
  const target = document.getElementById("eqbar-eq");
  const p = (cls, x, sym) => `\\htmlData{sym=${sym}}{\\htmlClass{eqp ${cls}}{${x}}}`;
  const src = `P(\\text{next}) = ${p("p-out", "\\operatorname{softmax}", "softmaxO")}\\Big(\\big(${p("p-word", "x_n", "xn")} ${p("p-res", "+", "res")} ${p("p-attn", "\\operatorname{softmax}", "softmaxA")}\\big(${p("p-word", "x_n", "xn")} ${p("p-query", "W_Q", "WQ")} (${p("p-word", "X", "X")} ${p("p-key", "W_K", "WK")})^{${p("p-key", "\\top", "T")}} ${p("p-attn", "/\\sqrt{2}", "sqrt")}\\big) ${p("p-word", "X", "X")} ${p("p-value", "W_V", "WV")}\\big) ${p("p-out", "W_O", "WO")}\\Big)`;
  window.katex.render(src, target, { ...KATEX_OPTS, macros: { ...MACROS } });
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
