// Chapter 13: rules versus the equation.
import * as M from "./model.js";
import { h, fmt, pct, sci, tile, matHTML, fmtTex } from "./ui.js";
import { corpus } from "./training.js";

const { VOCAB, INDEX: I, TRAINED: W } = M;
const ANIMALS = ["cat", "dog"];

// What Tinyspeak's three rules say comes next, or null if the phrase can't occur in Tinyspeak.
export function rulesSay(ids) {
  const C = corpus();
  const key = ids.slice(-3).map((t) => VOCAB[t]).join(" ");
  if (!C.data.some((d) => d.key === key)) return null;
  const w = key.split(" ");
  const last = w.at(-1), prev = w.at(-2), prev2 = w.at(-3);
  const dist = new Array(M.V).fill(0);
  if (last === "." || last === "chases" || last === "sees") dist[I[last === "." ? "the" : "the"]] = 1;
  else if (last === "runs" || last === "sleeps") dist[I["."]] = 1;
  else if (last === "the") {
    if (prev === "chases" || prev === "sees") dist[I[prev2 === "cat" ? "dog" : "cat"]] = 1;
    else { dist[I.cat] = 0.5; dist[I.dog] = 0.5; }
  } else if (ANIMALS.includes(last)) {
    if (prev2 === "chases" || prev2 === "sees") dist[I["."]] = 1;
    else for (const [v, p] of M.VERB_HABITS[last]) dist[I[v]] = p;
  }
  return dist;
}

const wordName = (t) => (VOCAB[t] === "." ? "“.”" : VOCAB[t]);
const summary = (p, min = 0.02) => [...p.keys()].filter((t) => p[t] >= min).sort((a, b) => p[b] - p[a]).map((t) => `${wordName(t)} ${pct(p[t])}`).join(", ");

export function rules(root) {
  const C = corpus();
  const rows = C.data.slice().sort((a, b) => a.ctx.length - b.ctx.length || a.key.localeCompare(b.key));
  let agree = 0, maxDiff = 0;
  const body = h("tbody");
  for (const d of rows) {
    const r = rulesSay(d.ctx), p = M.forward(W, d.ctx).probs;
    const same = r.every((rv, t) => (rv >= 0.02) === (p[t] >= 0.02));
    const diff = Math.max(...r.map((rv, t) => Math.abs(rv - p[t])));
    maxDiff = Math.max(maxDiff, diff);
    if (same) agree++;
    body.append(h("tr", {}, h("td", {}, h("span", { class: "tok-strip", style: { minHeight: 0, gap: "4px" } }, d.ctx.map((t) => tile(t)))), h("td", {}, summary(r)), h("td", {}, summary(p)), h("td", { class: same ? "good" : "bad" }, same ? "✓" : "✗")));
  }
  const showFirst = ["the", "the cat", "cat chases the", "dog chases the", "chases the cat", "dog . the"];
  const preview = h("tbody");
  for (const key of showFirst) {
    const d = rows.find((x) => x.key === key);
    const r = rulesSay(d.ctx), p = M.forward(W, d.ctx).probs;
    preview.append(h("tr", {}, h("td", {}, h("span", { class: "tok-strip", style: { minHeight: 0, gap: "4px" } }, d.ctx.map((t) => tile(t)))), h("td", {}, summary(r)), h("td", {}, summary(p))));
  }
  root.append(
    h("p", { class: "w-title" }, "The rules against the equation, on every phrase Tinyspeak allows"),
    h("div", { class: "scroll-x" }, h("table", { class: "dtable", style: { margin: "8px 0 12px" } }, h("thead", {}, h("tr", {}, h("th", {}, "phrase"), h("th", {}, "the rules say"), h("th", {}, "the equation says"))), preview)),
    h("p", { class: "status" }, `They agree on which words can come next in ${agree} of ${rows.length} phrases. The biggest difference in any single probability is ${Math.round(maxDiff * 100)} percentage points (that gap comes from rounding the 44 numbers to one decimal place so the equation is readable; before rounding, the model matched every probability to within about 2 points).`),
    h("details", { class: "more" }, h("summary", {}, `Show all ${rows.length} phrases`),
      h("div", { class: "scroll-x" }, h("table", { class: "dtable" }, h("thead", {}, h("tr", {}, h("th", {}, "phrase"), h("th", {}, "the rules say"), h("th", {}, "the equation says"), h("th", {}, "agree"))), body))));
}

// Stretch Tinyspeak's rules to a phrase that can't occur, using only its last word(s).
// Returns { words: [...allowed next words], why } or { words: null, why }.
export function stretchRules(ids) {
  const w = ids.map((t) => VOCAB[t]);
  const last = w.at(-1), prev = w.at(-2);
  if (last === ".") return { words: ["the"], why: "every sentence starts with “the”" };
  if (last === "runs" || last === "sleeps") return { words: ["."], why: `“${last}” ends the sentence` };
  if (last === "chases" || last === "sees") return { words: ["the"], why: `“${last}” is followed by “the”` };
  if (last === "the") return { words: ["cat", "dog"], why: "“the” is followed by an animal" };
  if (prev === "the") return { words: ["runs", "sleeps", "chases", "sees", "."], why: "after “the” and an animal comes an action, or the end of the sentence" };
  return { words: null, why: "in Tinyspeak an animal only ever comes right after “the”, so no rule applies" };
}

export function strange(root) {
  const all = [];
  for (let n = 1; n <= 3; n++) {
    const rec = (pre) => { if (pre.length === n) { all.push(pre); return; } for (let t = 0; t < M.V; t++) rec([...pre, t]); };
    rec([]);
  }
  const odd = all.filter((ids) => rulesSay(ids) === null);
  const curated = ["dog dog dog", "sleeps sleeps sleeps", "cat sees", "the the the", "cat . dog", "runs the", "sees cat"].map(M.tokenize);
  let k = 0, confident = 0, covered = 0, follows = 0;
  for (const ids of odd) {
    const p = M.forward(W, ids).probs, t = M.argmax(p), st = stretchRules(ids);
    if (p[t] > 0.9) confident++;
    if (st.words) { covered++; if (st.words.includes(VOCAB[t])) follows++; }
  }
  const show = h("div");
  const next = h("button", { class: "primary" }, "Show another strange phrase");
  const draw = () => {
    const ids = k < curated.length ? curated[k] : odd[Math.floor(Math.random() * odd.length)];
    const p = M.forward(W, ids).probs, top = VOCAB[M.argmax(p)], st = stretchRules(ids);
    show.innerHTML = "";
    const ruleLine = h("p", { class: "out-line" }, h("strong", {}, "The rules: "), "this phrase can't occur in Tinyspeak. ");
    if (st.words) ruleLine.append(`Stretching them by the last word: ${st.why}, so they'd suggest ${st.words.map((x) => `“${x}”`).join(" or ")}.`);
    else ruleLine.append(`Even stretching them doesn't help: ${st.why}.`);
    const eqLine = h("p", { class: "out-line" }, h("strong", {}, "The equation: "), summary(p, 0.05) + ". ");
    if (st.words) eqLine.append(h("span", { class: st.words.includes(top) ? "good" : "bad" }, st.words.includes(top) ? "Its top answer fits the stretched rule." : "Its top answer doesn't fit the stretched rule."));
    show.append(h("div", { class: "tok-strip" }, ids.map((t) => tile(t))), ruleLine, eqLine);
  };
  next.addEventListener("click", () => { k++; draw(); });
  root.append(h("p", { class: "w-title" }, `${odd.length} phrases that can't occur in Tinyspeak`), show, h("div", { class: "w-row" }, next),
    h("p", { class: "status" }, `Across all ${odd.length}: for ${covered}, a rule can be stretched by looking at the last word, and the equation's top answer fits that stretched rule in ${follows} of them. For the other ${odd.length - covered}, no rule applies at all. The equation answers every one, and is more than 90% sure of its answer for ${confident}. None of those answers was ever checked against anything; they're simply what the formula happens to give.`));
  draw();
}

export function same(root) {
  let Wq = W.WQ.map((r) => r.map((v) => v * 2));
  let Wk = W.WK.map((r) => r.map((v) => v / 2));
  const left = h("div"), right = h("div"), check = h("p", { class: "status" });
  const b = h("button", { class: "primary" }, "Scramble them a different way");
  const all = [];
  for (let n = 1; n <= 3; n++) { const rec = (pre) => { if (pre.length === n) { all.push(pre); return; } for (let t = 0; t < M.V; t++) rec([...pre, t]); }; rec([]); }
  const draw = () => {
    const alt = { ...M.cloneWeights(W), WQ: Wq, WK: Wk };
    left.innerHTML = ""; right.innerHTML = "";
    left.append(h("p", { class: "w-title" }, "The trained model"), h("div", { class: "w-row" }, matHTML(W.WQ, { cls: "c-query" }), matHTML(W.WK, { cls: "c-key" })));
    right.append(h("p", { class: "w-title" }, "A different model"), h("div", { class: "w-row" }, matHTML(Wq, { cls: "c-query", d: 2 }), matHTML(Wk, { cls: "c-key", d: 2 })));
    let worst = 0;
    for (const ids of all) { const p = M.forward(W, ids).probs, q = M.forward(alt, ids).probs; for (let t = 0; t < M.V; t++) worst = Math.max(worst, Math.abs(p[t] - q[t])); }
    check.innerHTML = "";
    check.append(`Largest difference in any prediction between the two models, over all 584 possible inputs: `, h("strong", {}, worst < 1e-12 ? "0" : sci(worst, 1)), worst < 1e-9 ? " (identical, up to the last digits of rounding inside the computer)." : ".");
  };
  b.addEventListener("click", () => {
    // pick a random invertible 2×2 matrix A; use W_Q·A and W_K·(A⁻¹)ᵀ
    let A, det;
    do { A = [[Math.random() * 4 - 2, Math.random() * 4 - 2], [Math.random() * 4 - 2, Math.random() * 4 - 2]]; det = A[0][0] * A[1][1] - A[0][1] * A[1][0]; } while (Math.abs(det) < 0.4);
    const inv = [[A[1][1] / det, -A[0][1] / det], [-A[1][0] / det, A[0][0] / det]];
    const invT = [[inv[0][0], inv[1][0]], [inv[0][1], inv[1][1]]];
    const mul = (X, Y) => X.map((r) => [r[0] * Y[0][0] + r[1] * Y[1][0], r[0] * Y[0][1] + r[1] * Y[1][1]]);
    Wq = mul(W.WQ, A); Wk = mul(W.WK, invT);
    draw();
  });
  root.append(h("p", { class: "w-help", html: "The query and key matrices of two models. The first pair is the trained model's; the second starts as “double W<sub>Q</sub>, halve W<sub>K</sub>”." }),
    h("div", { class: "w-grid two" }, left, right), check, h("div", { class: "w-row" }, b));
  draw();
}
