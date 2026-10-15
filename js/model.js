// The tiny language model behind the whole site.
// Everything the page shows is computed from this file, so the numbers in the
// lessons, the playground and the final equation always agree.

export const VOCAB = ["the", "cat", "dog", "runs", "sleeps", "chases", "sees", "."];
export const V = VOCAB.length; // 8 words
export const D = 2;            // each word is a list of 2 numbers
export const WINDOW = 3;       // the model reads the last 3 words
export const INDEX = Object.fromEntries(VOCAB.map((w, i) => [w, i]));

// The 44 numbers found by training (rounded to one decimal place).
// This is θ* — the fitted model.
export const TRAINED = {
  E: [
    [-3.8, 1.5],  // the
    [0.8, -2.3],  // cat
    [0.1, -2.3],  // dog
    [0.6, 3.5],   // runs
    [-3.0, 5.1],  // sleeps
    [3.7, 2.1],   // chases
    [3.7, 2.1],   // sees
    [3.1, -2.1],  // .
  ],
  WQ: [[-2.1, 0.5], [-1.5, 1.4]],
  WK: [[2.3, -1.4], [-1.8, 0.0]],
  WV: [[0.3, 1.0], [1.3, 1.8]],
  WO: [
    [6.1, -3.7, -4.2, -0.9, 0.4, -1.4, -0.2, 1.1],
    [-2.5, 3.0, 0.1, -4.1, -4.7, -3.9, -4.4, 3.9],
  ],
};

export const PARAM_NAMES = ["E", "WQ", "WK", "WV", "WO"];
export const SHAPES = { E: [V, D], WQ: [D, D], WK: [D, D], WV: [D, D], WO: [D, V] };
export const N_PARAMS = PARAM_NAMES.reduce((n, k) => n + SHAPES[k][0] * SHAPES[k][1], 0); // 44

export function cloneWeights(w) {
  const o = {};
  for (const k of PARAM_NAMES) o[k] = w[k].map((r) => r.slice());
  return o;
}

// ---------- small linear-algebra helpers ----------
export const vecMat = (x, M) => M[0].map((_, j) => x.reduce((s, xi, i) => s + xi * M[i][j], 0));
export const dot = (a, b) => a.reduce((s, ai, i) => s + ai * b[i], 0);
export function softmax(z) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

// ---------- words <-> indices ----------
export function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/\./g, " . ")
    .split(/\s+/)
    .filter(Boolean)
    .filter((w) => w in INDEX)
    .map((w) => INDEX[w]);
}
export const detok = (ids) => ids.map((i) => VOCAB[i]).join(" ").replace(/ \./g, ".");

// ---------- the forward pass, with every intermediate value kept ----------
// F(X) = softmax( ( x_n + softmax( x_n W_Q (X W_K)^T / sqrt(2) ) X W_V ) W_O )
export function forward(w, ids) {
  const ctx = ids.slice(-WINDOW);
  const X = ctx.map((t) => w.E[t].slice());
  const n = X.length - 1;
  const xn = X[n];
  const Q = X.map((x) => vecMat(x, w.WQ));
  const K = X.map((x) => vecMat(x, w.WK));
  const Vv = X.map((x) => vecMat(x, w.WV));
  const q = Q[n];
  const scores = K.map((k) => dot(q, k));
  const scaled = scores.map((s) => s / Math.sqrt(D));
  const attn = softmax(scaled);
  const z = [0, 0];
  Vv.forEach((v, i) => { z[0] += attn[i] * v[0]; z[1] += attn[i] * v[1]; });
  const r = [xn[0] + z[0], xn[1] + z[1]];
  const logits = vecMat(r, w.WO);
  const probs = softmax(logits);
  return { ctx, X, Q, K, V: Vv, q, scores, scaled, attn, z, r, logits, probs };
}

export function argmax(a) {
  let k = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i;
  return k;
}

// Deterministic pseudo-random numbers so every visitor sees the same "random" run.
export function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

export function sampleFrom(p, r) {
  const u = r();
  let c = 0;
  for (let i = 0; i < p.length; i++) { c += p[i]; if (u < c) return i; }
  return p.length - 1;
}

// Generate words one at a time. mode: "greedy" (always the likeliest) or "sample".
export function generate(w, ids, nWords, mode = "sample", r = Math.random) {
  const seq = ids.slice();
  const steps = [];
  for (let i = 0; i < nWords; i++) {
    const { probs } = forward(w, seq);
    const t = mode === "greedy" ? argmax(probs) : sampleFrom(probs, r);
    steps.push({ token: t, p: probs[t] });
    seq.push(t);
  }
  return { seq, steps };
}

// ---------- the tiny language ("Tinyspeak") the model was trained on ----------
export const VERB_HABITS = {
  cat: [["sleeps", 0.35], ["sees", 0.25], ["runs", 0.2], ["chases", 0.2]],
  dog: [["chases", 0.35], ["runs", 0.25], ["sees", 0.2], ["sleeps", 0.2]],
};
function pick(r, dist) {
  let u = r(), c = 0;
  for (const [k, p] of dist) { c += p; if (u < c) return k; }
  return dist[dist.length - 1][0];
}
export function sentence(r) {
  const s = pick(r, [["cat", 0.5], ["dog", 0.5]]);
  const v = pick(r, VERB_HABITS[s]);
  if (v === "runs" || v === "sleeps") return ["the", s, v, "."];
  return ["the", s, v, "the", s === "cat" ? "dog" : "cat", "."];
}

// The training text: 20,000 short passages of 1–3 sentences.
// Returns the passages and, for every 1–3 word context, how often each next word followed it.
export function buildCorpus(seed = 7, nDocs = 20000) {
  const r = rng(seed);
  const counts = new Map();
  const docs = [];
  let nSent = 0;
  for (let d = 0; d < nDocs; d++) {
    const k = 1 + Math.floor(r() * 3);
    let doc = [];
    for (let i = 0; i < k; i++) doc = doc.concat(sentence(r));
    nSent += k;
    if (docs.length < 40) docs.push(doc);
    for (let i = 1; i < doc.length; i++) {
      const key = doc.slice(Math.max(0, i - WINDOW), i).join(" ");
      if (!counts.has(key)) counts.set(key, new Array(V).fill(0));
      counts.get(key)[INDEX[doc[i]]]++;
    }
  }
  const data = [...counts.entries()].map(([key, c]) => {
    const tot = c.reduce((a, b) => a + b, 0);
    return { key, ctx: key.split(" ").map((w) => INDEX[w]), counts: c, target: c.map((x) => x / tot), w: tot };
  });
  const total = data.reduce((a, d) => a + d.w, 0);
  data.forEach((d) => { d.w /= total; });
  let floor = 0; // the lowest loss any model could reach on this data
  for (const d of data) for (const p of d.target) if (p > 0) floor -= d.w * p * Math.log(p);
  return { docs, data, nSent, nDocs, nTokens: total, floor };
}

// ---------- training ----------
// Loss = average surprise, -log P(actual next word).
export function loss(w, data) {
  let L = 0;
  for (const d of data) {
    const { probs } = forward(w, d.ctx);
    for (let t = 0; t < V; t++) if (d.target[t] > 0) L -= d.w * d.target[t] * Math.log(probs[t] + 1e-12);
  }
  return L;
}

const zeros = () => {
  const g = {};
  for (const k of PARAM_NAMES) g[k] = Array.from({ length: SHAPES[k][0] }, () => new Array(SHAPES[k][1]).fill(0));
  return g;
};

// Exact gradient of the loss (what you'd get by nudging each of the 44 numbers
// a tiny bit and watching the loss, computed all at once with the chain rule).
export function gradient(w, data) {
  const g = zeros();
  let L = 0;
  const s2 = Math.sqrt(D);
  for (const d of data) {
    const f = forward(w, d.ctx);
    const { X, K, V: Vv, q, attn, r, probs } = f;
    const n = X.length - 1;
    for (let t = 0; t < V; t++) if (d.target[t] > 0) L -= d.w * d.target[t] * Math.log(probs[t] + 1e-12);
    const dl = probs.map((p, t) => d.w * (p - d.target[t]));
    for (let i = 0; i < D; i++) for (let j = 0; j < V; j++) g.WO[i][j] += r[i] * dl[j];
    const dr = [0, 1].map((i) => w.WO[i].reduce((s, o, j) => s + o * dl[j], 0));
    const dz = dr;
    const dX = X.map(() => [0, 0]);
    dX[n][0] += dr[0]; dX[n][1] += dr[1];
    const da = Vv.map((v) => dot(v, dz));
    const sa = attn.reduce((s, a, i) => s + a * da[i], 0);
    const ds = attn.map((a, i) => a * (da[i] - sa));
    const dq = [0, 0];
    X.forEach((x, i) => {
      const dv = [attn[i] * dz[0], attn[i] * dz[1]];
      const dk = [ds[i] * q[0] / s2, ds[i] * q[1] / s2];
      dq[0] += ds[i] * K[i][0] / s2; dq[1] += ds[i] * K[i][1] / s2;
      for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) {
        g.WV[a][b] += x[a] * dv[b];
        g.WK[a][b] += x[a] * dk[b];
        dX[i][a] += w.WV[a][b] * dv[b] + w.WK[a][b] * dk[b];
      }
    });
    for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) {
      g.WQ[a][b] += X[n][a] * dq[b];
      dX[n][a] += w.WQ[a][b] * dq[b];
    }
    d.ctx.slice(-WINDOW).forEach((t, i) => { g.E[t][0] += dX[i][0]; g.E[t][1] += dX[i][1]; });
  }
  return { loss: L, grad: g };
}

export function randomWeights(seed, scale = 0.8) {
  const r = rng(seed);
  const w = {};
  for (const k of PARAM_NAMES) {
    w[k] = Array.from({ length: SHAPES[k][0] }, () => Array.from({ length: SHAPES[k][1] }, () => (r() * 2 - 1) * scale));
  }
  return w;
}

// Adam: a standard way of turning gradients into small, well-sized steps.
export class Trainer {
  constructor(w, data, lr = 0.05) {
    this.w = cloneWeights(w);
    this.data = data;
    this.lr = lr;
    this.t = 0;
    this.m = zeros();
    this.v = zeros();
    this.history = [];
  }
  step(n = 1) {
    let L = 0;
    for (let s = 0; s < n; s++) {
      const { loss: l, grad } = gradient(this.w, this.data);
      L = l;
      this.t++;
      const b1 = 0.9, b2 = 0.999;
      for (const k of PARAM_NAMES) {
        const W = this.w[k], G = grad[k], M = this.m[k], S = this.v[k];
        for (let i = 0; i < W.length; i++) for (let j = 0; j < W[i].length; j++) {
          M[i][j] = b1 * M[i][j] + (1 - b1) * G[i][j];
          S[i][j] = b2 * S[i][j] + (1 - b2) * G[i][j] * G[i][j];
          const mh = M[i][j] / (1 - b1 ** this.t), sh = S[i][j] / (1 - b2 ** this.t);
          W[i][j] -= this.lr * mh / (Math.sqrt(sh) + 1e-8);
        }
      }
      this.history.push(l);
    }
    return L;
  }
}

export function roundWeights(w, places = 1) {
  const f = 10 ** places;
  const o = cloneWeights(w);
  for (const k of PARAM_NAMES) o[k] = o[k].map((r) => r.map((v) => Math.round(v * f) / f));
  return o;
}

// Does a stream of words obey Tinyspeak? Returns [good sentences, finished sentences].
// With loose = true, the object only has to be an animal (it may repeat the subject).
export function grammarCheck(ids, loose = false) {
  const words = ids.map((i) => VOCAB[i]);
  const sents = [];
  let cur = [];
  for (const w of words) { if (w === ".") { sents.push(cur); cur = []; } else cur.push(w); }
  let ok = 0;
  for (const s of sents) {
    const isN = (x) => x === "cat" || x === "dog";
    if (s.length === 3 && s[0] === "the" && isN(s[1]) && (s[2] === "runs" || s[2] === "sleeps")) ok++;
    else if (s.length === 5 && s[0] === "the" && isN(s[1]) && (s[2] === "chases" || s[2] === "sees") && s[3] === "the" && isN(s[4]) && (loose || s[4] !== s[1])) ok++;
  }
  return [ok, sents.length];
}

// ---------- the fully expanded equation ----------
// Writes P(next word = target) for a context of n words as one formula that uses
// only numbers, the input symbols, +, ×, ÷ and e^( ). Nothing is abbreviated.
export function expandedFormula(w, n, target, { symbolic = true, ids = null } = {}) {
  const f = (v) => (Math.round(v * 10) / 10).toString().replace("-", "−");
  const sub = "₀₁₂₃₄₅₆₇₈₉";
  const x = (i, j) => (symbolic || !ids ? `x${sub[i + 1]}${sub[j + 1]}` : f(w.E[ids[i]][j]));
  const lin = (i, M, col) => `(${x(i, 0)}×${f(M[0][col])} + ${x(i, 1)}×${f(M[1][col])})`;
  const q = (col) => lin(n - 1, w.WQ, col);
  const score = (i) => `(${q(0)}×${lin(i, w.WK, 0)} + ${q(1)}×${lin(i, w.WK, 1)})÷√2`;
  const ex = (i) => `exp(${score(i)})`;
  const denomA = Array.from({ length: n }, (_, i) => ex(i)).join(" + ");
  const zc = (col) => "(" + Array.from({ length: n }, (_, i) => `${ex(i)}×${lin(i, w.WV, col)}`).join(" + ") + `)÷(${denomA})`;
  const rc = (col) => `(${x(n - 1, col)} + ${zc(col)})`;
  const logit = (t) => `(${rc(0)}×${f(w.WO[0][t])} + ${rc(1)}×${f(w.WO[1][t])})`;
  const num = `exp${logit(target)}`;
  const den = Array.from({ length: V }, (_, t) => `exp${logit(t)}`).join(" + ");
  return `P(${VOCAB[target]}) = ${num} ÷ (${den})`;
}
