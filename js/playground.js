// Chapter 10: run the equation on anything.
import * as M from "./model.js";
import { h, fmt, fmtTex, pct, tex, matTex, vecTex, tile, bars, store } from "./ui.js";

const { VOCAB, INDEX: I } = M;

// Weights used by the playground. The training demo can send new ones here.
export const playWeights = store(M.cloneWeights(M.TRAINED));
export const playSource = store("trained");

export function playground(root) {
  let ids = [I.the, I.dog, I.chases, I.the];
  const keys = h("div", { class: "tiles" });
  VOCAB.forEach((w, i) => keys.append(h("button", { class: "tile", onclick: () => { ids.push(i); draw(); } }, w)));
  const strip = h("div", { class: "tok-strip", "aria-live": "polite" });
  const winNote = h("p", { class: "w-help", style: { margin: "6px 0 0" } });
  const barBox = h("div");
  const status = h("p", { class: "status" });
  const picked = h("p", { class: "status", "aria-live": "polite", style: { marginTop: "10px", fontWeight: "700" } });
  const src = h("p", { class: "status" });
  const bLikely = h("button", { class: "primary" }, "Add the likeliest word");
  const bRandom = h("button", {}, "Draw a word (weighted by probability)");
  const bWrite = h("button", {}, "Write 20 words");
  const bBack = h("button", { class: "ghost" }, "Remove last word");
  const bClear = h("button", { class: "ghost" }, "Clear");
  const traceBox = h("div", { class: "trace" });
  const traceDet = h("details", { class: "more" }, h("summary", {}, "Show the calculation"), traceBox);
  const editBox = h("div");
  const editDet = h("details", { class: "more" }, h("summary", {}, "Edit the 44 numbers"), editBox);

  root.append(
    h("p", { class: "w-title" }, "The model's keyboard"),
    keys,
    h("div", { class: "w-row" }, bBack, bClear),
    strip, winNote,
    h("div", { class: "w-grid two", style: { marginTop: "16px" } },
      h("div", {}, h("p", { class: "w-title" }, "What comes next?"), h("p", { class: "w-help" }, "Click a word to add it."), barBox),
      h("div", {}, h("div", { class: "w-row", style: { flexDirection: "column", alignItems: "stretch" } }, bLikely, bRandom, bWrite), h("p", { class: "w-help", style: { margin: "4px 0 0" } }, "Drawing works like a raffle: each word holds tickets in proportion to its probability, so likely words usually win, but any word with tickets can."), picked, status, src)),
    traceDet, editDet);

  const W = () => playWeights.get();

  const draw = () => {
    if (!ids.length) ids = [I.the];
    if (ids.length > 80) ids = ids.slice(-80);
    strip.innerHTML = "";
    ids.forEach((t, k) => strip.append(tile(t, k >= ids.length - 3 ? "window-mark" : "")));
    const f = M.forward(W(), ids);
    winNote.textContent = `The model reads the outlined words: “${f.ctx.map((t) => VOCAB[t]).join(" ")}”.`;
    const top = M.argmax(f.probs);
    bars(barBox, VOCAB.map((w, t) => ({ label: h("span", { class: "tile" + (t === top ? " on" : "") }, w), p: f.probs[t], t })), { onPick: (it) => { ids.push(it.t); draw(); } });
    const [ok, tot] = M.grammarCheck(ids);
    status.innerHTML = "";
    if (tot) status.append(`Grammar check: ${ok} of ${tot} finished sentence${tot === 1 ? "" : "s"} follow Tinyspeak's rules.`);
    if (traceDet.open) drawTrace(f);
  };

  const drawTrace = (f) => {
    const w = W();
    const n = f.ctx.length - 1;
    traceBox.innerHTML = "";
    const step = (title, src) => {
      const b = h("div", { class: "scroll-x" });
      tex(b, src, true);
      traceBox.append(h("div", { class: "st" }, h("h4", {}, title), b));
    };
    const rowsTex = (Mx, d = 2) => `\\begin{pmatrix} ${Mx.map((r) => r.map((v) => fmtTex(v, d)).join(" & ")).join(" \\\\ ")} \\end{pmatrix}`;
    const labs = f.ctx.map((t) => `\\text{${VOCAB[t]}}`).join(" \\\\ ");
    step("1. The words as numbers", `\\Xm = \\htmlClass{c-word}{${rowsTex(f.X, 1)}} \\begin{matrix} ${labs} \\end{matrix}`);
    step("2. The last word's query", `\\qv = \\xn \\WQ = \\htmlClass{c-word}{${vecTex(f.X[n], 1)}} \\htmlClass{c-query}{${matTex(w.WQ)}} = \\htmlClass{c-query}{${vecTex(f.q)}}`);
    step("3. Every word's key", `\\Xm\\WK = \\htmlClass{c-key}{${rowsTex(f.K)}}`);
    step("4. Scores, scaled", `\\qv\\,(\\Xm\\WK)^{\\T} / \\rt = ${vecTex(f.scores)} / 1.414 = ${vecTex(f.scaled)}`);
    step("5. Attention weights", `\\av = \\smA(${f.scaled.map((v) => fmtTex(v)).join(",\\ ")}) = \\htmlClass{c-attn}{${vecTex(f.attn, 3)}}`);
    step("6. Values, blended", `\\Xm\\WV = \\htmlClass{c-value}{${rowsTex(f.V)}}, \\qquad \\zv = \\av\\,\\Xm\\WV = \\htmlClass{c-value}{${vecTex(f.z)}}`);
    step("7. Add the last word back", `r = \\xn + \\zv = ${vecTex(f.X[n], 1)} + ${vecTex(f.z)} = ${vecTex(f.r)}`);
    step("8. A score for every word", `r\\,\\WO = \\htmlClass{c-out}{${vecTex(f.logits, 1)}}`);
    step("9. Probabilities", `\\smO(r\\,\\WO) = ${vecTex(f.probs, 3)} \\quad \\text{for } (${VOCAB.map((x) => `\\text{${x}}`).join(",")})`);
  };
  traceDet.addEventListener("toggle", () => { if (traceDet.open) drawTrace(M.forward(W(), ids)); });

  // Say what was picked and how likely it was, so the randomness is visible.
  const q = (t) => `“${VOCAB[t]}”`;
  const fine = (p) => (p > 0.99 && p < 0.9995 ? `${(p * 100).toFixed(1)}%` : pct(p));
  bLikely.addEventListener("click", () => {
    const pr = M.forward(W(), ids).probs, t = M.argmax(pr);
    ids.push(t); draw();
    picked.textContent = `Added ${q(t)}, the likeliest word (${fine(pr[t])}).`;
  });
  bRandom.addEventListener("click", () => {
    const pr = M.forward(W(), ids).probs, t = M.sampleFrom(pr, Math.random), top = M.argmax(pr);
    ids.push(t); draw();
    picked.textContent = t === top
      ? (pr[t] > 0.95 ? `Drew ${q(t)}, which held ${fine(pr[t])} of the tickets. The model was nearly certain, so the draw almost always lands here.` : `Drew ${q(t)}, which held ${fine(pr[t])} of the tickets: the favorite won this time. Remove it and draw again from the same place to see other words win sometimes.`)
      : `Drew ${q(t)}, which held only ${fine(pr[t])} of the tickets: not the favorite ${q(top)} (${fine(pr[top])}), but it had a chance and won this time.`;
  });
  bWrite.addEventListener("click", () => { ids = M.generate(W(), ids, 20, "sample", Math.random).seq; draw(); picked.textContent = "Added 20 words, each drawn the same weighted way."; });
  bBack.addEventListener("click", () => { ids.pop(); draw(); });
  bClear.addEventListener("click", () => { ids = [I.the]; draw(); });

  // ---------- weight editor ----------
  const NAMES = { E: "E (word numbers)", WQ: "W_Q (query)", WK: "W_K (key)", WV: "W_V (value)", WO: "W_O (output)" };
  const CLS = { E: "c-word", WQ: "c-query", WK: "c-key", WV: "c-value", WO: "c-out" };
  const resetB = h("button", { class: "ghost" }, "Reset to the trained numbers");
  const grid = h("div", { class: "weditor" });
  editBox.append(h("p", { class: "w-help" }, "Change any number and everything above updates at once. Changed numbers are outlined."), grid, h("div", { class: "w-row" }, resetB));
  const inputs = [];
  const buildEditor = () => {
    grid.innerHTML = ""; inputs.length = 0;
    const w = W();
    for (const k of M.PARAM_NAMES) {
      const tbl = h("table");
      const [r, c] = M.SHAPES[k];
      if (k === "WO") tbl.append(h("tr", {}, VOCAB.map((x) => h("td", { class: "lab", style: { textAlign: "center" } }, x))));
      for (let i = 0; i < r; i++) {
        const tr = h("tr");
        if (k === "E") tr.append(h("td", { class: "lab" }, VOCAB[i]));
        for (let j = 0; j < c; j++) {
          const inp = h("input", { type: "number", step: 0.1, value: w[k][i][j], "aria-label": `${k} row ${i + 1} column ${j + 1}` });
          inp.addEventListener("input", () => {
            const v = parseFloat(inp.value);
            if (!isFinite(v)) return;
            const nw = M.cloneWeights(W()); nw[k][i][j] = v; playWeights.set(nw); playSource.set("edited");
            inp.classList.toggle("changed", Math.abs(v - M.TRAINED[k][i][j]) > 1e-9);
          });
          inputs.push({ inp, k, i, j });
          tr.append(h("td", {}, inp));
        }
        tbl.append(tr);
      }
      grid.append(h("div", { class: "grp" }, h("h5", { class: CLS[k], "data-sym": k }, NAMES[k]), tbl));
    }
    inputs.forEach(({ inp, k, i, j }) => inp.classList.toggle("changed", Math.abs(W()[k][i][j] - M.TRAINED[k][i][j]) > 1e-9));
  };
  resetB.addEventListener("click", () => { playWeights.set(M.cloneWeights(M.TRAINED)); playSource.set("trained"); buildEditor(); });
  playSource.sub((sname) => {
    src.textContent = sname === "trained" ? "Using the trained 44 numbers from chapter 9." : sname === "edited" ? "Using your edited numbers." : "Using numbers from the training demo in chapter 11.";
  });
  playWeights.sub(() => draw());
  playSource.sub((sname) => { if (sname === "fromTraining") buildEditor(); });
  buildEditor();
  draw();
}
