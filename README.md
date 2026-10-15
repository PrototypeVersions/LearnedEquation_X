# Finally, understand exactly how AI works

*One equation that writes sentences, and how to read it.*

A complete tiny language model, written as a single equation with 44 numbers, and a step-by-step page that teaches a newcomer to read every symbol in it.

**Live site:** https://prototypeversions.github.io/LearnedEquation_X/

Created by [Blake A. Myers](https://sites.google.com/view/blake-a-myers/home).

## What's on the page

1. Fitting a model to data (least squares, with draggable squares)
2. How to read the symbols
3. Words become numbers (embeddings on a plane)
4. Multiplying by a matrix (step-by-step and as a transformation of the plane)
5. The dot product
6. Softmax
7. Attention, with every number shown
8. Making the prediction
9. The whole equation: compact, with all 44 numbers, and fully expanded (~12,000 characters)
10. A playground: type any words, generate sentences, edit the numbers
11. Training live in the browser, from random numbers
12. How training works, step by step: walking downhill on the line, one real training step with every slope, backpropagation as "predicted minus actual" passed backwards, plain steps vs. Adam
13. From 44 numbers to billions
14. Is the equation the best description? Rules vs. equation, the 553 inputs the rules never mention, equations with identical behavior, and three possible answers

## The model

- Vocabulary: `the cat dog runs sleeps chases sees .`
- Each word is 2 numbers; the model reads the last 3 words.
- One attention head plus a residual connection:

```
P(next word | x₁…xₙ) = softmax( ( xₙ + softmax( xₙ W_Q (X W_K)ᵀ / √2 ) X W_V ) W_O )
```

- Trained on 39,845 sentences of "Tinyspeak", a made-up language with three rules. With its numbers rounded to one decimal place, it writes grammatical sentences such as *the dog chases the cat. the cat sleeps.*

All numbers on the page are computed live from [`js/model.js`](js/model.js).

## Files

- `index.html` — the page and its text
- `css/site.css` — styles (light and dark)
- `js/model.js` — the model, training text, training (with exact gradients), generation, expanded formula
- `js/lessons.js`, `js/playground.js`, `js/training.js`, `js/trainsteps.js`, `js/deep.js` — the interactive pieces
- `vendor/katex/` — [KaTeX](https://katex.org) for typesetting math (MIT license)

No build step: it's static HTML, CSS and JavaScript modules, served directly by GitHub Pages.
