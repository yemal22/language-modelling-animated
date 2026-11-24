# Language modelling — four animated decks

Independent study companion to **Shamsuddeen Muhammad, From Counting Words to Pretrained Transformers**, supplied lecture dated 24 November 2026. Prepared for Morel KPAVODE.

Open the [course home](https://yemal22.github.io/language-modelling-animated/), then choose one of four separate decks:

1. [Language models and n-grams](https://yemal22.github.io/language-modelling-animated/part-1/) — 15 slides.
2. [Neural LMs, RNNs, LSTM and GRU](https://yemal22.github.io/language-modelling-animated/part-2/) — 20 slides.
3. [Encoder–decoder models and attention](https://yemal22.github.io/language-modelling-animated/part-3/) — 15 slides.
4. [The Transformer](https://yemal22.github.io/language-modelling-animated/part-4/) — 22 slides.

## Interaction

- Right arrow / Space / click: reveal or advance. Left arrow: reverse.
- Page Down / Page Up: change slide. Home / End: beginning / end.
- O: slide overview. F: fullscreen. N: detailed speaker explanations and sources.
- Sliders and selectors change computed examples without advancing the deck.
- `#12/3` selects slide 12, build 3; `#12` opens it fully built.
- White lead-ins match the requested reference; `?blank=0` disables them.

## Architecture walkthroughs

The decks include RNN input/recurrent/output paths, numeric unrolling, gradient products, LSTM cell/hidden states and all gates, a GRU with an explicit update convention, encoder–decoder feedback, teacher forcing, attention scores/softmax/weighted sum, Q/K/V projections, causal masks, multi-head concatenation, residuals and normalisation, position-wise FFNs, cross-attention, and original Transformer encoder/decoder stacks.

All displayed toy values come from functions in `math.js`. They are chosen teaching parameters, not measurements from a pretrained model. The complete training corpus used by the bigram demo is printed on its slide. The course PDF is credited, not redistributed. Source papers are cited per slide and on the home page. Original course simplifications are clarified where needed: fixed context versus fixed sentence length, gate conventions, gradient guarantees, alignment versus explanation, and parallel training versus sequential generation.

## Edit and verify

No build dependencies. `content.js` holds the 72 slides; `scenes.js` holds editable SVG architecture diagrams; `math.js` holds computations. `engine.js` implements the reversible presentation state. `style.css` uses the white/ink/slate palette, Carlito/Calibri typography and broad spacing of the requested reference.

```sh
node build.mjs
npm test
python3 -m http.server 8000
```

Open `http://localhost:8000/`. ES modules require a web server; opening HTML with `file://` is not supported.

Visual reference: [delamarifer / Human–Machine Perception](https://github.com/delamarifer/human-machine-perception), especially Day 5 plan B, and the earlier EgoFalls deck. The engine and diagrams here are original implementations. Carlito fonts retain their SIL Open Font License in `assets/OFL.txt`.
