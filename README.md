# Language modelling — eight animated decks

Independent study companion to **Shamsuddeen Muhammad, From Counting Words to Pretrained Transformers**, supplied lecture dated 24 November 2026. Prepared for Morel KPAVODE.

Open the [course home](https://yemal22.github.io/language-modelling-animated/), then choose one of eight separate decks:

1. [Language models and n-grams](https://yemal22.github.io/language-modelling-animated/part-1/) — 15 slides.
2. [Neural LMs, RNNs, LSTM and GRU](https://yemal22.github.io/language-modelling-animated/part-2/) — 20 slides.
3. [Encoder–decoder models and attention](https://yemal22.github.io/language-modelling-animated/part-3/) — 15 slides.
4. [The Transformer](https://yemal22.github.io/language-modelling-animated/part-4/) — 22 slides.

5. [Contextual embeddings and tokenisation](https://yemal22.github.io/language-modelling-animated/part-5/) — 18 slides.
6. [Pretraining and adaptive pretraining](https://yemal22.github.io/language-modelling-animated/part-6/) — 22 slides.
7. [African language models and evidence](https://yemal22.github.io/language-modelling-animated/part-7/) — 18 slides.
8. [Fine-tuning methods and efficient adaptation](https://yemal22.github.io/language-modelling-animated/part-8/) — 18 slides.

Parts 5–8 continue the supplied second lecture, From Counting Words to Pretrained Transformers (2). The first four decks keep their existing endpoints. The original PDFs are not redistributed.

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

No build dependencies. `content.js` and continuation-content.js hold the 148 slides; `scenes.js` holds editable SVG architecture diagrams; `math.js` holds computations. `engine.js` implements the reversible presentation state. `style.css` uses the white/ink/slate palette, Carlito/Calibri typography and broad spacing of the requested reference.

```sh
node build.mjs
npm test
python3 -m http.server 8000
```

Open `http://localhost:8000/`. ES modules require a web server; opening HTML with `file://` is not supported.

Visual reference: [delamarifer / Human–Machine Perception](https://github.com/delamarifer/human-machine-perception), especially Day 5 plan B, and the earlier EgoFalls deck. The engine and diagrams here are original implementations. Carlito fonts retain their SIL Open Font License in `assets/OFL.txt`.

## Lecture 2 examples and published evidence

The continuation has executable examples for contextual attention visibility, a character convolution, an ELMo layer mixture, character BPE training and unseen-word encoding, selected-token cross-entropy, ULMFiT pooling, data-mixture arithmetic, adapter output, LoRA matrix products, learning rates and parameter counts. Chosen teaching parameters are distinguished from checkpoint outputs.

Four published-result charts have explicit sources:

| Chart | Source | Quantities |
| --- | --- | --- |
| Extra pretraining | Gururangan et al. (2020), Table 5 | Five-seed test means and SD; micro-F1 for CHEMPROT/RCT, macro-F1 otherwise |
| MAFT versus LAFT | Alabi et al. (2022), Table 3 | Average NER test F1 over 13 languages and stored-backbone parameter counts |
| Social-domain DAPT | Belay et al. (2025), final paper Table 3 | Reported average macro-F1 for each dataset; language sets differ |
| Lugha-Llama mixtures | Buzaaba et al. (2025), Table 1 | IrokoBench averages excluding English and French, same 8B base family |

The values are in continuation-math.js. These charts reproduce reported measurements rather than new experiments. Architecture and data facts for InkubaLM and AfriqueLLM cite their papers. The course does not present one global ranking across unrelated benchmarks.

Continuation-scenes.js contains the new editable diagrams. The build writes content-addressed browser assets so a new page loads a consistent module graph. Tests include tokenizer reconstruction, causal visibility, layer-weight normalisation, LoRA merged/parallel equivalence, published-value spot checks and local import resolution.
