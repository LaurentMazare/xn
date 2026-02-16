# wasm-pocket-tts

WebAssembly build of [Pocket TTS](../pocket-tts/) — run text-to-speech directly in the browser.

Try it online [here](https://laurentmazare.github.io/pocket-tts).

## Prerequisites

Install [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/):

```bash
cargo install wasm-pack
```

## Build

From the `wasm-pocket-tts/` directory:

```bash
make build
```

This runs `wasm-pack build` and copies `index.html` into `pkg/`.

## Run

Serve the `pkg/` directory with any HTTP server, for example:

```bash
cd wasm-pocket-tts/pkg
python3 -m http.server 8080
```

Then open http://localhost:8080 in your browser. The page will download the
model weights from HuggingFace on first use (~240 MB) and cache them for subsequent generations.

## Todo

- Handle long prompts, see `split_into_best_sentences` in
  [tts_model.py](https://github.com/kyutai-labs/pocket-tts/blob/aca7dc8db698e5885fe9dd4850bacfa757b429b1/pocket_tts/models/tts_model.py#L893).
- Voice cloning.
