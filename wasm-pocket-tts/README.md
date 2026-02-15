# wasm-pocket-tts

WebAssembly build of [Pocket TTS](../pocket-tts/) — run text-to-speech directly in the browser.

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
