import init, { Decoder } from './wasm_pocket_tts.js';

const HF_BASE = 'https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/main';
const MODEL_URL = `${HF_BASE}/tts_b6369a24.safetensors`;

let decoder = null;
let port = null;
let decodeStart = 0;
let decodeSteps = 0;

// Start WASM init immediately so it's ready when gen_worker signals us.
const wasmReady = init();

function post(type, data = {}, transferables = []) {
  self.postMessage({ type, ...data }, transferables);
}

async function onPortMessage(e) {
  const { type, ...data } = e.data;

  if (type === 'init') {
    await wasmReady;
    // Fetch model weights from browser cache (gen_worker already downloaded them).
    const resp = await fetch(MODEL_URL);
    const modelBytes = new Uint8Array(await resp.arrayBuffer());
    decoder = new Decoder(modelBytes);
    decoder.init_decoder();
    post('decoder_ready');
    return;
  }

  if (type === 'gen_start') {
    decoder.init_decoder();
    decodeStart = performance.now();
    decodeSteps = 0;
    return;
  }

  if (type === 'decode') {
    const audio = decoder.decode_latent_step(data.latent);
    post('chunk', { data: audio, step: data.step }, [audio.buffer]);
    decodeSteps++;
    return;
  }

  if (type === 'gen_done') {
    const decodeElapsed = performance.now() - decodeStart;
    console.log(`[decode_worker] ${decodeSteps} decode steps in ${decodeElapsed.toFixed(0)}ms (${(decodeElapsed / decodeSteps).toFixed(1)}ms/step)`);
    post('done');
    return;
  }
}

self.onmessage = (e) => {
  if (e.data.type === 'init_port') {
    port = e.ports[0];
    port.onmessage = onPortMessage;
    return;
  }
};
