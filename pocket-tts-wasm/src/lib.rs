use std::sync::Mutex;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format!($($t)*)))
}

use ptts::flow_lm::{self, FlowLMState};
use ptts::mimi::MimiDecoderState;
use ptts::transformer::{LayerAttentionState, StreamingMHAState, StreamingTransformerState};
use ptts::tts_model::{TTSConfig, TTSModel, TTSState, prepare_text_prompt};
use xn::nn::VB;
use xn::quantized::Q80F32;
use xn::{BackendQ, CPU, CpuDevice, Tensor, TypedTensor, Unquantized};

/// Tokenizer that returns pre-set token IDs (set from JS before each generation).
struct PresetTokenizer {
    tokens: Mutex<Vec<u32>>,
}

impl PresetTokenizer {
    fn new() -> Self {
        Self { tokens: Mutex::new(Vec::new()) }
    }

    fn set_tokens(&self, tokens: Vec<u32>) {
        *self.tokens.lock().unwrap() = tokens;
    }
}

impl ptts::Tokenizer for PresetTokenizer {
    fn encode(&self, _text: &str) -> Vec<u32> {
        self.tokens.lock().unwrap().clone()
    }

    fn decode(&self, _tokens: &[u32]) -> String {
        String::new()
    }
}

/// Wrapper to allow sharing a PresetTokenizer via Arc while implementing the Tokenizer trait.
struct SharedTokenizer(std::sync::Arc<PresetTokenizer>);

impl ptts::Tokenizer for SharedTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.0.encode(text)
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.0.decode(tokens)
    }
}

struct WasmRng {
    inner: Box<rand::rngs::StdRng>,
    distr: rand_distr::Normal<f32>,
}

impl WasmRng {
    fn new(temperature: f32) -> Self {
        use rand::SeedableRng;
        let std = temperature.sqrt();
        let distr = rand_distr::Normal::new(0f32, std).unwrap();
        let rng = rand::rngs::StdRng::seed_from_u64(42);
        Self { inner: Box::new(rng), distr }
    }
}

impl flow_lm::Rng for WasmRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

fn remap_key(name: &str) -> Option<String> {
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
        || name.contains("learnt_padding")
    {
        return None;
    }

    let mut name = name.to_string();
    name = name.replace(
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    );
    name = name.replace(
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    );
    name = name.replace("flow_lm.backbone.", "flow_lm.transformer.");
    name = name.replace("flow_lm.flow.", "flow_lm.flow_net.");
    name = name.replace("mimi.model.", "mimi.");

    Some(name)
}

/// Underlying type-erased transformer state, shared across all supported quantizations
/// (all of them use `T = f32, B = CpuDevice`).
type RawState = StreamingTransformerState<f32, CpuDevice>;

fn wrap_state<Q: BackendQ<T = f32, B = CpuDevice>>(raw: RawState) -> TTSState<Q> {
    TTSState { flow_lm_state: FlowLMState { transformer_state: raw } }
}

/// Creates a new transformer state with a larger seq_budget, copying the used KV entries
/// from a cached state (which was allocated with a smaller budget).
fn resize_state(cached: &RawState, new_seq_budget: usize) -> xn::Result<RawState> {
    console_log!("[resize_state] resizing to seq_budget={new_seq_budget}");
    let mut new_layer_states = Vec::new();
    for layer_state in cached.layer_states.iter() {
        match layer_state {
            LayerAttentionState::FlowLm(mha_state) => {
                let current_end = mha_state.current_end;
                let b = mha_state.k_cache.dim(0usize)?;
                let h = mha_state.k_cache.dim(2usize)?;
                let d = mha_state.k_cache.dim(3usize)?;
                let new_k = Tensor::zeros((b, new_seq_budget, h, d), &CPU)?;
                let new_v = Tensor::zeros((b, new_seq_budget, h, d), &CPU)?;
                if current_end > 0 {
                    let k_used = mha_state.k_cache.narrow(1, 0..current_end)?.contiguous()?;
                    let v_used = mha_state.v_cache.narrow(1, 0..current_end)?.contiguous()?;
                    new_k.slice_set(&k_used, 1usize, 0)?;
                    new_v.slice_set(&v_used, 1usize, 0)?;
                }
                new_layer_states.push(LayerAttentionState::FlowLm(StreamingMHAState {
                    k_cache: new_k,
                    v_cache: new_v,
                    current_end,
                }));
            }
            other => {
                new_layer_states.push(other.clone());
            }
        }
    }
    Ok(StreamingTransformerState { layer_states: new_layer_states })
}

/// Quantization variants exposed to JS.
#[derive(Clone, Copy, Debug)]
enum Quant {
    F32,
    Q8,
}

impl Quant {
    fn parse(s: &str) -> xn::Result<Self> {
        match s {
            "f32" => Ok(Self::F32),
            "q8" => Ok(Self::Q8),
            other => xn::bail!("unsupported quantization '{other}'"),
        }
    }
}

enum ModelInner {
    F32(TTSModel<Unquantized<f32, CpuDevice>>),
    Q8(TTSModel<Q80F32>),
}

enum StateInner {
    F32(TTSState<Unquantized<f32, CpuDevice>>),
    Q8(TTSState<Q80F32>),
}

/// Dispatch a block of code over the currently active (model, state) pair. Within the
/// block, `$m` is `&TTSModel<Q>` and `$s` is `&mut TTSState<Q>` for the matching `Q`.
macro_rules! dispatch {
    ($inner:expr, $state:expr, |$m:ident, $s:ident| $body:block) => {
        match ($inner, $state) {
            (ModelInner::F32($m), StateInner::F32($s)) => $body,
            (ModelInner::Q8($m), StateInner::Q8($s)) => $body,
            _ => xn::bail!("model/state quantization mismatch"),
        }
    };
}

struct GenState {
    tts_state: StateInner,
    mimi_state: MimiDecoderState<f32, CpuDevice>,
    prev_latent: Tensor<f32, CpuDevice>,
    rng: WasmRng,
    max_frames: usize,
    frames_after_eos: usize,
    eos_countdown: Option<usize>,
    step: usize,
}

#[wasm_bindgen]
pub struct Model {
    inner: ModelInner,
    tokenizer: std::sync::Arc<PresetTokenizer>,
    cfg: TTSConfig,
    gen_state: Option<GenState>,
    voice_states: Vec<RawState>,
}

impl Model {
    pub fn new_(model_weights: &[u8], quant: &str) -> xn::Result<Model> {
        let quant = Quant::parse(quant)?;
        console_log!("[new] loading model with quant={quant:?}");
        let cfg = TTSConfig::v202601(0.7);

        let is_gguf = model_weights.len() >= 4 && &model_weights[..4] == b"GGUF";
        let vb = if is_gguf {
            console_log!("[new] detected gguf format");
            let cursor = std::io::Cursor::new(model_weights.to_vec());
            VB::load_gguf_with_key_map(cursor, CPU, remap_key)?
        } else {
            console_log!("[new] detected safetensors format");
            VB::from_bytes_with_key_map(vec![model_weights.to_vec()], CPU, remap_key)?
        };
        let root = vb.root();
        let tokenizer = std::sync::Arc::new(PresetTokenizer::new());
        let tokenizer_box: Box<dyn ptts::Tokenizer + Send + Sync> =
            Box::new(SharedTokenizer(std::sync::Arc::clone(&tokenizer)));

        let inner = match quant {
            Quant::F32 => ModelInner::F32(TTSModel::<Unquantized<f32, CpuDevice>>::load(
                &root,
                tokenizer_box,
                &cfg,
            )?),
            Quant::Q8 => ModelInner::Q8(TTSModel::<Q80F32>::load(&root, tokenizer_box, &cfg)?),
        };

        Ok(Model { inner, tokenizer, cfg, gen_state: None, voice_states: Vec::new() })
    }

    /// Load a pre-computed KV cache state from a safetensors buffer.
    /// The file contains `transformer.layers.{i}.self_attn/cache` (shape [2, 1, seq, 16, 64])
    /// and `transformer.layers.{i}.self_attn/current_end` (single f32) for each layer.
    /// Returns the voice index for use with start_generation.
    pub fn add_voice_(&mut self, state_bytes: &[u8]) -> xn::Result<usize> {
        console_log!("[add_voice] loading safetensors, {} bytes", state_bytes.len());
        let tensors = xn::safetensors::load_from_buffer(state_bytes, &CPU)?;
        let num_layers = 6;
        let mut layer_states = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let cache_name = format!("transformer.layers.{i}.self_attn/cache");

            let cache = match tensors.get(&cache_name) {
                Some(TypedTensor::F32(t)) => t,
                _ => xn::bail!("expected f32 tensor: {cache_name}"),
            };

            // cache shape: (2, batch, seq_len, num_heads, head_dim)
            let (two, batch, seq_len, num_heads, head_dim) = cache.dims5()?;
            if two != 2 {
                xn::bail!("expected first dim of size 2 in cache tensor");
            }
            let k_cache = cache
                .narrow(0, 0..1)?
                .contiguous()?
                .reshape((batch, seq_len, num_heads, head_dim))?;
            let v_cache = cache
                .narrow(0, 1..2)?
                .contiguous()?
                .reshape((batch, seq_len, num_heads, head_dim))?;
            layer_states.push(LayerAttentionState::FlowLm(StreamingMHAState {
                k_cache,
                v_cache,
                current_end: seq_len,
            }));
        }

        let raw = StreamingTransformerState { layer_states };
        let idx = self.voice_states.len();
        self.voice_states.push(raw);
        Ok(idx)
    }

    pub fn start_generation_(
        &mut self,
        voice_index: usize,
        token_ids: &[u32],
        frames_after_eos: usize,
        temperature: f32,
    ) -> xn::Result<()> {
        console_log!(
            "[start_generation] voice_index={} num_tokens={} frames_after_eos={} temperature={}",
            voice_index,
            token_ids.len(),
            frames_after_eos,
            temperature
        );
        self.tokenizer.set_tokens(token_ids.to_vec());

        let num_tokens = token_ids.len();
        let max_frames = ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
        let seq_budget = num_tokens + 512 + max_frames;

        // Resize the cached voice state (small budget) into a full-sized state.
        if voice_index >= self.voice_states.len() {
            xn::bail!("invalid voice index: {voice_index}");
        }
        let cached = &self.voice_states[voice_index];
        let raw = resize_state(cached, seq_budget)?;

        let mut tts_state = match &self.inner {
            ModelInner::F32(_) => StateInner::F32(wrap_state(raw)),
            ModelInner::Q8(_) => StateInner::Q8(wrap_state(raw)),
        };

        console_log!("[start_generation] running prompt_text...");
        let mimi_state = dispatch!(&self.inner, &mut tts_state, |m, s| {
            m.prompt_text(s, token_ids)?;
            m.init_mimi_state(1, 250)?
        });
        console_log!("[start_generation] prompt_text done, starting generation loop");

        let rng = WasmRng::new(temperature);

        let ldim = self.cfg.flow_lm.ldim;
        let nan_data: Vec<f32> = vec![f32::NAN; ldim];
        let prev_latent = Tensor::from_vec(nan_data, (1, 1, ldim), &CPU)?;

        self.gen_state = Some(GenState {
            tts_state,
            mimi_state,
            prev_latent,
            rng,
            max_frames,
            frames_after_eos,
            eos_countdown: None,
            step: 0,
        });
        Ok(())
    }

    pub fn generation_step_(&mut self) -> xn::Result<Option<js_sys::Float32Array>> {
        let mut state = match self.gen_state.take() {
            Some(s) => s,
            None => return Ok(None),
        };

        if state.step >= state.max_frames {
            return Ok(None);
        }

        let (next_latent, audio_chunk, is_eos) =
            dispatch!(&self.inner, &mut state.tts_state, |m, s| {
                let (next_latent, is_eos) =
                    m.generate_step(s, &state.prev_latent, &mut state.rng)?;
                let audio_chunk = m.decode_latent(&next_latent, &mut state.mimi_state)?;
                (next_latent, audio_chunk, is_eos)
            });

        if is_eos && state.eos_countdown.is_none() {
            state.eos_countdown = Some(state.frames_after_eos);
        }

        let done = if let Some(ref mut countdown) = state.eos_countdown {
            if *countdown == 0 {
                true
            } else {
                *countdown -= 1;
                false
            }
        } else {
            false
        };

        state.prev_latent = next_latent;
        state.step += 1;

        let audio = audio_chunk.narrow(0, ..1)?.contiguous()?;
        let pcm = audio.to_vec()?;
        let result = js_sys::Float32Array::from(pcm.as_slice());

        if !done {
            self.gen_state = Some(state);
        }

        Ok(Some(result))
    }
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new(model_weights: &[u8], quant: &str) -> Result<Model, JsError> {
        Self::new_(model_weights, quant).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn add_voice(&mut self, voice_weights: &[u8]) -> Result<usize, JsError> {
        self.add_voice_(voice_weights).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Prepare text for generation: capitalize, add punctuation, pad short text.
    /// Returns [processed_text, frames_after_eos] as a JS array.
    pub fn prepare_text(&self, text: &str) -> js_sys::Array {
        let (processed, frames_after_eos) = prepare_text_prompt(text);
        let arr = js_sys::Array::new();
        arr.push(&JsValue::from_str(&processed));
        arr.push(&JsValue::from_f64(frames_after_eos as f64));
        arr
    }

    pub fn start_generation(
        &mut self,
        voice_index: usize,
        token_ids: &[u32],
        frames_after_eos: usize,
        temperature: f32,
    ) -> Result<(), JsError> {
        self.start_generation_(voice_index, token_ids, frames_after_eos, temperature)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn generation_step(&mut self) -> Result<Option<js_sys::Float32Array>, JsError> {
        self.generation_step_().map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn sample_rate(&self) -> usize {
        match &self.inner {
            ModelInner::F32(m) => m.sample_rate(),
            ModelInner::Q8(m) => m.sample_rate(),
        }
    }
}

/// CPU SIMD features the wasm module was compiled with. The relevant one
/// for browser builds is `simd128`; `avx`/`neon`/`f16c` are reported for
/// completeness so it's clear which native-target builds enabled them.
#[wasm_bindgen]
pub fn cpu_features() -> js_sys::Object {
    let obj = js_sys::Object::new();
    let set = |k: &str, v: bool| {
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str(k), &JsValue::from_bool(v));
    };
    set("avx", xn::with_avx());
    set("neon", xn::with_neon());
    set("simd128", xn::with_simd128());
    set("f16c", xn::with_f16c());
    obj
}
