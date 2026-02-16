use std::sync::Mutex;
use wasm_bindgen::prelude::*;

use pocket_tts::flow_lm::{self, FlowLMState};
use pocket_tts::mimi::MimiState;
use pocket_tts::mimi_transformer::{LayerAttentionState, StreamingTransformerState};
use pocket_tts::transformer::StreamingMHAState;
use pocket_tts::tts_model::{TTSConfig, TTSModel, TTSState, prepare_text_prompt};
use xn::nn::VB;
use xn::{CPU, CpuDevice, Tensor};

/// Tokenizer that returns pre-set token IDs (set from JS before each generation).
struct PresetTokenizer {
    tokens: Mutex<Vec<u32>>,
}

impl PresetTokenizer {
    fn new() -> Self {
        Self {
            tokens: Mutex::new(Vec::new()),
        }
    }

    fn set_tokens(&self, tokens: Vec<u32>) {
        *self.tokens.lock().unwrap() = tokens;
    }
}

impl pocket_tts::Tokenizer for PresetTokenizer {
    fn encode(&self, _text: &str) -> Vec<u32> {
        self.tokens.lock().unwrap().clone()
    }
}

/// Wrapper to allow sharing a PresetTokenizer via Arc while implementing the Tokenizer trait.
struct SharedTokenizer(std::sync::Arc<PresetTokenizer>);

impl pocket_tts::Tokenizer for SharedTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.0.encode(text)
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
        Self {
            inner: Box::new(rng),
            distr,
        }
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

/// Budget for the small KV cache used when caching voice states.
/// Only needs to fit the audio prompt tokens (~50-100), not the full generation.
const VOICE_CACHE_SEQ_BUDGET: usize = 512;

/// Creates a new TTSState with a larger seq_budget, copying the used KV entries
/// from a cached state (which was allocated with a smaller budget).
fn resize_tts_state(
    cached: &TTSState<f32, CpuDevice>,
    new_seq_budget: usize,
) -> xn::Result<TTSState<f32, CpuDevice>> {
    let mut new_layer_states = Vec::new();
    for layer_state in &cached.flow_lm_state.transformer_state.layer_states {
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
    Ok(TTSState {
        flow_lm_state: FlowLMState {
            transformer_state: StreamingTransformerState {
                layer_states: new_layer_states,
            },
        },
    })
}

struct GenState {
    tts_state: TTSState<f32, CpuDevice>,
    mimi_state: MimiState<f32, CpuDevice>,
    prev_latent: Tensor<f32, CpuDevice>,
    rng: WasmRng,
    max_frames: usize,
    frames_after_eos: usize,
    eos_countdown: Option<usize>,
    step: usize,
}

#[wasm_bindgen]
pub struct Model {
    inner: TTSModel<f32, CpuDevice>,
    tokenizer: std::sync::Arc<PresetTokenizer>,
    cfg: TTSConfig,
    gen_state: Option<GenState>,
    voice_states: Vec<TTSState<f32, CpuDevice>>,
}

impl Model {
    pub fn new_(model_weights: &[u8]) -> xn::Result<Model> {
        let cfg = TTSConfig::v202601(0.7);

        let vb = VB::from_bytes_with_key_map(vec![model_weights.to_vec()], CPU, remap_key)?;
        let root = vb.root();
        let tokenizer = std::sync::Arc::new(PresetTokenizer::new());
        let tokenizer_box: Box<dyn pocket_tts::Tokenizer + Send + Sync> =
            Box::new(SharedTokenizer(std::sync::Arc::clone(&tokenizer)));
        let model: TTSModel<f32, CpuDevice> = TTSModel::load(&root, tokenizer_box, &cfg)?;

        Ok(Model {
            inner: model,
            tokenizer,
            cfg,
            gen_state: None,
            voice_states: Vec::new(),
        })
    }

    fn load_voice_emb(voice_weights: &[u8]) -> xn::Result<Tensor<f32, CpuDevice>> {
        use xn::error::Context;
        let voice_vb = VB::from_bytes(vec![voice_weights.to_vec()], CPU)?;
        let voice_names = voice_vb.tensor_names();
        let voice_key = voice_names
            .first()
            .context("no tensors found in voice embedding file")?
            .to_string();
        let voice_td = voice_vb
            .get_tensor(&voice_key)
            .context("voice tensor not found")?;
        let voice_shape = voice_td.shape.clone();
        let voice_dims = voice_shape.dims().to_vec();

        let voice_emb: Tensor<f32, CpuDevice> = voice_vb.tensor(&voice_key, voice_shape)?;
        let voice_emb = if voice_dims.len() == 2 {
            voice_emb.reshape((1, voice_dims[0], voice_dims[1]))?
        } else {
            voice_emb
        };
        Ok(voice_emb)
    }

    /// Load a voice embedding, run prompt_audio, and cache the resulting state.
    /// Returns the voice index for use with start_generation.
    pub fn add_voice_(&mut self, voice_weights: &[u8]) -> xn::Result<usize> {
        let voice_emb = Self::load_voice_emb(voice_weights)?;
        let mut tts_state = self.inner.init_flow_lm_state(1, VOICE_CACHE_SEQ_BUDGET)?;
        self.inner.prompt_audio(&mut tts_state, &voice_emb)?;
        let idx = self.voice_states.len();
        self.voice_states.push(tts_state);
        Ok(idx)
    }

    pub fn start_generation_(
        &mut self,
        voice_index: usize,
        token_ids: &[u32],
        frames_after_eos: usize,
        temperature: f32,
    ) -> xn::Result<()> {
        self.tokenizer.set_tokens(token_ids.to_vec());

        let num_tokens = token_ids.len();
        let max_frames = ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
        let seq_budget = num_tokens + 512 + max_frames;

        // Resize the cached voice state (small budget) into a full-sized state.
        let cached = &self.voice_states[voice_index];
        let mut tts_state = resize_tts_state(cached, seq_budget)?;
        let mimi_state = self.inner.init_mimi_state(1, 250)?;

        self.inner.prompt_text(&mut tts_state, token_ids)?;

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

        let (next_latent, is_eos) =
            self.inner
                .generate_step(&mut state.tts_state, &state.prev_latent, &mut state.rng)?;

        let audio_chunk = self
            .inner
            .decode_latent(&next_latent, &mut state.mimi_state)?;

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
    pub fn new(model_weights: &[u8]) -> Result<Model, JsError> {
        Self::new_(model_weights).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn add_voice(&mut self, voice_weights: &[u8]) -> Result<usize, JsError> {
        self.add_voice_(voice_weights)
            .map_err(|e| JsError::new(&e.to_string()))
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
        self.generation_step_()
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn sample_rate(&self) -> usize {
        self.inner.sample_rate()
    }
}
