use std::sync::Mutex;
use wasm_bindgen::prelude::*;

use pocket_tts::flow_lm;
use pocket_tts::mimi::MimiState;
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
    voice_emb: Tensor<f32, CpuDevice>,
    tokenizer: std::sync::Arc<PresetTokenizer>,
    cfg: TTSConfig,
    gen_state: Option<GenState>,
}

impl Model {
    pub fn new_(model_weights: &[u8], voice_weights: &[u8]) -> xn::Result<Model> {
        use xn::error::Context;
        let cfg = TTSConfig::v202601(0.7);

        let vb = VB::from_bytes_with_key_map(vec![model_weights.to_vec()], CPU, remap_key)?;
        let root = vb.root();
        let tokenizer = std::sync::Arc::new(PresetTokenizer::new());
        let tokenizer_box: Box<dyn pocket_tts::Tokenizer + Send + Sync> =
            Box::new(SharedTokenizer(std::sync::Arc::clone(&tokenizer)));
        let model: TTSModel<f32, CpuDevice> = TTSModel::load(&root, tokenizer_box, &cfg)?;

        // Load voice embedding
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

        Ok(Model {
            inner: model,
            voice_emb,
            tokenizer,
            cfg,
            gen_state: None,
        })
    }

    pub fn start_generation_(
        &mut self,
        token_ids: &[u32],
        frames_after_eos: usize,
        temperature: f32,
    ) -> xn::Result<()> {
        self.tokenizer.set_tokens(token_ids.to_vec());

        let num_tokens = token_ids.len();
        let max_frames = ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
        let seq_budget = num_tokens + 512 + max_frames;

        let mut tts_state = self.inner.init_flow_lm_state(1, seq_budget)?;
        let mimi_state = self.inner.init_mimi_state(1, 250)?;

        self.inner.prompt_audio(&mut tts_state, &self.voice_emb)?;
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
    pub fn new(model_weights: &[u8], voice_weights: &[u8]) -> Result<Model, JsError> {
        Self::new_(model_weights, voice_weights).map_err(|e| JsError::new(&e.to_string()))
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
        token_ids: &[u32],
        frames_after_eos: usize,
        temperature: f32,
    ) -> Result<(), JsError> {
        self.start_generation_(token_ids, frames_after_eos, temperature)
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
