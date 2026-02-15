use std::sync::Mutex;
use wasm_bindgen::prelude::*;

use pocket_tts::flow_lm;
use pocket_tts::tts_model::{TTSConfig, TTSModel, prepare_text_prompt};
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

#[wasm_bindgen]
pub struct Model {
    inner: TTSModel<f32, CpuDevice>,
    voice_emb: Tensor<f32, CpuDevice>,
    tokenizer: std::sync::Arc<PresetTokenizer>,
    cfg: TTSConfig,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new(model_weights: &[u8], voice_weights: &[u8]) -> Result<Model, JsError> {
        let cfg = TTSConfig::v202601(0.7);

        let vb = VB::from_bytes_with_key_map(vec![model_weights.to_vec()], CPU, remap_key)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let root = vb.root();
        let tokenizer = std::sync::Arc::new(PresetTokenizer::new());
        let tokenizer_box: Box<dyn pocket_tts::Tokenizer + Send + Sync> =
            Box::new(SharedTokenizer(std::sync::Arc::clone(&tokenizer)));
        let model: TTSModel<f32, CpuDevice> =
            TTSModel::load(&root, tokenizer_box, &cfg).map_err(|e| JsError::new(&e.to_string()))?;

        // Load voice embedding
        let voice_vb = VB::from_bytes(vec![voice_weights.to_vec()], CPU)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let voice_names = voice_vb.tensor_names();
        let voice_key = voice_names
            .first()
            .ok_or_else(|| JsError::new("no tensors found in voice embedding file"))?
            .to_string();
        let voice_td = voice_vb
            .get_tensor(&voice_key)
            .ok_or_else(|| JsError::new("voice tensor not found"))?;
        let voice_shape = voice_td.shape.clone();
        let voice_dims = voice_shape.dims().to_vec();

        let voice_emb: Tensor<f32, CpuDevice> = voice_vb
            .tensor(&voice_key, voice_shape)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let voice_emb = if voice_dims.len() == 2 {
            voice_emb
                .reshape((1, voice_dims[0], voice_dims[1]))
                .map_err(|e| JsError::new(&e.to_string()))?
        } else {
            voice_emb
        };

        Ok(Model {
            inner: model,
            voice_emb,
            tokenizer,
            cfg,
        })
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

    /// Generate audio from token IDs.
    /// Returns Float32Array of PCM audio samples at 24kHz.
    pub fn generate(
        &self,
        token_ids: &[u32],
        frames_after_eos: usize,
        temperature: f32,
    ) -> Result<js_sys::Float32Array, JsError> {
        let e = |e: xn::Error| JsError::new(&e.to_string());

        // Store token IDs in the preset tokenizer so it returns them if called.
        self.tokenizer.set_tokens(token_ids.to_vec());

        let num_tokens = token_ids.len();
        let max_frames = ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
        let seq_budget = num_tokens + 512 + max_frames;

        let mut tts_state = self.inner.init_flow_lm_state(1, seq_budget).map_err(e)?;
        let mut mimi_state = self.inner.init_mimi_state(1, 250).map_err(e)?;

        // Prompt with voice
        self.inner
            .prompt_audio(&mut tts_state, &self.voice_emb)
            .map_err(e)?;

        // Prompt with text
        self.inner
            .prompt_text(&mut tts_state, token_ids)
            .map_err(e)?;

        let mut rng = WasmRng::new(temperature);

        // BOS marker: NaN tensor [1, 1, ldim]
        let ldim = self.cfg.flow_lm.ldim;
        let nan_data: Vec<f32> = vec![f32::NAN; ldim];
        let mut prev_latent: Tensor<f32, CpuDevice> =
            Tensor::from_vec(nan_data, (1, 1, ldim), &CPU).map_err(e)?;

        let mut eos_countdown: Option<usize> = None;
        let mut audio_chunks: Vec<Tensor<f32, CpuDevice>> = Vec::new();

        for _step in 0..max_frames {
            let (next_latent, is_eos) = self
                .inner
                .generate_step(&mut tts_state, &prev_latent, &mut rng)
                .map_err(e)?;

            let audio_chunk = self
                .inner
                .decode_latent(&next_latent, &mut mimi_state)
                .map_err(e)?;
            audio_chunks.push(audio_chunk);

            if is_eos && eos_countdown.is_none() {
                eos_countdown = Some(frames_after_eos);
            }

            if let Some(ref mut countdown) = eos_countdown {
                if *countdown == 0 {
                    break;
                }
                *countdown -= 1;
            }

            prev_latent = next_latent;
        }

        // Concatenate audio
        let audio_refs: Vec<&Tensor<f32, CpuDevice>> = audio_chunks.iter().collect();
        let audio = Tensor::cat(&audio_refs, 2).map_err(e)?;
        let audio = audio.narrow(0, ..1).map_err(e)?.contiguous().map_err(e)?;
        let pcm = audio.to_vec().map_err(e)?;

        Ok(js_sys::Float32Array::from(pcm.as_slice()))
    }

    pub fn sample_rate(&self) -> usize {
        self.inner.sample_rate()
    }
}
