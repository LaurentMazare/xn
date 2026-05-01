use numpy::{PyArray1, PyReadonlyArray1};
use ptts::tts_model::{MimiEnc, TTSConfig, TTSModel, TTSState};
use pyo3::prelude::*;
use std::sync::Arc;
use xn::nn::VB;
use xn::{Tensor, Unquantized, error::Context};

struct StdRng {
    inner: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
}

impl StdRng {
    fn new(temperature: f32, seed: u64) -> Self {
        use rand::SeedableRng;
        let distr = rand_distr::Normal::new(0f32, temperature.sqrt()).unwrap();
        let inner = rand::rngs::StdRng::seed_from_u64(seed);
        Self { inner, distr }
    }
}

impl ptts::flow_lm::Rng for StdRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

trait PyRes<R> {
    fn w(self) -> PyResult<R>;
}

impl<R, E: Into<xn::Error>> PyRes<R> for Result<R, E> {
    fn w(self) -> PyResult<R> {
        self.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.into().to_string()))
    }
}

#[macro_export]
macro_rules! py_bail {
    ($msg:literal $(,)?) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($msg)))
    };
    ($err:expr $(,)?) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($err)))
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($fmt, $($arg)*)))
    };
}

const VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

struct ModelB<B: xn::Backend> {
    inner: Arc<TTSModel<Unquantized<f32, B>>>,
    mimi_enc: MimiEnc<Unquantized<f32, B>>,
    voices: std::collections::HashMap<String, Tensor<f32, B>>,
}

impl<B: xn::Backend> ModelB<B> {
    fn get_state_for_audio(
        &self,
        audio_prompt: &[f32],
        cfg_coef: Option<f32>,
        max_seq_len: usize,
    ) -> xn::Result<ModelStateB<B>> {
        let expected_len = self.inner.sample_rate() * 10;
        if audio_prompt.len() != expected_len {
            xn::bail!(
                "audio_prompt must have exactly {expected_len} samples (10s at {}Hz), got {}",
                self.inner.sample_rate(),
                audio_prompt.len()
            );
        }
        let dev = self.inner.device();
        let pcm = xn::Tensor::from_vec(audio_prompt.to_vec(), (1, 1, ()), dev)?;
        let voice_emb = self.mimi_enc.encode_audio(&pcm)?;
        let mut state = self.inner.init_flow_lm_state(1, max_seq_len)?;
        self.inner.prompt_audio(&mut state, &voice_emb)?;
        let cfg_state = if let Some(cfg_coef) = cfg_coef {
            let null_pcm = pcm.zeros_like()?;
            let null_emb = self.mimi_enc.encode_audio(&null_pcm)?;
            let mut null_state = self.inner.init_flow_lm_state(1, max_seq_len)?;
            self.inner.prompt_audio(&mut null_state, &null_emb)?;
            Some((cfg_coef, null_state))
        } else {
            None
        };
        Ok(ModelStateB { model: Arc::clone(&self.inner), state, cfg_state })
    }

    fn get_state_for_voice(&self, voice: &str, max_seq_len: usize) -> xn::Result<ModelStateB<B>> {
        let voice_emb = match self.voices.get(voice) {
            Some(emb) => emb,
            None => {
                let available: Vec<_> = self.voices.keys().collect();
                xn::bail!("unknown voice '{voice}'. Available voices: {available:?}")
            }
        };
        let mut state = self.inner.init_flow_lm_state(1, max_seq_len)?;
        self.inner.prompt_audio(&mut state, voice_emb)?;
        Ok(ModelStateB { model: Arc::clone(&self.inner), state, cfg_state: None })
    }

    fn voices(&self) -> Vec<String> {
        self.voices.keys().cloned().collect()
    }

    fn sample_rate(&self) -> usize {
        self.inner.sample_rate()
    }
}

// Poor man's type erasure, that's especially painful with pyo3 where
// the [pyclass] attribute is on a struct to get a proper object.
enum ModelV {
    Cpu(ModelB<xn::CpuDevice>),
    #[cfg(feature = "cuda")]
    Cuda(ModelB<xn::CudaDevice>),
}

#[pyclass]
struct Model(ModelV);

#[pymethods]
impl Model {
    #[pyo3(signature = (audio_prompt, cfg_coef=None, max_seq_len=2048))]
    fn get_state_for_audio(
        &self,
        audio_prompt: PyReadonlyArray1<'_, f32>,
        cfg_coef: Option<f32>,
        max_seq_len: usize,
    ) -> PyResult<ModelState> {
        let audio_prompt = audio_prompt.as_slice()?;
        let inner = match &self.0 {
            ModelV::Cpu(m) => {
                ModelStateV::Cpu(m.get_state_for_audio(audio_prompt, cfg_coef, max_seq_len).w()?)
            }
            #[cfg(feature = "cuda")]
            ModelV::Cuda(m) => {
                ModelStateV::Cuda(m.get_state_for_audio(audio_prompt, cfg_coef, max_seq_len).w()?)
            }
        };
        Ok(ModelState(inner))
    }

    #[pyo3(signature = (voice, max_seq_len=2048))]
    fn get_state_for_voice(&self, voice: &str, max_seq_len: usize) -> PyResult<ModelState> {
        let inner = match &self.0 {
            ModelV::Cpu(m) => ModelStateV::Cpu(m.get_state_for_voice(voice, max_seq_len).w()?),
            #[cfg(feature = "cuda")]
            ModelV::Cuda(m) => ModelStateV::Cuda(m.get_state_for_voice(voice, max_seq_len).w()?),
        };
        Ok(ModelState(inner))
    }

    fn voices(&self) -> Vec<String> {
        match &self.0 {
            ModelV::Cpu(m) => m.voices(),
            #[cfg(feature = "cuda")]
            ModelV::Cuda(m) => m.voices(),
        }
    }

    fn sample_rate(&self) -> usize {
        match &self.0 {
            ModelV::Cpu(m) => m.sample_rate(),
            #[cfg(feature = "cuda")]
            ModelV::Cuda(m) => m.sample_rate(),
        }
    }
}

struct ModelStateB<B: xn::Backend> {
    model: Arc<TTSModel<Unquantized<f32, B>>>,
    state: TTSState<Unquantized<f32, B>>,
    cfg_state: Option<(f32, TTSState<Unquantized<f32, B>>)>,
}

enum ModelStateV {
    Cpu(ModelStateB<xn::CpuDevice>),
    #[cfg(feature = "cuda")]
    Cuda(ModelStateB<xn::CudaDevice>),
}

#[pyclass]
struct ModelState(ModelStateV);

/// Returns true if a python signal has been raised, e.g. on an interrupt.
fn check_py_interrupt() -> xn::Result<()> {
    if Python::attach(|py| py.check_signals().is_err()) {
        xn::bail!("interrupted by python signal");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_generate<B: xn::Backend>(
    model: Arc<TTSModel<Unquantized<f32, B>>>,
    mut state: TTSState<Unquantized<f32, B>>,
    mut cfg_state: Option<(f32, TTSState<Unquantized<f32, B>>)>,
    tokens: Vec<u32>,
    temperature: f32,
    seed: u64,
    frames_after_eos: usize,
    check_py_interrupts_every: Option<usize>,
) -> Result<Vec<f32>, xn::Error> {
    let device = model.device();
    let num_tokens = tokens.len();
    let max_frames = ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
    let mut rng = StdRng::new(temperature, seed);
    let mut mimi_state = model.init_mimi_state(1, 250)?;

    model.prompt_text(&mut state, &tokens)?;
    if let Some((_, cfg_state)) = cfg_state.as_mut() {
        model.prompt_text_null(cfg_state)?
    }

    let ldim = model.flow_lm.ldim;
    let nan_data = vec![f32::NAN; ldim];
    let mut prev_latent = Tensor::from_vec(nan_data, (1, 1, ldim), device)?;

    let (latent_tx, latent_rx) = std::sync::mpsc::channel();

    let decode_model = Arc::clone(&model);
    let decode_handle = std::thread::spawn(move || -> Result<Tensor<f32, B>, xn::Error> {
        let mut audio_chunks = Vec::new();
        while let Ok(latent) = latent_rx.recv() {
            let audio_chunk = decode_model.decode_latent(&latent, &mut mimi_state)?;
            audio_chunks.push(audio_chunk);
        }
        let audio_refs: Vec<_> = audio_chunks.iter().collect();
        let audio = Tensor::cat(&audio_refs, 2)?;
        let audio = audio.narrow(0, ..1)?.contiguous()?;
        Ok(audio)
    });

    let mut eos_countdown: Option<usize> = None;
    for step in 0..max_frames {
        let (next_latent, is_eos) = match cfg_state.as_mut() {
            Some((cfg_coef, cfg_state)) => {
                model.generate_step_cfg(&mut state, cfg_state, *cfg_coef, &prev_latent, &mut rng)?
            }
            None => model.generate_step(&mut state, &prev_latent, &mut rng)?,
        };
        // Rather than raising an error when the channel is closed, we break out of the loop
        // so as to get a proper error message from the decode thread.
        if latent_tx.send(next_latent.clone()).is_err() {
            break;
        }

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
        if let Some(check_step) = check_py_interrupts_every.as_ref()
            && step % check_step == 0
        {
            check_py_interrupt()?
        }
    }
    drop(latent_tx);

    let audio = decode_handle.join().map_err(|_| xn::Error::msg("decode thread panicked"))??;
    let pcm = audio.to_vec()?;
    Ok(pcm)
}

impl<B: xn::Backend> ModelStateB<B> {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            state: self.state.clone(),
            cfg_state: self.cfg_state.clone(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<u32>> {
        self.model.flow_lm.conditioner.tokenize(text).w()
    }

    fn generate_audio<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        temperature: f32,
        seed: u64,
        pad_to: Option<usize>,
        check_py_interrupts_every: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let model = Arc::clone(&self.model);
        let state = self.state.clone();
        let cfg_state = self.cfg_state.clone();
        let text = text.to_string();

        let pcm = py
            .detach(move || -> xn::Result<Vec<f32>> {
                let (text, frames_after_eos) = ptts::tts_model::prepare_text_prompt(&text);
                let mut tokens = model.flow_lm.conditioner.tokenize(&text)?;
                if let Some(pad_to) = pad_to
                    && tokens.len() < pad_to
                {
                    let learnt_padding_id = match model.flow_lm.conditioner.learnt_padding_id() {
                        Some(id) => id,
                        None => {
                            xn::bail!("model does not have a learnt padding token, cannot pad to {pad_to} tokens")
                        }
                    };
                    tokens.resize(pad_to, learnt_padding_id);
                }
                run_generate(model, state, cfg_state, tokens, temperature, seed, frames_after_eos, check_py_interrupts_every)
            })
            .w()?;

        Ok(PyArray1::from_vec(py, pcm))
    }

    fn generate_audio_for_tokens<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<i32>,
        temperature: f32,
        seed: u64,
        frames_after_eos: usize,
        check_py_interrupts_every: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let model = Arc::clone(&self.model);
        let state = self.state.clone();
        let cfg_state = self.cfg_state.clone();
        let mut new_tokens = Vec::with_capacity(tokens.len());
        for token in tokens {
            if token < 0 {
                let token = match model.flow_lm.conditioner.learnt_padding_id() {
                    Some(id) => id,
                    None => py_bail!(
                        "model does not have a learnt padding token, cannot use negative token value {token}"
                    ),
                };
                new_tokens.push(token);
            } else {
                new_tokens.push(token as u32);
            }
        }
        let pcm = py
            .detach(move || -> xn::Result<Vec<f32>> {
                run_generate(
                    model,
                    state,
                    cfg_state,
                    new_tokens,
                    temperature,
                    seed,
                    frames_after_eos,
                    check_py_interrupts_every,
                )
            })
            .w()?;

        Ok(PyArray1::from_vec(py, pcm))
    }
}

#[pymethods]
impl ModelState {
    fn clone(&self) -> Self {
        match &self.0 {
            ModelStateV::Cpu(s) => ModelState(ModelStateV::Cpu(s.clone())),
            #[cfg(feature = "cuda")]
            ModelStateV::Cuda(s) => ModelState(ModelStateV::Cuda(s.clone())),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<u32>> {
        match &self.0 {
            ModelStateV::Cpu(s) => s.tokenize(text),
            #[cfg(feature = "cuda")]
            ModelStateV::Cuda(s) => s.tokenize(text),
        }
    }

    #[pyo3(signature = (text, temperature=0.7, seed=4242424242424242, pad_to=None, check_py_interrupts_every=5))]
    fn generate_audio<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        temperature: f32,
        seed: u64,
        pad_to: Option<usize>,
        check_py_interrupts_every: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        match &self.0 {
            ModelStateV::Cpu(s) => {
                s.generate_audio(py, text, temperature, seed, pad_to, check_py_interrupts_every)
            }
            #[cfg(feature = "cuda")]
            ModelStateV::Cuda(s) => {
                s.generate_audio(py, text, temperature, seed, pad_to, check_py_interrupts_every)
            }
        }
    }

    #[pyo3(signature = (tokens, temperature=0.7, seed=4242424242424242, frames_after_eos=1, check_py_interrupts_every=5))]
    fn generate_audio_for_tokens<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<i32>,
        temperature: f32,
        seed: u64,
        frames_after_eos: usize,
        check_py_interrupts_every: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        match &self.0 {
            ModelStateV::Cpu(s) => s.generate_audio_for_tokens(
                py,
                tokens,
                temperature,
                seed,
                frames_after_eos,
                check_py_interrupts_every,
            ),
            #[cfg(feature = "cuda")]
            ModelStateV::Cuda(s) => s.generate_audio_for_tokens(
                py,
                tokens,
                temperature,
                seed,
                frames_after_eos,
                check_py_interrupts_every,
            ),
        }
    }
}

struct SpTokenizer(sentencepiece::SentencePieceProcessor);

impl ptts::Tokenizer for SpTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let pieces = self.0.encode(text).unwrap_or_default();
        pieces.iter().map(|p| p.id).collect()
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.0.decode_piece_ids(tokens).unwrap_or_default()
    }
}

fn remap_key(name: &str) -> Option<String> {
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
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

fn load_voice_embedding<B: xn::Backend>(
    voice_path: &std::path::Path,
    device: &B,
) -> Result<Tensor<f32, B>, xn::Error> {
    let voice_vb = VB::load(&[voice_path], device.clone())?;
    let voice_names = voice_vb.tensor_names();
    let voice_key = voice_names.first().context("no tensors found in voice embedding file")?;
    let voice_shape = voice_vb.shape(voice_key).context("voice tensor not found")?;
    let voice_dims = voice_shape.dims();
    let voice_emb: Tensor<f32, B> = voice_vb.tensor(voice_key, voice_shape.clone())?;
    if voice_dims.len() == 2 {
        Ok(voice_emb.reshape((1, voice_dims[0], voice_dims[1]))?)
    } else {
        Ok(voice_emb)
    }
}

fn load_model_<B: xn::Backend>(
    temperature: f32,
    repo_id: String,
    model_file: String,
    config: Option<String>,
    eos_threshold: Option<f32>,
    dev: B,
) -> xn::Result<ModelB<B>> {
    let (model_path, tokenizer_path, cfg, voices) = match config {
        Some(config_path) => {
            let config_path = std::fs::canonicalize(&config_path).map_err(xn::Error::msg)?;
            let parent = config_path.parent().context("config path has no parent")?;
            let model_path = parent.join("model.safetensors");
            let tokenizer_path = parent.join("tokenizer.model");
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| xn::Error::msg(e).with_path(&config_path))?;
            let cfg: TTSConfig = serde_json::from_str(&config_str)
                .map_err(|e| xn::Error::msg(e).with_path(&config_path))?;
            (model_path, tokenizer_path, cfg, std::collections::HashMap::new())
        }
        None => {
            use hf_hub::{Repo, RepoType, api::sync::Api};

            let api = Api::new().map_err(xn::Error::msg)?;
            let repo = api.repo(Repo::new(repo_id, RepoType::Model));

            let model_path =
                repo.get(&model_file).map_err(|e| xn::Error::msg(e).with_path(&model_file))?;
            let tokenizer_path = repo.get("tokenizer.model").map_err(xn::Error::msg)?;

            let mut voices = std::collections::HashMap::new();
            for &voice in VOICES {
                let voice_file = format!("embeddings/{voice}.safetensors");
                if let Ok(voice_path) = repo.get(&voice_file)
                    && let Ok(voice_emb) = load_voice_embedding(&voice_path, &dev)
                {
                    voices.insert(voice.to_string(), voice_emb);
                }
            }

            let cfg = TTSConfig::v202601(temperature);
            (model_path, tokenizer_path, cfg, voices)
        }
    };

    let tokenizer_path = tokenizer_path.to_str().context("invalid tokenizer path")?;
    let sp = sentencepiece::SentencePieceProcessor::open(tokenizer_path)
        .map_err(|e| xn::Error::msg(e).with_path(tokenizer_path))?;
    let tokenizer = SpTokenizer(sp);

    let vb = VB::load_with_key_map(&[&model_path], dev, remap_key)
        .map_err(|e| e.with_path(&model_path))?
        .root();
    let model = TTSModel::load(&vb, Box::new(tokenizer), &cfg)?;
    let model = if let Some(eos_threshold) = eos_threshold {
        model.with_eos_threshold(eos_threshold)
    } else {
        model
    };
    let mimi_enc = MimiEnc::load(&vb, &cfg)?;
    vb.check_all_used_with_ignore(|v| {
        v == "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding"
            || v.starts_with("mimi.quantizer")
    })?;

    Ok(ModelB { inner: Arc::new(model), mimi_enc, voices })
}

#[pyfunction]
#[pyo3(signature = (temperature=0.7, repo_id="kyutai/pocket-tts", model_file="tts_b6369a24.safetensors", config=None, eos_threshold=None, device=None))]
fn load_model(
    py: Python<'_>,
    temperature: f32,
    repo_id: &str,
    model_file: &str,
    config: Option<&str>,
    eos_threshold: Option<f32>,
    device: Option<&str>,
) -> PyResult<Model> {
    let repo_id = repo_id.to_string();
    let model_file = model_file.to_string();
    let config = config.map(|s| s.to_string());
    py.detach(move || match device {
        None | Some("cpu") => {
            let model = load_model_(
                temperature,
                repo_id,
                model_file,
                config,
                eos_threshold,
                xn::CpuDevice,
            )?;
            Ok(Model(ModelV::Cpu(model)))
        }
        #[cfg(feature = "cuda")]
        Some("cuda") => {
            let dev = xn::CudaDevice::new(0)?;
            let model = load_model_(temperature, repo_id, model_file, config, eos_threshold, dev)?;
            Ok(Model(ModelV::Cuda(model)))
        }
        Some(d) => Err(xn::Error::msg(format!("unknown device '{d}'"))),
    })
    .w()
}

#[pyfunction]
fn get_num_threads() -> usize {
    xn::utils::get_num_threads()
}

#[pyfunction]
fn set_num_threads(num_threads: usize) {
    xn::utils::set_num_threads(num_threads);
}

#[pyfunction]
fn prepare_text_prompt(text: &str) -> String {
    ptts::tts_model::prepare_text_prompt(text).0
}

#[pyfunction]
fn cuda_available() -> bool {
    cfg!(feature = "cuda")
}

#[pymodule(name = "ptts")]
fn ptts_(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<ModelState>()?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_text_prompt, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    Ok(())
}
