use numpy::{PyArray1, PyReadonlyArray1};
use pocket_tts::tts_model::{TTSConfig, TTSModel, prepare_text_prompt};
use pyo3::prelude::*;
use std::sync::Arc;
use xn::Tensor;
use xn::nn::VB;

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

impl pocket_tts::flow_lm::Rng for StdRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

trait PyRes<R> {
    #[allow(unused)]
    fn w(self) -> PyResult<R>;
    #[allow(unused)]
    fn w_f<P: AsRef<std::path::Path>>(self, p: P) -> PyResult<R>;
}

impl<R, E: Into<xn::Error>> PyRes<R> for Result<R, E> {
    fn w(self) -> PyResult<R> {
        self.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.into().to_string()))
    }
    fn w_f<P: AsRef<std::path::Path>>(self, p: P) -> PyResult<R> {
        self.map_err(|e| {
            let e = e.into().to_string();
            let msg = format!("{:?}: {e}", p.as_ref());
            pyo3::exceptions::PyValueError::new_err(msg)
        })
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

const VOICES: &[&str] = &[
    "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
];

#[pyclass]
struct Model {
    inner: Arc<TTSModel<f32, xn::CpuDevice>>,
    voices: std::collections::HashMap<String, Tensor<f32, xn::CpuDevice>>,
}

#[pymethods]
impl Model {
    #[pyo3(signature = (audio_prompt, max_seq_len=2048))]
    fn get_state_for_audio(
        &self,
        audio_prompt: PyReadonlyArray1<'_, f32>,
        max_seq_len: usize,
    ) -> PyResult<ModelState> {
        let expected_len = self.inner.sample_rate() * 10;
        let audio_prompt = audio_prompt.as_slice()?;
        if audio_prompt.len() != expected_len {
            py_bail!(
                "audio_prompt must have exactly {expected_len} samples (10s at {}Hz), got {}",
                self.inner.sample_rate(),
                audio_prompt.len()
            );
        }
        let pcm = xn::Tensor::from_vec(audio_prompt.to_vec(), (1, 1, ()), &xn::CpuDevice).w()?;
        let voice_emb = self.inner.encode_audio(&pcm).w()?;
        let mut state = self.inner.init_flow_lm_state(1, max_seq_len).w()?;
        self.inner.prompt_audio(&mut state, &voice_emb).w()?;
        Ok(ModelState {
            model: Arc::clone(&self.inner),
            state,
        })
    }

    #[pyo3(signature = (voice, max_seq_len=2048))]
    fn get_state_for_voice(&self, voice: &str, max_seq_len: usize) -> PyResult<ModelState> {
        let voice_emb = match self.voices.get(voice) {
            Some(emb) => emb,
            None => {
                let available: Vec<_> = self.voices.keys().collect();
                py_bail!("unknown voice '{voice}'. Available voices: {available:?}")
            }
        };
        let mut state = self.inner.init_flow_lm_state(1, max_seq_len).w()?;
        self.inner.prompt_audio(&mut state, voice_emb).w()?;
        Ok(ModelState {
            model: Arc::clone(&self.inner),
            state,
        })
    }

    fn voices(&self) -> Vec<String> {
        self.voices.keys().cloned().collect()
    }

    fn sample_rate(&self) -> usize {
        self.inner.sample_rate()
    }
}

#[pyclass]
struct ModelState {
    #[allow(unused)]
    model: Arc<TTSModel<f32, xn::CpuDevice>>,
    #[allow(unused)]
    state: pocket_tts::tts_model::TTSState<f32, xn::CpuDevice>,
}

#[pymethods]
impl ModelState {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            state: self.state.clone(),
        }
    }

    #[pyo3(signature = (text, temperature=0.7, seed=4242424242424242))]
    fn generate_audio<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        temperature: f32,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let model = Arc::clone(&self.model);
        let mut state = self.state.clone();
        let text = text.to_string();

        let pcm = py
            .detach(move || -> Result<Vec<f32>, xn::Error> {
                let (text, frames_after_eos) = prepare_text_prompt(&text);
                let tokens = model.flow_lm.conditioner.tokenize(&text)?;
                let num_tokens = tokens.len();
                let max_frames = ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;

                let mut rng = StdRng::new(temperature, seed);
                let mut mimi_state = model.init_mimi_state(1, 250)?;

                model.prompt_text(&mut state, &tokens)?;

                let ldim = model.flow_lm.ldim;
                let nan_data: Vec<f32> = vec![f32::NAN; ldim];
                let mut prev_latent: Tensor<f32, xn::CpuDevice> =
                    Tensor::from_vec(nan_data, (1, 1, ldim), &xn::CpuDevice)?;

                let (latent_tx, latent_rx) = std::sync::mpsc::channel();

                let decode_model = Arc::clone(&model);
                let decode_handle =
                    std::thread::spawn(move || -> Result<Tensor<f32, xn::CpuDevice>, xn::Error> {
                        let mut audio_chunks = Vec::new();
                        while let Ok(latent) = latent_rx.recv() {
                            let audio_chunk =
                                decode_model.decode_latent(&latent, &mut mimi_state)?;
                            audio_chunks.push(audio_chunk);
                        }
                        let audio_refs: Vec<_> = audio_chunks.iter().collect();
                        let audio = Tensor::cat(&audio_refs, 2)?;
                        let audio = audio.narrow(0, ..1)?.contiguous()?;
                        Ok(audio)
                    });

                let mut eos_countdown: Option<usize> = None;
                for _step in 0..max_frames {
                    let (next_latent, is_eos) =
                        model.generate_step(&mut state, &prev_latent, &mut rng)?;
                    latent_tx
                        .send(next_latent.clone())
                        .map_err(|e| xn::Error::Msg(e.to_string()))?;

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
                drop(latent_tx);

                let audio = decode_handle
                    .join()
                    .map_err(|_| xn::Error::Msg("decode thread panicked".to_string()))??;
                let pcm = audio.to_vec()?;
                Ok(pcm)
            })
            .w()?;

        Ok(PyArray1::from_vec(py, pcm))
    }
}

struct SpTokenizer(sentencepiece::SentencePieceProcessor);

impl pocket_tts::Tokenizer for SpTokenizer {
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

fn load_voice_embedding(
    voice_path: &std::path::Path,
) -> Result<Tensor<f32, xn::CpuDevice>, xn::Error> {
    let voice_vb = VB::load(&[voice_path], xn::CpuDevice)?;
    let voice_names = voice_vb.tensor_names();
    let voice_key = voice_names
        .first()
        .ok_or_else(|| xn::Error::Msg("no tensors found in voice embedding file".into()))?;
    let voice_td = voice_vb
        .get_tensor(voice_key)
        .ok_or_else(|| xn::Error::Msg("voice tensor not found".into()))?;
    let voice_shape = &voice_td.shape;
    let voice_dims = voice_shape.dims();
    let voice_emb: Tensor<f32, xn::CpuDevice> = voice_vb.tensor(voice_key, voice_shape.clone())?;
    if voice_dims.len() == 2 {
        Ok(voice_emb.reshape((1, voice_dims[0], voice_dims[1]))?)
    } else {
        Ok(voice_emb)
    }
}

fn load_model_(
    temperature: f32,
    repo_id: String,
    model_file: String,
    config: Option<String>,
) -> xn::Result<Model> {
    let (model_path, tokenizer_path, cfg, voices) = match config {
        Some(config_path) => {
            let config_path =
                std::fs::canonicalize(&config_path).map_err(|e| xn::Error::Msg(e.to_string()))?;
            let parent = config_path
                .parent()
                .ok_or_else(|| xn::Error::Msg("config path has no parent".into()))?;
            let model_path = parent.join("model.safetensors");
            let tokenizer_path = parent.join("tokenizer.model");
            let config_str =
                std::fs::read_to_string(&config_path).map_err(|e| xn::Error::Msg(e.to_string()))?;
            let cfg: TTSConfig =
                serde_json::from_str(&config_str).map_err(|e| xn::Error::Msg(e.to_string()))?;
            (
                model_path,
                tokenizer_path,
                cfg,
                std::collections::HashMap::new(),
            )
        }
        None => {
            use hf_hub::{Repo, RepoType, api::sync::Api};

            let api = Api::new().map_err(|e| xn::Error::Msg(e.to_string()))?;
            let repo = api.repo(Repo::new(repo_id, RepoType::Model));

            let model_path = repo
                .get(&model_file)
                .map_err(|e| xn::Error::Msg(e.to_string()))?;
            let tokenizer_path = repo
                .get("tokenizer.model")
                .map_err(|e| xn::Error::Msg(e.to_string()))?;

            let mut voices = std::collections::HashMap::new();
            for &voice in VOICES {
                let voice_file = format!("embeddings/{voice}.safetensors");
                if let Ok(voice_path) = repo.get(&voice_file)
                    && let Ok(voice_emb) = load_voice_embedding(&voice_path)
                {
                    voices.insert(voice.to_string(), voice_emb);
                }
            }

            let cfg = TTSConfig::v202601(temperature);
            (model_path, tokenizer_path, cfg, voices)
        }
    };

    let tokenizer_path = tokenizer_path
        .to_str()
        .ok_or_else(|| xn::Error::Msg("invalid tokenizer path".into()))?;
    let sp = sentencepiece::SentencePieceProcessor::open(tokenizer_path)
        .map_err(|e| xn::Error::Msg(e.to_string()))?;
    let tokenizer = SpTokenizer(sp);

    let dev = xn::CpuDevice;
    let vb = VB::load_with_key_map(&[&model_path], dev, remap_key)?.root();
    let model: TTSModel<f32, xn::CpuDevice> = TTSModel::load(&vb, Box::new(tokenizer), &cfg)?;
    vb.check_all_used_with_ignore(|v| {
        v == "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding"
            || v.starts_with("mimi.quantizer")
    })?;

    Ok(Model {
        inner: Arc::new(model),
        voices,
    })
}

#[pyfunction]
#[pyo3(signature = (temperature=0.7, repo_id="kyutai/pocket-tts", model_file="tts_b6369a24.safetensors", config=None))]
fn load_model(
    py: Python<'_>,
    temperature: f32,
    repo_id: &str,
    model_file: &str,
    config: Option<&str>,
) -> PyResult<Model> {
    let repo_id = repo_id.to_string();
    let model_file = model_file.to_string();
    let config = config.map(|s| s.to_string());
    py.detach(move || load_model_(temperature, repo_id, model_file, config))
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

#[pymodule]
fn ptts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<ModelState>()?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    Ok(())
}
