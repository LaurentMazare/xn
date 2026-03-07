#[path = "audio_helpers.rs"]
mod audio_helpers;

use anyhow::{Context, Result};
use clap::Parser;
use pocket_tts::tts_model::{TTSConfig, TTSModel, prepare_text_prompt, split_into_best_sentences};
use xn::nn::VB;
use xn::{Backend, Tensor};

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

#[derive(Parser, Debug)]
#[command(name = "pocket-tts")]
#[command(about = "Generate speech from text using Pocket TTS")]
struct Args {
    /// Text to synthesize
    text: String,

    #[arg(long)]
    config: Option<String>,

    #[arg(long)]
    cfg_coef: Option<f32>,

    /// Output WAV file path
    #[arg(short, long, default_value = "output.wav")]
    output: std::path::PathBuf,

    /// Voice to use
    #[arg(short, long)]
    voice: Option<String>,

    /// Sampling temperature
    #[arg(short, long, default_value_t = 0.7)]
    temperature: f32,

    /// Sampling seed
    #[arg(short, long, default_value_t = 4242424242424242)]
    seed: u64,

    /// Use the cpu device even if cuda is available
    #[arg(long, default_value_t = false)]
    cpu: bool,

    #[arg(long)]
    chrome_tracing: bool,

    #[arg(long)]
    rng_values: Option<String>,

    #[arg(long)]
    wait_to_decode: bool,
}

const VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

enum Voice {
    Safetensors(std::path::PathBuf),
    Audio(String),
}

fn download_files(voice: &str) -> Result<(std::path::PathBuf, std::path::PathBuf, Voice)> {
    use hf_hub::{Repo, RepoType, api::sync::Api};
    let repo_id = "kyutai/pocket-tts";
    tracing::info!(?repo_id, "downloading weights...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    let model_path = repo.get("tts_b6369a24.safetensors").context("model weights not found")?;
    tracing::info!(?model_path, "model weights downloaded");

    let tokenizer_path = repo.get("tokenizer.model").context("tokenizer not found")?;
    tracing::info!(?tokenizer_path, "tokenizer downloaded");

    let voice = if VOICES.contains(&voice) {
        let voice_file = format!("embeddings/{voice}.safetensors");
        let voice_path = repo
            .get(&voice_file)
            .with_context(|| format!("voice embedding '{voice}' not found"))?;
        tracing::info!(?voice_path, "voice embedding downloaded");
        Voice::Safetensors(voice_path)
    } else {
        Voice::Audio(voice.to_string())
    };
    Ok((model_path, tokenizer_path, voice))
}

fn remap_key(name: &str) -> Option<String> {
    // Skip keys we don't need
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
    {
        return None;
    }

    let mut name = name.to_string();

    // Order matters: more specific replacements first
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

fn init_tracing(chrome_tracing: bool) -> Option<tracing_chrome::FlushGuard> {
    use tracing_subscriber::prelude::*;

    if chrome_tracing {
        let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new().build();
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::Layer::new().with_target(false))
            .with(chrome_layer)
            .init();
        Some(guard)
    } else {
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::Layer::new().with_target(false))
            .init();
        None
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _guard = init_tracing(args.chrome_tracing);

    #[cfg(feature = "cuda")]
    {
        if args.cpu {
            tracing::info!("using cpu backend");
            run_for_device(args, xn::CPU)?;
        } else {
            tracing::info!("using cuda backend");
            let dev = xn::cuda_backend::Device::new(0)?;
            unsafe {
                dev.disable_event_tracking();
            }
            run_for_device(args, dev)?;
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        tracing::info!("using cpu backend");
        run_for_device(args, xn::CPU)?;
    }

    Ok(())
}

enum Rng {
    StdRng { inner: Box<rand::rngs::StdRng>, distr: rand_distr::Normal<f32> },
    FromFile { values: Vec<f32>, index: usize },
}

impl Rng {
    pub fn std_rng(temperature: f32, seed: u64) -> Result<Self> {
        use rand::SeedableRng;
        let std = temperature.sqrt();
        let distr = rand_distr::Normal::new(0f32, std)?;
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Ok(Self::StdRng { inner: Box::new(rng), distr })
    }

    pub fn from_file(path: &str) -> Result<Self> {
        let file = std::fs::read_to_string(path)?;
        let values = serde_json::from_str::<Vec<f32>>(&file)?;
        Ok(Self::FromFile { values, index: 0 })
    }
}

impl pocket_tts::flow_lm::Rng for Rng {
    fn sample(&mut self) -> f32 {
        match self {
            Self::StdRng { inner, distr } => {
                use rand::Rng;
                inner.sample(*distr)
            }
            Self::FromFile { values, index } => {
                if *index >= values.len() {
                    *index = 0;
                }
                let val = values[*index];
                *index += 1;
                val
            }
        }
    }
}

fn spawn<F, R>(f: F) -> std::thread::JoinHandle<R>
where
    F: FnOnce() -> Result<R>,
    F: Send + 'static,
    R: Send + 'static,
{
    std::thread::spawn(move || match f() {
        Err(e) => {
            tracing::error!(?e, "thread error");
            std::process::exit(1);
        }
        Ok(res) => res,
    })
}

fn run_for_device<Dev: Backend>(args: Args, dev: Dev) -> Result<()> {
    let (model_path, tokenizer_path, voice, cfg) = match args.config.as_ref() {
        Some(config) => {
            let config = std::fs::canonicalize(config)?;
            let parent = config.parent().context("config path has no parent")?;
            let model_path = parent.join("model.safetensors");
            let tokenizer_path = parent.join("tokenizer.model");
            tracing::info!(?config, "using local config");
            let config: pocket_tts::tts_model::TTSConfig =
                serde_json::from_str(&std::fs::read_to_string(config)?)?;
            let voice = args.voice.map(Voice::Audio);
            (model_path, tokenizer_path, voice, config)
        }
        None => {
            let voice = args.voice.unwrap_or("alba".to_string());
            if !VOICES.contains(&voice.as_str()) && !std::path::PathBuf::from(&voice).exists() {
                anyhow::bail!("unknown voice '{voice}'. Available voices: {}", VOICES.join(", "));
            }

            let (model_path, tokenizer_path, voice) = download_files(&voice)?;
            (model_path, tokenizer_path, Some(voice), TTSConfig::v202601(args.temperature))
        }
    };

    let tokenizer_path = tokenizer_path.to_str().context("invalid tokenizer path")?;
    let sp = sentencepiece::SentencePieceProcessor::open(tokenizer_path)?;
    let tokenizer = SpTokenizer(sp);
    let chunks = split_into_best_sentences(&tokenizer, &args.text, None);

    let mut rng = match args.rng_values {
        Some(path) => Rng::from_file(&path)?,
        None => Rng::std_rng(args.temperature, args.seed)?,
    };

    tracing::info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        xn::with_avx(),
        xn::with_neon(),
        xn::with_simd128(),
        xn::with_f16c()
    );

    let vb = VB::load_with_key_map(&[&model_path], dev.clone(), remap_key)?.root();
    let model: TTSModel<f32, Dev> = TTSModel::load(&vb, Box::new(tokenizer), &cfg)?;
    vb.check_all_used_with_ignore(|v| {
        v == "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding"
            || v.starts_with("mimi.quantizer")
    })?;
    tracing::info!("model loaded successfully!");

    let mut max_seq_budget = 0;
    let mut all_tokens = vec![];
    for chunk in chunks.iter() {
        let (text, frames_after_eos) = prepare_text_prompt(chunk);
        let tokens = model.flow_lm.conditioner.tokenize(&text)?;
        let num_tokens = tokens.len();
        tracing::info!(?text, ?num_tokens, "processing text");
        let max_frames = ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
        let seq_budget = num_tokens + 512 + max_frames;
        max_seq_budget = max_seq_budget.max(seq_budget);
        all_tokens.push((tokens, max_frames, frames_after_eos));
    }
    // Init states
    let mut tts_state = model.init_flow_lm_state(1, max_seq_budget)?;
    let mut cfg_state = match args.cfg_coef {
        Some(1.0) | None => None,
        Some(coef) => {
            tracing::info!(?coef, "using custom cfg coefficient");
            let null_state = model.init_flow_lm_state(1, max_seq_budget)?;
            Some((coef, null_state))
        }
    };
    let mimi_state = model.init_mimi_state(1, 250)?;

    // Load voice embedding
    if let Some(voice) = voice {
        let voice_emb = match voice {
            Voice::Safetensors(voice_path) => {
                let voice_vb = VB::load(&[&voice_path], dev.clone())?;
                let voice_names = voice_vb.tensor_names();
                let voice_key =
                    voice_names.first().context("no tensors found in voice embedding file")?;
                let voice_td = voice_vb.get_tensor(voice_key).context("voice tensor not found")?;
                let voice_shape = &voice_td.shape;
                let voice_dims = voice_shape.dims();

                // Load as raw tensor and reshape to [1, T, dim]
                let voice_emb: Tensor<f32, Dev> =
                    voice_vb.tensor(voice_key, voice_shape.clone())?;
                if voice_dims.len() == 2 {
                    voice_emb.reshape((1, voice_dims[0], voice_dims[1]))?
                } else {
                    voice_emb
                }
            }
            Voice::Audio(path) => {
                tracing::info!("loading voice from audio file {}", path);
                let (pcm, sample_rate) = audio_helpers::pcm_decode(&path)?;
                let sample_rate = sample_rate as usize;
                let pcm = if sample_rate != cfg.mimi.sample_rate {
                    audio_helpers::resample(&pcm, sample_rate, cfg.mimi.sample_rate)?
                } else {
                    pcm
                };
                tracing::info!("loaded audio with {} samples", pcm.len());
                // Trim it to 10s max.
                let pcm = if pcm.len() > cfg.mimi.sample_rate * 10 {
                    tracing::info!("trimming audio to 10 seconds");
                    pcm[..cfg.mimi.sample_rate * 10].to_vec()
                } else {
                    pcm
                };
                let pcm_tensor = Tensor::from_vec(pcm, (1, 1, ()), &dev)?;
                let encoded_audio = model.encode_audio(&pcm_tensor)?;
                tracing::info!(?encoded_audio, "encoded audio to latent");
                encoded_audio
            }
        };
        // Prompt with audio conditioning
        tracing::info!("prompting with voice conditioning ({} frames)...", voice_emb.dim(1usize)?);
        let start = std::time::Instant::now();
        model.prompt_audio(&mut tts_state, &voice_emb)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        tracing::info!("done prompting with voice conditioning in {elapsed_ms:.2}ms");
    }

    let start = std::time::Instant::now();
    tracing::info!("starting generation...");
    let mut all_audios = vec![];
    let model = std::sync::Arc::new(model);
    for (tokens, max_frames, frames_after_eos) in all_tokens.into_iter() {
        tracing::info!("prompting with text conditioning ({} tokens)...", tokens.len());
        let start = std::time::Instant::now();
        let mut tts_state = tts_state.clone();
        let mut mimi_state = mimi_state.clone();
        model.prompt_text(&mut tts_state, &tokens)?;
        if let Some((_, ref mut null_state)) = cfg_state {
            model.prompt_text_null(null_state)?;
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        tracing::info!("done prompting with text conditioning in {elapsed_ms:.2}ms");

        // BOS marker: NaN tensor [1, 1, ldim]
        let ldim = cfg.flow_lm.ldim;
        let nan_data: Vec<f32> = vec![f32::NAN; ldim];
        let mut prev_latent: Tensor<f32, Dev> = Tensor::from_vec(nan_data, (1, 1, ldim), &dev)?;

        let mut eos_countdown: Option<usize> = None;

        let (latent_tx, latent_rx) = std::sync::mpsc::channel();
        let is_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let jh = spawn({
            let wait_to_decode = args.wait_to_decode;
            let model = model.clone();
            let is_done = is_done.clone();
            move || {
                let mut audio_chunks: Vec<Tensor<f32, Dev>> = Vec::new();
                if wait_to_decode {
                    tracing::info!("waiting for generation to finish before decoding...");
                    while !is_done.load(std::sync::atomic::Ordering::SeqCst) {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                    }
                }
                while let Ok(next_latent) = latent_rx.recv() {
                    // Decode latent to audio
                    let audio_chunk = model.decode_latent(&next_latent, &mut mimi_state)?;
                    audio_chunks.push(audio_chunk);
                }
                // Concatenate audio
                let audio_refs: Vec<&Tensor<f32, Dev>> = audio_chunks.iter().collect();
                let audio = Tensor::cat(&audio_refs, 2)?;
                let audio = audio.narrow(0, ..1)?.contiguous()?;
                Ok::<_, anyhow::Error>(audio)
            }
        });

        for step in 0..max_frames {
            let (next_latent, is_eos) = match cfg_state.as_mut() {
                Some((coef, null_state)) => model.generate_step_cfg(
                    &mut tts_state,
                    null_state,
                    *coef,
                    &prev_latent,
                    &mut rng,
                )?,
                None => model.generate_step(&mut tts_state, &prev_latent, &mut rng)?,
            };
            latent_tx.send(next_latent.clone())?;

            if is_eos && eos_countdown.is_none() {
                eos_countdown = Some(frames_after_eos);
            }

            if let Some(ref mut countdown) = eos_countdown {
                if *countdown == 0 {
                    tracing::info!(?step, "reached eos");
                    break;
                }
                *countdown -= 1;
            }

            prev_latent = next_latent;

            if (step + 1) % 25 == 0 {
                tracing::info!(?step, ?max_frames, "generation progress");
            }
        }
        std::mem::drop(latent_tx); // Close channel to signal generation thread to finish
        is_done.store(true, std::sync::atomic::Ordering::SeqCst);
        let audio = jh.join().map_err(|_| anyhow::anyhow!("cannot join thread"))?;
        all_audios.push(audio);
    }
    let all_audios = all_audios.iter().collect::<Vec<&Tensor<f32, Dev>>>();
    let audio = Tensor::cat(&all_audios, 2)?;
    let pcm = audio.to_vec()?;
    let duration = pcm.len() as f64 / cfg.mimi.sample_rate as f64;

    let elapsed = start.elapsed().as_secs_f64();
    let rtf = duration / elapsed;
    tracing::info!("generated {duration:.2}s in {elapsed:.2}s (RTF={rtf:.3})");

    // Write WAV
    let output_file = std::fs::File::create(&args.output)?;
    let mut writer = std::io::BufWriter::new(output_file);
    pocket_tts::wav::write_pcm_as_wav(&mut writer, &pcm, cfg.mimi.sample_rate as u32, 1)?;
    tracing::info!("saving output to {}", args.output.display());
    Ok(())
}
