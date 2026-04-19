// JNI / native session for the pocket-tts Android demo.
//
// Public surface (exposed to Kotlin under the class `sh.gradium.ptts.Ptts`):
//   nativeInit(weights_dir, voice) -> handle : Long
//   nativeGenerate(handle, text, temperature, seed)
//   nativeNextChunk(handle) -> FloatArray?         // null signals EOS
//   nativeStats(handle) -> FloatArray              // 6 numbers, see stats()
//   nativeSampleRate(handle) -> i32
//   nativeFree(handle)
//
// The Rust core runs single-threaded; callers must keep JNI calls on one
// background thread. Anchored on Unquantized<f32, CpuDevice>, matching the
// `ptts::TTSConfig::v202601` default when the CLI is invoked with --cpu.

mod tokenizer;

use log::info;
use ptts::flow_lm::FlowLMState;
use ptts::transformer::{LayerAttentionState, StreamingMHAState, StreamingTransformerState};
use ptts::tts_model::{
    TTSConfig, TTSModel, TTSState, prepare_text_prompt, split_into_best_sentences,
};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;
use tokenizer::Unigram;
use xn::nn::VB;
use xn::{CPU, CpuDevice, Tensor, TypedTensor, Unquantized};

type Model = TTSModel<Unquantized<f32, CpuDevice>>;
type State = TTSState<Unquantized<f32, CpuDevice>>;

// Matches the public HF repo at huggingface.co/kyutai/pocket-tts-without-voice-cloning
// (no auth required). Voice files under embeddings_v2/ are pre-primed KV-cache
// states — loaded via load_voice_kv_cache() rather than prompt_audio().
const VOICES: &[&str] = &["alba", "marius", "javert", "fantine", "cosette", "eponine", "azelma"];

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

struct Rng {
    inner: Box<rand::rngs::StdRng>,
    distr: rand_distr::Normal<f32>,
}

impl Rng {
    fn new(temperature: f32, seed: u64) -> Self {
        use rand::SeedableRng;
        let std = temperature.sqrt();
        let distr = rand_distr::Normal::new(0f32, std).unwrap();
        let inner = Box::new(rand::rngs::StdRng::seed_from_u64(seed));
        Self { inner, distr }
    }
}

impl ptts::flow_lm::Rng for Rng {
    fn sample(&mut self) -> f32 {
        use rand::Rng as _;
        self.inner.sample(self.distr)
    }
}

struct Chunk {
    tokens: Vec<u32>,
    max_frames: usize,
    frames_after_eos: usize,
}

/// State for a single in-flight generation: two worker threads (backbone +
/// decode) feeding a PCM queue the JNI caller drains chunk-by-chunk. Mirrors
/// the CLI's `spawn(|| decode loop)` pattern from pocket-tts/examples/.
struct Run {
    pcm_rx: mpsc::Receiver<Vec<f32>>,
    backbone: Option<JoinHandle<Result<(), String>>>,
    decoder: Option<JoinHandle<Result<(), String>>>,
    step_count: Arc<AtomicUsize>,
    step_total_ns: Arc<AtomicU64>,
    started_at: Instant,
    first_chunk_at: Option<Instant>,
    duration_s: f64,
}

pub struct Session {
    model: Arc<Model>,
    voice_path: std::path::PathBuf,
    seq_budget: usize,
    voice_cached: Option<State>,
    sample_rate: usize,
    run: Option<Run>,
    stats: Stats,
}

#[derive(Default, Clone, Copy)]
struct Stats {
    total_elapsed_s: f64,
    duration_s: f64,
    rtf: f64,
    avg_step_ms: f64,
    first_audio_s: f64,
    peak_rss_mb: f64,
}

impl Session {
    pub fn new(weights_dir: &str, voice: &str) -> Result<Self, String> {
        if !VOICES.contains(&voice) {
            return Err(format!("unknown voice '{voice}'. Available: {}", VOICES.join(", ")));
        }
        let dir = std::path::Path::new(weights_dir);
        let model_path = dir.join("tts_b6369a24.safetensors");
        let tokenizer_path = dir.join("tokenizer.model");
        let voice_path = dir.join("embeddings_v2").join(format!("{voice}.safetensors"));
        for p in [&model_path, &tokenizer_path, &voice_path] {
            if !p.exists() {
                return Err(format!("missing file: {}", p.display()));
            }
        }

        let tok = Unigram::from_file(tokenizer_path.to_str().unwrap())
            .map_err(|e| format!("tokenizer load: {e}"))?;

        // Force the rayon global pool to be created up-front with the full
        // hardware thread count, so the UI can later dial the actual gemm
        // parallelism anywhere in 1..=hw via `set_threads` (which only flips
        // xn's atomic; the pool size is fixed after the first matmul).
        //
        // Default floor: 4 threads — big.LITTLE on Pixel 8a has a slow A520
        // cluster that drags gemm down if we include it. `PTTS_THREADS` env
        // var and the `setThreads` JNI method both override.
        let hw = xn::get_num_cpus();
        let _ = rayon::ThreadPoolBuilder::new().num_threads(hw).build_global();
        let threads = std::env::var("PTTS_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(1);
        xn::set_num_threads(threads);

        info!(
            "backends: avx={} neon={} simd128={} f16c={} | cpu threads: detected={} rayon={}",
            xn::with_avx(),
            xn::with_neon(),
            xn::with_simd128(),
            xn::with_f16c(),
            xn::get_num_cpus(),
            xn::get_num_threads(),
        );

        let cfg = TTSConfig::v202601(0.7);
        let vb = VB::load_with_key_map(&[&model_path], CPU, remap_key)
            .map_err(|e| format!("load weights: {e}"))?
            .root();
        let model: Model =
            Model::load(&vb, Box::new(tok), &cfg).map_err(|e| format!("model load: {e}"))?;
        let sample_rate = model.sample_rate();

        Ok(Self {
            model: Arc::new(model),
            voice_path,
            seq_budget: 0,
            voice_cached: None,
            sample_rate,
            run: None,
            stats: Stats::default(),
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Cap the number of rayon workers gemm uses per matmul. Safe to call
    /// between generations; the underlying pool is already max-sized.
    pub fn set_threads(&self, n: usize) {
        let n = n.clamp(1, xn::get_num_cpus());
        xn::set_num_threads(n);
        info!("matmul parallelism set to {n}");
    }

    pub fn num_cpus(&self) -> usize {
        xn::get_num_cpus()
    }

    /// Tokenize the input, build per-chunk token streams and spawn the
    /// backbone + decoder threads. The JNI caller drains PCM via `next_chunk`.
    pub fn start(&mut self, text: &str, temperature: f32, seed: u64) -> Result<(), String> {
        // Drop any previous run (force threads to finish / unblock).
        self.stop_run();

        let tokenizer_ref = self
            .model
            .flow_lm
            .conditioner
            .tokenizer
            .as_deref()
            .ok_or_else(|| "model has no tokenizer".to_string())?;
        let sentences = split_into_best_sentences(tokenizer_ref, text, None);

        let mut max_seq_budget = 0usize;
        let mut chunks: Vec<Chunk> = Vec::new();
        for sentence in sentences.iter() {
            let (prepared, frames_after_eos) = prepare_text_prompt(sentence);
            let tokens = self
                .model
                .flow_lm
                .conditioner
                .tokenize(&prepared)
                .map_err(|e| format!("tokenize: {e}"))?;
            let n = tokens.len();
            let max_frames = ((n as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
            let seq_budget = n + 512 + max_frames;
            max_seq_budget = max_seq_budget.max(seq_budget);
            info!("prepared chunk ({} tokens, max_frames={max_frames}): {prepared}", n);
            chunks.push(Chunk { tokens, max_frames, frames_after_eos });
        }
        if chunks.is_empty() {
            return Err("no sentences".into());
        }

        // Lazily load the voice's pre-primed KV-cache state. Kept at its
        // original small size; resized once in the backbone thread per chunk
        // (so the JNI thread doesn't stall on a ~27 MB alloc+copy).
        if self.voice_cached.is_none() {
            let t0 = Instant::now();
            let cached = load_voice_kv_cache(&self.voice_path)?;
            info!("voice state loaded in {:.2}ms", t0.elapsed().as_secs_f64() * 1000.0);
            self.voice_cached = Some(cached);
        }
        let cached_voice = self.voice_cached.as_ref().unwrap().clone();
        self.seq_budget = max_seq_budget;

        // Two channels:
        //   backbone --latent--> decoder --pcm--> JNI caller
        // The backbone is one thread tight around `generate_step`; decoder is a
        // second thread tight around `decode_latent`. Matches the
        // `pocket-tts/examples/pocket_tts.rs` CLI pattern (spawn()'d decode).
        let (latent_tx, latent_rx) = mpsc::channel::<(Tensor<f32, CpuDevice>, usize)>();
        let (pcm_tx, pcm_rx) = mpsc::sync_channel::<Vec<f32>>(8);

        let step_count = Arc::new(AtomicUsize::new(0));
        let step_total_ns = Arc::new(AtomicU64::new(0));

        let backbone = {
            let model = self.model.clone();
            let step_count = step_count.clone();
            let step_total_ns = step_total_ns.clone();
            std::thread::Builder::new()
                .name("ptts-backbone".into())
                .spawn(move || -> Result<(), String> {
                    let ldim = model.flow_lm.ldim;
                    for chunk in chunks.into_iter() {
                        let t_resize = Instant::now();
                        let mut tts_state = resize_tts_state(&cached_voice, max_seq_budget)?;
                        let before = layer_current_end(&tts_state);
                        let t_prompt = Instant::now();
                        model
                            .prompt_text(&mut tts_state, &chunk.tokens)
                            .map_err(|e| format!("prompt_text: {e}"))?;
                        let after = layer_current_end(&tts_state);
                        info!(
                            "chunk init: resize={:.1}ms prompt_text={:.1}ms \
                             (budget={}, tokens={}, current_end before={:?} after={:?})",
                            (t_prompt - t_resize).as_secs_f64() * 1000.0,
                            t_prompt.elapsed().as_secs_f64() * 1000.0,
                            max_seq_budget,
                            chunk.tokens.len(),
                            before,
                            after,
                        );
                        let nan_data: Vec<f32> = vec![f32::NAN; ldim];
                        let mut prev_latent = Tensor::from_vec(nan_data, (1, 1, ldim), &CPU)
                            .map_err(|e| format!("init bos tensor: {e}"))?;
                        let mut rng = Rng::new(temperature, seed);
                        let mut eos_countdown: Option<usize> = None;
                        for step in 0..chunk.max_frames {
                            let t0 = Instant::now();
                            let (next_latent, is_eos) = model
                                .generate_step(&mut tts_state, &prev_latent, &mut rng)
                                .map_err(|e| format!("generate_step: {e}"))?;
                            step_total_ns
                                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            step_count.fetch_add(1, Ordering::Relaxed);

                            // Hand the latent to the decoder; bail if the
                            // consumer went away (JNI caller closed early).
                            if latent_tx.send((next_latent.clone(), step)).is_err() {
                                return Ok(());
                            }
                            if is_eos && eos_countdown.is_none() {
                                eos_countdown = Some(chunk.frames_after_eos);
                            }
                            if let Some(c) = eos_countdown.as_mut() {
                                if *c == 0 {
                                    break;
                                }
                                *c -= 1;
                            }
                            prev_latent = next_latent;
                        }
                    }
                    Ok(())
                })
                .map_err(|e| format!("spawn backbone: {e}"))?
        };

        let decoder = {
            let model = self.model.clone();
            std::thread::Builder::new()
                .name("ptts-decoder".into())
                .spawn(move || -> Result<(), String> {
                    let mut mimi_state = model
                        .init_mimi_state(1, 250)
                        .map_err(|e| format!("init_mimi_state: {e}"))?;
                    while let Ok((latent, _step)) = latent_rx.recv() {
                        let audio_chunk = model
                            .decode_latent(&latent, &mut mimi_state)
                            .map_err(|e| format!("decode_latent: {e}"))?;
                        let audio = audio_chunk
                            .narrow(0, ..1)
                            .and_then(|t| t.contiguous())
                            .map_err(|e| format!("slice audio: {e}"))?;
                        let pcm: Vec<f32> =
                            audio.to_vec().map_err(|e| format!("audio.to_vec: {e}"))?;
                        if pcm.is_empty() {
                            continue;
                        }
                        if pcm_tx.send(pcm).is_err() {
                            return Ok(());
                        }
                    }
                    Ok(())
                })
                .map_err(|e| format!("spawn decoder: {e}"))?
        };

        self.run = Some(Run {
            pcm_rx,
            backbone: Some(backbone),
            decoder: Some(decoder),
            step_count,
            step_total_ns,
            started_at: Instant::now(),
            first_chunk_at: None,
            duration_s: 0.0,
        });
        self.stats = Stats::default();
        Ok(())
    }

    /// Block until the next PCM chunk is ready, or return `Ok(None)` on EOS.
    pub fn next_chunk(&mut self) -> Result<Option<Vec<f32>>, String> {
        let sr = self.sample_rate as f64;
        let Some(run) = self.run.as_mut() else { return Ok(None) };
        match run.pcm_rx.recv() {
            Ok(pcm) => {
                run.duration_s += pcm.len() as f64 / sr;
                if run.first_chunk_at.is_none() {
                    run.first_chunk_at = Some(Instant::now());
                }
                Ok(Some(pcm))
            }
            Err(_) => {
                // Channel closed → both threads finished (or errored). Drain
                // their Results so we propagate errors, then finalize stats.
                self.finalize_run();
                Ok(None)
            }
        }
    }

    fn finalize_run(&mut self) {
        let Some(mut run) = self.run.take() else { return };
        let bb_res = run.backbone.take().and_then(|h| h.join().ok());
        let dec_res = run.decoder.take().and_then(|h| h.join().ok());

        let total_elapsed_s = run.started_at.elapsed().as_secs_f64();
        let duration_s = run.duration_s;
        let step_count = run.step_count.load(Ordering::Relaxed);
        let step_total_ns = run.step_total_ns.load(Ordering::Relaxed);
        let avg_step_ms =
            if step_count > 0 { (step_total_ns as f64 / step_count as f64) / 1.0e6 } else { 0.0 };
        let first_audio_s = run
            .first_chunk_at
            .map(|t| t.saturating_duration_since(run.started_at).as_secs_f64())
            .unwrap_or(0.0);
        let rtf = if total_elapsed_s > 0.0 { duration_s / total_elapsed_s } else { 0.0 };
        self.stats = Stats {
            total_elapsed_s,
            duration_s,
            rtf,
            avg_step_ms,
            first_audio_s,
            peak_rss_mb: peak_rss_mb(),
        };
        info!(
            "generated {:.2}s in {:.2}s (RTF={:.3}), {} steps, avg backbone {:.1} ms, \
             first audio {:.2}s, peak RSS {:.1} MB",
            duration_s,
            total_elapsed_s,
            rtf,
            step_count,
            avg_step_ms,
            first_audio_s,
            self.stats.peak_rss_mb,
        );
        if let Some(Err(e)) = bb_res {
            log::warn!("backbone thread error: {e}");
        }
        if let Some(Err(e)) = dec_res {
            log::warn!("decoder thread error: {e}");
        }
    }

    fn stop_run(&mut self) {
        if self.run.is_some() {
            // Drop the PCM receiver by clearing `run`; this unblocks pcm_tx
            // sends on the decoder thread so it exits, which closes latent_rx
            // and unblocks the backbone.
            self.run = None;
        }
    }

    pub fn stats(&self) -> [f32; 6] {
        [
            self.stats.rtf as f32,
            self.stats.avg_step_ms as f32,
            self.stats.total_elapsed_s as f32,
            self.stats.duration_s as f32,
            self.stats.first_audio_s as f32,
            self.stats.peak_rss_mb as f32,
        ]
    }
}

/// Load an `embeddings_v2/{voice}.safetensors` file and build a small TTSState
/// whose flow LM KV cache matches the file exactly (no extra budget). Mirrors
/// `pocket-tts-wasm::Model::add_voice_`.
fn load_voice_kv_cache(path: &std::path::Path) -> Result<State, String> {
    let tensors = xn::safetensors::load_from_file(path, &CPU)
        .map_err(|e| format!("load voice safetensors: {e}"))?;
    const NUM_LAYERS: usize = 6;
    let mut layer_states = Vec::with_capacity(NUM_LAYERS);
    let mut primed_seq_lens = [0usize; NUM_LAYERS];
    for i in 0..NUM_LAYERS {
        let cache_name = format!("transformer.layers.{i}.self_attn/cache");
        let cache = match tensors.get(&cache_name) {
            Some(TypedTensor::F32(t)) => t,
            _ => return Err(format!("expected f32 tensor '{cache_name}' in voice file")),
        };
        let (two, batch, seq_len, num_heads, head_dim) =
            cache.dims5().map_err(|e| format!("cache shape: {e}"))?;
        primed_seq_lens[i] = seq_len;
        if two != 2 {
            return Err("voice cache first dim must be 2 (k/v)".into());
        }
        let k_cache = cache
            .narrow(0, 0..1)
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape((batch, seq_len, num_heads, head_dim)))
            .map_err(|e| format!("split k: {e}"))?;
        let v_cache = cache
            .narrow(0, 1..2)
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape((batch, seq_len, num_heads, head_dim)))
            .map_err(|e| format!("split v: {e}"))?;
        layer_states.push(LayerAttentionState::FlowLm(StreamingMHAState {
            k_cache,
            v_cache,
            current_end: seq_len,
        }));
    }
    info!(
        "voice state: primed seq_len per layer = {:?} (current_end starts at this value; \
         prompt_text then appends new K/V to positions seq_len..seq_len+n_text_tokens)",
        primed_seq_lens,
    );
    Ok(TTSState {
        flow_lm_state: FlowLMState {
            transformer_state: StreamingTransformerState { layer_states },
        },
    })
}

/// Grow a cached voice state's KV buffers to the requested sequence budget,
/// preserving the primed prefix. Mirrors `pocket-tts-wasm::resize_tts_state`.
fn resize_tts_state(cached: &State, new_seq_budget: usize) -> Result<State, String> {
    let mut new_layer_states = Vec::new();
    for layer_state in cached.flow_lm_state.transformer_state.layer_states.iter() {
        match layer_state {
            LayerAttentionState::FlowLm(mha) => {
                let current_end = mha.current_end;
                let b = mha.k_cache.dim(0usize).map_err(|e| e.to_string())?;
                let h = mha.k_cache.dim(2usize).map_err(|e| e.to_string())?;
                let d = mha.k_cache.dim(3usize).map_err(|e| e.to_string())?;
                let new_k = Tensor::zeros((b, new_seq_budget, h, d), &CPU)
                    .map_err(|e| format!("alloc k: {e}"))?;
                let new_v = Tensor::zeros((b, new_seq_budget, h, d), &CPU)
                    .map_err(|e| format!("alloc v: {e}"))?;
                if current_end > 0 {
                    let k_used = mha
                        .k_cache
                        .narrow(1, 0..current_end)
                        .and_then(|t| t.contiguous())
                        .map_err(|e| format!("slice k: {e}"))?;
                    let v_used = mha
                        .v_cache
                        .narrow(1, 0..current_end)
                        .and_then(|t| t.contiguous())
                        .map_err(|e| format!("slice v: {e}"))?;
                    new_k.slice_set(&k_used, 1usize, 0).map_err(|e| format!("slice_set k: {e}"))?;
                    new_v.slice_set(&v_used, 1usize, 0).map_err(|e| format!("slice_set v: {e}"))?;
                }
                new_layer_states.push(LayerAttentionState::FlowLm(StreamingMHAState {
                    k_cache: new_k,
                    v_cache: new_v,
                    current_end,
                }));
            }
            other => new_layer_states.push(other.clone()),
        }
    }
    Ok(TTSState {
        flow_lm_state: FlowLMState {
            transformer_state: StreamingTransformerState { layer_states: new_layer_states },
        },
    })
}

fn layer_current_end(state: &State) -> Vec<usize> {
    state
        .flow_lm_state
        .transformer_state
        .layer_states
        .iter()
        .map(|s| match s {
            LayerAttentionState::FlowLm(mha) => mha.current_end,
            _ => 0,
        })
        .collect()
}

fn peak_rss_mb() -> f64 {
    #[cfg(target_family = "unix")]
    unsafe {
        let mut usage = core::mem::MaybeUninit::<libc::rusage>::uninit();
        if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) != 0 {
            return 0.0;
        }
        // Linux/Android report kB; we target Android so this is always kB.
        usage.assume_init().ru_maxrss as f64 / 1024.0
    }
    #[cfg(not(target_family = "unix"))]
    {
        0.0
    }
}

// A thread-safe wrapper handed over to JNI as a raw pointer.
pub struct Handle(pub Mutex<Session>);

// ----- JNI glue -------------------------------------------------------------

#[cfg(target_os = "android")]
mod jni_bindings {
    use super::*;
    use jni::JNIEnv;
    use jni::objects::{JClass, JString};
    use jni::sys::{jfloatArray, jint, jlong};

    fn android_log_init() {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let cfg = android_logger::Config::default()
                .with_max_level(log::LevelFilter::Info)
                .with_tag("ptts");
            android_logger::init_once(cfg);
        });
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_sh_gradium_ptts_Ptts_nativeInit(
        mut env: JNIEnv,
        _class: JClass,
        weights_dir: JString,
        voice: JString,
    ) -> jlong {
        android_log_init();
        let weights_dir: String = match env.get_string(&weights_dir) {
            Ok(s) => s.into(),
            Err(_) => return 0,
        };
        let voice: String = match env.get_string(&voice) {
            Ok(s) => s.into(),
            Err(_) => return 0,
        };
        match Session::new(&weights_dir, &voice) {
            Ok(s) => Box::into_raw(Box::new(Handle(Mutex::new(s)))) as jlong,
            Err(e) => {
                let _ = env.throw_new("java/lang/RuntimeException", e);
                0
            }
        }
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_sh_gradium_ptts_Ptts_nativeGenerate(
        mut env: JNIEnv,
        _class: JClass,
        handle: jlong,
        text: JString,
        temperature: jni::sys::jfloat,
        seed: jlong,
    ) {
        if handle == 0 {
            return;
        }
        let text: String = match env.get_string(&text) {
            Ok(s) => s.into(),
            Err(_) => return,
        };
        let h = unsafe { &*(handle as *const Handle) };
        let mut sess = h.0.lock().unwrap();
        if let Err(e) = sess.start(&text, temperature as f32, seed as u64) {
            let _ = env.throw_new("java/lang/RuntimeException", e);
        }
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_sh_gradium_ptts_Ptts_nativeNextChunk(
        mut env: JNIEnv,
        _class: JClass,
        handle: jlong,
    ) -> jfloatArray {
        if handle == 0 {
            return std::ptr::null_mut();
        }
        let h = unsafe { &*(handle as *const Handle) };
        let mut sess = h.0.lock().unwrap();
        match sess.next_chunk() {
            Ok(None) => std::ptr::null_mut(),
            Ok(Some(pcm)) => {
                let arr = env.new_float_array(pcm.len() as jni::sys::jsize).unwrap();
                env.set_float_array_region(&arr, 0, &pcm).unwrap();
                arr.into_raw()
            }
            Err(e) => {
                let _ = env.throw_new("java/lang/RuntimeException", e);
                std::ptr::null_mut()
            }
        }
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_sh_gradium_ptts_Ptts_nativeStats(
        env: JNIEnv,
        _class: JClass,
        handle: jlong,
    ) -> jfloatArray {
        if handle == 0 {
            return std::ptr::null_mut();
        }
        let h = unsafe { &*(handle as *const Handle) };
        let sess = h.0.lock().unwrap();
        let s = sess.stats();
        let arr = env.new_float_array(s.len() as jni::sys::jsize).unwrap();
        env.set_float_array_region(&arr, 0, &s).unwrap();
        arr.into_raw()
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_sh_gradium_ptts_Ptts_nativeSampleRate(
        _env: JNIEnv,
        _class: JClass,
        handle: jlong,
    ) -> jint {
        if handle == 0 {
            return 0;
        }
        let h = unsafe { &*(handle as *const Handle) };
        let sess = h.0.lock().unwrap();
        sess.sample_rate() as jint
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_sh_gradium_ptts_Ptts_nativeSetThreads(
        _env: JNIEnv,
        _class: JClass,
        handle: jlong,
        n: jint,
    ) {
        if handle == 0 {
            return;
        }
        let h = unsafe { &*(handle as *const Handle) };
        let sess = h.0.lock().unwrap();
        sess.set_threads(n as usize);
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_sh_gradium_ptts_Ptts_nativeNumCpus(
        _env: JNIEnv,
        _class: JClass,
    ) -> jint {
        xn::get_num_cpus() as jint
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_sh_gradium_ptts_Ptts_nativeFree(
        _env: JNIEnv,
        _class: JClass,
        handle: jlong,
    ) {
        if handle == 0 {
            return;
        }
        let _ = unsafe { Box::from_raw(handle as *mut Handle) };
    }
}
