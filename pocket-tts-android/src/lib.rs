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
use ptts::tts_model::{
    TTSConfig, TTSModel, TTSState, prepare_text_prompt, split_into_best_sentences,
};
use std::sync::Mutex;
use std::time::Instant;
use tokenizer::Unigram;
use xn::nn::VB;
use xn::{CPU, CpuDevice, Tensor, Unquantized};

type Model = TTSModel<Unquantized<f32, CpuDevice>>;
type State = TTSState<Unquantized<f32, CpuDevice>>;
type Mimi = ptts::mimi::MimiState<f32, CpuDevice>;

const VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

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

struct GenLoop {
    chunks: std::vec::IntoIter<Chunk>,
    base_state: State,
    cur: Option<ActiveChunk>,
}

struct ActiveChunk {
    tts_state: State,
    mimi_state: Mimi,
    prev_latent: Tensor<f32, CpuDevice>,
    rng: Rng,
    max_frames: usize,
    frames_after_eos: usize,
    eos_countdown: Option<usize>,
    step: usize,
}

pub struct Session {
    model: Model,
    voice_path: std::path::PathBuf,
    seq_budget: usize,
    prompted_base: Option<State>,
    gen_loop: Option<GenLoop>,
    stats: Stats,
    started_at: Option<Instant>,
    first_chunk_seen: bool,
    step_count: usize,
    step_total_ms: f64,
    temperature: f32,
    seed: u64,
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
            return Err(format!(
                "unknown voice '{voice}'. Available: {}",
                VOICES.join(", ")
            ));
        }
        let dir = std::path::Path::new(weights_dir);
        let model_path = dir.join("tts_b6369a24.safetensors");
        let tokenizer_path = dir.join("tokenizer.model");
        let voice_path = dir.join("embeddings").join(format!("{voice}.safetensors"));
        for p in [&model_path, &tokenizer_path, &voice_path] {
            if !p.exists() {
                return Err(format!("missing file: {}", p.display()));
            }
        }

        let tok = Unigram::from_file(tokenizer_path.to_str().unwrap())
            .map_err(|e| format!("tokenizer load: {e}"))?;

        info!(
            "backends: avx={} neon={} simd128={} f16c={}",
            xn::with_avx(),
            xn::with_neon(),
            xn::with_simd128(),
            xn::with_f16c(),
        );

        let cfg = TTSConfig::v202601(0.7);
        let vb = VB::load_with_key_map(&[&model_path], CPU, remap_key)
            .map_err(|e| format!("load weights: {e}"))?
            .root();
        let model: Model =
            Model::load(&vb, Box::new(tok), &cfg).map_err(|e| format!("model load: {e}"))?;

        Ok(Self {
            model,
            voice_path,
            seq_budget: 0,
            prompted_base: None,
            gen_loop: None,
            stats: Stats::default(),
            started_at: None,
            first_chunk_seen: false,
            step_count: 0,
            step_total_ms: 0.0,
            temperature: 0.7,
            seed: 4242424242424242,
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.model.sample_rate()
    }

    /// Tokenize the input, build per-chunk token streams and prime a fresh
    /// base state with voice conditioning. Must be called before the first
    /// `next_chunk`.
    pub fn start(&mut self, text: &str, temperature: f32, seed: u64) -> Result<(), String> {
        self.temperature = temperature;
        self.seed = seed;

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

        let need_reprime =
            self.prompted_base.is_none() || max_seq_budget > self.seq_budget;
        if need_reprime {
            let mut base_state = self
                .model
                .init_flow_lm_state(1, max_seq_budget)
                .map_err(|e| format!("init state: {e}"))?;
            let voice_vb = VB::load(&[&self.voice_path], CPU)
                .map_err(|e| format!("load voice: {e}"))?;
            let voice_names = voice_vb.tensor_names();
            let voice_key: String = voice_names
                .first()
                .ok_or_else(|| "empty voice file".to_string())?
                .to_string();
            let voice_shape = voice_vb
                .shape(&voice_key)
                .ok_or_else(|| format!("voice tensor '{voice_key}' missing"))?;
            let dims = voice_shape.dims().to_vec();
            let voice_emb: Tensor<f32, CpuDevice> = voice_vb
                .tensor(&voice_key, voice_shape)
                .map_err(|e| format!("read voice tensor: {e}"))?;
            let voice_emb = if dims.len() == 2 {
                voice_emb
                    .reshape((1, dims[0], dims[1]))
                    .map_err(|e| format!("reshape voice: {e}"))?
            } else {
                voice_emb
            };
            self.model
                .prompt_audio(&mut base_state, &voice_emb)
                .map_err(|e| format!("prompt_audio: {e}"))?;
            self.prompted_base = Some(base_state);
            self.seq_budget = max_seq_budget;
        }

        let base_state = self.prompted_base.as_ref().unwrap().clone();
        self.gen_loop = Some(GenLoop {
            chunks: chunks.into_iter(),
            base_state,
            cur: None,
        });
        self.stats = Stats::default();
        self.started_at = Some(Instant::now());
        self.first_chunk_seen = false;
        self.step_count = 0;
        self.step_total_ms = 0.0;
        Ok(())
    }

    /// Produce the next PCM chunk. Returns Ok(None) at end-of-stream.
    pub fn next_chunk(&mut self) -> Result<Option<Vec<f32>>, String> {
        loop {
            let Some(gen_loop) = self.gen_loop.as_mut() else {
                return Ok(None);
            };

            if gen_loop.cur.is_none() {
                let next_chunk = match gen_loop.chunks.next() {
                    None => {
                        self.finish_stats();
                        self.gen_loop = None;
                        return Ok(None);
                    }
                    Some(c) => c,
                };
                let mut tts_state = gen_loop.base_state.clone();
                self.model
                    .prompt_text(&mut tts_state, &next_chunk.tokens)
                    .map_err(|e| format!("prompt_text: {e}"))?;
                let mimi_state = self
                    .model
                    .init_mimi_state(1, 250)
                    .map_err(|e| format!("init_mimi_state: {e}"))?;
                let ldim = self.model.flow_lm.ldim;
                let nan_data: Vec<f32> = vec![f32::NAN; ldim];
                let prev_latent = Tensor::from_vec(nan_data, (1, 1, ldim), &CPU)
                    .map_err(|e| format!("init bos tensor: {e}"))?;
                gen_loop.cur = Some(ActiveChunk {
                    tts_state,
                    mimi_state,
                    prev_latent,
                    rng: Rng::new(self.temperature, self.seed),
                    max_frames: next_chunk.max_frames,
                    frames_after_eos: next_chunk.frames_after_eos,
                    eos_countdown: None,
                    step: 0,
                });
            }

            let cur = gen_loop.cur.as_mut().unwrap();
            if cur.step >= cur.max_frames {
                gen_loop.cur = None;
                continue;
            }

            let step_start = Instant::now();
            let (next_latent, is_eos) = self
                .model
                .generate_step(&mut cur.tts_state, &cur.prev_latent, &mut cur.rng)
                .map_err(|e| format!("generate_step: {e}"))?;
            let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
            self.step_count += 1;
            self.step_total_ms += step_ms;

            let audio_chunk = self
                .model
                .decode_latent(&next_latent, &mut cur.mimi_state)
                .map_err(|e| format!("decode_latent: {e}"))?;
            let audio = audio_chunk
                .narrow(0, ..1)
                .and_then(|t| t.contiguous())
                .map_err(|e| format!("slice audio: {e}"))?;
            let pcm: Vec<f32> = audio.to_vec().map_err(|e| format!("audio.to_vec: {e}"))?;

            if is_eos && cur.eos_countdown.is_none() {
                cur.eos_countdown = Some(cur.frames_after_eos);
            }
            let done_this_chunk = if let Some(c) = cur.eos_countdown.as_mut() {
                if *c == 0 {
                    true
                } else {
                    *c -= 1;
                    false
                }
            } else {
                false
            };

            cur.prev_latent = next_latent;
            cur.step += 1;
            if done_this_chunk {
                gen_loop.cur = None;
            }

            self.stats.duration_s += pcm.len() as f64 / self.sample_rate() as f64;
            if !self.first_chunk_seen && !pcm.is_empty() {
                if let Some(started) = self.started_at {
                    self.stats.first_audio_s = started.elapsed().as_secs_f64();
                }
                self.first_chunk_seen = true;
            }

            if pcm.is_empty() {
                continue;
            }
            return Ok(Some(pcm));
        }
    }

    fn finish_stats(&mut self) {
        if let Some(started) = self.started_at {
            self.stats.total_elapsed_s = started.elapsed().as_secs_f64();
        }
        if self.stats.total_elapsed_s > 0.0 {
            self.stats.rtf = self.stats.duration_s / self.stats.total_elapsed_s;
        }
        if self.step_count > 0 {
            self.stats.avg_step_ms = self.step_total_ms / self.step_count as f64;
        }
        self.stats.peak_rss_mb = peak_rss_mb();
        info!(
            "generated {:.2}s in {:.2}s (RTF={:.3}), avg step {:.2}ms, peak RSS {:.1} MB",
            self.stats.duration_s,
            self.stats.total_elapsed_s,
            self.stats.rtf,
            self.stats.avg_step_ms,
            self.stats.peak_rss_mb,
        );
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
