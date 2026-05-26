// Android JNI wrapper for xn-moshi streaming ASR.
//
// Public Kotlin surface (class `sh.gradium.xnmoshi.Asr`):
//   nativeInit(mimi_path, lm_path, tokenizer_path, config_path, dtype) -> handle
//   nativeStep(handle, pcm: FloatArray) -> String     // newly recognized text
//   nativeReset(handle)
//   nativeFrameSize() -> Int
//   nativeSampleRate() -> Int
//   nativeFree(handle)
//
// One handle = one streaming ASR session. The model is parameterized over
// `Q: BackendQ` (chosen via the dtype string at init) and erased behind a
// boxed `dyn Session` trait so the JNI side only deals with a raw pointer.

mod sentencepiece;

use anyhow::{Result, anyhow};
use sentencepiece::SpDecoder;
use std::sync::Mutex;
use xn::streaming::{StreamMask, StreamTensor};
use xn::{BackendQ, CpuDevice, DTypeQ, Tensor, Unquantized};
use xn_moshi::asr::{Asr, AsrState, AsrWord};

// Streaming-ASR constants from xn-moshi/examples/moshi.rs (ASR section).
const SAMPLE_RATE: usize = 24_000;
const FRAME_SIZE: usize = 1_920;
// Default if the config file doesn't carry `asr_delay_in_tokens`. Matches
// the CLI example's hardcoded 2.5s * 24000 / 1920.
const DEFAULT_ASR_DELAY_IN_TOKENS: usize = 31;
/// The xn_moshi `moshi::Config` struct silently drops `asr_delay_in_tokens`
/// (it isn't part of its serde fields), so we sniff it separately. Returns
/// the default if the config path is empty or the field is missing.
fn asr_delay_from_config(config_path: &str) -> Result<usize> {
    if config_path.is_empty() {
        return Ok(DEFAULT_ASR_DELAY_IN_TOKENS);
    }
    let s = std::fs::read_to_string(config_path)
        .map_err(|e| anyhow!("reading config {config_path}: {e}"))?;
    let v: serde_json::Value =
        serde_json::from_str(&s).map_err(|e| anyhow!("parsing config: {e}"))?;
    Ok(v.get("asr_delay_in_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(DEFAULT_ASR_DELAY_IN_TOKENS))
}

pub trait Session: Send {
    fn step(&mut self, pcm: &[f32]) -> Result<String>;
    fn reset(&mut self) -> Result<()>;
}

struct AsrSession<Q: BackendQ> {
    state: AsrState<Q>,
    sp: SpDecoder,
    decoded_text: String,
    /// Counts processed frames for periodic audio-level logging.
    frames_since_log: usize,
    /// Accumulated rms over [frames_since_log] frames, for averaging.
    rms_accum: f64,
    /// True until the first word has been emitted; flips the leading-space
    /// stripping rule.
    first_word: bool,
}

impl<Q: BackendQ> AsrSession<Q> {
    fn new(
        mimi_path: &str,
        lm_path: &str,
        tokenizer_path: &str,
        config_path: &str,
        language: &str,
        dev: Q::B,
    ) -> Result<Self> {
        let asr_delay_in_tokens = asr_delay_from_config(config_path)?;
        let config_opt = if config_path.is_empty() { None } else { Some(config_path) };
        let asr: Asr<Q> = Asr::load(mimi_path, lm_path, config_opt, asr_delay_in_tokens, 0.0, dev)
            .map_err(|e| anyhow!("Asr::load: {e}"))?;
        let mut state = asr.init_state(1).map_err(|e| anyhow!("Asr::init_state: {e}"))?;

        // `Asr::load` only feeds `delay` to the conditioner, so any
        // `languages_in_segment` conditioner falls back to its default value
        // (often "other" — biases multilingual models toward whichever
        // language dominated training). Build the right condition and install
        // it via the only public path that lets us mutate `state.condition`:
        // `reset_batch_idx`. Safe to call here because the state is fresh.
        if !language.is_empty() {
            let delay = -0.08 * asr_delay_in_tokens as f64;
            let cond = state
                .condition_sum(Some(language), delay)
                .map_err(|e| anyhow!("condition_sum: {e}"))?;
            match cond.as_ref() {
                Some(c) => {
                    state
                        .reset_batch_idx(0, None, Some(c))
                        .map_err(|e| anyhow!("reset_batch_idx with lang cond: {e}"))?;
                    log::info!("language conditioner installed: {language}");
                }
                None => log::info!(
                    "language={language} requested but config has no conditioners — ignored"
                ),
            }
        }

        let sp =
            SpDecoder::from_file(tokenizer_path).map_err(|e| anyhow!("tokenizer load: {e}"))?;
        Ok(Self {
            state,
            sp,
            decoded_text: String::new(),
            frames_since_log: 0,
            rms_accum: 0.0,
            first_word: true,
        })
    }
}

impl<Q: BackendQ> Session for AsrSession<Q> {
    fn step(&mut self, pcm: &[f32]) -> Result<String> {
        if pcm.len() != FRAME_SIZE {
            return Err(anyhow!("expected frame of {FRAME_SIZE} samples, got {}", pcm.len()));
        }
        // Audio-level sanity check: log mean rms / peak roughly once a second
        // (every ~12 frames at 24 kHz / 1920 = 12.5 fps). If this stays near
        // zero, the mic isn't actually delivering samples.
        let mut peak: f32 = 0.0;
        let mut sumsq: f64 = 0.0;
        for &s in pcm {
            sumsq += (s as f64) * (s as f64);
            if s.abs() > peak {
                peak = s.abs();
            }
        }
        let rms = (sumsq / pcm.len() as f64).sqrt();
        self.rms_accum += rms;
        self.frames_since_log += 1;
        if self.frames_since_log >= 12 {
            log::info!(
                "audio: avg_rms={:.4} peak={:.4} ({} frames)",
                self.rms_accum / self.frames_since_log as f64,
                peak,
                self.frames_since_log,
            );
            self.frames_since_log = 0;
            self.rms_accum = 0.0;
        }

        let dev = self.state.device().clone();
        let audio: Tensor<f32, Q::B> = Tensor::from_vec(pcm.to_vec(), (1, 1, FRAME_SIZE), &dev)
            .map_err(|e| anyhow!("audio tensor: {e}"))?;
        let pcm_t = StreamTensor::from_tensor(audio);
        let mask = StreamMask::all_active(1);
        let t0 = std::time::Instant::now();
        // The callback fires once per LM step *before* the forward pass — the
        // `text_tokens` slice is what's being fed in (i.e. the previously
        // sampled token), so it tells us what the model has been emitting.
        let step_results = self
            .state
            .step_pcm(&pcm_t, &mask, |_items, text_tokens, _audio_ids| {
                // text_tokens[0] is the previously sampled text token being
                // fed back in (the prediction from the prior step). PAD=3,
                // EOP=0, SILENCE_PAD=4 mean "no word"; any other id is a
                // SentencePiece piece that would have been added to a word.
                log::debug!("lm fed_token={}", text_tokens[0]);
            })
            .map_err(|e| anyhow!("step_pcm: {e}"))?;
        let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let total_words: usize = step_results.iter().map(|sr| sr.words.len()).sum();
        log::debug!("step_pcm: {step_ms:.1}ms results={} words={total_words}", step_results.len());

        let mut delta = String::new();
        for sr in step_results {
            for word in sr.words {
                if let AsrWord::Word { tokens, batch_idx, start_time } = word
                    && batch_idx == 0
                {
                    log::info!("word @ {start_time:.2}s: tokens={tokens:?}",);
                    // Decode just this word's tokens. Each word-initial piece
                    // carries the SP meta-space (▁), so concatenating decoded
                    // word strings naturally yields the correct spacing.
                    let s = self.sp.decode_piece_ids(&tokens);
                    log::info!("  decoded: {s:?}");
                    if self.first_word {
                        // Strip a single leading space from the very first
                        // emission (matches `print!` UX from the CLI).
                        delta.push_str(s.trim_start_matches(' '));
                        self.first_word = false;
                    } else {
                        // If the new word's decoded string doesn't already
                        // start with a space (some tokenizers, ours included,
                        // produce bare continuation pieces), insert one.
                        if !s.starts_with(' ') {
                            delta.push(' ');
                        }
                        delta.push_str(&s);
                    }
                }
            }
        }
        self.decoded_text.push_str(&delta);
        Ok(delta)
    }

    fn reset(&mut self) -> Result<()> {
        self.state.reset_state().map_err(|e| anyhow!("reset_state: {e}"))?;
        self.decoded_text.clear();
        self.first_word = true;
        self.frames_since_log = 0;
        self.rms_accum = 0.0;
        Ok(())
    }
}

fn make_session(
    dtype: DTypeQ,
    mimi_path: &str,
    lm_path: &str,
    tokenizer_path: &str,
    config_path: &str,
    language: &str,
) -> Result<Box<dyn Session>> {
    use xn::quantized::*;
    let dev = CpuDevice;
    let m = mimi_path;
    let l = lm_path;
    let t = tokenizer_path;
    let c = config_path;
    let g = language;
    // The set of dtypes supported on a CPU-only Android build. Mirrors the
    // CPU arm of `xn::Runner::run` minus BF16/F16 (xn-core marks those "not
    // yet supported on CPU") and minus FP8 (CUDA only).
    Ok(match dtype {
        DTypeQ::F32 => {
            Box::new(AsrSession::<Unquantized<f32, CpuDevice>>::new(m, l, t, c, g, dev)?)
        }
        DTypeQ::Q4_0 => Box::new(AsrSession::<Q40F32>::new(m, l, t, c, g, dev)?),
        DTypeQ::Q4_1 => Box::new(AsrSession::<Q41F32>::new(m, l, t, c, g, dev)?),
        DTypeQ::Q5_0 => Box::new(AsrSession::<Q50F32>::new(m, l, t, c, g, dev)?),
        DTypeQ::Q5_1 => Box::new(AsrSession::<Q51F32>::new(m, l, t, c, g, dev)?),
        DTypeQ::Q8_0 => Box::new(AsrSession::<Q80F32>::new(m, l, t, c, g, dev)?),
        DTypeQ::Q8_1 => Box::new(AsrSession::<Q81F32>::new(m, l, t, c, g, dev)?),
        DTypeQ::Q2K => Box::new(AsrSession::<Q2kF32>::new(m, l, t, c, g, dev)?),
        DTypeQ::Q3K => Box::new(AsrSession::<Q3kF32>::new(m, l, t, c, g, dev)?),
        DTypeQ::Q4K => Box::new(AsrSession::<Q4kF32>::new(m, l, t, c, g, dev)?),
        DTypeQ::Q5K => Box::new(AsrSession::<Q5kF32>::new(m, l, t, c, g, dev)?),
        DTypeQ::Q6K => Box::new(AsrSession::<Q6kF32>::new(m, l, t, c, g, dev)?),
        DTypeQ::Q8K => Box::new(AsrSession::<Q8kF32>::new(m, l, t, c, g, dev)?),
        other => return Err(anyhow!("dtype {other:?} is not supported on CPU/Android")),
    })
}

pub struct Handle(pub Mutex<Box<dyn Session>>);

pub fn frame_size() -> usize {
    FRAME_SIZE
}

pub fn sample_rate() -> usize {
    SAMPLE_RATE
}

// ---- JNI -------------------------------------------------------------------

#[cfg(target_os = "android")]
mod jni_bindings {
    use super::*;
    use jni::JNIEnv;
    use jni::objects::{JClass, JFloatArray, JString};
    use jni::sys::{jint, jlong, jstring};
    use std::str::FromStr;
    use std::sync::Once;

    fn log_init() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            android_logger::init_once(
                android_logger::Config::default()
                    .with_max_level(log::LevelFilter::Debug)
                    .with_tag("xn-moshi"),
            );
        });
    }

    fn threads_init() {
        // gemm's CPU backend reads from xn's set_num_threads atomic. Default
        // is 1 — devastating for inference RTF. Pixel-class chips are
        // big.LITTLE: don't include the slowest cluster (typically half the
        // cores), it drags the gemm down. Pick a sensible floor of 4.
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let hw = xn::get_num_cpus();
            let n = hw.saturating_sub(hw / 2).max(4).min(hw).max(1);
            xn::set_num_threads(n);
            log::info!("threads: hw={hw} using={n}");
        });
    }

    fn throw(env: &mut JNIEnv, msg: &str) {
        let _ = env.throw_new("java/lang/RuntimeException", msg);
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_sh_gradium_xnmoshi_Asr_nativeInit(
        mut env: JNIEnv,
        _class: JClass,
        mimi_path: JString,
        lm_path: JString,
        tokenizer_path: JString,
        config_path: JString,
        dtype: JString,
        language: JString,
    ) -> jlong {
        log_init();
        threads_init();
        let mimi_path: String = match env.get_string(&mimi_path) {
            Ok(s) => s.into(),
            Err(_) => return 0,
        };
        let lm_path: String = match env.get_string(&lm_path) {
            Ok(s) => s.into(),
            Err(_) => return 0,
        };
        let tokenizer_path: String = match env.get_string(&tokenizer_path) {
            Ok(s) => s.into(),
            Err(_) => return 0,
        };
        // Empty config_path => use the default `lm::Config::stt_2_6b()`.
        let config_path: String = match env.get_string(&config_path) {
            Ok(s) => s.into(),
            Err(_) => String::new(),
        };
        let dtype: String = match env.get_string(&dtype) {
            Ok(s) => s.into(),
            Err(_) => return 0,
        };
        let dtype = match DTypeQ::from_str(&dtype) {
            Ok(d) => d,
            Err(e) => {
                throw(&mut env, &format!("invalid dtype '{dtype}': {e}"));
                return 0;
            }
        };
        // Empty language => keep model's default condition (likely "other").
        let language: String = match env.get_string(&language) {
            Ok(s) => s.into(),
            Err(_) => String::new(),
        };
        log::info!(
            "init: dtype={dtype:?} cpus={} neon={} fp16-via-fp32={}",
            xn::get_num_cpus(),
            xn::with_neon(),
            xn::with_f16c(),
        );
        match make_session(dtype, &mimi_path, &lm_path, &tokenizer_path, &config_path, &language) {
            Ok(s) => Box::into_raw(Box::new(Handle(Mutex::new(s)))) as jlong,
            Err(e) => {
                throw(&mut env, &format!("init: {e:#}"));
                0
            }
        }
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_sh_gradium_xnmoshi_Asr_nativeStep(
        mut env: JNIEnv,
        _class: JClass,
        handle: jlong,
        pcm: JFloatArray,
    ) -> jstring {
        if handle == 0 {
            return std::ptr::null_mut();
        }
        let len = match env.get_array_length(&pcm) {
            Ok(n) => n as usize,
            Err(e) => {
                throw(&mut env, &format!("array length: {e}"));
                return std::ptr::null_mut();
            }
        };
        let mut buf = vec![0f32; len];
        if let Err(e) = env.get_float_array_region(&pcm, 0, &mut buf) {
            throw(&mut env, &format!("array region: {e}"));
            return std::ptr::null_mut();
        }
        let h = unsafe { &*(handle as *const Handle) };
        let mut sess = h.0.lock().unwrap();
        match sess.step(&buf) {
            Ok(s) => match env.new_string(s) {
                Ok(j) => j.into_raw(),
                Err(_) => std::ptr::null_mut(),
            },
            Err(e) => {
                throw(&mut env, &format!("step: {e:#}"));
                std::ptr::null_mut()
            }
        }
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_sh_gradium_xnmoshi_Asr_nativeReset(
        mut env: JNIEnv,
        _class: JClass,
        handle: jlong,
    ) {
        if handle == 0 {
            return;
        }
        let h = unsafe { &*(handle as *const Handle) };
        let mut sess = h.0.lock().unwrap();
        if let Err(e) = sess.reset() {
            throw(&mut env, &format!("reset: {e:#}"));
        }
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_sh_gradium_xnmoshi_Asr_nativeFrameSize(
        _env: JNIEnv,
        _class: JClass,
    ) -> jint {
        FRAME_SIZE as jint
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_sh_gradium_xnmoshi_Asr_nativeSampleRate(
        _env: JNIEnv,
        _class: JClass,
    ) -> jint {
        SAMPLE_RATE as jint
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_sh_gradium_xnmoshi_Asr_nativeFree(
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
