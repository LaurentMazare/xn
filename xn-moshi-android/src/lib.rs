// Android JNI wrapper for xn-moshi streaming ASR.
//
// Public Kotlin surface (class `sh.gradium.xnmoshi.Asr`):
//   nativeInit(mimi_path, lm_path, tokenizer_path, dtype) -> handle : Long
//   nativeStep(handle, pcm: FloatArray) -> String     // newly recognized text
//   nativeReset(handle)
//   nativeFrameSize(handle) -> Int
//   nativeSampleRate(handle) -> Int
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
const ASR_DELAY_SECS: f64 = 2.5;
// Token 3 is the SentencePiece separator (▁). The example re-inserts it
// between emitted words so the decoder restores spaces.
const SEPARATOR_TOKEN: u32 = 3;

pub trait Session: Send {
    fn step(&mut self, pcm: &[f32]) -> Result<String>;
    fn reset(&mut self) -> Result<()>;
}

struct AsrSession<Q: BackendQ> {
    state: AsrState<Q>,
    sp: SpDecoder,
    accumulated_tokens: Vec<u32>,
    decoded_len: usize,
}

impl<Q: BackendQ> AsrSession<Q> {
    fn new(mimi_path: &str, lm_path: &str, tokenizer_path: &str, dev: Q::B) -> Result<Self> {
        let asr_delay_in_tokens =
            (ASR_DELAY_SECS * SAMPLE_RATE as f64 / FRAME_SIZE as f64) as usize;
        let asr: Asr<Q> = Asr::load(mimi_path, lm_path, None, asr_delay_in_tokens, 0.0, dev)
            .map_err(|e| anyhow!("Asr::load: {e}"))?;
        let state = asr.init_state(1).map_err(|e| anyhow!("Asr::init_state: {e}"))?;
        let sp =
            SpDecoder::from_file(tokenizer_path).map_err(|e| anyhow!("tokenizer load: {e}"))?;
        Ok(Self { state, sp, accumulated_tokens: Vec::new(), decoded_len: 0 })
    }
}

impl<Q: BackendQ> Session for AsrSession<Q> {
    fn step(&mut self, pcm: &[f32]) -> Result<String> {
        if pcm.len() != FRAME_SIZE {
            return Err(anyhow!("expected frame of {FRAME_SIZE} samples, got {}", pcm.len()));
        }
        let dev = self.state.device().clone();
        let audio: Tensor<f32, Q::B> = Tensor::from_vec(pcm.to_vec(), (1, 1, FRAME_SIZE), &dev)
            .map_err(|e| anyhow!("audio tensor: {e}"))?;
        let pcm = StreamTensor::from_tensor(audio);
        let mask = StreamMask::all_active(1);
        let step_results =
            self.state.step_pcm(&pcm, &mask, |_, _, _| {}).map_err(|e| anyhow!("step_pcm: {e}"))?;

        for sr in step_results {
            for word in sr.words {
                if let AsrWord::Word { tokens, batch_idx, .. } = word
                    && batch_idx == 0
                {
                    // Re-insert the separator so SP decode produces a space
                    // between words. Matches `examples/moshi.rs` ASR loop.
                    self.accumulated_tokens.push(SEPARATOR_TOKEN);
                    self.accumulated_tokens.extend_from_slice(&tokens);
                }
            }
        }
        let full = self.sp.decode_piece_ids(&self.accumulated_tokens);
        if full.len() > self.decoded_len {
            let new = full[self.decoded_len..].to_string();
            self.decoded_len = full.len();
            Ok(new)
        } else {
            Ok(String::new())
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.state.reset_state().map_err(|e| anyhow!("reset_state: {e}"))?;
        self.accumulated_tokens.clear();
        self.decoded_len = 0;
        Ok(())
    }
}

fn make_session(
    dtype: DTypeQ,
    mimi_path: &str,
    lm_path: &str,
    tokenizer_path: &str,
) -> Result<Box<dyn Session>> {
    use xn::quantized::*;
    let dev = CpuDevice;
    // The set of dtypes supported on a CPU-only Android build. Mirrors the
    // CPU arm of `xn::Runner::run` minus BF16/F16 (xn-core marks those "not
    // yet supported on CPU") and minus FP8 (CUDA only).
    Ok(match dtype {
        DTypeQ::F32 => Box::new(AsrSession::<Unquantized<f32, CpuDevice>>::new(
            mimi_path,
            lm_path,
            tokenizer_path,
            dev,
        )?),
        DTypeQ::Q4_0 => {
            Box::new(AsrSession::<Q40F32>::new(mimi_path, lm_path, tokenizer_path, dev)?)
        }
        DTypeQ::Q4_1 => {
            Box::new(AsrSession::<Q41F32>::new(mimi_path, lm_path, tokenizer_path, dev)?)
        }
        DTypeQ::Q5_0 => {
            Box::new(AsrSession::<Q50F32>::new(mimi_path, lm_path, tokenizer_path, dev)?)
        }
        DTypeQ::Q5_1 => {
            Box::new(AsrSession::<Q51F32>::new(mimi_path, lm_path, tokenizer_path, dev)?)
        }
        DTypeQ::Q8_0 => {
            Box::new(AsrSession::<Q80F32>::new(mimi_path, lm_path, tokenizer_path, dev)?)
        }
        DTypeQ::Q8_1 => {
            Box::new(AsrSession::<Q81F32>::new(mimi_path, lm_path, tokenizer_path, dev)?)
        }
        DTypeQ::Q2K => {
            Box::new(AsrSession::<Q2kF32>::new(mimi_path, lm_path, tokenizer_path, dev)?)
        }
        DTypeQ::Q3K => {
            Box::new(AsrSession::<Q3kF32>::new(mimi_path, lm_path, tokenizer_path, dev)?)
        }
        DTypeQ::Q4K => {
            Box::new(AsrSession::<Q4kF32>::new(mimi_path, lm_path, tokenizer_path, dev)?)
        }
        DTypeQ::Q5K => {
            Box::new(AsrSession::<Q5kF32>::new(mimi_path, lm_path, tokenizer_path, dev)?)
        }
        DTypeQ::Q6K => {
            Box::new(AsrSession::<Q6kF32>::new(mimi_path, lm_path, tokenizer_path, dev)?)
        }
        DTypeQ::Q8K => {
            Box::new(AsrSession::<Q8kF32>::new(mimi_path, lm_path, tokenizer_path, dev)?)
        }
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
                    .with_max_level(log::LevelFilter::Info)
                    .with_tag("xn-moshi"),
            );
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
        dtype: JString,
    ) -> jlong {
        log_init();
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
        log::info!(
            "init: dtype={dtype:?} cpus={} neon={} fp16-via-fp32={}",
            xn::get_num_cpus(),
            xn::with_neon(),
            xn::with_f16c(),
        );
        match make_session(dtype, &mimi_path, &lm_path, &tokenizer_path) {
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
