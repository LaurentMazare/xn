use numpy::PyReadonlyArray1;
use pocket_tts::tts_model::TTSModel;
use pyo3::prelude::*;
use std::sync::Arc;

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

#[pyclass]
struct Model {
    inner: Arc<TTSModel<f32, xn::CpuDevice>>,
}

#[pymethods]
impl Model {
    #[pyo3(signature = (audio_prompt, max_seq_len=2048))]
    fn get_state_for_audio_prompt(
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
        let voice_emb =
            xn::Tensor::from_vec(audio_prompt.to_vec(), expected_len, &xn::CpuDevice).w()?;
        let mut state = self.inner.init_flow_lm_state(1, max_seq_len).w()?;
        self.inner.prompt_audio(&mut state, &voice_emb).w()?;
        Ok(ModelState {
            model: Arc::clone(&self.inner),
            state,
        })
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
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    Ok(())
}
