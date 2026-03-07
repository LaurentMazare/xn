use pyo3::prelude::*;

#[pymodule]
fn ptts(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
