//! INT8 → BF16 weight dequantization at load time.
//!
//! Reads a safetensors buffer that may contain INT8-quantized weight tensors
//! (produced by `quantize.py`) alongside their per-channel BF16 scale factors.
//! Dequantizes every I8 tensor back to BF16, then returns a clean safetensors
//! buffer identical in dtype to the original non-quantized model.
//!
//! Convention produced by `quantize.py`:
//!   - Quantized weight: same tensor name as original, stored as **I8**
//!   - Scale factor:     `{name}_scale`, stored as **BF16**, shape `[out_channels]`
//!
//! Dequantization (per output channel `c`):
//!   `bf16_weight[c, ..] = bf16(i8_weight[c, ..]) * bf16_scale[c]`

use std::collections::HashSet;

/// Check whether `buffer` contains any I8 tensors. If so, dequantize them to
/// BF16 and return a new safetensors buffer. If the buffer is already fully
/// float, return a copy without extra work.
pub fn dequantize_if_needed(buffer: &[u8]) -> Vec<u8> {
    let st = match safetensors::SafeTensors::deserialize(buffer) {
        Ok(st) => st,
        Err(_) => return buffer.to_vec(),
    };

    let has_i8 = st.iter().any(|(_, t)| t.dtype() == safetensors::Dtype::I8);
    if !has_i8 {
        return buffer.to_vec();
    }

    dequantize_inner(&st)
}

/// Perform the actual I8 → BF16 dequantization.
fn dequantize_inner(st: &safetensors::SafeTensors<'_>) -> Vec<u8> {
    let scale_names: HashSet<String> = st
        .iter()
        .filter(|(_, t)| t.dtype() == safetensors::Dtype::I8)
        .map(|(name, _)| format!("{name}_scale"))
        .collect();

    let mut views: Vec<(String, OwnedView)> = Vec::new();

    for (name, tensor) in st.iter() {
        let name = name.to_string();

        if scale_names.contains(&name) {
            continue;
        }

        if tensor.dtype() == safetensors::Dtype::I8 {
            let scale_name = format!("{name}_scale");
            let scale_tensor = match st.tensor(&scale_name) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let shape = tensor.shape().to_vec();
            let out_channels = shape[0];
            let elements_per_channel: usize = shape[1..].iter().product();

            // Read scale factors as little-endian bf16.
            let scale_bytes = scale_tensor.data();
            let scales: Vec<half::bf16> = scale_bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
                .collect();

            // Dequantize: cast i8 to bf16, multiply by bf16 scale.
            let i8_data = tensor.data();
            let total = out_channels * elements_per_channel;
            let mut bf16_bytes: Vec<u8> = Vec::with_capacity(total * 2);

            for ch in 0..out_channels {
                let s = scales[ch];
                let base = ch * elements_per_channel;
                for i in 0..elements_per_channel {
                    let q = i8_data[base + i] as i8;
                    let val = half::bf16::from_f32(q as f32) * s;
                    bf16_bytes.extend_from_slice(&val.to_le_bytes());
                }
            }

            views.push((name, OwnedView {
                data: bf16_bytes,
                shape,
                dtype: safetensors::Dtype::BF16,
            }));
        } else {
            views.push((name, OwnedView {
                data: tensor.data().to_vec(),
                shape: tensor.shape().to_vec(),
                dtype: tensor.dtype(),
            }));
        }
    }

    let view_refs: Vec<(&str, safetensors::tensor::TensorView<'_>)> = views
        .iter()
        .map(|(name, v)| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(v.dtype, v.shape.clone(), &v.data)
                    .expect("invalid tensor view"),
            )
        })
        .collect();

    safetensors::tensor::serialize(view_refs, &None).expect("serialization failed")
}

struct OwnedView {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: safetensors::Dtype,
}
