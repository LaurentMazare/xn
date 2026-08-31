// Per-call latency of single-query attention (autoregressive decode), fused vs composed.
//
// The composed variant is the transpose/matmul/softmax sequence a caller writes today; the
// fused variant is `Tensor::sdpa_decode`. Both read the same operands: a query of one position
// and a kv cache narrowed to its filled prefix, in the (b, pos, head, dim) layout the cache is
// written in.
//
// Run with RAYON_NUM_THREADS=1 and with default threads.
use xn::Result;

type Tensor = xn::Tensor<f32, xn::CpuDevice>;

/// Time one closure: `rounds` interleaved samples, keeping the minimum. The two variants are
/// sampled alternately by the caller so that any drift (thermal, scheduler) hits both equally,
/// and the minimum is what the machine actually does when nothing else interferes.
fn sample<F: FnMut() -> Result<()>>(iters: usize, mut f: F) -> Result<f64> {
    let start = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(start.elapsed().as_secs_f64() / iters as f64)
}

fn bench<A: FnMut() -> Result<()>, B: FnMut() -> Result<()>>(
    mut a: A,
    mut b: B,
) -> Result<(f64, f64)> {
    for _ in 0..50 {
        a()?;
        b()?;
    }
    // Pick the iteration count from a pilot sample so every case gets ~20ms per round
    // regardless of whether a call costs 1us or 100us.
    let pilot = sample(20, &mut a)?;
    let iters = ((0.02 / pilot) as usize).clamp(20, 20_000);
    let (mut ta, mut tb) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..9 {
        ta = ta.min(sample(iters, &mut a)?);
        tb = tb.min(sample(iters, &mut b)?);
    }
    Ok((ta, tb))
}

/// What a caller writes without the fused op.
fn composed(
    q: &Tensor,
    k: &xn::TensorView<f32, xn::CpuDevice>,
    v: &xn::TensorView<f32, xn::CpuDevice>,
    mask: Option<&Tensor>,
    scale: f32,
    b: usize,
    hd: usize,
) -> Result<Tensor> {
    let qt = xn::TensorView::from(q).transpose(1, 2)?;
    let kt = k.transpose(1, 2)?;
    let vt = v.transpose(1, 2)?;
    let attn = qt.matmul_t(&kt)?.scale(scale)?;
    let attn = match mask {
        Some(m) => attn.broadcast_add(m)?,
        None => attn,
    };
    let attn = attn.softmax()?;
    attn.matmul(&vt)?.transpose(1, 2)?.reshape((b, 1, hd))?.contiguous()
}

fn case(name: &str, b: usize, h: usize, d: usize, kv: usize, masked: bool) -> Result<()> {
    let dev = xn::CPU;
    // Allocate the cache at its full capacity and use the filled prefix, as decode does.
    let cache_kv = kv.next_power_of_two().max(kv + 64);
    let q = Tensor::zeros((b, 1, h, d), &dev)?;
    q.rand_uniform_(-1.0, 1.0)?;
    let kc = Tensor::zeros((b, cache_kv, h, d), &dev)?;
    kc.rand_uniform_(-1.0, 1.0)?;
    let vc = Tensor::zeros((b, cache_kv, h, d), &dev)?;
    vc.rand_uniform_(-1.0, 1.0)?;
    let k = kc.narrow(1, 0..kv)?;
    let v = vc.narrow(1, 0..kv)?;
    let scale = 1.0 / (d as f32).sqrt();
    // Sliding-window style: keep the most recent half of the context.
    let mask = if masked {
        let cut = kv / 2;
        Some(Tensor::from_vec(
            (0..kv).map(|j| if j < cut { f32::NEG_INFINITY } else { 0.0 }).collect(),
            (1, 1, 1, kv),
            &dev,
        )?)
    } else {
        None
    };

    let (c, f) = bench(
        || composed(&q, &k, &v, mask.as_ref(), scale, b, h * d).map(|_| ()),
        || q.sdpa_decode(&k, &v, mask.as_ref(), scale).map(|_| ()),
    )?;
    println!("{name:<38} {:>9.2} {:>9.2} {:>8.2}x", c * 1e6, f * 1e6, c / f);
    Ok(())
}

fn main() -> Result<()> {
    println!("rayon threads: {}", rayon::current_num_threads());
    println!("{:<38} {:>9} {:>9} {:>9}", "case (b,h,d,kv)", "composed", "fused", "speedup");
    println!("{:-<68}", "");

    // Real decode shapes from the models in this repo.
    case("mimi transformer   1,8,64,250", 1, 8, 64, 250, false)?;
    case("moshi lm           1,16,128,750", 1, 16, 128, 750, false)?;
    case("moshi lm masked    1,16,128,750", 1, 16, 128, 750, true)?;
    case("depformer          1,8,128,750", 1, 8, 128, 750, false)?;
    case("moshi lm (32h)     1,32,64,375", 1, 32, 64, 375, false)?;
    println!();

    // Context sweep at one shape, to separate fixed overhead from per-kv work.
    for kv in [1usize, 8, 32, 128, 512, 2048] {
        case(&format!("ctx sweep          1,16,128,{kv}"), 1, 16, 128, kv, false)?;
    }
    println!();

    // Batched decode.
    for b in [2usize, 8, 32] {
        case(&format!("batched            {b},16,128,750"), b, 16, 128, 750, false)?;
    }
    Ok(())
}
