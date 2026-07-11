// Per-call latency of small ops (streaming-inference shapes), to measure parallelism
// overhead. Run with RAYON_NUM_THREADS=1 and default threads.
use xn::Result;

type Tensor = xn::Tensor<f32, xn::CpuDevice>;

fn bench<F: FnMut() -> Result<()>>(name: &str, mut f: F) -> Result<()> {
    for _ in 0..100 {
        f()?
    }
    let iters = 5000;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    let el = start.elapsed().as_secs_f64() / iters as f64;
    println!("{name:<34} {:>9.2} us/call", el * 1e6);
    Ok(())
}

fn main() -> Result<()> {
    let dev = xn::CPU;
    println!("rayon threads: {}", rayon::current_num_threads());

    let a = Tensor::zeros((8, 512), &dev)?;
    a.rand_uniform_(-1.0, 1.0)?;
    let b = a.copy()?;
    bench("add (8x512)", || a.add(&b).map(|_| ()))?;
    bench("silu (8x512)", || a.silu().map(|_| ()))?;
    bench("softmax (8x512)", || a.softmax().map(|_| ()))?;
    bench("sum_keepdim (8x512)", || a.sum_keepdim(vec![1]).map(|_| ()))?;
    bench("argmax (8x512)", || a.argmax(1).map(|_| ()))?;
    bench("contiguous of t() (8x512)", || a.t()?.contiguous_always_copy().map(|_| ()))?;

    let c = Tensor::zeros((1, 512, 64), &dev)?;
    c.rand_uniform_(-1.0, 1.0)?;
    bench("transpose(1,2) (1x512x64)", || c.transpose(1, 2)?.contiguous_always_copy().map(|_| ()))?;
    let kern = Tensor::zeros((512, 512, 3), &dev)?;
    kern.rand_uniform_(-0.1, 0.1)?;
    bench("conv1d 512ch k3 L64 (step)", || c.conv1d(&kern, None, 1, 1, 1, 1).map(|_| ()))?;

    // Bigger ops for 1-thread throughput comparison.
    let big = Tensor::zeros(4 * 1024 * 1024, &dev)?;
    big.rand_uniform_(-1.0, 1.0)?;
    let big2 = big.copy()?;
    bench("add (4M)", || big.add(&big2).map(|_| ()))?;
    bench("silu (4M)", || big.silu().map(|_| ()))?;
    let sm = Tensor::zeros((1024, 1024), &dev)?;
    sm.rand_uniform_(-1.0, 1.0)?;
    bench("sum_keepdim (1024x1024)", || sm.sum_keepdim(vec![1]).map(|_| ()))?;
    bench("transpose 2d (1024x1024)", || sm.t()?.contiguous_always_copy().map(|_| ()))?;
    Ok(())
}
