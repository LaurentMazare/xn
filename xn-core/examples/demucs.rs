use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use xn::Tensor;
use xn::models::demucs::{Config, Demucs, DemucsStreamer};
use xn::nn::VB;

#[derive(Parser, Debug)]
#[command(name = "demucs")]
struct Args {
    #[arg(short = 'c', long)]
    ckpt_path: PathBuf,

    #[arg(long)]
    combined: PathBuf,

    #[arg(long)]
    reference: PathBuf,

    #[arg(long)]
    out: PathBuf,

    #[arg(long)]
    chrome_tracing: bool,

    #[arg(short = 'f', long, default_value_t = 1)]
    num_frames: usize,
}

fn load_config(path: &std::path::Path) -> Result<Config> {
    let config_path = path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read config from {}", config_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&config_str)?;

    Ok(Config {
        chin: json["chin"].as_u64().unwrap_or(2) as usize,
        chout: json["chout"].as_u64().unwrap_or(1) as usize,
        hidden: json["hidden"].as_u64().unwrap_or(48) as usize,
        depth: json["depth"].as_u64().unwrap_or(5) as usize,
        kernel_size: json["kernel_size"].as_u64().unwrap_or(8) as usize,
        stride: json["stride"].as_u64().unwrap_or(4) as usize,
        causal: json["causal"].as_bool().unwrap_or(true),
        resample: json["resample"].as_u64().unwrap_or(4) as usize,
        growth: json["growth"].as_f64().unwrap_or(2.0) as f32,
        max_hidden: json["max_hidden"].as_u64().unwrap_or(10_000) as usize,
        normalize: json["normalize"].as_bool().unwrap_or(true),
        glu: json["glu"].as_bool().unwrap_or(true),
        floor: json["floor"].as_f64().unwrap_or(1e-3) as f32,
        bias: json["bias"].as_bool().unwrap_or(true),
        sample_rate: json["sample_rate"].as_u64().unwrap_or(16_000) as usize,
    })
}

fn read_and_resample_audio<P: AsRef<std::path::Path>>(p: P, target_sr: usize) -> Result<Vec<f32>> {
    let (data, sr) = kaudio::pcm_decode(p.as_ref())?;
    let sr = sr as usize;
    let resampled_data =
        if sr != target_sr { kaudio::resample(&data, sr, target_sr)? } else { data };
    Ok(resampled_data)
}

fn init_tracing() -> tracing_chrome::FlushGuard {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::{prelude::*, registry::Registry};

    let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
    Registry::default().with(chrome_layer).init();
    guard
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _guard = if args.chrome_tracing { Some(init_tracing()) } else { None };

    println!("\nLoading config...");
    let config = load_config(&args.ckpt_path)?;
    println!(
        "  chin={}, chout={}, hidden={}, depth={}",
        config.chin, config.chout, config.hidden, config.depth
    );
    println!("  sample_rate={}, causal={}", config.sample_rate, config.causal);

    println!("\nLoading model weights...");
    let model_path = args.ckpt_path.join("checkpoint.safetensors");

    let dev = xn::CPU;
    let vb = VB::load(&[model_path], dev)?;
    let model: Demucs<f32, xn::CpuDevice> = Demucs::load(&vb.root(), config.clone())?;
    println!("Model loaded successfully!");

    let mut streamer = DemucsStreamer::new(
        model,
        &dev,
        args.num_frames,
        64,   // resample_lookahead
        256,  // resample_buffer
        10.0, // mean_decay_duration
        0.0,  // dry (0 = full processing, 1 = pass-through)
    )?;

    // Load audio files
    println!("\nLoading audio files...");
    let c_data = read_and_resample_audio(&args.combined, config.sample_rate)?;
    let r_data = read_and_resample_audio(&args.reference, config.sample_rate)?;
    println!("  Combined: {} samples", c_data.len());
    println!("  Reference: {} samples", r_data.len());

    let max_len = c_data.len().max(r_data.len());
    println!(
        "  Audio length: {} samples ({:.2}s)",
        max_len,
        max_len as f64 / config.sample_rate as f64
    );

    // Process in streaming fashion
    println!("\nProcessing audio...");
    let process_start = std::time::Instant::now();
    let mut out_chunks = Vec::new();
    let mut c_pos = 0;
    let mut r_pos = 0;
    let mut frame_count = 0;

    while c_pos < c_data.len() || r_pos < r_data.len() {
        let frame_size = streamer.current_frame_length();

        let mut frame_data = vec![0.0f32; config.chin * frame_size];

        if c_pos < c_data.len() {
            let c_end = (c_pos + frame_size).min(c_data.len());
            let c_chunk_len = c_end - c_pos;
            frame_data[..c_chunk_len].copy_from_slice(&c_data[c_pos..c_end]);
        }

        if r_pos < r_data.len() {
            let r_end = (r_pos + frame_size).min(r_data.len());
            let r_chunk_len = r_end - r_pos;
            frame_data[frame_size..frame_size + r_chunk_len].copy_from_slice(&r_data[r_pos..r_end]);
        }

        let frame: Tensor<f32, xn::CpuDevice> =
            Tensor::from_vec(frame_data, (config.chin, frame_size), &xn::CPU)?;
        let out = streamer.feed(&frame)?;

        if out.dim(1)? > 0 {
            out_chunks.push(out);
        }

        c_pos += frame_size;
        r_pos += frame_size;
        frame_count += 1;

        if frame_count % 100 == 0 {
            println!("  Processed {} frames...", frame_count);
        }
    }

    // Flush remaining
    let flush_out = streamer.flush()?;
    if flush_out.dim(1)? > 0 {
        out_chunks.push(flush_out);
    }

    let process_elapsed = process_start.elapsed();
    println!("  Processing completed in {:.2}s", process_elapsed.as_secs_f64());

    // Concatenate output
    println!("\nConcatenating output...");
    let out_refs: Vec<_> = out_chunks.iter().collect();
    let output = if out_refs.is_empty() {
        Tensor::zeros((config.chout, 0), &xn::CPU)?
    } else {
        Tensor::cat(&out_refs, 1)?
    };
    println!("  Output shape: {:?}", output.dims());

    // Extract PCM and write
    let output_pcm = output.to_vec()?;
    let output_len = output_pcm.len().min(max_len);
    let output_pcm: Vec<f32> = output_pcm.into_iter().take(output_len).collect();

    println!("\nWriting output WAV...");
    let output_file = std::fs::File::create(&args.out)?;
    let mut writer = std::io::BufWriter::new(output_file);
    kaudio::wav::write_pcm_as_wav(&mut writer, &output_pcm, config.sample_rate as u32, 1)?;
    println!("  Written {} samples to {}", output_pcm.len(), args.out.display());

    // Summary
    let audio_duration = max_len as f64 / config.sample_rate as f64;
    let rtf = process_elapsed.as_secs_f64() / audio_duration;
    let sr_ms = config.sample_rate as f64 / 1000.0;
    let initial_lag = streamer.initial_frame_length as f64 / sr_ms;

    println!("\nSummary:");
    println!("  Audio duration: {:.2}s", audio_duration);
    println!("  Processing time: {:.2}s", process_elapsed.as_secs_f64());
    println!("  Initial latency: {:.1}ms", initial_lag);
    println!("  Stride: {:.1}ms", streamer.stride as f64 / sr_ms);
    println!(
        "  RTF: {:.2} ({})",
        rtf,
        if rtf < 1.0 { "faster than realtime" } else { "slower than realtime" }
    );

    println!("\nDone!");
    Ok(())
}
