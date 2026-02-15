use anyhow::{Context, Result};
use clap::Parser;
use xn::models::mimi::{Config, Mimi, StreamMask, StreamTensor};
use xn::nn::VB;
use xn::{Backend, Tensor};

#[derive(Parser, Debug)]
#[command(name = "mimi")]
#[command(about = "Run Mimi audio tokenizer model")]
struct Args {
    /// Input audio file to process
    input: std::path::PathBuf,

    /// Output WAV file path
    #[arg(short, long, default_value = "output.wav")]
    output: std::path::PathBuf,

    /// Number of codebooks to use (default: 16)
    #[arg(short, long, default_value_t = 16)]
    codebooks: usize,

    /// Batch size for processing (duplicates input to simulate higher batch sizes)
    #[arg(short, long, default_value_t = 1)]
    batch_size: usize,

    /// Use the cpu device even if cuda is available.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    #[arg(long)]
    chrome_tracing: bool,

    #[arg(long)]
    audio_to_codes_only: bool,
}

fn download_model() -> Result<std::path::PathBuf> {
    use hf_hub::{Repo, RepoType, api::sync::Api};
    let repo_id = "kyutai/moshiko-candle-q8";
    println!("Downloading model from {repo_id}...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    let model_path = repo
        .get("tokenizer-e351c8d8-checkpoint125.safetensors")
        .context("model.safetensors not found")?;
    println!("  Found model.safetensors at {}", model_path.display());

    Ok(model_path)
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

    #[cfg(feature = "cuda")]
    {
        if args.cpu {
            println!("Using CPU despite CUDA being available");
            run_for_device(args, xn::CPU)?;
        } else {
            println!("Using CUDA backend");
            let dev = xn::cuda_backend::Device::new(0)?;
            unsafe {
                dev.disable_event_tracking();
            }
            run_for_device(args, dev)?;
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("Using CPU backend");
        run_for_device(args, xn::CPU)?;
    }

    Ok(())
}

fn run_for_device<Dev: Backend>(args: Args, dev: Dev) -> Result<()> {
    println!("Mimi Audio Tokenizer Example");
    println!("============================");
    println!("Input file: {}", args.input.display());
    println!("Output file: {}", args.output.display());
    println!("Codebooks: {}", args.codebooks);
    println!("Batch size: {}", args.batch_size);

    // Load and resample audio
    println!("\nLoading audio file...");
    let (pcm_data, sample_rate) = kaudio::pcm_decode(&args.input)?;
    println!(
        "  Loaded {} samples at {} Hz ({:.2}s)",
        pcm_data.len(),
        sample_rate,
        pcm_data.len() as f64 / sample_rate as f64
    );

    // Resample to 24000 Hz if needed
    let target_sample_rate: usize = 24000;
    let pcm_data = if sample_rate as usize != target_sample_rate {
        println!("  Resampling from {} Hz to {} Hz...", sample_rate, target_sample_rate);
        kaudio::resample(&pcm_data, sample_rate as usize, target_sample_rate)?
    } else {
        pcm_data
    };
    println!(
        "  Audio ready: {} samples at {} Hz ({:.2}s)",
        pcm_data.len(),
        target_sample_rate,
        pcm_data.len() as f64 / target_sample_rate as f64
    );

    // Download model weights
    let model_path = download_model()?;

    // Load model
    println!("\nLoading model weights...");
    let vb = VB::load(&[model_path], dev.clone())?;
    let config = Config::v0_1_no_weight_norm(Some(args.codebooks));
    println!("Config: sample_rate={}, frame_rate={}", config.sample_rate, config.frame_rate);

    let mut model: Mimi<f32, Dev> = Mimi::load(&vb.root(), config, &dev)?;
    println!("Model loaded successfully!");

    // Process audio in chunks of 1920 samples
    let chunk_size = 1920;
    let num_chunks = pcm_data.len().div_ceil(chunk_size);

    println!(
        "\nEncoding {} samples ({} chunks of {} samples each)...",
        pcm_data.len(),
        num_chunks,
        chunk_size
    );

    // Reset state once before processing consecutive chunks
    model.reset_state();
    let mask = StreamMask::empty();

    let encode_start = std::time::Instant::now();
    let mut all_codes: Vec<Tensor<i64, Dev>> = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let start_idx = chunk_idx * chunk_size;
        let end_idx = (start_idx + chunk_size).min(pcm_data.len());

        // Get chunk data, pad with zeros if needed
        let mut chunk_data: Vec<f32> = pcm_data[start_idx..end_idx].to_vec();
        if chunk_data.len() < chunk_size {
            chunk_data.resize(chunk_size, 0.0);
        }

        // Duplicate chunk data for batch dimension
        let batch_data: Vec<f32> = chunk_data.repeat(args.batch_size);

        // Create tensor: shape [batch=batch_size, channels=1, time=1920]
        let audio: Tensor<f32, Dev> =
            Tensor::from_vec(batch_data, (args.batch_size, 1, chunk_size), &dev)?;
        let audio_stream = StreamTensor::from_tensor(audio);

        // Encode the audio to codes using streaming API
        let codes_stream = model.encode_step(&audio_stream, &mask)?;
        if let Some(codes) = codes_stream.as_option() {
            let mut codes = codes.copy()?;
            // Ensure codes have 3 dimensions [B, n_q, T] - unsqueeze if T=1 was collapsed
            if codes.rank() == 2 {
                codes = codes.unsqueeze(2)?;
            }
            all_codes.push(codes);

            if all_codes.len() == 1 {
                let code_dims = all_codes[0].dims();
                println!("  First chunk codes shape: {:?}", code_dims);
            }
        }

        if (chunk_idx + 1) % 50 == 0 || chunk_idx == num_chunks - 1 {
            println!("  Encoded chunk {}/{}", chunk_idx + 1, num_chunks);
        }
    }

    let encode_elapsed = encode_start.elapsed();
    println!(
        "  Encoding completed in {:.2}s ({:.2}x realtime)",
        encode_elapsed.as_secs_f64(),
        pcm_data.len() as f64 / target_sample_rate as f64 / encode_elapsed.as_secs_f64()
    );

    if args.audio_to_codes_only {
        println!("\nAudio to codes only mode, skipping decoding.");
        return Ok(());
    }

    // Concatenate all codes along the time dimension
    println!("\nConcatenating codes...");
    let code_refs: Vec<&Tensor<i64, Dev>> = all_codes.iter().collect();
    let all_codes = Tensor::cat(&code_refs, 2)?; // dim 2 is time
    let total_code_frames = all_codes.dims()[2];
    println!("  Total codes shape: {:?}", all_codes.dims());
    println!("CODES\n{all_codes}");

    // Decode all codes back to audio using streaming API
    println!("\nDecoding codes to audio ({} frames)...", total_code_frames);
    model.reset_state();
    let decode_start = std::time::Instant::now();
    let mut all_decoded: Vec<Tensor<f32, Dev>> = Vec::with_capacity(total_code_frames);

    for frame_idx in 0..total_code_frames {
        // Extract single frame: [B, n_q, 1]
        let codes_frame = all_codes.narrow(2, frame_idx..frame_idx + 1)?.contiguous()?;
        let codes_stream: StreamTensor<i64, Dev> = StreamTensor::from_tensor(codes_frame);

        let decoded_stream = model.decode_step(&codes_stream, &mask)?;
        if let Some(decoded) = decoded_stream.as_option() {
            all_decoded.push(decoded.copy()?);
        }

        if (frame_idx + 1) % 50 == 0 || frame_idx == total_code_frames - 1 {
            println!("  Decoded frame {}/{}", frame_idx + 1, total_code_frames);
        }
    }

    let decode_elapsed = decode_start.elapsed();
    println!(
        "  Decoding completed in {:.2}s ({:.2}x realtime)",
        decode_elapsed.as_secs_f64(),
        pcm_data.len() as f64 / target_sample_rate as f64 / decode_elapsed.as_secs_f64()
    );

    // Concatenate all decoded audio
    let decoded_refs: Vec<&Tensor<f32, Dev>> = all_decoded.iter().collect();
    let decoded_audio = Tensor::cat(&decoded_refs, 2)?; // dim 2 is time
    println!("  Decoded shape: {:?}", decoded_audio.dims());

    // Extract PCM data from first batch element only
    let decoded_audio = decoded_audio.narrow(0, ..1)?.contiguous()?; // [1, 1, time]
    let decoded_pcm = decoded_audio.to_vec()?;

    // Trim to original length (remove padding)
    let original_output_len = pcm_data.len();
    let decoded_pcm: Vec<f32> = decoded_pcm.into_iter().take(original_output_len).collect();

    // Write output WAV file
    println!("\nWriting output WAV file...");
    let output_file = std::fs::File::create(&args.output)?;
    let mut writer = std::io::BufWriter::new(output_file);
    kaudio::wav::write_pcm_as_wav(&mut writer, &decoded_pcm, target_sample_rate as u32, 1)?;
    println!("  Written {} samples to {}", decoded_pcm.len(), args.output.display());

    // Summary
    let total_elapsed = encode_elapsed + decode_elapsed;
    let audio_duration = pcm_data.len() as f64 / target_sample_rate as f64;
    println!("\nSummary:");
    println!("  Input duration: {:.2}s", audio_duration);
    println!("  Total processing time: {:.2}s", total_elapsed.as_secs_f64());
    println!(
        "  Overall realtime factor: {:.2}x (>1 means faster than realtime)",
        audio_duration / total_elapsed.as_secs_f64()
    );

    println!("\nDone!");
    Ok(())
}
