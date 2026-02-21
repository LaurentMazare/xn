use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use xn::nn::VB;
use xn::{Backend, Tensor};
use xn_moshi::mimi::{Config, Mimi};
use xn_moshi::streaming::{StreamMask, StreamTensor};

#[derive(Parser, Debug)]
#[command(name = "moshi")]
#[command(about = "Moshi audio processing tool")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Encode audio to codes and decode back to audio using Mimi streaming.
    AudioToAudio {
        /// Input audio file to process.
        input: std::path::PathBuf,

        /// Output WAV file path.
        #[arg(short, long, default_value = "output.wav")]
        output: std::path::PathBuf,

        /// Number of codebooks to use.
        #[arg(short, long, default_value_t = 16)]
        codebooks: usize,

        /// Use CPU even if CUDA is available.
        #[arg(long, default_value_t = false)]
        cpu: bool,

        /// Write a chrome tracing profile.
        #[arg(long)]
        chrome_tracing: bool,
    },
}

fn download_model() -> Result<std::path::PathBuf> {
    use hf_hub::{Repo, RepoType, api::sync::Api};
    let repo_id = "kyutai/moshiko-candle-q8";
    println!("Downloading model from {repo_id}...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
    let model_path = repo
        .get("tokenizer-e351c8d8-checkpoint125.safetensors")
        .context("model safetensors not found")?;
    println!("  Model at {}", model_path.display());
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
    let cli = Cli::parse();

    match cli.command {
        Command::AudioToAudio {
            input,
            output,
            codebooks,
            cpu,
            chrome_tracing,
        } => {
            let _guard = if chrome_tracing {
                Some(init_tracing())
            } else {
                None
            };

            #[cfg(feature = "cuda")]
            {
                if cpu {
                    println!("Using CPU");
                    audio_to_audio(input, output, codebooks, xn::CPU)?;
                } else {
                    println!("Using CUDA");
                    let dev = xn::cuda_backend::Device::new(0)?;
                    unsafe {
                        dev.disable_event_tracking();
                    }
                    audio_to_audio(input, output, codebooks, dev)?;
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = cpu;
                println!("Using CPU");
                audio_to_audio(input, output, codebooks, xn::CPU)?;
            }
        }
    }

    Ok(())
}

fn audio_to_audio<Dev: Backend>(
    input: std::path::PathBuf,
    output: std::path::PathBuf,
    codebooks: usize,
    dev: Dev,
) -> Result<()> {
    let target_sample_rate: usize = 24000;

    // --- Load audio ---
    println!("Loading audio from {}...", input.display());
    let (pcm_data, sample_rate) = kaudio::pcm_decode(&input)?;
    println!(
        "  {} samples at {} Hz ({:.2}s)",
        pcm_data.len(),
        sample_rate,
        pcm_data.len() as f64 / sample_rate as f64
    );

    let pcm_data = if sample_rate as usize != target_sample_rate {
        println!(
            "  Resampling {} Hz -> {} Hz",
            sample_rate, target_sample_rate
        );
        kaudio::resample(&pcm_data, sample_rate as usize, target_sample_rate)?
    } else {
        pcm_data
    };

    // --- Load model ---
    let model_path = download_model()?;
    println!("Loading model weights...");
    let vb = VB::load(&[model_path], dev.clone())?;
    let config = Config::v0_1(Some(codebooks));
    println!(
        "  sample_rate={}, frame_rate={}, codebooks={}",
        config.sample_rate, config.frame_rate, codebooks
    );
    let mut model: Mimi<f32, Dev> = Mimi::load(&vb.root(), config, &dev)?;
    println!("  Model loaded");

    // --- Streaming encode ---
    // Process in chunks of 1920 samples (= one frame at 12.5 Hz with product of ratios 8*6*5*4 = 960
    // and downsample stride 2, so 1920 audio samples -> 1 code frame).
    let chunk_size = 1920;
    let num_chunks = pcm_data.len().div_ceil(chunk_size);

    println!(
        "\nEncoding ({} chunks of {} samples)...",
        num_chunks, chunk_size
    );
    model.reset_state();
    let mask = StreamMask::empty();

    let encode_start = std::time::Instant::now();
    let mut all_codes: Vec<Tensor<i64, Dev>> = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(pcm_data.len());
        let mut chunk: Vec<f32> = pcm_data[start..end].to_vec();
        if chunk.len() < chunk_size {
            chunk.resize(chunk_size, 0.0);
        }

        let audio: Tensor<f32, Dev> = Tensor::from_vec(chunk, (1, 1, chunk_size), &dev)?;
        let codes_out = model.encode_step(&StreamTensor::from_tensor(audio), &mask)?;

        if let Some(codes) = codes_out.as_option() {
            let mut codes = codes.copy()?;
            if codes.rank() == 2 {
                codes = codes.unsqueeze(2)?;
            }
            all_codes.push(codes);
        }

        if (chunk_idx + 1) % 50 == 0 || chunk_idx == num_chunks - 1 {
            println!("  chunk {}/{}", chunk_idx + 1, num_chunks);
        }
    }

    let encode_elapsed = encode_start.elapsed();
    let audio_duration = pcm_data.len() as f64 / target_sample_rate as f64;
    println!(
        "  Done in {:.2}s ({:.1}x realtime)",
        encode_elapsed.as_secs_f64(),
        audio_duration / encode_elapsed.as_secs_f64()
    );

    // --- Display codes ---
    let code_refs: Vec<&Tensor<i64, Dev>> = all_codes.iter().collect();
    let all_codes = Tensor::cat(&code_refs, 2)?;
    let total_frames = all_codes.dims()[2];
    println!(
        "\nCodes shape: {:?} (batch, codebooks, frames)",
        all_codes.dims()
    );
    println!("{all_codes}");

    // --- Streaming decode ---
    println!("\nDecoding ({} frames)...", total_frames);
    model.reset_state();
    let decode_start = std::time::Instant::now();
    let mut all_decoded: Vec<Tensor<f32, Dev>> = Vec::with_capacity(total_frames);

    for frame_idx in 0..total_frames {
        let codes_frame = all_codes
            .narrow(2, frame_idx..frame_idx + 1)?
            .contiguous()?;
        let decoded = model.decode_step(&StreamTensor::from_tensor(codes_frame), &mask)?;

        if let Some(pcm) = decoded.as_option() {
            all_decoded.push(pcm.copy()?);
        }

        if (frame_idx + 1) % 50 == 0 || frame_idx == total_frames - 1 {
            println!("  frame {}/{}", frame_idx + 1, total_frames);
        }
    }

    let decode_elapsed = decode_start.elapsed();
    println!(
        "  Done in {:.2}s ({:.1}x realtime)",
        decode_elapsed.as_secs_f64(),
        audio_duration / decode_elapsed.as_secs_f64()
    );

    // --- Write output WAV ---
    let decoded_refs: Vec<&Tensor<f32, Dev>> = all_decoded.iter().collect();
    let decoded_audio = Tensor::cat(&decoded_refs, 2)?;
    println!("  Decoded shape: {:?}", decoded_audio.dims());

    let decoded_audio = decoded_audio.narrow(0, ..1)?.contiguous()?;
    let decoded_pcm = decoded_audio.to_vec()?;
    let decoded_pcm: Vec<f32> = decoded_pcm.into_iter().take(pcm_data.len()).collect();

    println!("\nWriting {} to {}...", decoded_pcm.len(), output.display());
    let file = std::fs::File::create(&output)?;
    let mut writer = std::io::BufWriter::new(file);
    kaudio::wav::write_pcm_as_wav(&mut writer, &decoded_pcm, target_sample_rate as u32, 1)?;

    // --- Summary ---
    let total = encode_elapsed + decode_elapsed;
    println!("\nSummary:");
    println!("  Input:    {:.2}s", audio_duration);
    println!("  Encode:   {:.2}s", encode_elapsed.as_secs_f64());
    println!("  Decode:   {:.2}s", decode_elapsed.as_secs_f64());
    println!(
        "  Total:    {:.2}s ({:.1}x realtime)",
        total.as_secs_f64(),
        audio_duration / total.as_secs_f64()
    );

    Ok(())
}
