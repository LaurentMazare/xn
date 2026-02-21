use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use xn::nn::VB;
use xn::{Backend, Tensor, WithDTypeF};
use xn_moshi::asr::{Asr, AsrMsg};
use xn_moshi::lm::{self, LmModel};
use xn_moshi::mimi::{self, Mimi};
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

    /// Run speech-to-text on an audio file.
    Asr {
        /// Input audio file to process.
        input: std::path::PathBuf,

        /// Sampling temperature (0 for greedy).
        #[arg(short, long, default_value_t = 0.0)]
        temperature: f64,

        /// Use CPU even if CUDA is available.
        #[arg(long, default_value_t = false)]
        cpu: bool,

        /// Use f32 for the LM instead of bf16 (bf16 is the default on CUDA).
        #[arg(long, default_value_t = false)]
        f32: bool,

        /// Batch size for computation (ASR output uses first element only).
        #[arg(short, long, default_value_t = 1)]
        batch_size: usize,

        /// Write a chrome tracing profile.
        #[arg(long)]
        chrome_tracing: bool,

        #[arg(long)]
        verbose: bool,
    },
}

fn download_mimi_model() -> Result<std::path::PathBuf> {
    use hf_hub::{Repo, RepoType, api::sync::Api};
    let repo_id = "kyutai/moshiko-candle-q8";
    println!("Downloading mimi model from {repo_id}...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
    let model_path = repo
        .get("tokenizer-e351c8d8-checkpoint125.safetensors")
        .context("mimi safetensors not found")?;
    println!("  Mimi at {}", model_path.display());
    Ok(model_path)
}

struct AsrFiles {
    lm: std::path::PathBuf,
    mimi: std::path::PathBuf,
    tokenizer: std::path::PathBuf,
}

fn download_asr_model() -> Result<AsrFiles> {
    use hf_hub::{Repo, RepoType, api::sync::Api};
    let repo_id = "kyutai/stt-2.6b-en-candle";
    println!("Downloading ASR model from {repo_id}...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
    let lm = repo
        .get("model.safetensors")
        .context("LM safetensors not found")?;
    println!("  LM at {}", lm.display());
    let mimi = repo
        .get("mimi-pytorch-e351c8d8@125.safetensors")
        .context("mimi safetensors not found")?;
    println!("  Mimi at {}", mimi.display());
    let tokenizer = repo
        .get("tokenizer_en_audio_4000.model")
        .context("tokenizer not found")?;
    println!("  Tokenizer at {}", tokenizer.display());
    Ok(AsrFiles {
        lm,
        mimi,
        tokenizer,
    })
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

        Command::Asr {
            input,
            temperature,
            cpu,
            f32: use_f32,
            batch_size,
            chrome_tracing,
            verbose,
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
                    run_asr::<f32, _>(input, temperature, batch_size, verbose, xn::CPU)?;
                } else {
                    let dev = xn::cuda_backend::Device::new(0)?;
                    unsafe {
                        dev.disable_event_tracking();
                    }
                    if use_f32 {
                        println!("Using CUDA (f32)");
                        run_asr::<f32, _>(input, temperature, batch_size, verbose, dev)?;
                    } else {
                        println!("Using CUDA (bf16)");
                        run_asr::<half::bf16, _>(input, temperature, batch_size, verbose, dev)?;
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = cpu;
                let _ = use_f32;
                println!("Using CPU");
                run_asr::<f32, _>(input, temperature, batch_size, verbose, xn::CPU)?;
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
    let model_path = download_mimi_model()?;
    println!("Loading model weights...");
    let vb = VB::load(&[model_path], dev.clone())?;
    let config = mimi::Config::v0_1(Some(codebooks));
    println!(
        "  sample_rate={}, frame_rate={}, codebooks={}",
        config.sample_rate, config.frame_rate, codebooks
    );
    let model: Mimi<f32, Dev> = Mimi::load(&vb.root(), config)?;
    println!("  Model loaded");

    // --- Streaming encode ---
    let chunk_size = 1920;
    let num_chunks = pcm_data.len().div_ceil(chunk_size);

    println!(
        "\nEncoding ({} chunks of {} samples)...",
        num_chunks, chunk_size
    );
    let mut enc_state = model.init_encode_state(1)?;
    let mask = StreamMask::all_active(1);

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
        let codes_out =
            model.encode_step(&StreamTensor::from_tensor(audio), &mut enc_state, &mask)?;

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
    let mut dec_state = model.init_decode_state(1)?;
    let decode_start = std::time::Instant::now();
    let mut all_decoded: Vec<Tensor<f32, Dev>> = Vec::with_capacity(total_frames);

    for frame_idx in 0..total_frames {
        let codes_frame = all_codes
            .narrow(2, frame_idx..frame_idx + 1)?
            .contiguous()?;
        let decoded = model.decode_step(
            &StreamTensor::from_tensor(codes_frame),
            &mut dec_state,
            &mask,
        )?;

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

fn run_asr<LmT: WithDTypeF, Dev: Backend>(
    input: std::path::PathBuf,
    temperature: f64,
    batch_size: usize,
    verbose: bool,
    dev: Dev,
) -> Result<()> {
    use std::io::Write;

    let target_sample_rate: usize = 24000;

    // --- Load audio ---
    println!("Loading audio from {}...", input.display());
    let (pcm_data, sample_rate) = kaudio::pcm_decode(&input)?;
    let audio_duration = pcm_data.len() as f64 / sample_rate as f64;
    println!(
        "  {} samples at {} Hz ({:.2}s)",
        pcm_data.len(),
        sample_rate,
        audio_duration
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

    // --- Download models ---
    let files = download_asr_model()?;

    // --- Load tokenizer ---
    let tokenizer_path = files.tokenizer.to_str().context("invalid tokenizer path")?;
    let sp = sentencepiece::SentencePieceProcessor::open(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to open tokenizer: {e}"))?;

    // --- Load mimi ---
    println!("Loading mimi weights...");
    let mimi_vb = VB::load(&[files.mimi], dev.clone())?;
    let mimi_config = mimi::Config::v0_1(Some(32));
    let mimi: Mimi<f32, Dev> = Mimi::load(&mimi_vb.root(), mimi_config)?;
    println!("  Mimi loaded");

    // --- Load LM ---
    println!("Loading LM weights...");
    let lm_vb = VB::load(&[files.lm], dev.clone())?;
    let lm_config = lm::Config::stt_2_6b();
    let lm: LmModel<LmT, Dev> = LmModel::load(&lm_vb.root(), &lm_config)?;
    println!("  LM loaded");

    // --- Create ASR ---
    let asr_delay_in_tokens = 31; // 2.5s * 12.5fps
    let asr = Asr::new(asr_delay_in_tokens, temperature, mimi, lm);
    let mut state = asr.init_state(batch_size)?;
    let mask = StreamMask::all_active(batch_size);

    // --- Process audio ---
    let chunk_size = 1920; // 80ms at 24kHz
    let num_chunks = pcm_data.len().div_ceil(chunk_size);
    let start_time = std::time::Instant::now();

    println!(
        "\nProcessing ({} chunks of {} samples, batch_size={})...",
        num_chunks, chunk_size, batch_size
    );
    println!("---");

    // Accumulate all text tokens (re-inserting the separator token 3 that
    // triggers word emission) so that SentencePiece can handle spacing.
    let mut all_text_tokens: Vec<u32> = vec![];
    let mut last_decoded_len = 0;

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(pcm_data.len());
        let mut chunk: Vec<f32> = pcm_data[start..end].to_vec();
        if chunk.len() < chunk_size {
            chunk.resize(chunk_size, 0.0);
        }

        // Replicate the same audio chunk across the batch.
        let chunk_batched: Vec<f32> = chunk.repeat(batch_size);
        let audio: Tensor<f32, Dev> =
            Tensor::from_vec(chunk_batched, (batch_size, 1, chunk_size), &dev)?;
        let pcm = StreamTensor::from_tensor(audio);
        let start_time = std::time::Instant::now();
        let msgs = asr.step_pcm(&pcm, &mut state, &mask, |_, _, _| {})?;
        if verbose {
            println!(
                "  chunk {}/{} processed in {:.2}ms",
                chunk_idx + 1,
                num_chunks,
                start_time.elapsed().as_secs_f64() * 1000.0
            );
        }

        for msg in msgs {
            if let AsrMsg::Word {
                tokens, batch_idx, ..
            } = msg
            {
                if batch_idx == 0 {
                    all_text_tokens.push(3); // re-insert space/separator token
                    all_text_tokens.extend_from_slice(&tokens);
                    let text = sp.decode_piece_ids(&all_text_tokens).unwrap_or_default();
                    let new_chars = text.len() - last_decoded_len;
                    if new_chars > 0 && !verbose {
                        print!("{}", &text[last_decoded_len..]);
                        std::io::stdout().flush()?;
                    }
                    last_decoded_len = text.len();
                }
            }
        }
    }

    println!();
    println!("---");
    if verbose {
        let decoded_text = sp.decode_piece_ids(&all_text_tokens).unwrap_or_default();
        println!("{decoded_text}\n---");
    }

    let elapsed = start_time.elapsed();
    let audio_duration = pcm_data.len() as f64 / target_sample_rate as f64;
    println!(
        "Done in {:.2}s ({:.1}x realtime)",
        elapsed.as_secs_f64(),
        audio_duration / elapsed.as_secs_f64()
    );

    Ok(())
}
