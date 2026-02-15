use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use rand::Rng;
use xn::models::llama::{Config, KvCache, Llama};
use xn::nn::VB;
use xn::{Backend, Tensor};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelSize {
    /// Tiny test model (~1M params, 2 layers) - for quick testing (no weights)
    Test,
    /// SmolLM 135M - small but capable
    Smol135m,
    /// SmolLM 360M
    Smol360m,
    /// TinyLlama 1.1B
    TinyLlama,
}

impl ModelSize {
    fn hf_repo(&self) -> Option<&'static str> {
        match self {
            ModelSize::Test => None,
            ModelSize::Smol135m => Some("HuggingFaceTB/SmolLM-135M"),
            ModelSize::Smol360m => Some("HuggingFaceTB/SmolLM-360M"),
            ModelSize::TinyLlama => Some("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        }
    }

    fn config(&self) -> Config {
        match self {
            ModelSize::Test => Config::tiny_test(),
            ModelSize::Smol135m => Config::smol_lm_135m(),
            ModelSize::Smol360m => Config::smol_lm_360m(),
            ModelSize::TinyLlama => Config::tiny_llama_1_1b(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "llama")]
#[command(about = "Run Llama model inference in autoregressive mode")]
struct Args {
    /// Model size to use
    #[arg(short = 's', long, value_enum, default_value_t = ModelSize::Smol135m)]
    model_size: ModelSize,

    /// Number of tokens to generate
    #[arg(short, long, default_value_t = 50)]
    max_tokens: usize,

    /// Text prompt (or comma-separated token ids if --raw-tokens is set)
    #[arg(short, long, default_value = "The quick brown fox")]
    prompt: String,

    /// Interpret prompt as comma-separated token IDs instead of text
    #[arg(long, default_value_t = false)]
    raw_tokens: bool,

    /// Use the cpu device even if cuda is available.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Sampling temperature (0 = greedy/argmax, higher = more random)
    #[arg(short, long, default_value_t = 0.7)]
    temperature: f32,

    /// Verbose output (show tensor loading)
    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    /// Enable chrome tracing (writes trace-*.json)
    #[arg(long)]
    chrome_tracing: bool,
}

struct ModelFiles {
    safetensor_paths: Vec<std::path::PathBuf>,
    tokenizer_path: std::path::PathBuf,
}

fn download_model(repo_id: &str) -> Result<ModelFiles> {
    use hf_hub::{Repo, RepoType, api::sync::Api};
    println!("Downloading model from {repo_id}...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    // Download tokenizer
    let tokenizer_path = repo.get("tokenizer.json").context("tokenizer.json not found")?;
    println!("  Found tokenizer.json");

    // Get list of safetensor files
    let mut safetensor_paths = Vec::new();

    // Try single model.safetensors first
    match repo.get("model.safetensors") {
        Ok(path) => {
            println!("  Found model.safetensors");
            safetensor_paths.push(path);
        }
        Err(_) => {
            // Try sharded format with different shard counts
            for i in 1..=10 {
                for total in [2, 3, 4, 5, 6, 7, 8] {
                    if let Ok(path) = repo.get(&format!("model-{i:05}-of-{total:05}.safetensors")) {
                        println!("  Found {}", path.display());
                        safetensor_paths.push(path);
                        break;
                    }
                }
            }
        }
    }

    if safetensor_paths.is_empty() {
        anyhow::bail!("No safetensor files found in {repo_id}");
    }

    Ok(ModelFiles { safetensor_paths, tokenizer_path })
}

fn sample_token<B: Backend>(
    logits: &Tensor<f32, B>,
    temperature: f32,
    rng: &mut impl Rng,
) -> Result<u32> {
    // logits shape: (1, seq_len, vocab_size)
    let vocab_size = logits.dims()[2];
    let data = logits.to_vec()?;
    // Get logits for the last token
    let start = data.len() - vocab_size;
    let last_logits = &data[start..];

    if temperature <= 0.0 {
        // Pure argmax
        let logit = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        Ok(logit)
    } else {
        // Temperature-scaled sampling with top-p
        let mut probs: Vec<(usize, f32)> =
            last_logits.iter().enumerate().map(|(i, &v)| (i, v / temperature)).collect();

        // Softmax
        let max_logit = probs.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = probs.iter().map(|(_, v)| (*v - max_logit).exp()).sum();
        for (_, v) in probs.iter_mut() {
            *v = (*v - max_logit).exp() / sum_exp;
        }

        // Sort by probability descending for top-p sampling
        probs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Top-p (nucleus) sampling with p=0.9
        let top_p = 0.9;
        let mut cumsum = 0.0;
        let mut cutoff_idx = probs.len();
        for (i, (_, p)) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Renormalize the top-p tokens
        let top_probs = &probs[..cutoff_idx];
        let sum: f32 = top_probs.iter().map(|(_, p)| p).sum();

        // Sample from the top-p distribution
        let mut rng_val: f32 = rng.random();
        for (idx, p) in top_probs {
            rng_val -= p / sum;
            if rng_val <= 0.0 {
                return Ok(*idx as u32);
            }
        }

        // Fallback to first token
        Ok(top_probs[0].0 as u32)
    }
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

fn run_for_device<Dev: xn::Backend>(args: Args, dev: Dev) -> Result<()> {
    let config = args.model_size.config();

    println!("Model: {:?}", args.model_size);
    println!("Config: {:?}", config);

    let (model, tokenizer): (Llama<f32, Dev>, _) = if let Some(repo_id) = args.model_size.hf_repo()
    {
        let model_files = download_model(repo_id)?;

        println!("Loading tokenizer...");
        let tokenizer = tokenizers::Tokenizer::from_file(&model_files.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        println!("Loading weights...");
        let vb = VB::load(&model_files.safetensor_paths, dev)?;
        let model = Llama::load(&vb.root(), &config)?;

        (model, Some(tokenizer))
    } else {
        anyhow::bail!("Test mode not supported without weights");
    };

    // Tokenize the prompt
    let mut tokens: Vec<u32> = if args.raw_tokens {
        args.prompt.split(',').filter_map(|s| s.trim().parse().ok()).collect()
    } else if let Some(ref tokenizer) = tokenizer {
        let encoding = tokenizer
            .encode(args.prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        encoding.get_ids().to_vec()
    } else {
        // Test mode without tokenizer
        vec![1, 2, 3]
    };

    if tokens.is_empty() {
        tokens = vec![1]; // Default BOS token
    }

    println!("\nPrompt: \"{}\"", args.prompt);
    println!("Tokenized: {:?} ({} tokens)", tokens, tokens.len());
    println!("Generating {} tokens (temperature={})...\n", args.max_tokens, args.temperature);

    // Autoregressive generation loop
    let mut rng = rand::rng();
    let mut kv_cache: Option<KvCache<f32, Dev>> = None;
    let mut pos = 0;
    let mut generated_tokens = Vec::new();
    let mut autoregressive_start: Option<std::time::Instant> = None;

    for step in 0..args.max_tokens {
        let input_tokens: Vec<u32> = if kv_cache.is_none() {
            // First forward pass: process all prompt tokens
            tokens.clone()
        } else {
            // Subsequent passes: only process the last generated token
            vec![*tokens.last().unwrap()]
        };

        let (logits, new_kv_cache) = model.forward(&input_tokens, pos, kv_cache.as_ref())?;
        kv_cache = Some(new_kv_cache);
        pos += input_tokens.len();

        // Start timing after the first forward pass (prefill)
        if autoregressive_start.is_none() {
            autoregressive_start = Some(std::time::Instant::now());
        }

        // Sample next token
        let next_token = sample_token(&logits, args.temperature, &mut rng)?;
        tokens.push(next_token);
        generated_tokens.push(next_token);

        // Decode and print the new token
        if let Some(ref tokenizer) = tokenizer {
            let decoded = tokenizer
                .decode(&[next_token], false)
                .unwrap_or_else(|_| format!("[{}]", next_token));
            print!("{}", decoded);
            use std::io::Write;
            std::io::stdout().flush().ok();
        } else {
            print!("[{}]", next_token);
        }

        // Stop if we hit an EOS token (common values: 0, 1, 2 depending on tokenizer)
        if next_token == 0 || next_token == 1 || next_token == 2 {
            println!("\n\n(stopped at EOS token)");
            break;
        }

        if step == args.max_tokens - 1 {
            println!("\n\n(reached max tokens)");
        }
    }

    // Print final summary
    let autoregressive_elapsed =
        autoregressive_start.map(|start| start.elapsed()).unwrap_or_default();
    let tokens_per_second = if autoregressive_elapsed.as_secs_f64() > 0.0 {
        generated_tokens.len() as f64 / autoregressive_elapsed.as_secs_f64()
    } else {
        0.0
    };
    println!(
        "\nGenerated {} tokens in {:.2}s ({:.2} tokens/sec)",
        generated_tokens.len(),
        autoregressive_elapsed.as_secs_f64(),
        tokens_per_second
    );
    println!("Token IDs: {:?}", generated_tokens);

    if let Some(ref tokenizer) = tokenizer {
        let full_text =
            tokenizer.decode(&tokens, false).unwrap_or_else(|_| "(decode error)".to_string());
        println!("\nFull text:\n{}", full_text);
    }

    Ok(())
}
