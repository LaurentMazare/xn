use anyhow::{Context, Result, bail};
use clap::Parser;
use xn::quantized::{GgmlDType, QStorage, QTensor, gguf_file};
use xn::{CPU, TypedTensor, safetensors};

#[derive(Parser, Debug)]
#[command(name = "quantize")]
#[command(about = "Quantize some models")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    Asr {
        /// Input safetensors file
        model: String,

        /// Output GGUF file
        output: std::path::PathBuf,

        /// Quantization to apply to flow_lm transformer linear weights (e.g. q8_0, q6k)
        #[arg(long)]
        quant: Option<String>,

        #[arg(long)]
        force_bf16: bool,
    },
}

fn is_excluded(name: &str) -> bool {
    name.starts_with("mimi.decoder") || name.starts_with("mimi.upsample.")
}

const QUANT_SUFFIXES: &[&str] = &[
    "linear_in.weight",
    "linear_out.weight",
    "self_attn.in_proj_weight",
    "self_attn.out_proj.weight",
];

fn parse_quant(s: &str) -> Result<GgmlDType> {
    let dtype = match s {
        "q8" | "q8_0" => GgmlDType::Q8_0,
        "q8_1" => GgmlDType::Q8_1,
        "q8k" => GgmlDType::Q8K,
        "q6k" => GgmlDType::Q6K,
        "q5" | "q5_0" => GgmlDType::Q5_0,
        "q5_1" => GgmlDType::Q5_1,
        "q5k" => GgmlDType::Q5K,
        "q4" | "q4_0" => GgmlDType::Q4_0,
        "q4_1" => GgmlDType::Q4_1,
        "q4k" => GgmlDType::Q4K,
        other => bail!("unsupported quantization option '{other}'"),
    };
    Ok(dtype)
}

fn should_quantize(name: &str) -> bool {
    if !name.starts_with("lm.transformer.layers.") {
        return false;
    }
    QUANT_SUFFIXES.iter().any(|s| name.ends_with(s))
}

fn tensor_to_f32(tensor: &TypedTensor<xn::CpuDevice>) -> Result<Vec<f32>> {
    let vec = match tensor {
        TypedTensor::F32(t) => t.to_vec()?,
        TypedTensor::F16(t) => t.to_vec()?.into_iter().map(|v| v.to_f32()).collect(),
        TypedTensor::BF16(t) => t.to_vec()?.into_iter().map(|v| v.to_f32()).collect(),
        TypedTensor::I64(_) | TypedTensor::U8(_) => bail!("cannot quantize non-float tensor"),
    };
    Ok(vec)
}

fn as_is_qtensor(tensor: &TypedTensor<xn::CpuDevice>, force_bf16: bool) -> Result<QTensor> {
    let shape = tensor.shape().clone();
    let qtensor = match tensor {
        TypedTensor::F32(t) => {
            if force_bf16 {
                let t = t.to::<half::bf16>()?;
                QTensor::new(QStorage::Cpu(Box::new(t.to_vec()?)), shape)?
            } else {
                QTensor::new(QStorage::Cpu(Box::new(t.to_vec()?)), shape)?
            }
        }
        TypedTensor::F16(t) => QTensor::new(QStorage::Cpu(Box::new(t.to_vec()?)), shape)?,
        TypedTensor::BF16(t) => QTensor::new(QStorage::Cpu(Box::new(t.to_vec()?)), shape)?,
        TypedTensor::I64(_) => bail!("I64 tensors cannot be stored in GGUF"),
        TypedTensor::U8(_) => bail!("U8 tensors cannot be stored in GGUF"),
    };
    Ok(qtensor)
}

fn dtype_label(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::F32 => "f32",
        GgmlDType::F16 => "f16",
        GgmlDType::BF16 => "bf16",
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q8_1 => "q8_1",
        GgmlDType::Q2K => "q2k",
        GgmlDType::Q3K => "q3k",
        GgmlDType::Q4K => "q4k",
        GgmlDType::Q5K => "q5k",
        GgmlDType::Q6K => "q6k",
        GgmlDType::Q8K => "q8k",
    }
}

struct AsrFiles {
    lm: std::path::PathBuf,
    mimi: std::path::PathBuf,
}

impl AsrFiles {
    fn download_or_local(path_str: &str) -> Result<Self> {
        let path = std::path::Path::new(path_str);
        if path.is_dir() {
            println!("looking for safetensors files in {path_str}...");
            let lm = path.join("model.safetensors");
            if !lm.exists() {
                anyhow::bail!("LM safetensors not found at {lm:?}")
            }
            let mimi = path.join("mimi.safetensors");
            if !mimi.exists() {
                anyhow::bail!("Mimi safetensors not found at {mimi:?}")
            }
            Ok(AsrFiles { lm, mimi })
        } else {
            use hf_hub::{Repo, RepoType, api::sync::Api};
            println!("downloading ASR model from Hugging Face Hub {path_str}...");
            let api = Api::new()?;
            let repo = api.repo(Repo::new(path_str.to_string(), RepoType::Model));
            let lm = repo.get("model.safetensors").context("LM safetensors not found")?;
            let mimi = repo.get("mimi.safetensors").context("mimi safetensors not found")?;
            Ok(AsrFiles { lm, mimi })
        }
    }
}

fn main() -> Result<()> {
    let args = Cli::parse();
    match args.command {
        Command::Asr { model, output, quant, force_bf16 } => {
            let quant_dtype = quant.as_deref().map(parse_quant).transpose()?;

            let AsrFiles { lm, mimi } = AsrFiles::download_or_local(&model)?;
            let lm_tensors = safetensors::load_from_file(&lm, &CPU)?;
            let mimi_tensors = safetensors::load_from_file(&mimi, &CPU)?;
            let tensors = lm_tensors
                .into_iter()
                .map(|(n, t)| (format!("lm.{n}"), t))
                .chain(mimi_tensors.into_iter().map(|(n, t)| (format!("mimi.{n}"), t)))
                .collect::<std::collections::HashMap<_, _>>();

            let mut names: Vec<&String> = tensors.keys().collect();
            names.sort();

            let mut qtensors: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
            for name in names {
                if is_excluded(name) {
                    println!("skipping {name}");
                    continue;
                }
                let tensor = &tensors[name];
                let qtensor = match quant_dtype {
                    Some(dtype) if should_quantize(name) => {
                        let data = tensor_to_f32(tensor)?;
                        QTensor::quantize_f32(&data, tensor.shape(), dtype)?
                    }
                    _ => as_is_qtensor(tensor, force_bf16)?,
                };
                let size_mb = qtensor.storage_size_in_bytes() as f64 / (1024.0 * 1024.0);
                println!(
                    "{name}: {:?} {} -> {} ({size_mb:.2} MB)",
                    tensor.shape().dims(),
                    format!("{:?}", tensor.dtype()).to_lowercase(),
                    dtype_label(qtensor.dtype()),
                );
                qtensors.push((name.clone(), qtensor));
            }

            let refs: Vec<(&str, &QTensor)> =
                qtensors.iter().map(|(n, t)| (n.as_str(), t)).collect();

            println!("writing {output:?}");
            {
                let output = std::fs::File::create(&output)?;
                let mut writer = std::io::BufWriter::new(output);
                gguf_file::write(&mut writer, &[], &refs)?;
                drop(writer);
            }
            let size = std::fs::metadata(&output)?.len();
            println!("wrote {output:?} ({size} bytes)");
        }
    }
    Ok(())
}
