/// NeuroRVQ inference CLI.
///
/// Build — CPU (default):
///   cargo build --release
///
/// Build — GPU:
///   cargo build --release --no-default-features --features wgpu
///
/// Usage:
///   # Minimal — modality auto-detected from YAML filename:
///   infer --weights model.safetensors --config flags/NeuroRVQ_EEG_v1.yml
///
///   # Override modality and some parameters:
///   infer --weights m.safetensors --config flags/NeuroRVQ_ECG_v1.yml \
///         --modality ECG --embed-dim 64 --depth-encoder 6 --mode reconstruct
///
///   # Foundation model mode:
///   infer --weights fm.safetensors --config flags/NeuroRVQ_EEG_v1.yml --mode fm

use std::{path::Path, time::Instant};
use clap::Parser;
use neurorvq_rs::{
    NeuroRVQEncoder, NeuroRVQFoundationModel, NeuroRVQConfig, ConfigOverrides, Modality,
    data, channels,
};

// ── Backend ───────────────────────────────────────────────────────────────────
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub use burn::backend::{Wgpu as B, wgpu::WgpuDevice as Device};
    pub fn device() -> Device { Device::DefaultDevice }
    #[cfg(feature = "metal")]
    pub const NAME: &str = "GPU (wgpu — Metal / MSL shaders)";
    #[cfg(feature = "vulkan")]
    pub const NAME: &str = "GPU (wgpu — Vulkan / SPIR-V shaders)";
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    pub const NAME: &str = "GPU (wgpu — WGSL shaders)";
}

#[cfg(feature = "ndarray")]
mod backend {
    pub use burn::backend::NdArray as B;
    pub type Device = burn::backend::ndarray::NdArrayDevice;
    pub fn device() -> Device { Device::Cpu }
    #[cfg(feature = "blas-accelerate")]
    pub const NAME: &str = "CPU (NdArray + Apple Accelerate)";
    #[cfg(not(feature = "blas-accelerate"))]
    pub const NAME: &str = "CPU (NdArray + Rayon)";
}

use backend::{B, device};

// ── CLI ───────────────────────────────────────────────────────────────────────
#[derive(Parser, Debug)]
#[command(about = "NeuroRVQ biosignal tokenizer inference (Burn 0.20.1)")]
struct Args {
    /// Safetensors weights file (converted from .pt).
    #[arg(long)]
    weights: String,

    /// YAML config file (e.g. flags/NeuroRVQ_EEG_v1.yml).
    /// Can be an upstream flag file — unknown keys are ignored.
    #[arg(long)]
    config: String,

    /// Signal modality: EEG, ECG, or EMG.
    /// Auto-detected from YAML filename if omitted.
    #[arg(long)]
    modality: Option<String>,

    /// Mode: "tokenize", "reconstruct", "forward", or "fm".
    #[arg(long, default_value = "tokenize")]
    mode: String,

    /// Print details.
    #[arg(long, short = 'v')]
    verbose: bool,

    // ── Config overrides (all optional; override values from YAML) ────────

    /// Patch size (samples per patch). EEG/EMG default: 200, ECG: 40.
    #[arg(long)]
    patch_size: Option<usize>,

    /// Maximum number of patches (n_channels × n_time). Default: 256 (EEG/EMG), 600 (ECG).
    #[arg(long)]
    n_patches: Option<usize>,

    /// Embedding dimension. Default: 200 (EEG/EMG), 40 (ECG).
    #[arg(long)]
    embed_dim: Option<usize>,

    /// Codebook vector dimension. Default: 128.
    #[arg(long)]
    code_dim: Option<usize>,

    /// Codebook vocabulary size. Default: 8192.
    #[arg(long)]
    n_code: Option<usize>,

    /// Decoder output dimension. Default: 200 (EEG/EMG), 40 (ECG).
    #[arg(long)]
    decoder_out_dim: Option<usize>,

    /// Output channels of encoder's multi-scale conv. Default: 8.
    #[arg(long)]
    out_chans_encoder: Option<usize>,

    /// Number of transformer blocks in the encoder. Default: 12.
    #[arg(long)]
    depth_encoder: Option<usize>,

    /// Number of transformer blocks in the decoder. Default: 3.
    #[arg(long)]
    depth_decoder: Option<usize>,

    /// Number of attention heads. Default: 10.
    #[arg(long)]
    num_heads: Option<usize>,

    /// MLP expansion ratio. Default: 4.0.
    #[arg(long)]
    mlp_ratio: Option<f64>,

    /// Layer-scale initial value. Default: 0.0 (disabled).
    #[arg(long)]
    init_values: Option<f64>,

    /// Init scale for output heads. Default: 0.001.
    #[arg(long)]
    init_scale: Option<f64>,

    /// Enable / disable QKV bias. Default: true.
    #[arg(long)]
    qkv_bias: Option<bool>,

    /// Override number of global electrodes (normally set from modality).
    #[arg(long)]
    n_global_electrodes: Option<usize>,
}

impl Args {
    /// Collect the optional CLI flags into a ConfigOverrides.
    fn overrides(&self) -> ConfigOverrides {
        ConfigOverrides {
            patch_size: self.patch_size,
            n_patches: self.n_patches,
            embed_dim: self.embed_dim,
            code_dim: self.code_dim,
            n_code: self.n_code,
            decoder_out_dim: self.decoder_out_dim,
            out_chans_encoder: self.out_chans_encoder,
            depth_encoder: self.depth_encoder,
            depth_decoder: self.depth_decoder,
            num_heads_tokenizer: self.num_heads,
            mlp_ratio_tokenizer: self.mlp_ratio,
            qkv_bias_tokenizer: self.qkv_bias,
            init_values_tokenizer: self.init_values,
            init_scale_tokenizer: self.init_scale,
            n_global_electrodes: self.n_global_electrodes,
        }
    }

    fn has_overrides(&self) -> bool {
        let o = self.overrides();
        o.patch_size.is_some() || o.n_patches.is_some() || o.embed_dim.is_some()
            || o.code_dim.is_some() || o.n_code.is_some() || o.decoder_out_dim.is_some()
            || o.out_chans_encoder.is_some() || o.depth_encoder.is_some()
            || o.depth_decoder.is_some() || o.num_heads_tokenizer.is_some()
            || o.mlp_ratio_tokenizer.is_some() || o.qkv_bias_tokenizer.is_some()
            || o.init_values_tokenizer.is_some() || o.init_scale_tokenizer.is_some()
            || o.n_global_electrodes.is_some()
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();
    let dev = device();
    let modality: Modality = match &args.modality {
        Some(m) => m.parse()?,
        None => {
            let cfg = NeuroRVQConfig::from_yaml(&args.config)?;
            let m = cfg.resolve_modality();
            eprintln!("Auto-detected modality: {m} (from config filename)");
            m
        }
    };

    println!("Backend  : {}", backend::NAME);
    println!("Modality : {modality}");

    match args.mode.as_str() {
        "fm" => run_fm(&args, modality, dev, t0),
        _ => run_tokenizer(&args, modality, dev, t0),
    }
}

fn run_tokenizer(
    args: &Args,
    modality: Modality,
    dev: backend::Device,
    t0: Instant,
) -> anyhow::Result<()> {
    let overrides = args.overrides();
    let ovr = if args.has_overrides() { Some(&overrides) } else { None };

    let (model, ms_weights) = NeuroRVQEncoder::<B>::load_full(
        Path::new(&args.config),
        Path::new(&args.weights),
        modality,
        ovr,
        dev.clone(),
    )?;

    println!("Model    : {}  ({ms_weights:.0} ms)", model.describe());

    let batch = make_dummy_batch(&model.config, modality, &dev);

    let t_inf = Instant::now();

    match args.mode.as_str() {
        "tokenize" => {
            let result = model.tokenize(&batch)?;
            let ms_infer = t_inf.elapsed().as_secs_f64() * 1000.0;
            println!("Tokens   : {} branches × {} RVQ levels  ({ms_infer:.1} ms)",
                result.branch_tokens.len(),
                result.branch_tokens[0].len(),
            );
            if args.verbose {
                for (br, tokens) in result.branch_tokens.iter().enumerate() {
                    for (lvl, indices) in tokens.iter().enumerate() {
                        println!("  Branch {br} Level {lvl}: {} indices, first 5: {:?}",
                            indices.len(), &indices[..5.min(indices.len())]);
                    }
                }
            }
        }
        "reconstruct" => {
            let result = model.reconstruct(&batch)?;
            let ms_infer = t_inf.elapsed().as_secs_f64() * 1000.0;
            println!("Output   : shape={:?}  ({ms_infer:.1} ms)", result.shape);
            if args.verbose {
                let mean: f64 = result.amplitude.iter().map(|&v| v as f64).sum::<f64>()
                    / result.amplitude.len() as f64;
                println!("  amp mean={mean:+.4}  len={}", result.amplitude.len());
            }
        }
        "forward" => {
            let result = model.forward(&batch)?;
            let ms_infer = t_inf.elapsed().as_secs_f64() * 1000.0;
            println!("Forward  : shape={:?}  ({ms_infer:.1} ms)", result.shape);
            if args.verbose {
                let orig_mean: f64 = result.original_std.iter().map(|&v| v as f64).sum::<f64>()
                    / result.original_std.len() as f64;
                let recon_mean: f64 = result.reconstructed_std.iter().map(|&v| v as f64).sum::<f64>()
                    / result.reconstructed_std.len() as f64;
                println!("  orig_std mean={orig_mean:+.6}  recon_std mean={recon_mean:+.6}");
            }
        }
        other => {
            anyhow::bail!("Unknown mode: {other}. Use 'tokenize', 'reconstruct', 'forward', or 'fm'.");
        }
    }

    print_timing(ms_weights, t0);
    Ok(())
}

fn run_fm(
    args: &Args,
    modality: Modality,
    dev: backend::Device,
    t0: Instant,
) -> anyhow::Result<()> {
    let (fm, ms_weights) = NeuroRVQFoundationModel::<B>::load(
        Path::new(&args.config),
        Path::new(&args.weights),
        modality,
        dev.clone(),
    )?;

    println!("FM       : {}  ({ms_weights:.0} ms)", fm.describe());

    let batch = make_dummy_batch_fm(&fm.config, modality, &dev);

    let t_inf = Instant::now();
    let result = fm.encode(&batch)?;
    let ms_infer = t_inf.elapsed().as_secs_f64() * 1000.0;

    println!("Features : {} branches × shape={:?}  ({ms_infer:.1} ms)",
        result.branch_features.len(), result.shape);

    if args.verbose {
        for (i, feats) in result.branch_features.iter().enumerate() {
            let mean: f64 = feats.iter().map(|&v| v as f64).sum::<f64>() / feats.len() as f64;
            println!("  Branch {i}: len={} mean={mean:+.6}", feats.len());
        }
    }

    print_timing(ms_weights, t0);
    Ok(())
}

fn make_dummy_batch(
    config: &neurorvq_rs::NeuroRVQConfig,
    modality: Modality,
    dev: &backend::Device,
) -> data::InputBatch<B> {
    let ch = channels::global_channels(modality);
    let n_channels = ch.len().min(16);
    let channel_names: Vec<&str> = ch[..n_channels].to_vec();
    let patch_size = config.patch_size;
    let n_time = channels::compute_n_time(config.n_patches, n_channels);
    let n_samples = n_time * patch_size;

    let signal = vec![0.0f32; n_channels * n_samples];
    data::build_batch_with_modality(
        signal, &channel_names, n_time, config.n_patches,
        n_channels, n_samples, modality, dev,
    )
}

fn make_dummy_batch_fm(
    config: &neurorvq_rs::NeuroRVQConfig,
    modality: Modality,
    dev: &backend::Device,
) -> data::InputBatch<B> {
    make_dummy_batch(config, modality, dev)
}

fn print_timing(ms_weights: f64, t0: Instant) {
    let ms_total = t0.elapsed().as_secs_f64() * 1000.0;
    println!("── Timing ───────────────────────────────────────────────────────");
    println!("  Weights  : {ms_weights:.0} ms");
    println!("  Total    : {ms_total:.0} ms");
}
