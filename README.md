# neurorvq-rs

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![Burn](https://img.shields.io/badge/burn-0.20-purple.svg)](https://burn.dev)

Pure-Rust inference for the [NeuroRVQ](https://github.com/KonstantinosBarmpas/NeuroRVQ) multi-scale biosignal tokenizer, built on [Burn 0.20](https://burn.dev).

NeuroRVQ tokenizes raw **EEG**, **ECG**, and **EMG** signals into discrete neural tokens using a multi-scale temporal encoder and Residual Vector Quantization (RVQ).

## Features

- **Full parity** with the Python reference for all three modalities (EEG, ECG, EMG)
- **Tokenizer pipeline**: encoder → encode heads → RVQ → decoder → decode heads → iFFT reconstruction
- **Standalone Foundation Model** encoder for fine-tuning workflows
- **Zero Python dependencies** — loads upstream YAML configs and safetensors weights directly
- **CPU** (NdArray + Rayon / Apple Accelerate) and **GPU** (wgpu — Metal / Vulkan) backends
- **CLI** with per-parameter overrides on top of YAML configs

## Architecture

```
Raw Signal [B, N, T]
    │
    ▼
┌─────────────────────────────────────┐
│  Multi-Scale Temporal Conv (4 branches) │  ← modality-specific kernels
│  EEG/ECG: 21/15/9/5   EMG: 51/17/8/5   │
└─────────────────────────────────────┘
    │ ×4 branches
    ▼
┌───────────────────────┐
│  Transformer Encoder  │  12 blocks, 10 heads, embed_dim=200
│  (spatial + temporal   │
│   positional embeds)   │
└───────────────────────┘
    │ ×4 branches
    ▼
┌───────────────────────┐
│  Encode Heads         │  Linear → Tanh → Linear (→ code_dim)
└───────────────────────┘
    │ ×4 branches
    ▼
┌───────────────────────┐
│  Residual VQ          │  8 levels (EEG/ECG) or 16 levels (EMG)
│  (L2-normalized       │  codebook_size=8192, code_dim=128
│   codebook lookup)    │
└───────────────────────┘
    │ ×4 branches           ← Token indices output here
    ▼
┌───────────────────────┐
│  Transformer Decoder  │  3 blocks (PatchEmbed 1×1 conv input)
└───────────────────────┘
    │ concat 4 branches
    ▼
┌───────────────────────┐
│  Decode Heads         │  Amplitude (GELU), Sin/Cos phase (Tanh)
└───────────────────────┘
    │
    ▼
  Inverse FFT → Reconstructed Signal
```

## Supported Modalities

| Modality | Channels | Patch Size | Conv Kernels (G1) | Conv Kernels (G2) | RVQ Levels |
|----------|----------|-----------|-------------------|-------------------|------------|
| **EEG**  | 103      | 200       | 21, 15, 9, 5     | 9, 7, 5, 3       | 8          |
| **ECG**  | 15       | 40        | 21, 15, 9, 5     | 9, 7, 5, 3       | 8          |
| **EMG**  | 16       | 200       | 51, 17, 8, 5     | 25, 9, 4, 3      | 16         |

## Quick Start

### Build

```bash
# CPU (default — NdArray + Rayon)
cargo build --release

# CPU with Apple Accelerate BLAS
cargo build --release --features blas-accelerate

# GPU (Metal on macOS)
cargo build --release --no-default-features --features metal
```

### CLI — Tokenize

```bash
# Auto-detects EEG modality from the YAML filename:
cargo run --release --bin infer -- \
    --config flags/NeuroRVQ_EEG_v1.yml \
    --weights model.safetensors

# Explicit modality + full reconstruction:
cargo run --release --bin infer -- \
    --config flags/NeuroRVQ_ECG_v1.yml \
    --weights model.safetensors \
    --modality ECG \
    --mode forward

# Foundation model encoder:
cargo run --release --bin infer -- \
    --config flags/NeuroRVQ_EEG_v1.yml \
    --weights fm_weights.safetensors \
    --mode fm
```

### CLI — Override Config Parameters

Any YAML value can be overridden from the command line:

```bash
cargo run --release --bin infer -- \
    --config flags/NeuroRVQ_EEG_v1.yml \
    --weights model.safetensors \
    --embed-dim 64 \
    --depth-encoder 6 \
    --depth-decoder 2 \
    --n-code 4096
```

Run `--help` for all available flags.

### Library API

```rust
use neurorvq_rs::{NeuroRVQEncoder, Modality, data, channels};
use std::path::Path;

// Load tokenizer (modality auto-detected from YAML filename)
let (model, load_ms) = NeuroRVQEncoder::<B>::load(
    Path::new("flags/NeuroRVQ_EEG_v1.yml"),
    Path::new("model.safetensors"),
    device,
)?;

// Build input batch
let batch = data::build_batch_with_modality(
    signal_data,
    &channel_names,
    n_time, model.config.n_patches,
    n_channels, n_samples,
    Modality::EEG, &device,
);

// Tokenize → 4 branches × 8 RVQ levels of token indices
let tokens = model.tokenize(&batch)?;

// Full forward: encode → quantize → decode → iFFT → standardized signals
let result = model.forward(&batch)?;
// result.original_std, result.reconstructed_std
```

### Foundation Model API

```rust
use neurorvq_rs::{NeuroRVQFoundationModel, Modality};

let (fm, _ms) = NeuroRVQFoundationModel::<B>::load(
    Path::new("flags/NeuroRVQ_EEG_v1.yml"),
    Path::new("fm_weights.safetensors"),
    Modality::EEG,
    device,
)?;

// 4 branch feature vectors
let features = fm.encode(&batch)?;

// Mean-pooled representation for classification
let pooled = fm.encode_pooled(&batch)?;
```

### Config Overrides (Library)

```rust
use neurorvq_rs::{NeuroRVQEncoder, ConfigOverrides, Modality};

let overrides = ConfigOverrides {
    embed_dim: Some(64),
    depth_encoder: Some(6),
    ..Default::default()
};

let (model, _ms) = NeuroRVQEncoder::<B>::load_full(
    Path::new("flags/NeuroRVQ_EEG_v1.yml"),
    Path::new("model.safetensors"),
    Modality::EEG,
    Some(&overrides),
    device,
)?;
```

## Weight Conversion

NeuroRVQ ships `.pt` (PyTorch pickle) weights. Convert to safetensors:

```python
import torch
from safetensors.torch import save_file

state_dict = torch.load("model.pt", map_location="cpu")
# Flatten nested state dicts if needed
save_file(state_dict, "model.safetensors")
```

## Benchmarks

**Platform:** Apple M4 Pro, 64 GB RAM, macOS (arm64)
**Backend:** NdArray + Rayon (CPU, multi-threaded)

| Configuration | Modality | NdArray CPU (ms) | wgpu GPU (ms) |
|---|---|---:|---:|
| EEG 4ch ×64t | EEG | 848 | 445 |
| EEG 16ch ×16t | EEG | 849 | 432 |
| EEG 64ch ×4t | EEG | 854 | 411 |
| ECG 4ch ×150t | ECG | 2212 | — |
| ECG 15ch ×40t | ECG | 2224 | — |
| EMG 4ch ×64t | EMG | 1343 | 617 |
| EMG 16ch ×16t | EMG | 1349 | 660 |

### Tokenize Latency

![Tokenize Latency](figures/tokenize_latency.svg)

### Encode Latency

![Encode Latency](figures/encode_latency.svg)

### Model Construction Time

![Construction Time](figures/construction_time.svg)

### EEG Scaling by Channel Count

![EEG Scaling](figures/eeg_scaling.svg)

### Key Observations

- **EEG** tokenization runs at ~**833 ms** per sample (256 total patches, 12-layer encoder + 3-layer decoder, 200-dim), largely independent of channel/time decomposition
- **ECG** is ~**2.6× slower** due to 600 total patches (vs 256) despite smaller embed_dim (40)
- **EMG** is ~**1.6× slower** than EEG due to 16 RVQ quantizer levels (vs 8) and larger conv kernels
- **Construction time** is ~12 ms for subsequent models (60 ms cold start for the first)
- **Latency is dominated by the transformer** — the total patch count is the primary scaling factor, not channel vs time decomposition
- Standard deviation is consistently **< 1%** of the mean, indicating stable performance

## Rust vs Python Comparison

**Platform:** Apple M4 Pro, 64 GB RAM, macOS (arm64)  
**Rust backends:** Burn NdArray+Rayon (CPU), Burn wgpu (GPU)  
**Python:** PyTorch 2.8.0 (CPU)

| Configuration | Modality | NdArray (ms) | wgpu (ms) | PyTorch (ms) | GPU speedup |
|---|---|---:|---:|---:|---:|
| EEG 4ch ×64t | EEG | 848 | **445** | 179 | 1.9× vs NdArray |
| EEG 16ch ×16t | EEG | 849 | **432** | 180 | 2.0× vs NdArray |
| EEG 64ch ×4t | EEG | 854 | **411** | 179 | 2.1× vs NdArray |
| ECG 15ch ×40t | ECG | 2224 | —¹ | 272 | — |
| EMG 4ch ×64t | EMG | 1343 | **617** | 255 | 2.2× vs NdArray |
| EMG 16ch ×16t | EMG | 1349 | **660** | 254 | 2.0× vs NdArray |

¹ ECG skipped on wgpu due to burn-wgpu shared memory limitation with embed_dim=40

### Tokenize Latency: All Backends

![Tokenize Comparison](figures/compare_all_tokenize.svg)

### Summary

| Comparison | Ratio |
|---|---:|
| **wgpu vs NdArray** | wgpu is **2.0× faster** (GPU acceleration) |
| **wgpu vs PyTorch** | PyTorch is **2.4× faster** than wgpu |
| **NdArray vs PyTorch** | PyTorch is **4.7× faster** than NdArray |

### Why PyTorch is faster

PyTorch 2.8 on Apple Silicon uses highly optimized **Apple Accelerate / AMX** BLAS kernels and **fused operators** for matrix multiplication, attention, and layer normalization. The Burn backends are improving rapidly but don’t yet have the same level of hardware-specific kernel fusion.

### Why Rust matters anyway

- **Zero Python dependencies** — no interpreter, no GIL, no pip
- **Embeddable** — real-time pipelines, mobile apps, edge devices, WASM
- **wgpu GPU already halves latency** vs CPU, and Burn’s GPU kernels are actively improving
- **Deterministic memory** — no garbage collector pauses
- **Single static binary** — deploy anywhere without a Python environment

### Re-run Benchmarks

```bash
# Rust only
cargo run --release --bin bench

# Python only
python3 scripts/bench_python.py

# Generate comparison charts
python3 scripts/compare_benchmarks.py
```

Results are written to `figures/`.

## Project Structure

```
src/
├── lib.rs                  # Public API re-exports
├── config.rs               # YAML config parser + ConfigOverrides
├── channels.rs             # EEG/ECG/EMG channel vocabularies
├── data.rs                 # Input batch construction
├── encoder.rs              # High-level NeuroRVQEncoder + FoundationModel
├── weights.rs              # Safetensors weight loading
├── model/
│   ├── foundation.rs       # NeuroRVQFM (encoder/decoder transformer)
│   ├── tokenizer.rs        # Full tokenizer pipeline with FFT/iFFT
│   ├── multi_scale_conv.rs # Modality-specific inception-style convolutions
│   ├── attention.rs        # Multi-head attention with QKV bias + QK norm
│   ├── encoder_block.rs    # Pre-norm transformer block with layer scale
│   ├── feedforward.rs      # MLP (GELU activation)
│   ├── norm.rs             # LayerNorm wrapper
│   ├── patch_embed.rs      # 1×1 Conv2d decoder patch embedding
│   ├── quantizer.rs        # L2-normalized vector quantizer
│   └── rvq.rs              # Residual Vector Quantization
├── bin/
│   ├── infer.rs            # CLI for tokenization / reconstruction
│   └── bench.rs            # Benchmark suite
flags/
├── NeuroRVQ_EEG_v1.yml    # Upstream EEG config
├── NeuroRVQ_ECG_v1.yml    # Upstream ECG config
└── NeuroRVQ_EMG_v1.yml    # Upstream EMG config
```

## License

Apache-2.0

## References

- [NeuroRVQ: Joint Neurophysiological Multi-Scale Temporal Tokenization and Reconstruction](https://github.com/KonstantinosBarmpas/NeuroRVQ)
- [Burn ML Framework](https://burn.dev)
