//! # neurorvq-rs — NeuroRVQ Biosignal Tokenizer inference in Rust
//!
//! Pure-Rust inference for the NeuroRVQ multi-scale biosignal tokenizer,
//! built on [Burn 0.20](https://burn.dev).
//!
//! NeuroRVQ tokenizes raw EEG/EMG/ECG signals into discrete neural tokens
//! using a multi-scale temporal encoder and Residual Vector Quantization (RVQ).
//!
//! ## Supported modalities
//!
//! - **EEG** (103 channels, fs=200 Hz, conv kernels: 21/15/9/5)
//! - **ECG** (15 channels, fs=200 Hz, conv kernels: 21/15/9/5)
//! - **EMG** (16 channels, fs=1000 Hz, conv kernels: 51/17/8/5)
//!
//! ## Quick start (Tokenizer)
//!
//! ```rust,ignore
//! use neurorvq_rs::{NeuroRVQEncoder, Modality};
//!
//! let (model, _ms) = NeuroRVQEncoder::<B>::load_with_modality(
//!     Path::new("config.yml"),
//!     Path::new("model.safetensors"),
//!     Modality::EEG,
//!     device,
//! )?;
//! ```
//!
//! ## Quick start (Foundation Model)
//!
//! ```rust,ignore
//! use neurorvq_rs::{NeuroRVQFoundationModel, Modality};
//!
//! let (fm, _ms) = NeuroRVQFoundationModel::<B>::load(
//!     Path::new("config.yml"),
//!     Path::new("fm_weights.safetensors"),
//!     Modality::EEG,
//!     device,
//! )?;
//! ```

pub mod channels;
pub mod config;
pub mod data;
pub mod encoder;
pub mod model;
pub mod weights;

// Flat re-exports
pub use encoder::{
    NeuroRVQEncoder, NeuroRVQFoundationModel,
    TokenResult, ReconstructionResult, ForwardResult, FMEncoderResult,
};
pub use config::{NeuroRVQConfig, ConfigOverrides, Modality};
pub use data::{InputBatch, build_batch, build_batch_with_modality, channel_wise_normalize};
pub use channels::{
    EEG_CHANNELS, EEG_VOCAB_SIZE,
    ECG_CHANNELS, ECG_VOCAB_SIZE,
    EMG_CHANNELS, EMG_VOCAB_SIZE,
    global_channels, global_vocab_size,
    channel_index, channel_indices,
    create_embedding_ix, filter_channels,
    compute_n_time, create_patches,
};
