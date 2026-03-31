# Changelog

## 0.1.0 (2025-03-31)

Initial release — full inference parity with [NeuroRVQ](https://github.com/KonstantinosBarmpas/NeuroRVQ) Python reference.

### Features

- **Three modalities**: EEG (103 ch), ECG (15 ch), EMG (16 ch) with modality-specific conv kernels and RVQ levels
- **Tokenizer pipeline**: multi-scale temporal conv → transformer encoder → encode heads → RVQ (8 or 16 levels) → transformer decoder → decode heads → inverse FFT reconstruction
- **Standalone Foundation Model** encoder with mean-pooled output for downstream tasks
- **YAML config loading**: reads upstream `flags/NeuroRVQ_*.yml` files directly; auto-detects modality from filename
- **CLI overrides**: any config parameter can be overridden from the command line
- **Safetensors weight loading** for both tokenizer and standalone FM weights
- **CPU backends**: NdArray + Rayon, NdArray + Apple Accelerate (BLAS)
- **GPU backends**: wgpu with Metal (macOS) or Vulkan support
- **Benchmark suite** with CSV output and SVG charts
