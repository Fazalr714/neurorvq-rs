#!/usr/bin/env python3
"""
Benchmark the Python NeuroRVQ reference implementation.

Mirrors the Rust bench.rs cases exactly:
  - EEG: 4/8/16/32/64 channels
  - ECG: 4/8/12/15 channels
  - EMG: 4/8/16 channels

Measures model construction and encode (tokenize) latency.
Outputs CSV to figures/benchmark_python.csv
"""

import sys, os, time, csv
import numpy as np
import torch
import yaml
from functools import partial
from torch import nn

# Add the NeuroRVQ repo to path
NEURORVQ_DIR = "/tmp/NeuroRVQ"
sys.path.insert(0, NEURORVQ_DIR)

WARMUP = 2
ITERS = 10

# ── Channel vocabularies (must match Rust) ─────────────────────────────────

EEG_CHANNELS = [
    'a1','a2','af3','af4','af7','af8','afz','c1','c2','c3','c4','c5','c6',
    'ccp1','ccp2','ccp3','ccp4','ccp5','ccp6','ccp7','ccp8',
    'cfc1','cfc2','cfc3','cfc4','cfc5','cfc6','cfc7','cfc8',
    'cp1','cp2','cp3','cp4','cp5','cp6','cpz','cz','eog',
    'f1','f10','f2','f3','f4','f5','f6','f7','f8','f9',
    'fc1','fc2','fc3','fc4','fc5','fc6','fcz','fp1','fp2','fpz',
    'ft7','ft8','fz','iz','loc','o1','o2','oz',
    'p08','p1','p10','p2','p3','p4','p5','p6','p7','p8','p9',
    'po1','po10','po2','po3','po4','po7','po8','po9','poz','pz',
    'roc','sp1','sp2',
    't1','t10','t2','t3','t4','t5','t6','t7','t8','t9',
    'tp10','tp7','tp8','tp9',
]
ECG_CHANNELS = ['avf','avl','avr','i','ii','iii','v1','v2','v3','v4','v5','v6','vx','vy','vz']
EMG_CHANNELS = ['c1','c10','c11','c12','c13','c14','c15','c16','c2','c3','c4','c5','c6','c7','c8','c9']


def create_embedding_ix(n_time, max_n_patches, n_channels, ch_indices):
    temp = torch.arange(max_n_patches - n_time, max_n_patches).repeat(n_channels).reshape(1, -1)
    spat = torch.tensor(ch_indices).repeat_interleave(n_time).reshape(1, -1)
    return temp, spat


def get_encoder_decoder_params(args):
    config = dict(
        patch_size=args['patch_size'], n_patches=args['n_patches'],
        n_global_electrodes=args['n_global_electrodes'],
        embed_dim=args['embed_dim'], num_heads=args['num_heads_tokenizer'],
        mlp_ratio=args['mlp_ratio_tokenizer'],
        qkv_bias=args['qkv_bias_tokenizer'], drop_rate=args['drop_rate_tokenizer'],
        attn_drop_rate=args['attn_drop_rate_tokenizer'],
        drop_path_rate=args['drop_path_rate_tokenizer'],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=args['init_values_tokenizer'],
        init_scale=args['init_scale_tokenizer'],
    )
    encoder_config = config.copy()
    encoder_config['in_chans'] = args['in_chans_encoder']
    encoder_config['depth'] = args['depth_encoder']
    encoder_config['num_classes'] = 0
    encoder_config['out_chans_encoder'] = args['out_chans_encoder']

    decoder_config = config.copy()
    decoder_config['in_chans'] = args['code_dim']
    decoder_config['depth'] = args['depth_decoder']
    decoder_config['num_classes'] = 0
    return encoder_config, decoder_config


def load_tokenizer(modality):
    """Construct a NeuroRVQ tokenizer for the given modality (no weights loaded)."""
    mod_dir = f"NeuroRVQ_{modality}"
    sys.path.insert(0, os.path.join(NEURORVQ_DIR, mod_dir))

    # Force reimport for each modality
    for key in list(sys.modules.keys()):
        if 'NeuroRVQ' in key and key != '__main__':
            del sys.modules[key]

    from NeuroRVQ import NeuroRVQTokenizer

    yml_path = os.path.join(NEURORVQ_DIR, "flags", f"NeuroRVQ_{modality}_v1.yml")
    with open(yml_path) as f:
        args = yaml.safe_load(f)

    if modality == "EEG":
        args['n_global_electrodes'] = len(EEG_CHANNELS)
    elif modality == "ECG":
        args['n_global_electrodes'] = len(ECG_CHANNELS)
    elif modality == "EMG":
        args['n_global_electrodes'] = len(EMG_CHANNELS)

    encoder_config, decoder_config = get_encoder_decoder_params(args)
    tokenizer = NeuroRVQTokenizer(
        encoder_config, decoder_config,
        n_code=args['n_code'],
        code_dim=args['code_dim'],
        decoder_out_dim=args['decoder_out_dim'],
    )
    tokenizer.eval()
    return tokenizer, args


def bench_case(modality, n_channels, n_time, label):
    if modality == "EEG":
        all_ch = EEG_CHANNELS
    elif modality == "ECG":
        all_ch = ECG_CHANNELS
    elif modality == "EMG":
        all_ch = EMG_CHANNELS

    n_channels = min(n_channels, len(all_ch))
    ch_indices = list(range(n_channels))

    # Measure construction
    t0 = time.perf_counter()
    tokenizer, args = load_tokenizer(modality)
    construct_ms = (time.perf_counter() - t0) * 1000

    patch_size = args['patch_size']
    n_patches = args['n_patches']
    n_samples = n_time * patch_size

    # Build input
    x = torch.randn(1, n_channels, n_samples) * 0.1
    temp_ix, spat_ix = create_embedding_ix(n_time, n_patches, n_channels, ch_indices)

    # --- Encode benchmark ---
    encode_times = []
    with torch.no_grad():
        for i in range(WARMUP + ITERS):
            x_in = x.clone()
            x_patched = x_in.reshape(1, n_channels, n_time, patch_size)
            t0 = time.perf_counter()
            quantize, code_ind, loss, usage = tokenizer.encode(
                x_patched, temp_ix.int(), spat_ix.int()
            )
            ms = (time.perf_counter() - t0) * 1000
            if i >= WARMUP:
                encode_times.append(ms)

    # --- Tokenize benchmark (encode + get indices) ---
    tok_times = []
    with torch.no_grad():
        for i in range(WARMUP + ITERS):
            x_in = x.clone()
            x_patched = x_in.reshape(1, n_channels, n_time, patch_size)
            t0 = time.perf_counter()
            output = tokenizer.get_tokens(x_patched, temp_ix.int(), spat_ix.int())
            # Force materialization
            _ = output['token'].cpu().numpy()
            ms = (time.perf_counter() - t0) * 1000
            if i >= WARMUP:
                tok_times.append(ms)

    enc_mean = np.mean(encode_times)
    enc_std = np.std(encode_times)
    tok_mean = np.mean(tok_times)
    tok_std = np.std(tok_times)

    return {
        'label': label,
        'modality': modality,
        'n_channels': n_channels,
        'n_time': n_time,
        'n_patches': n_patches,
        'patch_size': patch_size,
        'construct_ms': construct_ms,
        'encode_mean_ms': enc_mean,
        'encode_std_ms': enc_std,
        'tokenize_mean_ms': tok_mean,
        'tokenize_std_ms': tok_std,
    }


def main():
    print("NeuroRVQ Python benchmark")
    print("========================\n")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: CPU")
    print(f"Warmup: {WARMUP}, Iters: {ITERS}\n")

    cases = []
    # EEG
    for nch in [4, 8, 16, 32, 64]:
        nt = max(256 // nch, 1)
        cases.append(("EEG", nch, nt, f"EEG {nch}ch x{nt}t"))
    # ECG
    for nch in [4, 8, 12, 15]:
        nt = max(600 // nch, 1)
        cases.append(("ECG", nch, nt, f"ECG {nch}ch x{nt}t"))
    # EMG
    for nch in [4, 8, 16]:
        nt = max(256 // nch, 1)
        cases.append(("EMG", nch, nt, f"EMG {nch}ch x{nt}t"))

    results = []
    for i, (mod, nch, nt, label) in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {label} ... ", end="", flush=True)
        r = bench_case(mod, nch, nt, label)
        print(f"construct={r['construct_ms']:.0f}ms  "
              f"encode={r['encode_mean_ms']:.1f}+/-{r['encode_std_ms']:.1f}ms  "
              f"tokenize={r['tokenize_mean_ms']:.1f}+/-{r['tokenize_std_ms']:.1f}ms")
        results.append(r)

    # Write CSV
    os.makedirs("figures", exist_ok=True)
    csv_path = "figures/benchmark_python.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print(f"\nCSV: {csv_path}")
    print("Done!")


if __name__ == "__main__":
    main()
