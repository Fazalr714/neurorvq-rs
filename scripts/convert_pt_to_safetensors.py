#!/usr/bin/env python3
"""Convert NeuroRVQ .pt weights to .safetensors format for Rust inference.

Usage:
    python scripts/convert_pt_to_safetensors.py \
        --input pretrained_models/tokenizers/NeuroRVQ_EEG_tokenizer_v1.pt \
        --output weights/NeuroRVQ_EEG_tokenizer_v1.safetensors
"""

import argparse
import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Convert .pt to .safetensors")
    parser.add_argument("--input", required=True, help="Input .pt file")
    parser.add_argument("--output", required=True, help="Output .safetensors file")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    state_dict = torch.load(args.input, map_location="cpu")

    # If wrapped in a checkpoint dict, extract the model state
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Convert all tensors to float32 and ensure contiguous
    converted = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            t = tensor.float().contiguous()
            converted[key] = t
            print(f"  {key:80s}  {list(t.shape)}")
        else:
            print(f"  SKIP {key} (not a tensor)")

    print(f"\nSaving {len(converted)} tensors to {args.output}...")
    save_file(converted, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
