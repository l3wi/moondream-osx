#!/usr/bin/env python3
"""
Quantize Moondream3 to int2 or int3 for iOS deployment.

The default md3p-int4 is 6.96 GB because:
- MoE experts: int4
- Other weights: bf16 (unchanged)

This script quantizes ALL weights more aggressively:
- int3: ~50% smaller than int4 (~3.5 GB target)
- int2: ~50% smaller than int3 (~1.75 GB target, may lose quality)

Usage:
    python quantize_moondream3.py --bits 3 --output ./md3-int3
    python quantize_moondream3.py --bits 2 --output ./md3-int2
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download


def load_model_weights(model_path: Path) -> dict:
    """Load safetensors weights from a directory."""
    import safetensors.numpy

    weights = {}
    for sf_file in sorted(model_path.glob("*.safetensors")):
        print(f"Loading {sf_file.name}...")
        with safetensors.safe_open(sf_file, framework="numpy") as f:
            for key in f.keys():
                weights[key] = mx.array(f.get_tensor(key))

    return weights


def quantize_weight(weight: mx.array, bits: int, group_size: int = 64) -> tuple:
    """
    Quantize a single weight tensor.
    Returns (quantized_weight, scales, biases) for affine quantization.
    """
    # Only quantize 2D weights (linear layers)
    if weight.ndim != 2:
        return weight, None, None

    # Use MLX's built-in quantization
    return mx.quantize(weight, bits=bits, group_size=group_size)


def quantize_model_weights(weights: dict, bits: int, group_size: int = 64) -> dict:
    """
    Quantize all linear layer weights in the model.
    """
    quantized = {}
    stats = {"quantized": 0, "skipped": 0, "original_bytes": 0, "quantized_bytes": 0}

    for key, value in weights.items():
        original_size = value.nbytes
        stats["original_bytes"] += original_size

        # Quantize weight matrices (typically end with .weight and are 2D)
        if value.ndim == 2 and value.shape[0] > 64 and value.shape[1] > 64:
            # This looks like a weight matrix worth quantizing
            try:
                q_weight, scales, biases = mx.quantize(value, bits=bits, group_size=group_size)

                quantized[key] = q_weight
                quantized[key.replace(".weight", ".scales")] = scales
                quantized[key.replace(".weight", ".biases")] = biases

                new_size = q_weight.nbytes + scales.nbytes + biases.nbytes
                stats["quantized_bytes"] += new_size
                stats["quantized"] += 1

                print(f"  Quantized {key}: {value.shape} ({original_size/1e6:.1f}MB -> {new_size/1e6:.1f}MB)")
            except Exception as e:
                print(f"  Skipped {key}: {e}")
                quantized[key] = value
                stats["quantized_bytes"] += original_size
                stats["skipped"] += 1
        else:
            # Keep as-is (embeddings, small tensors, biases, etc.)
            quantized[key] = value
            stats["quantized_bytes"] += original_size
            stats["skipped"] += 1

    return quantized, stats


def save_quantized_weights(weights: dict, output_path: Path, config: dict):
    """Save quantized weights as safetensors."""
    import safetensors.numpy as sf
    import numpy as np

    output_path.mkdir(parents=True, exist_ok=True)

    # Convert MLX arrays to numpy for safetensors
    np_weights = {}
    for key, value in weights.items():
        np_weights[key] = np.array(value)

    # Save weights
    output_file = output_path / "model.safetensors"
    print(f"Saving to {output_file}...")
    sf.save_file(np_weights, str(output_file))

    # Save config
    config_file = output_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved quantized model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Quantize Moondream3 for iOS")
    parser.add_argument("--model", type=str, default="moondream/moondream3-preview",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4],
                        help="Quantization bits (2, 3, or 4)")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Quantization group size")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for quantized model")

    args = parser.parse_args()

    print(f"=== Moondream3 Quantization ===")
    print(f"Model: {args.model}")
    print(f"Bits: {args.bits}")
    print(f"Group size: {args.group_size}")
    print(f"Output: {args.output}")
    print()

    # Download or locate model
    if Path(args.model).exists():
        model_path = Path(args.model)
    else:
        print(f"Downloading {args.model}...")
        model_path = Path(snapshot_download(args.model))

    print(f"Model path: {model_path}")

    # Load weights
    print("\nLoading weights...")
    weights = load_model_weights(model_path)
    print(f"Loaded {len(weights)} weight tensors")

    # Load config
    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config = {}

    # Add quantization info to config
    config["quantization"] = {
        "bits": args.bits,
        "group_size": args.group_size,
        "method": "affine"
    }

    # Quantize
    print(f"\nQuantizing to {args.bits}-bit...")
    quantized_weights, stats = quantize_model_weights(
        weights,
        bits=args.bits,
        group_size=args.group_size
    )

    # Print stats
    print(f"\n=== Quantization Stats ===")
    print(f"Quantized layers: {stats['quantized']}")
    print(f"Skipped layers: {stats['skipped']}")
    print(f"Original size: {stats['original_bytes']/1e9:.2f} GB")
    print(f"Quantized size: {stats['quantized_bytes']/1e9:.2f} GB")
    print(f"Compression: {stats['original_bytes']/stats['quantized_bytes']:.1f}x")

    # Save
    output_path = Path(args.output)
    save_quantized_weights(quantized_weights, output_path, config)

    print(f"\n=== Done! ===")
    print(f"Quantized model saved to: {output_path}")
    print(f"To use in MLX Swift, update the model path to: {output_path}")


if __name__ == "__main__":
    main()
