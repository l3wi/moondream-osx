#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "safetensors>=0.4",
#     "mlx>=0.21",
#     "huggingface-hub>=0.20",
#     "numpy>=1.24",
# ]
# ///
"""
Convert Moondream3 BF16 model to int8 quantized format for MLX.

This script converts the BF16 weights from moondream/moondream3-preview to int8
quantized format compatible with MoondreamKit. Only MoE expert layers (blocks 4-23)
are quantized; vision, attention, and other layers remain in BF16.

Usage:
    uv run tools/convert_bf16_to_mlx.py --output /path/to/output
    uv run tools/convert_bf16_to_mlx.py --bits 4 --output /path/to/output  # for int4
"""

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file as torch_save_file


def find_hf_cache_dir(model_id: str = "moondream/moondream3-preview") -> Path:
    """Find the HuggingFace cache directory for the source model."""
    # Try to find in cache first
    cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir_name = f"models--{model_id.replace('/', '--')}"
    model_cache = cache_base / model_dir_name

    if model_cache.exists():
        # Find the latest snapshot
        snapshots_dir = model_cache / "snapshots"
        if snapshots_dir.exists():
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                return max(snapshots, key=lambda p: p.stat().st_mtime)

    # Not in cache, download it
    print(f"Downloading {model_id} from HuggingFace...")
    return Path(snapshot_download(model_id))


def transform_key(key: str) -> str:
    """Transform weight key from PyTorch format to MoondreamKit format.

    Transformations:
    - Strip 'model.' prefix if present
    - text.wte → text.wte.weight (for Embedding module)
    - tau.alpha → tau_alpha
    - tau.wq → tau_wq
    - tau.wv → tau_wv
    """
    new_key = key
    if new_key.startswith("model."):
        new_key = new_key[6:]  # Remove "model." prefix

    # Transform tau keys: tau.alpha → tau_alpha, etc.
    new_key = new_key.replace(".tau.alpha", ".tau_alpha")
    new_key = new_key.replace(".tau.wq", ".tau_wq")
    new_key = new_key.replace(".tau.wv", ".tau_wv")

    # Transform text.wte to text.wte.weight (for Embedding module)
    if new_key == "text.wte":
        new_key = "text.wte.weight"

    # Transform vision.proj_mlp.fc{1,2} → vision.proj_fc{1,2}
    new_key = new_key.replace("vision.proj_mlp.fc1", "vision.proj_fc1")
    new_key = new_key.replace("vision.proj_mlp.fc2", "vision.proj_fc2")

    return new_key


def is_moe_expert_layer(key: str) -> bool:
    """Check if the key corresponds to an MoE expert layer that should be quantized.

    Only MoE expert layers in blocks 4-23 are quantized.
    Source format: text.blocks.{4-23}.mlp.fc{1,2}.weight (shape: [64, out_dim, in_dim])
    """
    if not key.startswith("text.blocks."):
        return False

    # Extract block number - format is text.blocks.N.mlp.fc{1,2}.weight
    match = re.match(r"text\.blocks\.(\d+)\.mlp\.fc[12]\.weight", key)
    if not match:
        return False

    block_num = int(match.group(1))
    # MoE starts at layer 4 (0-indexed), so blocks 4-23
    return block_num >= 4


def quantize_tensor_mlx(tensor: torch.Tensor, bits: int = 8, group_size: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a tensor using MLX's quantize function.

    Args:
        tensor: PyTorch tensor in BF16 format
        bits: Quantization bits (4 or 8)
        group_size: Group size for quantization

    Returns:
        Tuple of (quantized_weight, scales, biases) as PyTorch tensors
        - quantized_weight: uint32 (packed)
        - scales: bfloat16
        - biases: bfloat16
    """
    # Convert bf16 -> float32 -> MLX array
    tensor_f32 = tensor.to(torch.float32).numpy()
    mlx_array = mx.array(tensor_f32)

    # Use MLX's quantize function
    q_weight, scales, biases = mx.quantize(mlx_array, group_size=group_size, bits=bits)

    # Evaluate to ensure computation is done
    mx.eval(q_weight, scales, biases)

    # Convert back to PyTorch tensors with correct dtypes
    # q_weight stays as uint32 (packed quantized values)
    # scales and biases should be bfloat16 for MLX compatibility
    q_torch = torch.from_numpy(np.asarray(q_weight).astype(np.uint32))
    scales_torch = torch.from_numpy(np.asarray(scales).astype(np.float32)).to(torch.bfloat16)
    biases_torch = torch.from_numpy(np.asarray(biases).astype(np.float32)).to(torch.bfloat16)

    return q_torch, scales_torch, biases_torch


def convert_model(
    source_dir: Path,
    output_dir: Path,
    bits: int = 8,
    group_size: int = 64,
) -> None:
    """Convert the model from BF16 to quantized format.

    Args:
        source_dir: Path to source model directory
        output_dir: Path to output directory
        bits: Quantization bits (4 or 8)
        group_size: Group size for quantization
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all safetensors files
    weight_files = sorted(source_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found in {source_dir}")

    print(f"Found {len(weight_files)} weight files")

    # Process all tensors
    all_weights = {}
    total_tensors = 0
    quantized_count = 0

    for file_idx, weight_file in enumerate(weight_files):
        print(f"\nProcessing [{file_idx + 1}/{len(weight_files)}]: {weight_file.name}")

        with safe_open(weight_file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"  Found {len(keys)} tensors")

            for key in keys:
                total_tensors += 1
                tensor = f.get_tensor(key)
                new_key = transform_key(key)

                if is_moe_expert_layer(new_key):
                    # This is an MoE expert weight - quantize it
                    # Source format: text.blocks.N.mlp.fc1.weight (shape: [64, out_dim, in_dim])
                    # Target format: text.blocks.N.mlp.fc1_q, fc1_scales, fc1_biases

                    print(f"    Quantizing: {new_key} ({tensor.shape}, {tensor.dtype})")

                    # Get the base key (e.g., text.blocks.4.mlp.fc1 from text.blocks.4.mlp.fc1.weight)
                    base_key = new_key.rsplit(".weight", 1)[0]

                    # Quantize using MLX - returns PyTorch tensors
                    q_weight, scales, biases = quantize_tensor_mlx(tensor, bits=bits, group_size=group_size)

                    # Store with correct key names (fc1_q, fc1_scales, fc1_biases)
                    all_weights[f"{base_key}_q"] = q_weight
                    all_weights[f"{base_key}_scales"] = scales
                    all_weights[f"{base_key}_biases"] = biases

                    print(f"      -> {base_key}_q: {q_weight.shape}, {q_weight.dtype}")
                    print(f"      -> {base_key}_scales: {scales.shape}, {scales.dtype}")

                    quantized_count += 1

                    # Free memory
                    del tensor, q_weight, scales, biases
                    gc.collect()
                else:
                    # Keep as-is (BF16 or whatever original dtype)
                    all_weights[new_key] = tensor

                # Progress indicator
                if total_tensors % 50 == 0:
                    print(f"    Processed {total_tensors} tensors...")

        # Force garbage collection after each file
        gc.collect()

    print(f"\nTotal tensors: {total_tensors}, Quantized: {quantized_count}")

    # Save the combined weights
    print(f"\nSaving weights to {output_dir / 'model.safetensors'}...")
    torch_save_file(all_weights, str(output_dir / "model.safetensors"))

    # Free memory
    del all_weights
    gc.collect()

    # Create config.json
    create_config(source_dir, output_dir, bits, group_size)

    # Copy tokenizer files from source or download from starmie-v1
    copy_tokenizer_files(source_dir, output_dir)

    print(f"\nConversion complete! Output saved to {output_dir}")


def copy_tokenizer_files(source_dir: Path, output_dir: Path) -> None:
    """Copy tokenizer files to output directory."""
    import shutil

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    # Try to copy from source directory first
    copied = False
    for filename in tokenizer_files:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)
            copied = True

    if copied:
        print("Copied tokenizer files from source directory")
        return

    # If not in source, try to download from starmie-v1
    print("Downloading tokenizer files from moondream/starmie-v1...")
    try:
        tokenizer_dir = Path(snapshot_download("moondream/starmie-v1"))
        for filename in tokenizer_files:
            src = tokenizer_dir / filename
            if src.exists():
                shutil.copy2(src, output_dir / filename)
        print("Copied tokenizer files from starmie-v1")
    except Exception as e:
        print(f"Warning: Could not copy tokenizer files: {e}")
        print("You may need to manually copy tokenizer files to the output directory")


def create_config(source_dir: Path, output_dir: Path, bits: int, group_size: int) -> None:
    """Create config.json with appropriate settings for the quantized model."""

    # Load source config
    source_config_path = source_dir / "config.json"
    if source_config_path.exists():
        with open(source_config_path) as f:
            config = json.load(f)
    else:
        # Fallback config based on moondream3 architecture
        config = {
            "architectures": ["HfMoondream"],
            "model_type": "moondream3",
            "torch_dtype": "bfloat16",
        }

    # Add/update text config with quantization settings
    if "text" not in config:
        config["text"] = {
            "dim": 2048,
            "ff_dim": 8192,
            "n_layers": 24,
            "vocab_size": 51200,
            "max_context": 4096,
            "n_heads": 32,
            "n_kv_heads": 32,
            "prefix_attn": 730,
            "moe": {
                "num_experts": 64,
                "start_layer": 4,
                "experts_per_token": 8,
                "expert_inner_dim": 1024
            }
        }

    # Set bits and group_size for MoE quantization
    config["text"]["bits"] = bits
    config["text"]["group_size"] = group_size

    # Add vision config if not present
    if "vision" not in config:
        config["vision"] = {
            "enc_dim": 1152,
            "enc_patch_size": 14,
            "enc_n_layers": 27,
            "enc_ff_dim": 4304,
            "enc_n_heads": 16,
            "proj_out_dim": 2048,
            "crop_size": 378,
            "in_channels": 3,
            "max_crops": 12,
            "overlap_margin": 4,
            "proj_inner_dim": 8192
        }

    # Add region config if not present
    if "region" not in config:
        config["region"] = {
            "dim": 2048,
            "coord_feat_dim": 256,
            "coord_out_dim": 1024,
            "size_feat_dim": 512,
            "size_out_dim": 2048,
            "group_size": None
        }

    # Add dtype field
    config["dtype"] = "bfloat16"

    # Write config
    output_config_path = output_dir / "config.json"
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created config.json with bits={bits}, group_size={group_size}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Moondream3 BF16 model to int8 quantized format for MLX"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="moondream/moondream3-preview",
        help="Source model ID or path (default: moondream/moondream3-preview)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory path",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Quantization bits (default: 8)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Group size for quantization (default: 64)",
    )

    args = parser.parse_args()

    # Find source directory
    if os.path.isdir(args.source):
        source_dir = Path(args.source)
    else:
        source_dir = find_hf_cache_dir(args.source)

    print(f"Source directory: {source_dir}")
    print(f"Output directory: {args.output}")
    print(f"Quantization: {args.bits}-bit, group_size={args.group_size}")

    convert_model(
        source_dir=source_dir,
        output_dir=Path(args.output),
        bits=args.bits,
        group_size=args.group_size,
    )


if __name__ == "__main__":
    main()
