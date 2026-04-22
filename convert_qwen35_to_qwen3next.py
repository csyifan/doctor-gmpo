#!/usr/bin/env python3
"""
Convert Qwen3.5-9B VLM checkpoint to a text-only qwen3_next checkpoint
compatible with sglang and transformers.

Key transforms:
  1. Strip vision encoder + mtp weights
  2. Remap: model.language_model.X  ->  model.X
  3. Merge split projections:
       in_proj_qkv + in_proj_z  ->  in_proj_qkvz   (cat dim=0)
       in_proj_b   + in_proj_a  ->  in_proj_ba      (cat dim=0)
"""

import json
import os
import shutil
import argparse
from collections import defaultdict
import torch
from safetensors.torch import load_file, save_file


def build_shard_groups(index):
    """
    Group weight keys by their linear-attention layer number so we can merge
    the split projections even when they span multiple shards.
    Returns a mapping:  layer_idx -> {suffix -> (full_original_key, shard_filename)}
    """
    groups = defaultdict(dict)
    for key, shard in index["weight_map"].items():
        if "linear_attn.in_proj_qkv" in key:
            layer_idx = int(key.split(".layers.")[1].split(".")[0])
            groups[layer_idx]["qkv"] = (key, shard)
        elif "linear_attn.in_proj_z" in key:
            layer_idx = int(key.split(".layers.")[1].split(".")[0])
            groups[layer_idx]["z"] = (key, shard)
        elif "linear_attn.in_proj_b." in key:
            layer_idx = int(key.split(".layers.")[1].split(".")[0])
            groups[layer_idx]["b"] = (key, shard)
        elif "linear_attn.in_proj_a." in key:
            layer_idx = int(key.split(".layers.")[1].split(".")[0])
            groups[layer_idx]["a"] = (key, shard)
    return groups


def remap_key(key):
    """
    Remap a text-backbone key to plain CausalLM key, or return None to skip.
    """
    if key.startswith("model.visual.") or key.startswith("model.image_") or key.startswith("model.patch_"):
        return None
    if key.startswith("mtp."):
        return None
    if key.startswith("model.language_model."):
        return key.replace("model.language_model.", "model.", 1)
    # lm_head.weight stays as-is
    return key


def is_split_proj(key):
    return any(x in key for x in [
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_b.",
        "linear_attn.in_proj_a.",
    ])


def convert(src: str, dst: str):
    os.makedirs(dst, exist_ok=True)

    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        index = json.load(f)

    shard_groups = build_shard_groups(index)

    # Pre-load all shards that contain split projections
    split_shards_needed = set()
    for layer_idx, parts in shard_groups.items():
        for _, (_, shard) in parts.items():
            split_shards_needed.add(shard)

    print(f"Pre-loading {len(split_shards_needed)} shards for projection merging...")
    split_shard_tensors = {}
    for shard in split_shards_needed:
        split_shard_tensors[shard] = load_file(os.path.join(src, shard))

    # Build merged qkvz / ba tensors, keyed by merged model-key
    merged = {}
    for layer_idx, parts in shard_groups.items():
        if "qkv" in parts and "z" in parts:
            qkv_key, qkv_shard = parts["qkv"]
            z_key, z_shard = parts["z"]
            qkv_w = split_shard_tensors[qkv_shard][qkv_key]
            z_w = split_shard_tensors[z_shard][z_key]
            merged_w = torch.cat([qkv_w, z_w], dim=0)
            new_key = f"model.layers.{layer_idx}.linear_attn.in_proj_qkvz.weight"
            merged[new_key] = merged_w
            print(f"  Layer {layer_idx}: in_proj_qkvz {merged_w.shape} "
                  f"(from qkv {qkv_w.shape} + z {z_w.shape})")

        if "b" in parts and "a" in parts:
            b_key, b_shard = parts["b"]
            a_key, a_shard = parts["a"]
            b_w = split_shard_tensors[b_shard][b_key]
            a_w = split_shard_tensors[a_shard][a_key]
            merged_w = torch.cat([b_w, a_w], dim=0)
            new_key = f"model.layers.{layer_idx}.linear_attn.in_proj_ba.weight"
            merged[new_key] = merged_w

    # Keys that are handled by merging (will be skipped in regular pass)
    skip_keys = set()
    for layer_idx, parts in shard_groups.items():
        for _, (key, _) in parts.items():
            skip_keys.add(key)

    # Now do the regular pass shard by shard
    all_shards = sorted(set(index["weight_map"].values()))
    new_index_map = {}

    # We'll assign merged tensors to shard 1 (or the first shard)
    merged_shard_name = all_shards[0]

    for shard in all_shards:
        print(f"Processing {shard} ...")
        tensors = load_file(os.path.join(src, shard))
        out = {}

        for key, tensor in tensors.items():
            if key in skip_keys:
                continue
            new_key = remap_key(key)
            if new_key is None:
                continue
            out[new_key] = tensor
            new_index_map[new_key] = shard

        if out:
            save_file(out, os.path.join(dst, shard))
        else:
            print(f"  (no output tensors for {shard})")

    # Append merged tensors to first shard (load existing, merge, re-save)
    if merged:
        first_shard_path = os.path.join(dst, merged_shard_name)
        if os.path.exists(first_shard_path):
            existing = load_file(first_shard_path)
        else:
            existing = {}
        existing.update(merged)
        save_file(existing, first_shard_path)
        for k in merged:
            new_index_map[k] = merged_shard_name

    # Write new index
    new_index = {"metadata": index.get("metadata", {}), "weight_map": new_index_map}
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    # Write config.json
    with open(os.path.join(src, "config.json")) as f:
        src_cfg = json.load(f)

    text_cfg = src_cfg["text_config"]
    rope_params = text_cfg.get("rope_parameters", {})
    layer_types_raw = text_cfg.get("layer_types", [])
    # qwen3_next uses "full_attention" / "linear_attention"
    layer_types = []
    for lt in layer_types_raw:
        if lt == "full_attention":
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")

    new_cfg = {
        "model_type": "qwen3_next",
        "architectures": ["Qwen3NextForCausalLM"],
        "vocab_size": text_cfg["vocab_size"],
        "hidden_size": text_cfg["hidden_size"],
        "intermediate_size": text_cfg["intermediate_size"],
        "num_hidden_layers": text_cfg["num_hidden_layers"],
        "num_attention_heads": text_cfg["num_attention_heads"],
        "num_key_value_heads": text_cfg["num_key_value_heads"],
        "head_dim": text_cfg["head_dim"],
        "hidden_act": text_cfg["hidden_act"],
        "max_position_embeddings": text_cfg["max_position_embeddings"],
        "rms_norm_eps": text_cfg["rms_norm_eps"],
        "use_cache": text_cfg["use_cache"],
        "attention_bias": text_cfg.get("attention_bias", False),
        "attention_dropout": text_cfg.get("attention_dropout", 0.0),
        "layer_types": layer_types,
        "linear_conv_kernel_dim": text_cfg["linear_conv_kernel_dim"],
        "linear_key_head_dim": text_cfg["linear_key_head_dim"],
        "linear_value_head_dim": text_cfg["linear_value_head_dim"],
        "linear_num_key_heads": text_cfg["linear_num_key_heads"],
        "linear_num_value_heads": text_cfg["linear_num_value_heads"],
        "mlp_only_layers": text_cfg.get("mlp_only_layers", []),
        "rope_theta": rope_params.get("rope_theta", 10000000),
        "partial_rotary_factor": rope_params.get("partial_rotary_factor", 0.25),
        "attn_output_gate": text_cfg.get("attn_output_gate", True),
        "tie_word_embeddings": src_cfg.get("tie_word_embeddings", False),
        "transformers_version": "4.57.1",
        "torch_dtype": "bfloat16",
        "eos_token_id": text_cfg.get("eos_token_id", 248044),
        "bos_token_id": text_cfg.get("bos_token_id", None),
    }

    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(new_cfg, f, indent=2)

    # Copy tokenizer files
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "merges.txt", "special_tokens_map.json"]:
        src_f = os.path.join(src, fname)
        if os.path.exists(src_f):
            shutil.copy2(src_f, os.path.join(dst, fname))
            print(f"Copied {fname}")

    print(f"\nDone! Converted model -> {dst}")
    print(f"  Total text weights: {len(new_index_map)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/nfs-stor/yifan.lu/ckpt/Qwen3.5-9B")
    parser.add_argument("--dst", default="/nfs-stor/yifan.lu/ckpt/Qwen3.5-9B-text")
    args = parser.parse_args()
    convert(args.src, args.dst)
