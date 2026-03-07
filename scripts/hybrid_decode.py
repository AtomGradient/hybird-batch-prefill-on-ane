#!/usr/bin/env python3
"""
Hybrid decode: read ANE prefill state → construct MLX cache → decode with MLX.

Usage:
    python hybrid_decode.py \
        --state-file /tmp/state.bin \
        --model-path /path/to/Qwen3.5-0.8B \
        --max-tokens 50 \
        [--temp 0.0] [--verbose]
"""

import argparse
import struct
import sys
import time

import numpy as np

# MLX imports
import mlx.core as mx
import mlx.nn as nn


# ============ Binary state file reader ============

HEADER_FMT = "<8s I I I I I I I I I I I I I I"
HEADER_SIZE = 64

def read_state(path: str):
    """Read binary state file produced by ane-lm --prefill-only --state-file."""
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
        fields = struct.unpack(HEADER_FMT, raw)

        magic = fields[0]
        if magic[:6] != b"ANELMS":
            raise ValueError(f"Bad magic: {magic!r}")

        header = {
            "version":           fields[1],
            "num_layers":        fields[2],
            "hidden_size":       fields[3],
            "vocab_size":        fields[4],
            "num_kv_heads":      fields[5],
            "head_dim":          fields[6],
            "lin_num_val_heads": fields[7],
            "lin_num_key_heads": fields[8],
            "lin_key_dim":       fields[9],
            "lin_val_dim":       fields[10],
            "lin_qkv_dim":       fields[11],
            "conv_kernel":       fields[12],
            "kv_capacity":       fields[13],
            "n_prompt":          fields[14],
        }

        if header["version"] != 1:
            raise ValueError(f"Unsupported version: {header['version']}")

        layers = []
        for L in range(header["num_layers"]):
            ltype = struct.unpack("<I", f.read(4))[0]

            if ltype == 0:
                # LinearAttention (DeltaNet)
                conv_pos = struct.unpack("<I", f.read(4))[0]

                H = header["lin_num_val_heads"]
                K = header["lin_key_dim"]
                V = header["lin_val_dim"]
                ssm_state = np.frombuffer(f.read(H * K * V * 4), dtype=np.float32).reshape(H, K, V)

                C = header["lin_qkv_dim"]
                ks = header["conv_kernel"] - 1
                conv_state = np.frombuffer(f.read(C * ks * 4), dtype=np.float32).reshape(C, ks)

                layers.append({
                    "type": "linear",
                    "conv_pos": conv_pos,
                    "ssm_state": ssm_state,
                    "conv_state": conv_state,
                })
            else:
                # FullAttention
                kv_len, kv_start = struct.unpack("<II", f.read(8))

                cap = header["kv_capacity"]
                nh = header["num_kv_heads"]
                hd = header["head_dim"]
                k_cache = np.frombuffer(f.read(cap * nh * hd * 4), dtype=np.float32).reshape(cap, nh, hd)
                v_cache = np.frombuffer(f.read(cap * nh * hd * 4), dtype=np.float32).reshape(cap, nh, hd)

                layers.append({
                    "type": "full",
                    "kv_len": kv_len,
                    "kv_start": kv_start,
                    "k_cache": k_cache,
                    "v_cache": v_cache,
                })

        logits = np.frombuffer(f.read(header["vocab_size"] * 4), dtype=np.float32).copy()

    return header, layers, logits


# ============ State conversion: ANE → MLX ============

def convert_ssm_state(ane_ssm: np.ndarray) -> mx.array:
    """
    ANE: ssm_state[H, K, V]  (row-major, SSM uses CblasTrans → [K,V] is the matrix)
    MLX: [1, H, V, K]
    """
    # [H, K, V] → [H, V, K] via transpose of last two dims, then add batch
    return mx.array(ane_ssm.transpose(0, 2, 1)[np.newaxis])


def convert_conv_state(ane_conv: np.ndarray, conv_pos: int, kernel_size: int) -> mx.array:
    """
    ANE: conv_state[C, kernel-1] circular buffer, conv_pos = oldest entry
    MLX: [1, kernel-1, C] in temporal order (oldest → newest)
    """
    C, ks = ane_conv.shape  # ks = kernel_size - 1
    mlx_conv = np.zeros((1, ks, C), dtype=np.float32)
    for t in range(ks):
        src_idx = (conv_pos + t) % ks
        mlx_conv[0, t, :] = ane_conv[:, src_idx]
    return mx.array(mlx_conv)


def convert_kv_cache(k_cache: np.ndarray, v_cache: np.ndarray,
                     kv_start: int, kv_len: int, capacity: int) -> tuple:
    """
    ANE: [capacity, num_kv_heads, head_dim] circular buffer (start, len)
    MLX KVCache: keys/values [1, num_kv_heads, seq_len, head_dim]
    """
    # Extract valid entries in temporal order
    indices = [(kv_start + i) % capacity for i in range(kv_len)]

    k_valid = k_cache[indices]  # [kv_len, nh, hd]
    v_valid = v_cache[indices]  # [kv_len, nh, hd]

    # [kv_len, nh, hd] → [1, nh, kv_len, hd]
    k_mlx = mx.array(k_valid.transpose(1, 0, 2)[np.newaxis])
    v_mlx = mx.array(v_valid.transpose(1, 0, 2)[np.newaxis])

    return k_mlx, v_mlx


# ============ MLX model loading + cache construction ============

def load_mlx_model(model_path: str):
    """Load MLX model and tokenizer. Tries mlx_vlm first (for VLM architectures like
    Qwen3.5ForConditionalGeneration), falls back to mlx_lm."""
    try:
        from mlx_vlm.utils import load
        model, processor = load(model_path)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        return model, tokenizer
    except Exception:
        from mlx_lm.utils import load
        model, tokenizer = load(model_path)
        return model, tokenizer


def get_text_model(model):
    """Unwrap VLM wrapper to get the text model with make_cache()."""
    if hasattr(model, "make_cache"):
        return model
    if hasattr(model, "language_model") and hasattr(model.language_model, "make_cache"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "make_cache"):
        return model.model
    raise RuntimeError("Cannot find text model with make_cache() on the loaded model")


def build_mlx_cache(model, header, layer_states):
    """Build MLX prompt cache from ANE state."""
    text_model = get_text_model(model)
    cache = text_model.make_cache()

    for L, (layer_cache, layer_state) in enumerate(zip(cache, layer_states)):
        if layer_state["type"] == "linear":
            # ArraysCache: slot 0 = conv state, slot 1 = SSM state
            conv_mlx = convert_conv_state(
                layer_state["conv_state"],
                layer_state["conv_pos"],
                header["conv_kernel"],
            )
            ssm_mlx = convert_ssm_state(layer_state["ssm_state"])

            layer_cache[0] = conv_mlx
            layer_cache[1] = ssm_mlx
        else:
            # KVCache: set keys, values, offset
            k_mlx, v_mlx = convert_kv_cache(
                layer_state["k_cache"],
                layer_state["v_cache"],
                layer_state["kv_start"],
                layer_state["kv_len"],
                header["kv_capacity"],
            )
            layer_cache.keys = k_mlx
            layer_cache.values = v_mlx
            layer_cache.offset = layer_state["kv_len"]

    return cache


# ============ Decode loop ============

def sample_token(logits: mx.array, temperature: float = 0.0) -> int:
    """Sample a single token from logits."""
    if temperature <= 0.0:
        return mx.argmax(logits, axis=-1).item()
    probs = mx.softmax(logits / temperature, axis=-1)
    return mx.random.categorical(probs).item()


def decode_loop(model, tokenizer, cache, first_logits: mx.array,
                max_tokens: int, temperature: float, verbose: bool = False):
    """Run decode loop starting from prefilled state.
    model should be the text model (unwrapped from VLM if needed)."""
    text_model = get_text_model(model)
    eos_ids = set()
    if hasattr(tokenizer, "eos_token_id"):
        eid = tokenizer.eos_token_id
        if isinstance(eid, list):
            eos_ids.update(eid)
        elif eid is not None:
            eos_ids.add(eid)

    # Also check for im_end token
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        try:
            im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end is not None:
                eos_ids.add(im_end)
        except Exception:
            pass

    # Sample first token from prefilled logits
    token = sample_token(first_logits, temperature)
    generated = [token]

    if verbose:
        print(f"[decode] First token: {token}", file=sys.stderr)

    t0 = time.perf_counter()

    for i in range(max_tokens - 1):
        if token in eos_ids:
            break

        # Forward one token through MLX model with cache
        input_ids = mx.array([[token]])
        out = text_model(input_ids, cache=cache)
        logits = out.logits if hasattr(out, 'logits') else out
        mx.eval(logits)

        logits_last = logits[0, -1, :]
        token = sample_token(logits_last, temperature)
        generated.append(token)

    decode_time = time.perf_counter() - t0

    # Remove trailing EOS if present
    if generated and generated[-1] in eos_ids:
        generated = generated[:-1]

    n_tokens = len(generated)
    tps = n_tokens / decode_time if decode_time > 0 else 0

    # Decode text
    text = tokenizer.decode(generated)

    return text, n_tokens, tps, decode_time


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="Hybrid decode: ANE prefill → MLX decode")
    parser.add_argument("--state-file", required=True, help="Path to ANE state binary file")
    parser.add_argument("--model-path", required=True, help="Path to MLX model directory")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # 1. Read ANE state
    t0 = time.perf_counter()
    header, layer_states, logits_np = read_state(args.state_file)
    read_ms = (time.perf_counter() - t0) * 1000
    print(f"State read: {read_ms:.1f} ms  ({header['n_prompt']} prompt tokens, "
          f"{header['num_layers']} layers)", file=sys.stderr)

    if args.verbose:
        for key, val in header.items():
            print(f"  {key}: {val}", file=sys.stderr)

    # 2. Load MLX model
    t0 = time.perf_counter()
    model, tokenizer = load_mlx_model(args.model_path)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"MLX model loaded: {load_ms:.1f} ms", file=sys.stderr)

    # 3. Build cache from ANE state
    t0 = time.perf_counter()
    cache = build_mlx_cache(model, header, layer_states)
    first_logits = mx.array(logits_np)
    mx.eval(first_logits)
    cache_ms = (time.perf_counter() - t0) * 1000
    print(f"Cache constructed: {cache_ms:.1f} ms", file=sys.stderr)

    # 4. Decode
    text, n_tokens, tps, decode_time = decode_loop(
        model, tokenizer, cache, first_logits,
        args.max_tokens, args.temp, args.verbose
    )

    print(f"\n==========", file=sys.stderr)
    print(text, file=sys.stderr)
    print(f"==========", file=sys.stderr)
    print(f"Decode: {n_tokens} tokens, {tps:.1f} tok/s ({decode_time*1000:.1f} ms)", file=sys.stderr)
    print(f"Overhead (read+cache): {read_ms + cache_ms:.1f} ms", file=sys.stderr)

    # Machine-readable output for benchmarking
    print(f"HYBRID_RESULT n_prompt={header['n_prompt']} "
          f"n_decode={n_tokens} decode_tps={tps:.2f} "
          f"read_ms={read_ms:.1f} cache_ms={cache_ms:.1f} "
          f"decode_ms={decode_time*1000:.1f}")


if __name__ == "__main__":
    main()
