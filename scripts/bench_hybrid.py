#!/usr/bin/env python3
"""
Benchmark: Pure ANE vs Pure MLX vs Hybrid (ANE prefill + MLX decode).

Usage:
    python bench_hybrid.py \
        --ane-bin ./build/ane-lm \
        --ane-model /path/to/Qwen3.5-0.8B \
        --mlx-model /path/to/Qwen3.5-0.8B \
        [--runs 3] [--max-tokens 20]

Runs 3 prompt lengths × 3 modes × N runs, outputs comparison table.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchResult:
    mode: str
    prompt_label: str
    n_prompt: int = 0
    prefill_tps: float = 0.0
    n_decode: int = 0
    decode_tps: float = 0.0
    total_ms: float = 0.0
    runs: list = field(default_factory=list)


PROMPTS = {
    "short":  "Hello",
    "medium": "Explain the theory of relativity in simple terms for a high school student.",
    "long":   ("Write a detailed technical comparison of Python, Rust, and C++ covering "
               "type systems, memory management, concurrency models, ecosystem maturity, "
               "and ideal use cases. Include code examples for each language showing "
               "how to implement a concurrent web scraper that respects rate limits. "
               "Also discuss the trade-offs between development speed and runtime performance."),
}


def run_cmd(cmd: list[str], timeout: int = 300) -> tuple[str, str, float]:
    """Run command, return (stdout, stderr, elapsed_sec)."""
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - t0
    return proc.stdout, proc.stderr, elapsed


def parse_ane_output(stderr: str) -> dict:
    """Parse ane-lm generate output for prompt/gen stats."""
    result = {}
    m = re.search(r"Prompt:\s*(\d+)\s*tokens?,\s*([\d.]+)\s*tokens?-per-sec", stderr)
    if m:
        result["n_prompt"] = int(m.group(1))
        result["prefill_tps"] = float(m.group(2))
    m = re.search(r"Generation:\s*(\d+)\s*tokens?,\s*([\d.]+)\s*tokens?-per-sec", stderr)
    if m:
        result["n_decode"] = int(m.group(1))
        result["decode_tps"] = float(m.group(2))
    # Prefill-only mode
    m = re.search(r"Prefill:\s*(\d+)\s*tokens?,\s*([\d.]+)\s*tokens?-per-sec", stderr)
    if m:
        result["n_prompt"] = int(m.group(1))
        result["prefill_tps"] = float(m.group(2))
    m = re.search(r"State saved.*\(([\d.]+)\s*ms\)", stderr)
    if m:
        result["save_ms"] = float(m.group(1))
    return result


def parse_mlx_output(stdout: str, stderr: str) -> dict:
    """Parse mlx_lm generate output."""
    result = {}
    combined = stdout + "\n" + stderr
    m = re.search(r"Prompt:\s*(\d+).*?([\d.]+)\s*tokens?-per-sec", combined)
    if m:
        result["n_prompt"] = int(m.group(1))
        result["prefill_tps"] = float(m.group(2))
    m = re.search(r"Generation:\s*(\d+).*?([\d.]+)\s*tokens?-per-sec", combined)
    if m:
        result["n_decode"] = int(m.group(1))
        result["decode_tps"] = float(m.group(2))
    return result


def parse_hybrid_output(stdout: str, stderr: str) -> dict:
    """Parse hybrid_decode.py output."""
    result = {}
    combined = stdout + "\n" + stderr
    m = re.search(r"HYBRID_RESULT.*?n_decode=(\d+)\s+decode_tps=([\d.]+)\s+"
                  r"read_ms=([\d.]+)\s+cache_ms=([\d.]+)\s+decode_ms=([\d.]+)", combined)
    if m:
        result["n_decode"] = int(m.group(1))
        result["decode_tps"] = float(m.group(2))
        result["read_ms"] = float(m.group(3))
        result["cache_ms"] = float(m.group(4))
        result["decode_ms"] = float(m.group(5))
    return result


def bench_pure_ane(ane_bin: str, model_path: str, prompt: str,
                   max_tokens: int) -> dict:
    """Run pure ANE generation."""
    cmd = [ane_bin, "generate",
           "--model", model_path,
           "--prompt", prompt,
           "--max-tokens", str(max_tokens),
           "--temp", "0"]
    _, stderr, elapsed = run_cmd(cmd)
    result = parse_ane_output(stderr)
    result["total_ms"] = elapsed * 1000
    return result


def _is_vlm(model_path: str) -> bool:
    """Check if model is a VLM (needs mlx_vlm instead of mlx_lm)."""
    import json
    config_path = os.path.join(model_path, "config.json")
    try:
        config = json.load(open(config_path))
        arch = config.get("architectures", [""])[0]
        return "ConditionalGeneration" in arch
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def bench_pure_mlx(model_path: str, prompt: str, max_tokens: int) -> dict:
    """Run pure MLX generation (supports both LM and VLM models)."""
    if _is_vlm(model_path):
        # VLM model: use inline script with mlx_vlm loader
        script = f'''
import sys, time, json
from mlx_vlm.utils import load
import mlx.core as mx

model, processor = load("{model_path}")
lm = model.language_model if hasattr(model, "language_model") else model
tokenizer = processor.tokenizer

prompt = """{prompt}"""
messages = [{{"role": "user", "content": prompt}}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(text)
n_prompt = len(input_ids)

# Prefill
from mlx_lm.models.cache import KVCache, ArraysCache
config = json.load(open("{model_path}/config.json"))
tc = config.get("text_config", config)
n_layers = tc["num_hidden_layers"]
interval = tc.get("full_attention_interval", 4)
cache = [ArraysCache(size=2) if (i+1) % interval != 0 else KVCache() for i in range(n_layers)]

t_pf = time.perf_counter()
x = mx.array([input_ids])
out = lm(x, cache=cache)
logits = out.logits if hasattr(out, "logits") else out
mx.eval(logits)
prefill_time = time.perf_counter() - t_pf
prefill_tps = n_prompt / prefill_time

token = int(mx.argmax(logits[0, -1, :]).item())
eos_ids = set()
eos = tc.get("eos_token_id", None)
if isinstance(eos, list): eos_ids.update(eos)
elif eos is not None: eos_ids.add(eos)

# Decode
generated = []
t_dec = time.perf_counter()
for _ in range({max_tokens}):
    if token in eos_ids: break
    generated.append(token)
    out = lm(mx.array([[token]]), cache=cache)
    logits = out.logits if hasattr(out, "logits") else out
    mx.eval(logits)
    token = int(mx.argmax(logits[0, -1, :]).item())
decode_time = time.perf_counter() - t_dec
n_decode = len(generated)
decode_tps = n_decode / decode_time if decode_time > 0 else 0

print(f"Prompt: {{n_prompt}} tokens, {{prefill_tps:.1f}} tokens-per-sec", file=sys.stderr)
print(f"Generation: {{n_decode}} tokens, {{decode_tps:.1f}} tokens-per-sec", file=sys.stderr)
'''
        cmd = [sys.executable, "-c", script]
    else:
        cmd = [sys.executable, "-m", "mlx_lm.generate",
               "--model", model_path,
               "--prompt", prompt,
               "--max-tokens", str(max_tokens),
               "--temp", "0"]
    stdout, stderr, elapsed = run_cmd(cmd)
    result = parse_mlx_output(stdout, stderr)
    result["total_ms"] = elapsed * 1000
    return result


def bench_hybrid(ane_bin: str, ane_model: str, mlx_model: str,
                 prompt: str, max_tokens: int, script_dir: str) -> dict:
    """Run hybrid pipeline: ANE prefill + MLX decode."""
    state_file = os.path.join(tempfile.gettempdir(), "ane_hybrid_state.bin")

    # Step 1: ANE prefill
    cmd_prefill = [ane_bin, "generate",
                   "--model", ane_model,
                   "--prompt", prompt,
                   "--prefill-only",
                   "--state-file", state_file,
                   "--temp", "0"]
    _, stderr_pf, elapsed_pf = run_cmd(cmd_prefill)
    pf_result = parse_ane_output(stderr_pf)

    # Step 2: MLX decode
    decode_script = os.path.join(script_dir, "hybrid_decode.py")
    cmd_decode = [sys.executable, decode_script,
                  "--state-file", state_file,
                  "--model-path", mlx_model,
                  "--max-tokens", str(max_tokens),
                  "--temp", "0"]
    stdout_dec, stderr_dec, elapsed_dec = run_cmd(cmd_decode)
    dec_result = parse_hybrid_output(stdout_dec, stderr_dec)

    # Combine
    result = {
        "n_prompt": pf_result.get("n_prompt", 0),
        "prefill_tps": pf_result.get("prefill_tps", 0),
        "n_decode": dec_result.get("n_decode", 0),
        "decode_tps": dec_result.get("decode_tps", 0),
        "save_ms": pf_result.get("save_ms", 0),
        "read_ms": dec_result.get("read_ms", 0),
        "cache_ms": dec_result.get("cache_ms", 0),
        "total_ms": (elapsed_pf + elapsed_dec) * 1000,
    }

    # Cleanup
    try:
        os.unlink(state_file)
    except OSError:
        pass

    return result


def avg_results(results: list[dict]) -> dict:
    """Average numeric fields across runs."""
    if not results:
        return {}
    avg = {}
    for key in results[0]:
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        if vals:
            avg[key] = sum(vals) / len(vals)
    return avg


def print_table(all_results: dict):
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print(f"{'Prompt':<8} {'Mode':<10} {'Prompt':>6} {'Prefill':>10} {'Decode':>6} "
          f"{'Decode':>10} {'Total':>10} {'Notes'}")
    print(f"{'':8} {'':10} {'tok':>6} {'tok/s':>10} {'tok':>6} "
          f"{'tok/s':>10} {'ms':>10}")
    print("-" * 90)

    for prompt_label in PROMPTS:
        for mode in ["ane", "mlx", "hybrid"]:
            key = f"{prompt_label}_{mode}"
            r = all_results.get(key, {})
            if not r:
                continue

            notes = ""
            if mode == "hybrid":
                save_ms = r.get("save_ms", 0)
                read_ms = r.get("read_ms", 0)
                cache_ms = r.get("cache_ms", 0)
                notes = f"save={save_ms:.0f}ms read={read_ms:.0f}ms cache={cache_ms:.0f}ms"

            print(f"{prompt_label:<8} {mode:<10} "
                  f"{r.get('n_prompt', 0):>6.0f} "
                  f"{r.get('prefill_tps', 0):>10.1f} "
                  f"{r.get('n_decode', 0):>6.0f} "
                  f"{r.get('decode_tps', 0):>10.1f} "
                  f"{r.get('total_ms', 0):>10.0f} "
                  f"{notes}")
        print()

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Benchmark: ANE vs MLX vs Hybrid")
    parser.add_argument("--ane-bin", required=True, help="Path to ane-lm binary")
    parser.add_argument("--ane-model", required=True, help="Path to ANE model directory")
    parser.add_argument("--mlx-model", required=True, help="Path to MLX model directory")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max decode tokens")
    parser.add_argument("--runs", type=int, default=3, help="Runs per config")
    parser.add_argument("--modes", nargs="+", default=["ane", "mlx", "hybrid"],
                        choices=["ane", "mlx", "hybrid"], help="Modes to benchmark")
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS.keys()),
                        choices=list(PROMPTS.keys()), help="Prompt lengths to test")
    args = parser.parse_args()

    script_dir = str(Path(__file__).parent)
    all_results = {}

    total_configs = len(args.prompts) * len(args.modes)
    current = 0

    for prompt_label in args.prompts:
        prompt = PROMPTS[prompt_label]

        for mode in args.modes:
            current += 1
            print(f"\n[{current}/{total_configs}] {prompt_label} / {mode} "
                  f"({args.runs} runs)...", file=sys.stderr)

            runs = []
            for run_idx in range(args.runs):
                print(f"  Run {run_idx + 1}/{args.runs}...", end="", file=sys.stderr)
                try:
                    if mode == "ane":
                        r = bench_pure_ane(args.ane_bin, args.ane_model,
                                           prompt, args.max_tokens)
                    elif mode == "mlx":
                        r = bench_pure_mlx(args.mlx_model, prompt, args.max_tokens)
                    else:
                        r = bench_hybrid(args.ane_bin, args.ane_model, args.mlx_model,
                                         prompt, args.max_tokens, script_dir)
                    runs.append(r)
                    print(f" done (decode {r.get('decode_tps', 0):.1f} t/s)",
                          file=sys.stderr)
                except Exception as e:
                    print(f" FAILED: {e}", file=sys.stderr)

            if runs:
                key = f"{prompt_label}_{mode}"
                all_results[key] = avg_results(runs)

    print_table(all_results)


if __name__ == "__main__":
    main()
