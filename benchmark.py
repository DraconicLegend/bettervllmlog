#!/usr/bin/env python3
import time
import json
import random
import string
import requests
from pathlib import Path

from transformers import AutoTokenizer

# ====== CONFIG (match your server) ======
VLLM_API_URL = "http://localhost:11434/v1/completions"  # Use completions endpoint
METRICS_URL  = "http://localhost:11434/metrics"  # Same port as API
MODEL_NAME = "InternVL3_5-14B"  # Model name from vllm serve
MODEL_FOR_TOKENIZER = "/home/jiaheng/hf_models/OpenGVLab/InternVL3_5-14B"

MAKE_PLOT  = True

# Prompt sweep configurations
# Set which experiment to run: 'reuse_vs_prefill' or 'prompt_length_vs_tpot'
EXPERIMENT_MODE = 'prompt_length_vs_tpot'  # Change to 'reuse_vs_prefill' for original experiment

# For 'reuse_vs_prefill' mode
N_TOKENS = 4000
REUSE_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# For 'prompt_length_vs_tpot' mode
PROMPT_LENGTHS = [500, 1000, 2000, 4000, 8000]  # Token counts to test
KV_CACHE_REUSE_RATE = 0.8  # Fixed reuse rate (e.g., 0.8 = 80% cache hit)

MAX_TOKENS = 128
TEMPERATURE = 0.0

SLEEP_AFTER_REQUEST = 0.4
SLEEP_AFTER_WARMUP  = 0.5

# ====== Metrics parsing ======
def parse_metrics(text):
    m = {
        "prefill_sum": None, "prefill_count": None,
        "decode_sum": None, "decode_count": None,
        "time_per_output_token_sum": None, "time_per_output_token_count": None,
        "time_to_first_token_sum": None, "time_to_first_token_count": None
    }
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        # Handle both v0 and v1 metric names
        if "prefill_time_seconds_sum" in line or "vllm:request_prefill_time_seconds_sum" in line:
            m["prefill_sum"] = float(line.strip().split()[-1])
        elif "prefill_time_seconds_count" in line or "vllm:request_prefill_time_seconds_count" in line:
            m["prefill_count"] = float(line.strip().split()[-1])
        elif "decode_time_seconds_sum" in line or "vllm:request_decode_time_seconds_sum" in line:
            m["decode_sum"] = float(line.strip().split()[-1])
        elif "decode_time_seconds_count" in line or "vllm:request_decode_time_seconds_count" in line:
            m["decode_count"] = float(line.strip().split()[-1])
        elif "time_per_output_token_seconds_sum" in line or "vllm:time_per_output_token_seconds_sum" in line:
            m["time_per_output_token_sum"] = float(line.strip().split()[-1])
        elif "time_per_output_token_seconds_count" in line or "vllm:time_per_output_token_seconds_count" in line:
            m["time_per_output_token_count"] = float(line.strip().split()[-1])
        elif "time_to_first_token_seconds_sum" in line or "vllm:time_to_first_token_seconds_sum" in line:
            m["time_to_first_token_sum"] = float(line.strip().split()[-1])
        elif "time_to_first_token_seconds_count" in line or "vllm:time_to_first_token_seconds_count" in line:
            m["time_to_first_token_count"] = float(line.strip().split()[-1])
    return m

def read_metrics():
    try:
        r = requests.get(METRICS_URL, timeout=5)
        r.raise_for_status()
        metrics = parse_metrics(r.text)
        # Validate that we got metrics
        if all(v is None for v in metrics.values()):
            print(f"[WARNING] No metrics found. Check that vLLM server is running on {METRICS_URL}")
        return metrics
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch metrics from {METRICS_URL}: {e}")
        print(f"[INFO] Make sure vLLM server is running on port 11434")
        raise

def delta_last_sample(before, after, key_sum, key_count):
    s0, c0 = before.get(key_sum), before.get(key_count)
    s1, c1 = after.get(key_sum),  after.get(key_count)
    if None in (s0, c0, s1, c1):
        return None
    dc, ds = c1 - c0, s1 - s0
    if abs(dc - 1.0) < 1e-6 and ds >= 0:
        return ds
    if dc > 0:
        return ds / dc
    return None

# ====== Prompt builders (token-level control) ======
def build_base_text(tok, n_tokens):
    seed = (
        "This is a synthetic context paragraph used to measure vLLM prefill time. "
        "We repeat sentences to reach a target token length. "
        "Numbers and punctuation help create stable tokenization: "
        "alpha-1 beta-2 gamma-3 delta-4 epsilon-5 zeta-6 eta-7 theta-8 iota-9 kappa-10. "
    )
    text = seed
    while True:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= n_tokens:
            return tok.decode(ids[:n_tokens], skip_special_tokens=True)
        text += seed

def random_tail(tok, tokens_needed):
    words = []
    target_words = int(tokens_needed * 1.2)
    for _ in range(target_words):
        wlen = random.randint(3, 9)
        words.append("".join(random.choices(string.ascii_lowercase, k=wlen)))
    tail = " " + " ".join(words)
    ids = tok(tail, add_special_tokens=False)["input_ids"][:tokens_needed]
    return tok.decode(ids, skip_special_tokens=True)

def build_prompt_with_reuse(tok, base_text, reuse_fraction):
    base_ids = tok(base_text, add_special_tokens=False)["input_ids"]
    N = len(base_ids)
    L = int(reuse_fraction * N)
    prefix_ids = base_ids[:L]
    tail_needed = max(N - L, 0)
    tail_text = random_tail(tok, tail_needed) if tail_needed > 0 else ""
    prefix_text = tok.decode(prefix_ids, skip_special_tokens=True)
    return prefix_text + tail_text, N, L

# ====== Request/measure ======
def completions_request(prompt):
    """Send a completions request to vLLM."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    try:
        r = requests.post(VLLM_API_URL, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP error: {e}")
        print(f"[ERROR] Response: {r.text[:500]}")
        raise

def measure_prefill_for_prompt(prompt):
    m0 = read_metrics()
    _ = completions_request(prompt)
    time.sleep(SLEEP_AFTER_REQUEST)
    m1 = read_metrics()

    prefill = delta_last_sample(m0, m1, "prefill_sum", "prefill_count")
    decode  = delta_last_sample(m0, m1, "decode_sum",  "decode_count")
    time_per_output_token = delta_last_sample(m0, m1, "time_per_output_token_sum", "time_per_output_token_count")
    time_to_first_token = delta_last_sample(m0, m1, "time_to_first_token_sum", "time_to_first_token_count")

    # Validate results
    if prefill is None or decode is None:
        print(f"[WARNING] Could not extract timing metrics. prefill={prefill}, decode={decode}")
        print(f"[DEBUG] Before metrics: {m0}")
        print(f"[DEBUG] After metrics: {m1}")

    return {
        "prefill": prefill if prefill is not None else 0.0,
        "decode": decode if decode is not None else 0.0,
        "time_per_output_token": time_per_output_token if time_per_output_token is not None else 0.0,
        "time_to_first_token": time_to_first_token if time_to_first_token is not None else 0.0,
    }

# ====== Cache Management ======
def reset_prefix_cache():
    """Reset the vLLM prefix cache before benchmark."""
    # Extract base URL (e.g., http://localhost:11434 from http://localhost:11434/v1/completions)
    base_url = VLLM_API_URL.split('/v1/')[0]
    reset_url = f"{base_url}/reset_prefix_cache"
    try:
        print(f"[*] Resetting prefix cache at {reset_url}...")
        response = requests.post(reset_url, timeout=10)
        response.raise_for_status()
        print("[✓] Prefix cache reset successful")
        time.sleep(0.5)  # Give cache time to reset
        return True
    except requests.exceptions.RequestException as e:
        print(f"[WARNING] Could not reset prefix cache: {e}")
        print(f"[INFO] Continuing without cache reset...")
        return False

# ====== Main ======
def main():
    # Reset prefix cache at the very beginning
    reset_prefix_cache()

    random.seed(1234)
    tok = AutoTokenizer.from_pretrained(MODEL_FOR_TOKENIZER, use_fast=True, trust_remote_code=True)

    results = []
    all_token_latencies = []

    if EXPERIMENT_MODE == 'reuse_vs_prefill':
        print(f"[*] Running experiment: Reuse Fraction vs Prefill Time")
        print(f"[*] Prompt length: {N_TOKENS} tokens")

        base_text = build_base_text(tok, N_TOKENS)
        base_ids = tok(base_text, add_special_tokens=False)["input_ids"]
        print(f"[*] Base prompt tokens: {len(base_ids)}")

        # Warm up to populate KV for full prefix
        print("[*] Warm-up …")
        _ = measure_prefill_for_prompt(base_text)
        time.sleep(SLEEP_AFTER_WARMUP)

        for f in REUSE_FRACTIONS:
            prompt, N, L = build_prompt_with_reuse(tok, base_text, f)
            print(f"[*] reuse={f:.2f}  LCP={L}/{N}")
            metrics = measure_prefill_for_prompt(prompt)

            print(f"    prefill={metrics['prefill']:.4f}s  decode={metrics['decode']:.4f}s")
            print(f"    TTFT={metrics['time_to_first_token']:.4f}s  "
                  f"time_per_token={metrics['time_per_output_token']:.4f}s")

            result_entry = {
                "reuse_fraction": f,
                "lcp_tokens": L,
                "prompt_tokens": N,
                "prefill_seconds": metrics['prefill'],
                "decode_seconds": metrics['decode'],
                "time_to_first_token": metrics['time_to_first_token'],
                "time_per_output_token": metrics['time_per_output_token'],
            }

            # Store for JSON output
            all_token_latencies.append({
                "reuse_fraction": f,
                "lcp_tokens": L,
                "prompt_tokens": N,
                "time_to_first_token": metrics['time_to_first_token'],
                "time_per_output_token": metrics['time_per_output_token'],
            })

            results.append(result_entry)

    elif EXPERIMENT_MODE == 'prompt_length_vs_tpot':
        print(f"[*] Running experiment: Prompt Length vs TPOT")
        print(f"[*] KV cache reuse rate: {KV_CACHE_REUSE_RATE:.1%}")

        # Build base texts for each length first
        base_texts = {}
        for prompt_length in PROMPT_LENGTHS:
            base_texts[prompt_length] = build_base_text(tok, prompt_length)

        # Initial warm-up with a mid-size prompt
        mid_length = PROMPT_LENGTHS[len(PROMPT_LENGTHS) // 2]
        print(f"[*] Initial warm-up with {mid_length} tokens…")
        _ = measure_prefill_for_prompt(base_texts[mid_length])
        time.sleep(SLEEP_AFTER_WARMUP)

        for prompt_length in PROMPT_LENGTHS:
            # Reset cache before each measurement to ensure clean state
            print(f"[*] Resetting cache for prompt_length={prompt_length}")
            reset_prefix_cache()
            time.sleep(0.5)

            base_text = base_texts[prompt_length]

            # Warm up this specific prompt length to populate cache
            print(f"[*] Warming up with full prompt ({prompt_length} tokens)")
            _ = measure_prefill_for_prompt(base_text)
            time.sleep(0.3)

            # Now measure with partial reuse
            prompt, N, L = build_prompt_with_reuse(tok, base_text, KV_CACHE_REUSE_RATE)

            print(f"[*] Measuring: prompt_length={prompt_length}  LCP={L}/{N}  reuse={KV_CACHE_REUSE_RATE:.2f}")
            metrics = measure_prefill_for_prompt(prompt)

            print(f"    prefill={metrics['prefill']:.4f}s  decode={metrics['decode']:.4f}s")
            print(f"    TTFT={metrics['time_to_first_token']:.4f}s  "
                  f"time_per_token={metrics['time_per_output_token']:.4f}s")

            result_entry = {
                "prompt_length": prompt_length,
                "reuse_fraction": KV_CACHE_REUSE_RATE,
                "lcp_tokens": L,
                "prompt_tokens": N,
                "prefill_seconds": metrics['prefill'],
                "decode_seconds": metrics['decode'],
                "time_to_first_token": metrics['time_to_first_token'],
                "time_per_output_token": metrics['time_per_output_token'],
            }

            # Store for JSON output
            all_token_latencies.append({
                "prompt_length": prompt_length,
                "reuse_fraction": KV_CACHE_REUSE_RATE,
                "lcp_tokens": L,
                "prompt_tokens": N,
                "time_to_first_token": metrics['time_to_first_token'],
                "time_per_output_token": metrics['time_per_output_token'],
            })

            results.append(result_entry)

    else:
        raise ValueError(f"Unknown EXPERIMENT_MODE: {EXPERIMENT_MODE}")

    # Determine output filenames based on experiment mode
    if EXPERIMENT_MODE == 'reuse_vs_prefill':
        output_csv = "kv_reuse_vs_prefill.csv"
        output_png = "kv_reuse_vs_prefill.png"
        output_json = "token_metrics_reuse.json"
    else:  # prompt_length_vs_tpot
        output_csv = f"prompt_length_vs_tpot_reuse{int(KV_CACHE_REUSE_RATE*100)}.csv"
        output_png = f"prompt_length_vs_tpot_reuse{int(KV_CACHE_REUSE_RATE*100)}.png"
        output_json = f"token_metrics_prompt_length_reuse{int(KV_CACHE_REUSE_RATE*100)}.json"

    # Save CSV
    import csv
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Saved: {output_csv}")

    # Save detailed token metrics to JSON
    with open(output_json, "w") as f:
        json.dump(all_token_latencies, f, indent=2)
    print(f"Saved token metrics: {output_json}")

    # Optional plot
    if MAKE_PLOT:
        try:
            import matplotlib.pyplot as plt

            if EXPERIMENT_MODE == 'reuse_vs_prefill':
                xs = [r["reuse_fraction"] for r in results]
                ys = [r["prefill_seconds"] for r in results]
                plt.figure()
                plt.title("vLLM Prefill Time vs. Prefix Reuse")
                plt.xlabel("Reuse fraction (LCP / N)")
                plt.ylabel("Prefill time (s)")
                plt.plot(xs, ys, marker="o")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(output_png, dpi=160)
                print(f"Saved plot: {output_png}")

            elif EXPERIMENT_MODE == 'prompt_length_vs_tpot':
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f"Prompt Length Impact (KV Cache Reuse: {KV_CACHE_REUSE_RATE:.0%})")

                xs = [r["prompt_length"] for r in results]

                # Plot 1: Time per output token vs prompt length
                axes[0, 0].plot(xs, [r["time_per_output_token"] for r in results], marker="o", color='blue')
                axes[0, 0].set_xlabel("Prompt Length (tokens)")
                axes[0, 0].set_ylabel("Time per Output Token (s)")
                axes[0, 0].set_title("TPOT vs Prompt Length")
                axes[0, 0].grid(True)

                # Plot 2: Time to first token vs prompt length
                axes[0, 1].plot(xs, [r["time_to_first_token"] for r in results], marker="o", color='green')
                axes[0, 1].set_xlabel("Prompt Length (tokens)")
                axes[0, 1].set_ylabel("Time to First Token (s)")
                axes[0, 1].set_title("TTFT vs Prompt Length")
                axes[0, 1].grid(True)

                # Plot 3: Prefill time vs prompt length
                axes[1, 0].plot(xs, [r["prefill_seconds"] for r in results], marker="o", color='orange')
                axes[1, 0].set_xlabel("Prompt Length (tokens)")
                axes[1, 0].set_ylabel("Prefill Time (s)")
                axes[1, 0].set_title("Prefill Time vs Prompt Length")
                axes[1, 0].grid(True)

                # Plot 4: Decode time vs prompt length
                axes[1, 1].plot(xs, [r["decode_seconds"] for r in results], marker="o", color='red')
                axes[1, 1].set_xlabel("Prompt Length (tokens)")
                axes[1, 1].set_ylabel("Decode Time (s)")
                axes[1, 1].set_title("Decode Time vs Prompt Length")
                axes[1, 1].grid(True)

                plt.tight_layout()
                plt.savefig(output_png, dpi=160)
                print(f"Saved plot: {output_png}")

        except Exception as e:
            print(f"[!] Plot failed: {e}")

if __name__ == "__main__":
    main()
