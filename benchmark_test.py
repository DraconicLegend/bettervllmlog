#!/usr/bin/env python3
"""Quick test to verify benchmark.py fixes work"""
import requests

# Test configuration
VLLM_API_URL = "http://localhost:11434/v1/completions"
METRICS_URL = "http://localhost:11434/metrics"
MODEL_NAME = "InternVL3_5-14B"

print("=" * 60)
print("Testing vLLM Benchmark Script Fixes")
print("=" * 60)

# Test 1: Metrics endpoint
print("\n[Test 1] Checking metrics endpoint...")
try:
    r = requests.get(METRICS_URL, timeout=5)
    r.raise_for_status()
    print(f"✓ Metrics endpoint reachable: {r.status_code}")

    # Check for metrics
    has_prefill = "prefill_time_seconds" in r.text
    has_decode = "decode_time_seconds" in r.text
    print(f"✓ Has prefill metrics: {has_prefill}")
    print(f"✓ Has decode metrics: {has_decode}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Completions endpoint
print("\n[Test 2] Testing completions endpoint...")
try:
    payload = {
        "model": MODEL_NAME,
        "prompt": "Hello, this is a test prompt.",
        "max_tokens": 10,
        "temperature": 0.0,
    }
    r = requests.post(VLLM_API_URL, json=payload, timeout=30)
    r.raise_for_status()
    result = r.json()

    print(f"✓ Request successful")
    print(f"✓ Model: {result.get('model', 'N/A')}")
    print(f"✓ Generated text: {result['choices'][0]['text'][:50]}...")
    print(f"✓ Tokens used: prompt={result['usage']['prompt_tokens']}, completion={result['usage']['completion_tokens']}")
except Exception as e:
    print(f"✗ Error: {e}")
    if 'r' in locals():
        print(f"Response: {r.text[:200]}")

# Test 3: Parse metrics after request
print("\n[Test 3] Checking metrics after request...")
try:
    r = requests.get(METRICS_URL, timeout=5)
    text = r.text

    # Parse metrics
    metrics = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if "prefill_time_seconds_sum" in line or "vllm:request_prefill_time_seconds_sum" in line:
            metrics["prefill_sum"] = float(line.strip().split()[-1])
        elif "decode_time_seconds_sum" in line or "vllm:request_decode_time_seconds_sum" in line:
            metrics["decode_sum"] = float(line.strip().split()[-1])

    print(f"✓ Parsed metrics: {metrics}")

    if metrics.get("prefill_sum", 0) > 0:
        print(f"✓ Metrics are being collected!")
    else:
        print(f"⚠ Metrics are zero (expected if first request)")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
print("\nIf all tests passed, benchmark.py should work.")
print("Run: python benchmark.py")
