#!/usr/bin/env python3
"""
Visualize Prometheus histogram metrics from vLLM.

Shows distribution of generation tokens, prefill times, decode times, etc.
"""

import requests
import re
from typing import Dict, List, Tuple


def parse_histogram(metrics_text: str, metric_name: str) -> Dict:
    """Parse a histogram metric from Prometheus text format."""

    # Find all bucket entries
    bucket_pattern = f'{metric_name}_bucket{{[^}}]*le="([^"]+)"[^}}]*}} (\\S+)'
    buckets = []

    for match in re.finditer(bucket_pattern, metrics_text):
        le = match.group(1)  # "less than or equal to" value
        count = float(match.group(2))

        if le == "+Inf":
            le = float('inf')
        else:
            le = float(le)

        buckets.append((le, count))

    # Get sum and count
    sum_match = re.search(f'{metric_name}_sum{{[^}}]*}} (\\S+)', metrics_text)
    count_match = re.search(f'{metric_name}_count{{[^}}]*}} (\\S+)', metrics_text)

    total_sum = float(sum_match.group(1)) if sum_match else 0
    total_count = float(count_match.group(1)) if count_match else 0

    return {
        'buckets': sorted(buckets),
        'sum': total_sum,
        'count': total_count,
        'average': total_sum / total_count if total_count > 0 else 0
    }


def calculate_percentiles(buckets: List[Tuple[float, float]], total_count: float) -> Dict[str, float]:
    """Calculate percentiles from histogram buckets."""
    if total_count == 0:
        return {}

    percentiles = {}
    targets = [50, 90, 95, 99]

    prev_le = 0
    prev_count = 0

    for le, cumulative_count in buckets:
        if le == float('inf'):
            continue

        # Calculate what percentile this bucket represents
        percentile = (cumulative_count / total_count) * 100

        # Check if we crossed any target percentiles
        for target in targets:
            if target not in percentiles and percentile >= target:
                # Linear interpolation within bucket
                if cumulative_count > prev_count:
                    target_count = (target / 100) * total_count
                    bucket_fraction = (target_count - prev_count) / (cumulative_count - prev_count)
                    percentiles[target] = prev_le + bucket_fraction * (le - prev_le)
                else:
                    percentiles[target] = le

        prev_le = le
        prev_count = cumulative_count

    return percentiles


def visualize_histogram(hist: Dict, name: str, unit: str = ""):
    """Print a visual representation of the histogram."""
    buckets = hist['buckets']
    total_count = hist['count']

    if total_count == 0:
        print(f"  No data for {name}")
        return

    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Total: {hist['sum']:.2f}{unit} across {int(total_count)} requests")
    print(f"Average: {hist['average']:.2f}{unit}")
    print()

    # Calculate percentiles
    percentiles = calculate_percentiles(buckets, total_count)
    if percentiles:
        print("Percentiles:")
        for p in [50, 90, 95, 99]:
            if p in percentiles:
                print(f"  P{p}: {percentiles[p]:.2f}{unit}")
        print()

    # Distribution
    print("Distribution:")
    prev_le = 0
    prev_count = 0

    for le, cumulative_count in buckets:
        if le == float('inf'):
            bucket_count = cumulative_count - prev_count
            if bucket_count > 0:
                pct = (bucket_count / total_count) * 100
                bar = '█' * int(pct / 2)
                print(f"  > {prev_le:>8}{unit:>2}: {int(bucket_count):>4} requests ({pct:>5.1f}%) {bar}")
            break

        bucket_count = cumulative_count - prev_count
        if bucket_count > 0:
            pct = (bucket_count / total_count) * 100
            bar = '█' * int(pct / 2)
            print(f"  {prev_le:>6.0f}-{le:<6.0f}{unit:>2}: {int(bucket_count):>4} requests ({pct:>5.1f}%) {bar}")

        prev_le = le
        prev_count = cumulative_count


def main():
    """Fetch and visualize vLLM histogram metrics."""
    print("Fetching metrics from vLLM...")

    try:
        response = requests.get('http://localhost:11434/metrics', timeout=5)
        metrics_text = response.text
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return

    print("✅ Metrics fetched successfully!\n")

    # Available histograms
    histograms = [
        ('vllm:request_generation_tokens', 'Generation Tokens per Request', 'tokens'),
        ('vllm:request_prompt_tokens', 'Prompt Tokens per Request', 'tokens'),
        ('vllm:request_prefill_time_seconds', 'Prefill Time per Request', 's'),
        ('vllm:request_decode_time_seconds', 'Decode Time per Request', 's'),
        ('vllm:time_to_first_token_seconds', 'Time to First Token (TTFT)', 's'),
        ('vllm:e2e_request_latency_seconds', 'End-to-End Latency', 's'),
        ('vllm:request_time_per_output_token_seconds', 'Time per Output Token', 's'),
    ]

    for metric_name, display_name, unit in histograms:
        hist = parse_histogram(metrics_text, metric_name)
        if hist['count'] > 0:
            visualize_histogram(hist, display_name, unit)

    print(f"\n{'='*70}")
    print("For raw Prometheus format, run:")
    print("  curl http://localhost:11434/metrics | grep -A 20 'metric_name'")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
