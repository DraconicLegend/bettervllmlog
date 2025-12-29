#!/usr/bin/env python3
"""
Parse metrics snapshots to calculate per-request timing.

This script reads the metrics snapshot files saved before/after each request
and calculates the exact timing for each request.
"""

import re
import sys
from pathlib import Path
from datetime import datetime


def parse_metrics_file(filepath):
    """Parse a Prometheus metrics file and extract vLLM timing metrics."""
    try:
        with open(filepath, 'r') as f:
            text = f.read()

        metrics = {}

        # Prefill time
        prefill_sum = re.search(r'vllm:request_prefill_time_seconds_sum{[^}]*} (\S+)', text)
        prefill_count = re.search(r'vllm:request_prefill_time_seconds_count{[^}]*} (\S+)', text)
        if prefill_sum and prefill_count:
            metrics['prefill_sum'] = float(prefill_sum.group(1))
            metrics['prefill_count'] = float(prefill_count.group(1))

        # Decode time
        decode_sum = re.search(r'vllm:request_decode_time_seconds_sum{[^}]*} (\S+)', text)
        decode_count = re.search(r'vllm:request_decode_time_seconds_count{[^}]*} (\S+)', text)
        if decode_sum and decode_count:
            metrics['decode_sum'] = float(decode_sum.group(1))
            metrics['decode_count'] = float(decode_count.group(1))

        # TTFT (time to first token)
        ttft_sum = re.search(r'vllm:time_to_first_token_seconds_sum{[^}]*} (\S+)', text)
        ttft_count = re.search(r'vllm:time_to_first_token_seconds_count{[^}]*} (\S+)', text)
        if ttft_sum and ttft_count:
            metrics['ttft_sum'] = float(ttft_sum.group(1))
            metrics['ttft_count'] = float(ttft_count.group(1))

        return metrics
    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)
        return None


def find_request_pairs(snapshots_dir):
    """Find before/after snapshot pairs for each request."""
    snapshots_dir = Path(snapshots_dir)

    # Group files by request_id
    requests = {}

    for filepath in snapshots_dir.glob("*.txt"):
        filename = filepath.name
        # Parse filename: {request_id}_{stage}_{timestamp}.txt
        parts = filename.replace('.txt', '').split('_')

        if len(parts) >= 3:
            # Request ID is everything before the second-to-last underscore
            # Stage is second-to-last part (before/after)
            # Timestamp is last part

            # Find the stage (before/after)
            if 'before' in filename:
                stage = 'before'
                stage_idx = filename.index('_before_')
                request_id = filename[:stage_idx]
            elif 'after' in filename:
                stage = 'after'
                stage_idx = filename.index('_after_')
                request_id = filename[:stage_idx]
            else:
                continue

            if request_id not in requests:
                requests[request_id] = {}

            requests[request_id][stage] = filepath

    return requests


def calculate_request_metrics(request_id, before_file, after_file):
    """Calculate timing metrics for a request from before/after snapshots."""
    print(f"\n{'='*70}")
    print(f"Request: {request_id}")
    print(f"{'='*70}")

    before_metrics = parse_metrics_file(before_file) if before_file else None
    after_metrics = parse_metrics_file(after_file) if after_file else None

    if not before_metrics:
        print(f"⚠️  Could not parse 'before' snapshot: {before_file}")
        return

    if not after_metrics:
        print(f"⚠️  Could not parse 'after' snapshot: {after_file}")
        return

    # Calculate deltas
    prefill_delta = after_metrics.get('prefill_sum', 0) - before_metrics.get('prefill_sum', 0)
    decode_delta = after_metrics.get('decode_sum', 0) - before_metrics.get('decode_sum', 0)
    ttft_delta = after_metrics.get('ttft_sum', 0) - before_metrics.get('ttft_sum', 0)

    print(f"Snapshot (before): {before_file}")
    print(f"Snapshot (after):  {after_file}")
    print(f"\n📊 CALCULATED METRICS:")
    print(f"  Prefill Time: {prefill_delta:.3f}s")
    print(f"  Decode Time: {decode_delta:.3f}s")
    print(f"  Time to First Token (TTFT): {ttft_delta:.3f}s")

    if prefill_delta == 0 and decode_delta == 0 and ttft_delta == 0:
        print(f"\n⚠️  WARNING: All deltas are zero - metrics may not have changed between snapshots")

    return {
        'request_id': request_id,
        'prefill_time': prefill_delta,
        'decode_time': decode_delta,
        'ttft': ttft_delta,
        'before_file': str(before_file),
        'after_file': str(after_file)
    }


def main():
    snapshots_dir = Path("/home/jiaheng/vllm_log/metrics_snapshots")

    if not snapshots_dir.exists():
        print(f"Error: Snapshots directory not found: {snapshots_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing metrics snapshots from: {snapshots_dir}")

    requests = find_request_pairs(snapshots_dir)

    if not requests:
        print("No request snapshot pairs found.")
        return

    print(f"\nFound {len(requests)} request(s) with snapshots\n")

    results = []
    for request_id, files in sorted(requests.items()):
        before_file = files.get('before')
        after_file = files.get('after')

        if before_file and after_file:
            result = calculate_request_metrics(request_id, before_file, after_file)
            if result:
                results.append(result)
        else:
            print(f"\n⚠️  Incomplete snapshot pair for request: {request_id}")
            if before_file:
                print(f"   Found 'before': {before_file}")
            if after_file:
                print(f"   Found 'after': {after_file}")

    # Summary
    if results:
        print(f"\n{'='*70}")
        print(f"SUMMARY - {len(results)} request(s) analyzed")
        print(f"{'='*70}")

        total_prefill = sum(r['prefill_time'] for r in results)
        total_decode = sum(r['decode_time'] for r in results)
        total_ttft = sum(r['ttft'] for r in results)

        print(f"Total Prefill Time: {total_prefill:.3f}s")
        print(f"Total Decode Time: {total_decode:.3f}s")
        print(f"Total TTFT: {total_ttft:.3f}s")

        if len(results) > 0:
            print(f"\nAverage Prefill Time: {total_prefill/len(results):.3f}s")
            print(f"Average Decode Time: {total_decode/len(results):.3f}s")
            print(f"Average TTFT: {total_ttft/len(results):.3f}s")


if __name__ == "__main__":
    main()
