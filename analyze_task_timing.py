#!/usr/bin/env python3
"""
Analyze task logs and fetch corresponding timing metrics.

Correlates task log files with Prometheus metrics to show:
- Which requests were made during a task
- Prefill and decode times for those requests
"""

import re
import json
from datetime import datetime
from pathlib import Path
import requests


def parse_task_log(log_file: Path):
    """Parse task log to extract request IDs and timestamps."""
    content = log_file.read_text()

    requests = []

    # Find all request entries
    pattern = r'Received request (chatcmpl-[a-f0-9]+):'
    for match in re.finditer(pattern, content):
        request_id = match.group(1)
        requests.append({
            'request_id': request_id,
            'position': match.start()
        })

    # Extract task start/end times
    start_match = re.search(r'Started: (\S+)', content)
    end_match = re.search(r'Task ended: (\S+)', content)

    task_info = {
        'log_file': log_file.name,
        'task_id': log_file.stem,
        'start_time': start_match.group(1) if start_match else None,
        'end_time': end_match.group(1) if end_match else None,
        'requests': requests,
        'request_count': len(requests)
    }

    return task_info


def fetch_current_metrics():
    """Fetch current metrics from vLLM Prometheus endpoint."""
    try:
        response = requests.get('http://localhost:11434/metrics', timeout=5)
        response.raise_for_status()
        text = response.text

        # Parse key metrics
        metrics = {}

        # TTFT (prefill time)
        ttft_sum = re.search(r'vllm:time_to_first_token_seconds_sum{[^}]*} (\S+)', text)
        ttft_count = re.search(r'vllm:time_to_first_token_seconds_count{[^}]*} (\S+)', text)
        if ttft_sum and ttft_count:
            total = float(ttft_sum.group(1))
            count = float(ttft_count.group(1))
            metrics['ttft_avg'] = total / count if count > 0 else 0
            metrics['ttft_total'] = total

        # Decode time
        decode_sum = re.search(r'vllm:request_decode_time_seconds_sum{[^}]*} (\S+)', text)
        decode_count = re.search(r'vllm:request_decode_time_seconds_count{[^}]*} (\S+)', text)
        if decode_sum and decode_count:
            total = float(decode_sum.group(1))
            count = float(decode_count.group(1))
            metrics['decode_avg'] = total / count if count > 0 else 0
            metrics['decode_total'] = total

        # Prefill time (alternative metric)
        prefill_sum = re.search(r'vllm:request_prefill_time_seconds_sum{[^}]*} (\S+)', text)
        prefill_count = re.search(r'vllm:request_prefill_time_seconds_count{[^}]*} (\S+)', text)
        if prefill_sum and prefill_count:
            total = float(prefill_sum.group(1))
            count = float(prefill_count.group(1))
            metrics['prefill_avg'] = total / count if count > 0 else 0
            metrics['prefill_total'] = total
            metrics['prefill_count'] = count

        # E2E latency
        e2e_sum = re.search(r'vllm:e2e_request_latency_seconds_sum{[^}]*} (\S+)', text)
        e2e_count = re.search(r'vllm:e2e_request_latency_seconds_count{[^}]*} (\S+)', text)
        if e2e_sum and e2e_count:
            total = float(e2e_sum.group(1))
            count = float(e2e_count.group(1))
            metrics['e2e_avg'] = total / count if count > 0 else 0

        # Total requests
        total_req = re.search(r'vllm:request_success_total{[^}]*finished_reason="stop"[^}]*} (\S+)', text)
        if total_req:
            metrics['total_requests'] = int(float(total_req.group(1)))

        return metrics

    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return {}


def analyze_task_logs(log_dir: Path = Path('/home/jiaheng/vllm_log/task_logs')):
    """Analyze all task logs and show summary with timing."""
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return

    log_files = sorted(log_dir.glob('task_*.log'))

    if not log_files:
        print("No task log files found.")
        return

    print("=" * 70)
    print("Task Log Analysis with Timing Metrics")
    print("=" * 70)
    print()

    # Fetch current metrics
    current_metrics = fetch_current_metrics()

    if current_metrics:
        print("CURRENT VLLM METRICS:")
        print("-" * 70)
        print(f"Total Requests Completed: {current_metrics.get('total_requests', 'N/A')}")
        print(f"Average Prefill Time (TTFT): {current_metrics.get('ttft_avg', 0):.3f}s")
        print(f"Average Decode Time: {current_metrics.get('decode_avg', 0):.3f}s")
        print(f"Average E2E Latency: {current_metrics.get('e2e_avg', 0):.3f}s")
        print()

    print("TASK LOGS:")
    print("-" * 70)

    for log_file in log_files:
        task_info = parse_task_log(log_file)

        print(f"\nTask: {task_info['task_id']}")
        print(f"  File: {task_info['log_file']}")
        print(f"  Start: {task_info['start_time']}")
        print(f"  End: {task_info['end_time']}")
        print(f"  Requests: {task_info['request_count']}")

        if task_info['requests']:
            print(f"  Request IDs:")
            for i, req in enumerate(task_info['requests'][:5], 1):  # Show first 5
                print(f"    {i}. {req['request_id']}")
            if len(task_info['requests']) > 5:
                print(f"    ... and {len(task_info['requests']) - 5} more")

    print()
    print("=" * 70)
    print("\nTo see detailed per-request metrics, run:")
    print("  python3 /home/jiaheng/vllm_log/metrics_logger.py")
    print("\nTo monitor metrics continuously:")
    print("  python3 /home/jiaheng/vllm_log/metrics_logger.py --continuous --interval 60")
    print()


def main():
    analyze_task_logs()


if __name__ == '__main__':
    main()
