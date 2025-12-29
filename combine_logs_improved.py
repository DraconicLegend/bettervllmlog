#!/usr/bin/env python3
"""
Improved log combinator that:
1. Matches stats with task log by request ID and timestamp
2. Extracts KV cache hit rate from task log
3. Combines all timing and cache information
"""

import json
import re
from pathlib import Path
from datetime import datetime
import shutil


def find_latest_task_log(task_logs_dir):
    """Find the most recently modified task log file."""
    task_logs_path = Path(task_logs_dir)
    task_log_files = list(task_logs_path.glob("task_*.log"))

    if not task_log_files:
        raise FileNotFoundError(f"No task log files found in {task_logs_dir}")

    latest_task_log = max(task_log_files, key=lambda f: f.stat().st_mtime)
    return latest_task_log


def read_vllm_stats(stats_file):
    """Read all entries from vllm_request_stats.log file."""
    stats_entries = []

    with open(stats_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    stats_entries.append(entry)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line[:50]}...")

    return stats_entries


def parse_task_log(task_log_file):
    """
    Parse task log to extract:
    - Request IDs
    - Timestamps (received and completed)
    - KV cache hit rates over time
    - Aborted requests
    """
    requests = {}
    kv_cache_samples = []

    with open(task_log_file, 'r') as f:
        current_request_id = None

        for line in f:
            # Match "Request #N (Step M): chatcmpl-xxx"
            req_match = re.search(r'Request #(\d+) \(Step (\d+)\): (chatcmpl-[a-f0-9]+)', line)
            if req_match:
                request_num = int(req_match.group(1))
                step_num = int(req_match.group(2))
                request_id = req_match.group(3)
                current_request_id = request_id

                requests[request_id] = {
                    'request_num': request_num,
                    'step_num': step_num,
                    'request_id': request_id,
                    'received_time': None,
                    'completed_time': None,
                    'status': 'pending',
                    'kv_cache_samples': []
                }

            # Extract timestamp from "Time: 2025-11-08T16:00:55.904560"
            time_match = re.search(r'Time: (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)', line)
            if time_match and current_request_id and requests[current_request_id]['received_time'] is None:
                requests[current_request_id]['received_time'] = time_match.group(1)

            # Match "Generated response chatcmpl-xxx"
            gen_match = re.search(r'Generated response (chatcmpl-[a-f0-9]+)', line)
            if gen_match:
                request_id = gen_match.group(1)
                if request_id in requests:
                    requests[request_id]['status'] = 'completed'
                    # Extract timestamp from this line
                    timestamp_match = re.search(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match:
                        requests[request_id]['completed_time'] = timestamp_match.group(1)

            # Match "Aborted request(s) chatcmpl-xxx"
            abort_match = re.search(r'Aborted request\(s\) (chatcmpl-[a-f0-9]+)', line)
            if abort_match:
                request_id = abort_match.group(1)
                if request_id in requests:
                    requests[request_id]['status'] = 'aborted'

            # Extract KV cache hit rate from metrics logs
            # "Engine 000: ... GPU KV cache usage: 3.4%, Prefix cache hit rate: 65.6%"
            kv_match = re.search(
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*'
                r'GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%',
                line
            )
            if kv_match:
                timestamp_str = kv_match.group(1)
                kv_usage = float(kv_match.group(2))
                hit_rate = float(kv_match.group(3))

                kv_sample = {
                    'timestamp': timestamp_str,
                    'kv_cache_usage': kv_usage,
                    'prefix_cache_hit_rate': hit_rate
                }
                kv_cache_samples.append(kv_sample)

                # Associate with current request if active
                if current_request_id and current_request_id in requests:
                    if requests[current_request_id]['status'] == 'pending':
                        requests[current_request_id]['kv_cache_samples'].append(kv_sample)

    return requests, kv_cache_samples


def match_stats_with_requests(stats_entries, requests):
    """
    Match vllm stats with task log requests by timestamp proximity.
    Stats are logged when requests complete.
    """
    matched_data = []

    for stats in stats_entries:
        stats_time = datetime.fromisoformat(stats['timestamp'])

        # Find the request that completed closest to this stats timestamp
        best_match = None
        min_time_diff = None

        for req_id, req_data in requests.items():
            if req_data['status'] != 'completed':
                continue

            if req_data['completed_time']:
                try:
                    # Parse completed time (may be in different format)
                    completed_time = datetime.fromisoformat(req_data['completed_time'])
                except:
                    continue

                time_diff = abs((stats_time - completed_time).total_seconds())

                # Match if within 5 seconds
                if time_diff <= 5:
                    if min_time_diff is None or time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_match = req_id

        matched_entry = {
            'stats': stats,
            'request': requests.get(best_match) if best_match else None,
            'request_id': best_match,
            'time_diff': min_time_diff
        }
        matched_data.append(matched_entry)

    return matched_data


def write_combined_log(output_file, task_log_file, matched_data, all_requests, kv_cache_samples):
    """Write comprehensive combined log with all timing and cache info."""

    with open(output_file, 'w') as out:
        # Header
        out.write("=" * 100 + "\n")
        out.write(f"COMPREHENSIVE VLLM REQUEST ANALYSIS\n")
        out.write(f"Generated: {datetime.now().isoformat()}\n")
        out.write(f"Task Log: {task_log_file.name}\n")
        out.write("=" * 100 + "\n\n")

        # Summary
        completed_count = sum(1 for r in all_requests.values() if r['status'] == 'completed')
        aborted_count = sum(1 for r in all_requests.values() if r['status'] == 'aborted')

        out.write("SUMMARY\n")
        out.write("-" * 100 + "\n")
        out.write(f"Total Requests:        {len(all_requests)}\n")
        out.write(f"  Completed:           {completed_count}\n")
        out.write(f"  Aborted:             {aborted_count}\n")
        out.write(f"Stats Entries:         {len(matched_data)}\n")
        out.write("\n")

        # Per-request detailed analysis
        out.write("=" * 100 + "\n")
        out.write("DETAILED REQUEST ANALYSIS (Matched Stats + Task Log + KV Cache)\n")
        out.write("=" * 100 + "\n\n")

        for i, match in enumerate(matched_data, 1):
            stats = match['stats']
            req_data = match['request']

            out.write("─" * 100 + "\n")
            out.write(f"REQUEST #{i}\n")
            out.write("─" * 100 + "\n")

            if req_data:
                out.write(f"Request ID:             {req_data['request_id']}\n")
                out.write(f"Request Number:         #{req_data['request_num']} (Step {req_data['step_num']})\n")
                out.write(f"Status:                 {req_data['status'].upper()}\n")
                out.write(f"Received Time:          {req_data['received_time']}\n")
                out.write(f"Completed Time:         {req_data['completed_time']}\n")
                if match['time_diff'] is not None:
                    out.write(f"Time Match Diff:        {match['time_diff']:.3f}s\n")
            else:
                out.write(f"Request ID:             {match['request_id'] or 'UNMATCHED'}\n")
                out.write(f"Status:                 Could not match with task log\n")

            out.write(f"\nSTATS FROM vllm_request_stats.log:\n")
            out.write(f"  Stats Timestamp:      {stats['timestamp']}\n")
            out.write(f"  Finish Reason:        {stats['finish_reason']}\n")
            out.write(f"  E2E Latency:          {stats['e2e_latency']:.3f}s\n")
            out.write(f"  Prompt Tokens:        {stats['num_prompt_tokens']}\n")
            out.write(f"  Generation Tokens:    {stats['num_generation_tokens']}\n")
            out.write(f"  Max Tokens Param:     {stats['max_tokens_param']}\n")
            out.write(f"\n  TIMING BREAKDOWN:\n")
            out.write(f"    Queued Time:        {stats['queued_time']:.3f}s\n")
            out.write(f"    Prefill Time:       {stats['prefill_time']:.3f}s\n")
            out.write(f"    Decode Time:        {stats['decode_time']:.3f}s\n")
            out.write(f"    Inference Time:     {stats['inference_time']:.3f}s\n")
            out.write(f"    Mean Time/Token:    {stats['mean_time_per_output_token']:.6f}s\n")

            # KV Cache information
            if req_data and req_data['kv_cache_samples']:
                out.write(f"\n  KV CACHE METRICS (during request processing):\n")
                out.write(f"    Samples Collected:  {len(req_data['kv_cache_samples'])}\n")

                # Show first and last sample
                first_sample = req_data['kv_cache_samples'][0]
                last_sample = req_data['kv_cache_samples'][-1]

                out.write(f"    Initial State (at start):\n")
                out.write(f"      Time:             {first_sample['timestamp']}\n")
                out.write(f"      KV Cache Usage:   {first_sample['kv_cache_usage']:.1f}%\n")
                out.write(f"      Hit Rate:         {first_sample['prefix_cache_hit_rate']:.1f}%\n")

                if len(req_data['kv_cache_samples']) > 1:
                    out.write(f"    Final State (at end):\n")
                    out.write(f"      Time:             {last_sample['timestamp']}\n")
                    out.write(f"      KV Cache Usage:   {last_sample['kv_cache_usage']:.1f}%\n")
                    out.write(f"      Hit Rate:         {last_sample['prefix_cache_hit_rate']:.1f}%\n")

                    # Calculate deltas
                    cache_delta = last_sample['kv_cache_usage'] - first_sample['kv_cache_usage']
                    hit_rate_delta = last_sample['prefix_cache_hit_rate'] - first_sample['prefix_cache_hit_rate']

                    out.write(f"    Changes During Request:\n")
                    out.write(f"      Cache Usage Δ:    {cache_delta:+.1f}%\n")
                    out.write(f"      Hit Rate Δ:       {hit_rate_delta:+.1f}%\n")

                # Calculate token breakdown based on hit rate
                num_prompt_tokens = stats['num_prompt_tokens']
                # Use the average hit rate from samples during this request
                avg_hit_rate = sum(s['prefix_cache_hit_rate'] for s in req_data['kv_cache_samples']) / len(req_data['kv_cache_samples'])

                cached_tokens = int(num_prompt_tokens * (avg_hit_rate / 100.0))
                new_tokens = num_prompt_tokens - cached_tokens

                out.write(f"\n  TOKEN BREAKDOWN (Prompt Tokens):\n")
                out.write(f"    Total Prompt Tokens:    {num_prompt_tokens}\n")
                out.write(f"    Cached Tokens (hits):   {cached_tokens} ({avg_hit_rate:.1f}%)\n")
                out.write(f"    New Tokens (computed):  {new_tokens} ({100-avg_hit_rate:.1f}%)\n")

                # Calculate time savings from cache hits
                if cached_tokens > 0 and stats['prefill_time'] > 0:
                    # Estimate time per token if all were new
                    time_per_new_token = stats['prefill_time'] / new_tokens if new_tokens > 0 else 0
                    if time_per_new_token > 0:
                        estimated_time_without_cache = num_prompt_tokens * time_per_new_token
                        time_saved = estimated_time_without_cache - stats['prefill_time']
                        out.write(f"    Estimated Time Saved:   {time_saved:.3f}s (by caching {cached_tokens} tokens)\n")

            out.write("\n")

        # Aborted requests section
        aborted_requests = [r for r in all_requests.values() if r['status'] == 'aborted']
        if aborted_requests:
            out.write("\n" + "=" * 100 + "\n")
            out.write("ABORTED REQUESTS (No stats available)\n")
            out.write("=" * 100 + "\n\n")

            for req in aborted_requests:
                out.write("─" * 100 + "\n")
                out.write(f"Request ID:             {req['request_id']}\n")
                out.write(f"Request Number:         #{req['request_num']} (Step {req['step_num']})\n")
                out.write(f"Status:                 ABORTED\n")
                out.write(f"Received Time:          {req['received_time']}\n")

                if req['kv_cache_samples']:
                    out.write(f"KV Cache Samples:       {len(req['kv_cache_samples'])} collected before abort\n")
                    last_sample = req['kv_cache_samples'][-1]
                    out.write(f"  Last KV Cache Usage:  {last_sample['kv_cache_usage']:.1f}%\n")
                    out.write(f"  Last Hit Rate:        {last_sample['prefix_cache_hit_rate']:.1f}%\n")

                out.write("\n")

        # Overall statistics
        out.write("\n" + "=" * 100 + "\n")
        out.write("AGGREGATE STATISTICS\n")
        out.write("=" * 100 + "\n\n")

        if matched_data:
            all_stats = [m['stats'] for m in matched_data]

            total_e2e = sum(s['e2e_latency'] for s in all_stats)
            total_prefill = sum(s['prefill_time'] for s in all_stats)
            total_decode = sum(s['decode_time'] for s in all_stats)
            total_prompt_tokens = sum(s['num_prompt_tokens'] for s in all_stats)
            total_gen_tokens = sum(s['num_generation_tokens'] for s in all_stats)

            avg_e2e = total_e2e / len(all_stats)
            avg_prefill = total_prefill / len(all_stats)
            avg_decode = total_decode / len(all_stats)

            out.write(f"Completed Requests:     {len(all_stats)}\n")
            out.write(f"Total Prompt Tokens:    {total_prompt_tokens}\n")
            out.write(f"Total Generated Tokens: {total_gen_tokens}\n")
            out.write(f"\nAVERAGE TIMINGS:\n")
            out.write(f"  Avg E2E Latency:      {avg_e2e:.3f}s\n")
            out.write(f"  Avg Prefill Time:     {avg_prefill:.3f}s\n")
            out.write(f"  Avg Decode Time:      {avg_decode:.3f}s\n")

            if kv_cache_samples:
                avg_kv_usage = sum(s['kv_cache_usage'] for s in kv_cache_samples) / len(kv_cache_samples)
                avg_hit_rate = sum(s['prefix_cache_hit_rate'] for s in kv_cache_samples) / len(kv_cache_samples)

                # Calculate total cached vs new tokens
                total_cached_tokens = int(total_prompt_tokens * (avg_hit_rate / 100.0))
                total_new_tokens = total_prompt_tokens - total_cached_tokens

                out.write(f"\nKV CACHE STATISTICS:\n")
                out.write(f"  Total Samples:        {len(kv_cache_samples)}\n")
                out.write(f"  Avg Cache Usage:      {avg_kv_usage:.1f}%\n")
                out.write(f"  Avg Hit Rate:         {avg_hit_rate:.1f}%\n")
                out.write(f"\nTOKEN BREAKDOWN (across all requests):\n")
                out.write(f"  Total Prompt Tokens:  {total_prompt_tokens}\n")
                out.write(f"  Cached Tokens:        {total_cached_tokens} ({avg_hit_rate:.1f}%)\n")
                out.write(f"  New Tokens:           {total_new_tokens} ({100-avg_hit_rate:.1f}%)\n")

                # Estimate total time saved from caching
                if total_cached_tokens > 0 and total_prefill > 0 and total_new_tokens > 0:
                    avg_time_per_new_token = total_prefill / total_new_tokens
                    estimated_time_without_cache = total_prompt_tokens * avg_time_per_new_token
                    total_time_saved = estimated_time_without_cache - total_prefill
                    out.write(f"  Est. Total Time Saved: {total_time_saved:.3f}s (by caching)\n")


def main():
    task_logs_dir = "/home/jiaheng/vllm_log/task_logs"
    stats_file = Path("/home/jiaheng/vllm_log/profiler_output/vllm_request_stats.log")
    output_dir = "/home/jiaheng/vllm_log/combined_logs"

    print("=" * 100)
    print("IMPROVED VLLM Log Combinator (with KV Cache Hit Rate)")
    print("=" * 100)
    print()

    if not stats_file.exists():
        print(f"Error: Stats file not found: {stats_file}")
        return

    # Find latest task log
    try:
        latest_task_log = find_latest_task_log(task_logs_dir)
        print(f"Latest task log: {latest_task_log.name}")
        print(f"  Modified: {datetime.fromtimestamp(latest_task_log.stat().st_mtime).isoformat()}")
        print()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Read stats
    print("Reading vllm_request_stats.log...")
    stats_entries = read_vllm_stats(stats_file)
    print(f"  Found {len(stats_entries)} stats entries")

    # Parse task log
    print("Parsing task log...")
    all_requests, kv_cache_samples = parse_task_log(latest_task_log)
    completed = sum(1 for r in all_requests.values() if r['status'] == 'completed')
    aborted = sum(1 for r in all_requests.values() if r['status'] == 'aborted')
    print(f"  Found {len(all_requests)} total requests ({completed} completed, {aborted} aborted)")
    print(f"  Found {len(kv_cache_samples)} KV cache samples")

    # Match stats with requests
    print("Matching stats with requests by timestamp...")
    matched_data = match_stats_with_requests(stats_entries, all_requests)

    # Generate output filename
    if stats_entries:
        first_timestamp = datetime.fromisoformat(stats_entries[0]['timestamp']).strftime('%Y%m%d_%H%M%S')
    else:
        first_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write combined log
    combined_filename = output_path / f"combined_analysis_{first_timestamp}.log"
    renamed_stats_filename = output_path / f"vllm_request_stats_{first_timestamp}.log"

    print(f"\nWriting combined analysis...")
    write_combined_log(combined_filename, latest_task_log, matched_data, all_requests, kv_cache_samples)

    # Copy stats file
    shutil.copy2(stats_file, renamed_stats_filename)

    print(f"\n✓ Combined analysis:    {combined_filename}")
    print(f"✓ Stats log archived:   {renamed_stats_filename}")
    print(f"\n✓ All operations completed successfully!")


if __name__ == "__main__":
    main()
