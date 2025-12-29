#!/usr/bin/env python3
"""
Combine vllm_request_stats.log with the latest task log file.
Renames the stats log to include the first timestamp from the file.
"""

import json
from pathlib import Path
from datetime import datetime
import shutil


def find_latest_task_log(task_logs_dir):
    """Find the most recently modified task log file."""
    task_logs_path = Path(task_logs_dir)
    task_log_files = list(task_logs_path.glob("task_*.log"))

    if not task_log_files:
        raise FileNotFoundError(f"No task log files found in {task_logs_dir}")

    # Sort by modification time, newest first
    latest_task_log = max(task_log_files, key=lambda f: f.stat().st_mtime)
    return latest_task_log


def read_vllm_stats(stats_file):
    """Read all entries from vllm_request_stats.log file."""
    stats_entries = []

    with open(stats_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    entry = json.loads(line)
                    stats_entries.append(entry)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line[:50]}...")

    return stats_entries


def get_first_timestamp(stats_entries):
    """Extract the first timestamp from stats entries."""
    if not stats_entries:
        return None

    first_entry = stats_entries[0]
    timestamp_str = first_entry.get('timestamp')

    if timestamp_str:
        # Parse ISO format timestamp and convert to filename-friendly format
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime('%Y%m%d_%H%M%S')

    return None


def combine_logs(stats_file, task_log_file, output_dir):
    """
    Combine vllm_request_stats.log with task log file.

    Creates a combined file with:
    1. Task log content (for context)
    2. Separator
    3. Detailed stats from vllm_request_stats.log in readable format

    Renames the stats log based on first timestamp.
    """
    stats_entries = read_vllm_stats(stats_file)

    if not stats_entries:
        print(f"Warning: No valid entries found in {stats_file}")
        return

    # Get first timestamp for naming
    timestamp = get_first_timestamp(stats_entries)
    if not timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    combined_filename = output_path / f"combined_stats_{timestamp}.log"
    renamed_stats_filename = output_path / f"vllm_request_stats_{timestamp}.log"

    # Write combined log
    with open(combined_filename, 'w') as out:
        # Write header
        out.write("=" * 80 + "\n")
        out.write(f"COMBINED LOG - Generated at {datetime.now().isoformat()}\n")
        out.write(f"Task Log: {task_log_file.name}\n")
        out.write(f"Stats File: {stats_file.name}\n")
        out.write(f"First Request Timestamp: {stats_entries[0]['timestamp']}\n")
        out.write("=" * 80 + "\n\n")

        # Write task log content
        out.write("=" * 80 + "\n")
        out.write("TASK LOG CONTENT\n")
        out.write("=" * 80 + "\n\n")

        with open(task_log_file, 'r') as task_f:
            out.write(task_f.read())

        # Write separator
        out.write("\n\n" + "=" * 80 + "\n")
        out.write("DETAILED REQUEST STATISTICS (from vllm_request_stats.log)\n")
        out.write("=" * 80 + "\n\n")

        # Write each stats entry in readable format
        for i, entry in enumerate(stats_entries, 1):
            out.write(f"{'─' * 80}\n")
            out.write(f"REQUEST #{i}\n")
            out.write(f"{'─' * 80}\n")
            out.write(f"Timestamp:              {entry['timestamp']}\n")
            out.write(f"Finish Reason:          {entry['finish_reason']}\n")
            out.write(f"E2E Latency:            {entry['e2e_latency']:.3f}s\n")
            out.write(f"Prompt Tokens:          {entry['num_prompt_tokens']}\n")
            out.write(f"Generation Tokens:      {entry['num_generation_tokens']}\n")
            out.write(f"Max Tokens Param:       {entry['max_tokens_param']}\n")
            out.write(f"\nTIMING BREAKDOWN:\n")
            out.write(f"  Queued Time:          {entry['queued_time']:.3f}s\n")
            out.write(f"  Prefill Time:         {entry['prefill_time']:.3f}s\n")
            out.write(f"  Decode Time:          {entry['decode_time']:.3f}s\n")
            out.write(f"  Inference Time:       {entry['inference_time']:.3f}s\n")
            out.write(f"  Mean Time/Token:      {entry['mean_time_per_output_token']:.6f}s\n")
            out.write(f"\n")

        # Write summary statistics
        out.write("\n" + "=" * 80 + "\n")
        out.write("SUMMARY STATISTICS\n")
        out.write("=" * 80 + "\n\n")

        total_requests = len(stats_entries)
        avg_e2e = sum(e['e2e_latency'] for e in stats_entries) / total_requests
        avg_prefill = sum(e['prefill_time'] for e in stats_entries) / total_requests
        avg_decode = sum(e['decode_time'] for e in stats_entries) / total_requests
        total_prompt_tokens = sum(e['num_prompt_tokens'] for e in stats_entries)
        total_gen_tokens = sum(e['num_generation_tokens'] for e in stats_entries)

        out.write(f"Total Completed Requests:     {total_requests}\n")
        out.write(f"Total Prompt Tokens:          {total_prompt_tokens}\n")
        out.write(f"Total Generation Tokens:      {total_gen_tokens}\n")
        out.write(f"\nAVERAGE TIMINGS:\n")
        out.write(f"  Avg E2E Latency:            {avg_e2e:.3f}s\n")
        out.write(f"  Avg Prefill Time:           {avg_prefill:.3f}s\n")
        out.write(f"  Avg Decode Time:            {avg_decode:.3f}s\n")
        out.write(f"\n")

    # Copy and rename the original stats file
    shutil.copy2(stats_file, renamed_stats_filename)

    print(f"✓ Combined log created: {combined_filename}")
    print(f"✓ Stats log copied to:  {renamed_stats_filename}")
    print(f"\nSummary:")
    print(f"  - Combined {total_requests} completed requests")
    print(f"  - First request timestamp: {stats_entries[0]['timestamp']}")
    print(f"  - Task log: {task_log_file.name}")

    return combined_filename, renamed_stats_filename


def main():
    # Configuration
    task_logs_dir = "/home/jiaheng/vllm_log/task_logs"
    stats_file = Path("/home/jiaheng/vllm_log/profiler_output/vllm_request_stats.log")
    output_dir = "/home/jiaheng/vllm_log/combined_logs"

    print("=" * 80)
    print("VLLM Log Combiner")
    print("=" * 80)
    print()

    # Check if stats file exists
    if not stats_file.exists():
        print(f"Error: Stats file not found: {stats_file}")
        return

    # Find latest task log
    try:
        latest_task_log = find_latest_task_log(task_logs_dir)
        print(f"Found latest task log: {latest_task_log.name}")
        print(f"  Modified: {datetime.fromtimestamp(latest_task_log.stat().st_mtime).isoformat()}")
        print()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Combine logs
    try:
        combine_logs(stats_file, latest_task_log, output_dir)
        print("\n✓ All operations completed successfully!")
    except Exception as e:
        print(f"\nError during log combination: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
