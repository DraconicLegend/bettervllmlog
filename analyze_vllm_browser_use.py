#!/usr/bin/env python3
"""
Extract prefix cache hit rate for each request from task logs and match with
corresponding request statistics from vllm_request_stats.log to create visualizations.
For each request, we collect all cache hit rates while "Running: 1 reqs" and take the median.
"""

import json
import re
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def extract_prompt_content(text):
    """
    Extract system prompt and user prompt content from chat template format.
    Returns tuple: (system_prompt_word_count, user_prompt_word_count)
    Count words instead of characters for more meaningful metrics.
    """
    system_prompt_word_count = 0
    user_prompt_word_count = 0

    # Extract system prompt content between <|im_start|>system and <|im_end|>
    # Note: There should only be ONE system prompt per request
    system_pattern = r'<\|im_start\|>system\s*(.*?)\s*<\|im_end\|>'
    system_match = re.search(system_pattern, text, re.DOTALL)
    if system_match:
        system_content = system_match.group(1).strip()
        # Count words by splitting on whitespace
        system_prompt_word_count = len(system_content.split())

    # Extract user prompt content between <|im_start|>user and <|im_end|>
    # Note: There should only be ONE user prompt per request
    user_pattern = r'<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>'
    user_match = re.search(user_pattern, text, re.DOTALL)
    if user_match:
        user_content = user_match.group(1).strip()
        # Count words by splitting on whitespace
        user_prompt_word_count = len(user_content.split())

    return system_prompt_word_count, user_prompt_word_count


def extract_cache_hit_rates_per_request(task_log_path):
    """
    Extract cache hit rates per request.
    Returns a list of tuples: (request_id, start_time, end_time, median_cache_hit_rate, cache_samples)
    """
    request_data = []
    current_request = None
    cache_samples = []
    # Store prompts by request ID since "Received request" comes before "Added request"
    prompts_by_request_id = {}

    with open(task_log_path, 'r') as f:
        for line in f:
            # Check for "Received request" which contains the prompt
            received_match = re.search(r'Received request ([\w-]+): prompt: \'(.+)', line)
            if received_match:
                request_id = received_match.group(1)
                # Extract prompt content starting after "prompt: '"
                prompt_content = received_match.group(2)
                prompts_by_request_id[request_id] = prompt_content

            # Check for new request being added
            add_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO vllm\.v1\.engine\.async_llm - Added request ([\w-]+)', line)
            if add_match:
                # If we have a previous request with cache samples, save it
                if current_request and cache_samples:
                    median_rate = np.median([s['rate'] for s in cache_samples])

                    # Extract prompt word counts using the stored prompt for this request ID
                    prompt_text = prompts_by_request_id.get(current_request['id'], '')
                    system_word_count, user_word_count = extract_prompt_content(prompt_text)

                    request_data.append({
                        'request_id': current_request['id'],
                        'start_time': current_request['start_time'],
                        'end_time': cache_samples[-1]['timestamp'],
                        'median_cache_hit_rate': median_rate,
                        'num_samples': len(cache_samples),
                        'cache_samples': cache_samples.copy(),
                        'system_prompt_word_count': system_word_count,
                        'user_prompt_word_count': user_word_count
                    })

                # Start tracking new request
                timestamp = datetime.strptime(add_match.group(1), '%Y-%m-%d %H:%M:%S')
                current_request = {
                    'id': add_match.group(2),
                    'start_time': timestamp
                }
                cache_samples = []

            # Check for cache hit rate with "Running: 1 reqs"
            running_match = re.search(
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*Running: 1 reqs.*Prefix cache hit rate: ([\d.]+)%',
                line
            )
            if running_match and current_request:
                timestamp = datetime.strptime(running_match.group(1), '%Y-%m-%d %H:%M:%S')
                hit_rate = float(running_match.group(2))
                cache_samples.append({
                    'timestamp': timestamp,
                    'rate': hit_rate
                })

    # Don't forget the last request
    if current_request and cache_samples:
        median_rate = np.median([s['rate'] for s in cache_samples])

        # Extract prompt word counts using the stored prompt for this request ID
        prompt_text = prompts_by_request_id.get(current_request['id'], '')
        system_word_count, user_word_count = extract_prompt_content(prompt_text)

        request_data.append({
            'request_id': current_request['id'],
            'start_time': current_request['start_time'],
            'end_time': cache_samples[-1]['timestamp'],
            'median_cache_hit_rate': median_rate,
            'num_samples': len(cache_samples),
            'cache_samples': cache_samples.copy(),
            'system_prompt_word_count': system_word_count,
            'user_prompt_word_count': user_word_count
        })

    return request_data


def extract_request_stats(stats_log_path):
    """Extract request statistics from vllm_request_stats.log."""
    stats_data = []

    with open(stats_log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                timestamp = datetime.strptime(data['timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
                stats_data.append({
                    'timestamp': timestamp,
                    'prefill_time': data['prefill_time'],
                    'decode_time': data['decode_time'],
                    'num_prompt_tokens': data['num_prompt_tokens'],
                    'num_generation_tokens': data['num_generation_tokens'],
                    'e2e_latency': data['e2e_latency'],
                    'finish_reason': data.get('finish_reason', 'unknown')
                })
            except (json.JSONDecodeError, KeyError) as e:
                continue

    return stats_data


def match_requests_to_stats(request_data, stats_data, time_window_seconds=30):
    """
    Match each request from task log to its corresponding stats entry.
    We look for stats entries where the timestamp falls within the request's timeframe.
    """
    matched_data = []

    for req in request_data:
        req_start = req['start_time']
        req_end = req['end_time']

        # Find stats entries that fall within or near this request's timeframe
        best_match = None
        min_time_diff = float('inf')

        for stat in stats_data:
            stat_time = stat['timestamp']

            # Check if stat timestamp is within the request window or shortly after
            if req_start <= stat_time <= req_end + pd.Timedelta(seconds=time_window_seconds):
                # Calculate time difference from request end (completion time)
                time_diff = abs((stat_time - req_end).total_seconds())

                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_match = stat

        # If we found a match, add it to our dataset
        if best_match:
            matched_data.append({
                'request_id': req['request_id'],
                'request_start': req_start,
                'request_end': req_end,
                'completion_timestamp': best_match['timestamp'],
                'cache_hit_rate': req['median_cache_hit_rate'],
                'num_cache_samples': req['num_samples'],
                'prefill_time': best_match['prefill_time'],
                'decode_time': best_match['decode_time'],
                'num_prompt_tokens': best_match['num_prompt_tokens'],
                'num_generation_tokens': best_match['num_generation_tokens'],
                'e2e_latency': best_match['e2e_latency'],
                'finish_reason': best_match['finish_reason'],
                'time_diff_seconds': min_time_diff,
                'system_prompt_word_count': req.get('system_prompt_word_count', 0),
                'user_prompt_word_count': req.get('user_prompt_word_count', 0)
            })

    return matched_data


def create_visualizations(matched_data, output_prefix):
    """Create and save visualization plots."""
    if not matched_data:
        print("No matched data to visualize!")
        return

    df = pd.DataFrame(matched_data)

    # Set global font sizes
    plt.rcParams.update({
        'font.size': 14,           # Base font size
        'axes.titlesize': 16,      # Subplot title size
        'axes.labelsize': 14,      # Axis label size
        'xtick.labelsize': 12,     # X-axis tick label size
        'ytick.labelsize': 12,     # Y-axis tick label size
        'legend.fontsize': 12,     # Legend font size
        'figure.titlesize': 20     # Main title size
    })

    # Create a figure with multiple subplots - 4x2 layout for additional plots
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(f'vLLM Performance Analysis - {output_prefix.split("/")[-1]}',
                 fontsize=16, fontweight='bold')

    # 1. Cache Hit Rate over Time (per request)
    ax1 = axes[0, 0]
    ax1.plot(df['completion_timestamp'], df['cache_hit_rate'],
             marker='o', linestyle='-', linewidth=2, markersize=6, color='#2ecc71')
    ax1.set_xlabel('Request Completion Time')
    ax1.set_ylabel('Cache Hit Rate (%)')
    ax1.set_title('Median Prefix Cache Hit Rate Per Request')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Prefill Time vs Cache Hit Rate
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['cache_hit_rate'], df['prefill_time'],
                          c=df['num_prompt_tokens'], cmap='viridis',
                          alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Cache Hit Rate (%)')
    ax2.set_ylabel('Prefill Time (s)')
    ax2.set_title('Prefill Time vs Cache Hit Rate')
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Num Prompt Tokens')

    # 3. Decode Time vs Cache Hit Rate
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['cache_hit_rate'], df['decode_time'],
                          c=df['num_generation_tokens'], cmap='plasma',
                          alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Cache Hit Rate (%)')
    ax3.set_ylabel('Decode Time (s)')
    ax3.set_title('Decode Time vs Cache Hit Rate')
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Num Generation Tokens')

    # 4. Number of Tokens over Time
    ax4 = axes[1, 1]
    ax4.plot(df['completion_timestamp'], df['num_prompt_tokens'],
             marker='o', linestyle='-', label='Prompt Tokens',
             alpha=0.7, markersize=5, linewidth=2, color='#3498db')
    ax4.plot(df['completion_timestamp'], df['num_generation_tokens'],
             marker='s', linestyle='-', label='Generation Tokens',
             alpha=0.7, markersize=5, linewidth=2, color='#e74c3c')
    ax4.set_xlabel('Request Completion Time')
    ax4.set_ylabel('Number of Tokens')
    ax4.set_title('Token Counts Per Request')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 5. E2E Latency vs Cache Hit Rate
    ax5 = axes[2, 0]
    scatter = ax5.scatter(df['cache_hit_rate'], df['e2e_latency'],
                          c=df['num_prompt_tokens'], cmap='coolwarm',
                          alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    ax5.set_xlabel('Cache Hit Rate (%)')
    ax5.set_ylabel('E2E Latency (s)')
    ax5.set_title('End-to-End Latency vs Cache Hit Rate')
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Num Prompt Tokens')

    # 6. Prefill and Decode Time per Request
    ax6 = axes[2, 1]
    ax6.plot(df['completion_timestamp'], df['prefill_time'],
             marker='o', linestyle='-', label='Prefill Time',
             alpha=0.8, markersize=6, linewidth=2, color='#FF6B6B')
    ax6.plot(df['completion_timestamp'], df['decode_time'],
             marker='s', linestyle='-', label='Decode Time',
             alpha=0.8, markersize=6, linewidth=2, color='#4ECDC4')
    ax6.set_xlabel('Request Completion Time')
    ax6.set_ylabel('Time (s)')
    ax6.set_title('Prefill & Decode Time Per Request')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 7. System vs User Prompt Word Count over Time
    ax7 = axes[3, 0]
    if df['system_prompt_word_count'].sum() > 0 or df['user_prompt_word_count'].sum() > 0:
        ax7.plot(df['completion_timestamp'], df['system_prompt_word_count'],
                 marker='o', linestyle='-', label='System Prompt',
                 alpha=0.7, markersize=5, linewidth=2, color='#9b59b6')
        ax7.plot(df['completion_timestamp'], df['user_prompt_word_count'],
                 marker='s', linestyle='-', label='User Prompt',
                 alpha=0.7, markersize=5, linewidth=2, color='#f39c12')
        ax7.set_xlabel('Request Completion Time')
        ax7.set_ylabel('Word Count')
        ax7.set_title('System vs User Prompt Word Count')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax7.text(0.5, 0.5, 'No prompt word count data available',
                 ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        ax7.set_title('System vs User Prompt Word Count')

    # 8. Prompt Composition Breakdown
    ax8 = axes[3, 1]
    if df['system_prompt_word_count'].sum() > 0 or df['user_prompt_word_count'].sum() > 0:
        df['total_prompt_word_count'] = df['system_prompt_word_count'] + df['user_prompt_word_count']
        x_positions = list(range(len(df)))

        # Create bars with explicit width to ensure all are visible
        bar_width = 0.8
        bars1 = ax8.bar(x_positions, df['system_prompt_word_count'],
                        width=bar_width, label='System Prompt', alpha=0.8, color='#9b59b6')
        bars2 = ax8.bar(x_positions, df['user_prompt_word_count'],
                        width=bar_width, bottom=df['system_prompt_word_count'],
                        label='User Prompt', alpha=0.8, color='#f39c12')

        # Add text annotation for bars with 0 total to make them visible
        for i, (sys_count, user_count) in enumerate(zip(df['system_prompt_word_count'],
                                                          df['user_prompt_word_count'])):
            total = sys_count + user_count
            if total == 0:
                ax8.text(i, 0, '0', ha='center', va='bottom', fontsize=8, color='red')

        ax8.set_xlabel('Request Index')
        ax8.set_ylabel('Word Count')
        ax8.set_title(f'Prompt Composition per Request (Words) - Total: {len(df)} requests')
        ax8.set_xticks(x_positions)
        ax8.set_xticklabels([f'{i+1}' for i in x_positions], fontsize=8)
        ax8.set_xlim(-0.5, len(df) - 0.5)  # Ensure all bars are within view
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')
    else:
        ax8.text(0.5, 0.5, 'No prompt word count data available',
                 ha='center', va='center', transform=ax8.transAxes, fontsize=12)
        ax8.set_title('Prompt Composition per Request (Words)')

    plt.tight_layout()

    # Save the figure
    fig_path = f'{output_prefix}_analysis.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")

    return fig_path


def save_matched_data(matched_data, output_prefix):
    """Save matched data to CSV and JSON formats."""
    if not matched_data:
        print("No data to save!")
        return

    df = pd.DataFrame(matched_data)

    # Save to CSV
    csv_path = f'{output_prefix}_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")

    # Save to JSON
    json_path = f'{output_prefix}_data.json'
    with open(json_path, 'w') as f:
        json.dump(matched_data, f, indent=2, default=str)
    print(f"Data saved to: {json_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (PER REQUEST)")
    print("="*60)
    print(f"Total requests matched: {len(matched_data)}")
    print(f"\nCache Hit Rate (median per request):")
    print(f"  Min:  {df['cache_hit_rate'].min():.2f}%")
    print(f"  Max:  {df['cache_hit_rate'].max():.2f}%")
    print(f"  Mean: {df['cache_hit_rate'].mean():.2f}%")
    print(f"  Std:  {df['cache_hit_rate'].std():.2f}%")

    print(f"\nPrefill Time (seconds):")
    print(f"  Min:  {df['prefill_time'].min():.4f}")
    print(f"  Max:  {df['prefill_time'].max():.4f}")
    print(f"  Mean: {df['prefill_time'].mean():.4f}")
    print(f"  Std:  {df['prefill_time'].std():.4f}")

    print(f"\nDecode Time (seconds):")
    print(f"  Min:  {df['decode_time'].min():.4f}")
    print(f"  Max:  {df['decode_time'].max():.4f}")
    print(f"  Mean: {df['decode_time'].mean():.4f}")
    print(f"  Std:  {df['decode_time'].std():.4f}")

    print(f"\nPrompt Tokens:")
    print(f"  Min:  {df['num_prompt_tokens'].min()}")
    print(f"  Max:  {df['num_prompt_tokens'].max()}")
    print(f"  Mean: {df['num_prompt_tokens'].mean():.0f}")

    print(f"\nGeneration Tokens:")
    print(f"  Min:  {df['num_generation_tokens'].min()}")
    print(f"  Max:  {df['num_generation_tokens'].max()}")
    print(f"  Mean: {df['num_generation_tokens'].mean():.0f}")

    print(f"\nE2E Latency (seconds):")
    print(f"  Min:  {df['e2e_latency'].min():.4f}")
    print(f"  Max:  {df['e2e_latency'].max():.4f}")
    print(f"  Mean: {df['e2e_latency'].mean():.4f}")

    # Prompt word count statistics
    if df['system_prompt_word_count'].sum() > 0 or df['user_prompt_word_count'].sum() > 0:
        print(f"\nSystem Prompt Word Count:")
        print(f"  Min:  {df['system_prompt_word_count'].min()}")
        print(f"  Max:  {df['system_prompt_word_count'].max()}")
        print(f"  Mean: {df['system_prompt_word_count'].mean():.0f}")

        print(f"\nUser Prompt Word Count:")
        print(f"  Min:  {df['user_prompt_word_count'].min()}")
        print(f"  Max:  {df['user_prompt_word_count'].max()}")
        print(f"  Mean: {df['user_prompt_word_count'].mean():.0f}")

        df['total_prompt_word_count'] = df['system_prompt_word_count'] + df['user_prompt_word_count']
        print(f"\nTotal Prompt Word Count:")
        print(f"  Min:  {df['total_prompt_word_count'].min()}")
        print(f"  Max:  {df['total_prompt_word_count'].max()}")
        print(f"  Mean: {df['total_prompt_word_count'].mean():.0f}")

        # Show which requests have 0 word counts (if any)
        zero_count_requests = df[df['total_prompt_word_count'] == 0]
        if len(zero_count_requests) > 0:
            print(f"\nWarning: {len(zero_count_requests)} request(s) have 0 total word count:")
            for idx, row in zero_count_requests.iterrows():
                print(f"  Request {idx+1}: {row['request_id'][:20]}...")

    # Correlation analysis
    print(f"\nCorrelation Analysis:")
    corr_prefill = df['cache_hit_rate'].corr(df['prefill_time'])
    corr_decode = df['cache_hit_rate'].corr(df['decode_time'])
    corr_e2e = df['cache_hit_rate'].corr(df['e2e_latency'])
    print(f"  Cache Hit Rate vs Prefill Time:  {corr_prefill:.4f}")
    print(f"  Cache Hit Rate vs Decode Time:   {corr_decode:.4f}")
    print(f"  Cache Hit Rate vs E2E Latency:   {corr_e2e:.4f}")
    print("="*60 + "\n")

    return csv_path, json_path


def main():
    # Find the latest task log
    task_logs_dir = Path('task_logs')
    task_logs = sorted(task_logs_dir.glob('task_*.log'),
                      key=lambda p: p.stat().st_mtime, reverse=True)

    if not task_logs:
        print("No task logs found!")
        return

    latest_task_log = task_logs[0]
    print(f"Using task log: {latest_task_log}")

    # Extract timestamp from filename for output naming
    log_name = latest_task_log.stem  # e.g., 'task_20251110_203149'
    timestamp_match = re.search(r'task_(\d{8}_\d{6})', log_name)
    if timestamp_match:
        output_prefix = f'browser_use_results/vllm_analysis_{timestamp_match.group(1)}'
    else:
        output_prefix = f'browser_use_results/vllm_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # Path to request stats log
    stats_log = Path('profiler_output/vllm_request_stats.log')

    if not stats_log.exists():
        print(f"Stats log not found: {stats_log}")
        return

    print("\nExtracting per-request cache hit rates from task log...")
    request_data = extract_cache_hit_rates_per_request(latest_task_log)
    print(f"Found {len(request_data)} requests with cache hit rate data")

    if request_data:
        print("\nSample request data:")
        for i, req in enumerate(request_data[:3]):
            print(f"  Request {i+1}: {req['request_id'][:20]}... - "
                  f"Median cache hit rate: {req['median_cache_hit_rate']:.1f}% "
                  f"({req['num_samples']} samples), "
                  f"System prompt: {req.get('system_prompt_word_count', 0)} words, "
                  f"User prompt: {req.get('user_prompt_word_count', 0)} words")

    print("\nExtracting request statistics...")
    stats_data = extract_request_stats(stats_log)
    print(f"Found {len(stats_data)} request stat entries")

    print("\nMatching requests to statistics...")
    matched_data = match_requests_to_stats(request_data, stats_data, time_window_seconds=30)
    print(f"Matched {len(matched_data)} requests")

    if matched_data:
        print("\nSaving matched data...")
        save_matched_data(matched_data, output_prefix)

        print("\nCreating visualizations...")
        create_visualizations(matched_data, output_prefix)

        print(f"\n✓ Analysis complete! Files saved with prefix: {output_prefix}")
    else:
        print("\n✗ No matching data found. Try adjusting the time window.")


if __name__ == '__main__':
    main()
