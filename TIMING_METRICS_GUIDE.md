# vLLM Timing Metrics Guide

## Overview

This guide explains how to capture and monitor **prefill time** (time to first token) and **decode time** (generation time) for your vLLM server.

## Key Metrics

vLLM exposes the following timing metrics via Prometheus:

1. **Prefill Time (TTFT)** - Time to First Token
   - Metric: `vllm:time_to_first_token_seconds`
   - Also: `vllm:request_prefill_time_seconds`
   - This is the time spent processing the prompt (prefill phase)

2. **Decode Time** - Generation Time
   - Metric: `vllm:request_decode_time_seconds`
   - Time spent generating output tokens

3. **Time Per Output Token (TPOT)**
   - Metric: `vllm:time_per_output_token_seconds`
   - Average time to generate each token

4. **E2E Request Latency**
   - Metric: `vllm:e2e_request_latency_seconds`
   - Total request time (prefill + decode + overhead)

## Current Setup

Your vLLM server is running with:
- Port: `11434`
- Metrics endpoint: `http://localhost:11434/metrics`
- Log directory: `/home/jiaheng/vllm_log/`

## Three Ways to Record Timing Metrics

### Method 1: Query Prometheus Metrics (Recommended)

The easiest way to get accurate timing metrics:

```bash
# Get current metrics snapshot
python3 /home/jiaheng/vllm_log/metrics_logger.py

# Continuous monitoring (updates every 60 seconds)
python3 /home/jiaheng/vllm_log/metrics_logger.py --continuous --interval 60
```

Output includes:
- Average prefill time per request
- Average decode time per request
- Token counts
- Per-token latency
- E2E latency

Files saved to: `/home/jiaheng/vllm_log/metrics/`

### Method 2: Analyze Task Logs

View timing metrics correlated with your task logs:

```bash
python3 /home/jiaheng/vllm_log/analyze_task_timing.py
```

This shows:
- Current vLLM metrics
- All task logs with request counts
- Request IDs for each task

### Method 3: Direct Prometheus Query

Query the metrics endpoint directly:

```bash
# Get all metrics
curl http://localhost:11434/metrics

# Filter for timing metrics only
curl http://localhost:11434/metrics | grep -E "time|latency"

# Get specific metric
curl http://localhost:11434/metrics | grep "vllm:time_to_first_token_seconds"
```

## Understanding the Metrics

### Example Output

From your current metrics:

```
Average Prefill Time (TTFT): 2.513s
Average Decode Time: 29.594s
Average E2E Latency: 32.107s
Total Requests: 18
```

This means:
- **Prefill (2.5s)**: Processing the prompt takes ~2.5 seconds
- **Decode (29.6s)**: Generating the response takes ~29.6 seconds
- **E2E (32.1s)**: Total time including overhead is ~32.1 seconds

### Per-Request Breakdown

For each request:
1. **Queue Time**: Time waiting in queue (usually negligible)
2. **Prefill Time**: Processing input tokens → first output token
3. **Decode Time**: Generating all output tokens
4. **Total = Queue + Prefill + Decode**

## Prometheus Histogram Metrics

The metrics are stored as histograms with:
- `_count`: Number of requests
- `_sum`: Total time across all requests
- `_bucket`: Distribution across time ranges

To get average: `sum / count`

## Automating Metrics Collection

### Option 1: Periodic Snapshots

Add to cron for hourly snapshots:

```bash
# Edit crontab
crontab -e

# Add line (runs every hour)
0 * * * * python3 /home/jiaheng/vllm_log/metrics_logger.py
```

### Option 2: Continuous Monitoring

Run in background with screen/tmux:

```bash
# Start screen session
screen -S vllm_metrics

# Run continuous monitoring
python3 /home/jiaheng/vllm_log/metrics_logger.py --continuous --interval 60

# Detach: Ctrl+A, then D
# Reattach: screen -r vllm_metrics
```

### Option 3: Systemd Service

Create `/etc/systemd/system/vllm-metrics.service`:

```ini
[Unit]
Description=vLLM Metrics Logger
After=network.target

[Service]
Type=simple
User=jiaheng
WorkingDirectory=/home/jiaheng/vllm_log
ExecStart=/usr/bin/python3 /home/jiaheng/vllm_log/metrics_logger.py --continuous --interval 60
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm-metrics
sudo systemctl start vllm-metrics
```

## Metrics Files

### Text Format
Location: `/home/jiaheng/vllm_log/metrics/metrics_YYYYMMDD_HHMMSS.txt`

Human-readable report with:
- Overall statistics
- Timing metrics
- Delta metrics (new requests since last check)

### JSON Format
Location: `/home/jiaheng/vllm_log/metrics/metrics_YYYYMMDD_HHMMSS.json`

Machine-readable format for programmatic access:
```json
{
  "timestamp": "2025-11-06T17:43:07.965598",
  "current": {
    "ttft": {"count": 18, "sum": 45.238, "average": 2.513},
    "decode_time": {"count": 18, "sum": 532.693, "average": 29.594},
    ...
  },
  "deltas": {...}
}
```

## Integrating with Task Logs

Your task logs (in `/home/jiaheng/vllm_log/task_logs/`) currently show:
- Request received events
- Request/response content

To add timing metrics to task logs, you can either:

1. **Use separate metrics files** (current approach - recommended)
   - Keep task logs clean
   - Query metrics independently
   - Correlate by timestamp/request ID

2. **Enhance task handler** (future enhancement)
   - Modify `/home/jiaheng/vllm_log/browser_use_task_handler.py`
   - Add response completion logging
   - Compute timing from log timestamps

## Troubleshooting

### Metrics endpoint not accessible
```bash
# Check if vLLM is running
ps aux | grep vllm

# Check if port is open
curl http://localhost:11434/metrics
```

### Missing timing data in task logs

The task logs only capture request receipt events, not completion events. Use the metrics logger to get timing information:

```bash
python3 /home/jiaheng/vllm_log/metrics_logger.py
```

### Want per-request timing

To get timing for each individual request:
1. Note the request ID from task logs
2. Record metrics before and after the request
3. Calculate delta

Or use vLLM's `--enable-log-requests` flag (already enabled in your setup) and parse the completion logs.

## Summary

**To record prefill and decode times:**

```bash
# Quick snapshot
python3 /home/jiaheng/vllm_log/metrics_logger.py

# Continuous monitoring
python3 /home/jiaheng/vllm_log/metrics_logger.py --continuous --interval 60

# View with task correlation
python3 /home/jiaheng/vllm_log/analyze_task_timing.py
```

Metrics are saved to:
- `/home/jiaheng/vllm_log/metrics/*.txt` (human-readable)
- `/home/jiaheng/vllm_log/metrics/*.json` (machine-readable)

## Files Created

1. **metrics_logger.py** - Main metrics collection script
2. **analyze_task_timing.py** - Correlate metrics with task logs
3. **enhanced_task_handler_with_metrics.py** - Alternative handler (not currently used)
4. **TIMING_METRICS_GUIDE.md** - This guide

## Next Steps

1. Run `metrics_logger.py` periodically or continuously
2. Analyze timing patterns in your workload
3. Optimize based on prefill vs decode time bottlenecks
4. Monitor for performance regressions

For questions or issues, refer to:
- vLLM metrics documentation: https://docs.vllm.ai/en/latest/serving/metrics.html
- Prometheus query syntax: https://prometheus.io/docs/prometheus/latest/querying/basics/
