# Metrics Snapshots Guide

## Overview

This solution captures **full Prometheus metrics snapshots** before and after each request, saving them to files for post-processing analysis. This completely sidesteps the real-time blocking issues with vLLM's metrics endpoint.

## How It Works

### 1. During Server Runtime

When a request is received:
```
Request arrives → Save metrics snapshot to file (async curl)
Request processing...
Request completes → Save another snapshot to file (async curl)
```

Snapshot files are saved to: `/home/jiaheng/vllm_log/metrics_snapshots/`

Filename format: `{request_id}_{stage}_{timestamp}.txt`
- Example before: `chatcmpl-abc123_before_20251106_225030_123456.txt`
- Example after: `chatcmpl-abc123_after_20251106_225055_789012.txt`

### 2. After Completion

Run the parser script to analyze all snapshots:
```bash
python3 /home/jiaheng/vllm_log/parse_metrics_snapshots.py
```

The parser will:
1. Find all before/after snapshot pairs
2. Parse the Prometheus metrics from each file
3. Calculate deltas (after - before) for each request
4. Display per-request timing and summary statistics

## Advantages

✅ **No blocking**: `curl` runs asynchronously in background
✅ **Accurate**: Full metrics snapshot captured when server is idle
✅ **Complete data**: Full Prometheus output saved for later analysis
✅ **Flexible**: Can re-parse snapshots with different logic
✅ **Debuggable**: Raw metrics files available for inspection
✅ **No race conditions**: Each snapshot is independent

## Usage

### 1. Restart the vLLM Server

```bash
cd /home/jiaheng/vllm_log
./start_vllm_with_per_request_logging.sh
```

### 2. Make Some Requests

Use the server normally. Snapshots will be saved automatically.

### 3. Analyze the Results

```bash
python3 /home/jiaheng/vllm_log/parse_metrics_snapshots.py
```

### 4. Check Individual Snapshots

If you want to inspect the raw metrics:
```bash
ls -lh /home/jiaheng/vllm_log/metrics_snapshots/
cat /home/jiaheng/vllm_log/metrics_snapshots/chatcmpl-*_after_*.txt
```

## Output Example

```
Analyzing metrics snapshots from: /home/jiaheng/vllm_log/metrics_snapshots

Found 5 request(s) with snapshots

======================================================================
Request: chatcmpl-fe60bc421eef4d8a8462311492f0aa2a
======================================================================
Snapshot (before): .../chatcmpl-fe60bc421eef4d8a8462311492f0aa2a_before_20251106_224700_123456.txt
Snapshot (after):  .../chatcmpl-fe60bc421eef4d8a8462311492f0aa2a_after_20251106_224725_789012.txt

📊 CALCULATED METRICS:
  Prefill Time: 2.345s
  Decode Time: 18.234s
  Time to First Token (TTFT): 2.450s

[... more requests ...]

======================================================================
SUMMARY - 5 request(s) analyzed
======================================================================
Total Prefill Time: 12.345s
Total Decode Time: 85.234s
Total TTFT: 13.450s

Average Prefill Time: 2.469s
Average Decode Time: 17.047s
Average TTFT: 2.690s
```

## Cleaning Up Old Snapshots

To save disk space, periodically clean old snapshots:
```bash
# Remove snapshots older than 7 days
find /home/jiaheng/vllm_log/metrics_snapshots/ -name "*.txt" -mtime +7 -delete

# Or remove all snapshots
rm -f /home/jiaheng/vllm_log/metrics_snapshots/*.txt
```

## Troubleshooting

### No snapshots being created?
- Check permissions: `ls -la /home/jiaheng/vllm_log/metrics_snapshots/`
- Ensure curl is installed: `which curl`
- Check if metrics endpoint is accessible: `curl http://localhost:11434/metrics`

### Snapshots are empty?
- The curl might be timing out while server is busy
- Wait a few seconds and check again - curl runs asynchronously
- Increase the timeout in the code if needed

### All deltas are zero?
- Metrics might not have changed (possible if server was just started)
- Check the timestamps in filenames - ensure they're different
- Look at the raw snapshot files to verify they contain data
