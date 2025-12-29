# Simple Setup: Add Timing to Task Logs

## The Simple Solution

Your task logs already capture request events. This enhancement adds **timing metrics directly into your existing task log files** - no separate tools needed!

## What Gets Added to Your Task Logs

### Before (current):
```
----------------------------------------------------------------------
Request #1 (Step 1): chatcmpl-786bad27c1cf4c529d7cd2181e58d9ca
Time: 2025-11-06T17:35:20.020602
----------------------------------------------------------------------
2025-11-06 17:35:20,016 INFO vllm.entrypoints.logger - Received request...
```

### After (with timing):
```
----------------------------------------------------------------------
Request #1 (Step 1): chatcmpl-786bad27c1cf4c529d7cd2181e58d9ca
Time: 2025-11-06T17:35:20.020602
----------------------------------------------------------------------
2025-11-06 17:35:20,016 INFO vllm.entrypoints.logger - Received request...
...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
⏱️  TIMING METRICS - Request #1: chatcmpl-786bad27c1cf4c529d7cd2181e58d9ca
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Completed: 2025-11-06T17:35:35.973000
Total Latency: 15.952s
Output Tokens: 220
Average Time/Token: 0.073s

Estimated Breakdown:
  Prefill (est): ~1.595s
  Decode (est): ~14.357s
Finish Reason: stop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

## How to Enable

### Option 1: Update Your vLLM Start Script (Recommended)

Replace the handler file in your logging config:

```bash
# Edit your vLLM start script
nano /home/jiaheng/start_vllm_with_per_request_logging.sh

# Change this line:
# --log-config /home/jiaheng/vllm_log/vllm_logging.json

# To:
# --log-config /home/jiaheng/vllm_log/vllm_logging_with_timing.json
```

Then restart vLLM.

### Option 2: Replace the Handler File

Simply replace the old handler with the new one:

```bash
# Backup original
cd /home/jiaheng/vllm_log
cp browser_use_task_handler.py browser_use_task_handler.py.backup

# Replace with enhanced version
cp browser_use_task_handler_with_timing.py browser_use_task_handler.py

# Restart vLLM
```

### Option 3: Use New Config File

```bash
# Restart vLLM with new config
vllm serve /home/jiaheng/hf_models/OpenGVLab/InternVL3_5-14B \
  --served-model-name InternVL3_5-14B \
  --host 0.0.0.0 \
  --port 11434 \
  --max-model-len 40960 \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --enable-log-requests \
  --enable-log-outputs \
  --max-log-len 200000 \
  --log-config /home/jiaheng/vllm_log/vllm_logging_with_timing.json
```

## What You Get

✅ **All timing in task logs** - No need to check separate files
✅ **Per-request breakdown** - See timing for each request
✅ **Automatic calculation** - Prefill and decode time estimated
✅ **Same file location** - Your task logs at `/home/jiaheng/vllm_log/task_logs/`
✅ **No extra tools** - Everything in one place

## Limitations

The timing breakdown shows:
- **Total Latency**: ✅ Accurate (measured from request start to completion)
- **Output Tokens**: ✅ Accurate (counted from response)
- **Prefill/Decode Split**: ⚠️ Estimated (rough heuristic)

For **exact prefill and decode times from Prometheus metrics**, you can still use:
```bash
python3 /home/jiaheng/vllm_log/metrics_logger.py
```

But for most use cases, the estimates in the task logs are good enough!

## Files Created

1. **browser_use_task_handler_with_timing.py** - Enhanced handler with timing
2. **vllm_logging_with_timing.json** - Updated logging config
3. **SIMPLE_TIMING_SETUP.md** - This guide

## Original Files (Backups)

- `vllm_logging.json.backup` - Your original config
- `browser_use_task_handler.py` - Original handler (still works)

## Summary

**Before**: Request logs only
**After**: Request logs + timing metrics
**Change**: One line in your vLLM start script or config
**Result**: Complete timing info directly in your task logs!
