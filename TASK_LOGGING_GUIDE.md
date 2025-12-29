# Browser-Use Task-Based Logging

## Overview

VLLM logs are now grouped by **browser-use task** instead of individual requests.

**One agent task = One log file** (containing all VLLM requests from that task)

## How It Works

The system automatically detects when a new browser-use task starts by checking:

1. **Step reset**: When Step number goes back to 1 (new task started)
2. **Idle timeout**: More than 5 minutes since last request (assumed new task)
3. **First request**: Very first request creates initial task file

## File Structure

```
/home/jiaheng/vllm_log/
├── server.log              # All requests (original, for reference)
├── task_logs/              # NEW: One file per browser-use task
│   ├── task_20251030_211200.log    # Task 1 (all its requests)
│   ├── task_20251030_214500.log    # Task 2 (all its requests)
│   └── task_20251030_220000.log    # Task 3 (all its requests)
└── uvicorn.log             # Uvicorn logs
```

## Task Log Format

Each task file contains:

```
======================================================================
Browser-Use Task Log
Task ID: task_20251030_211200
Started: 2025-10-30T21:12:00.123456
======================================================================

----------------------------------------------------------------------
Request #1 (Step 1): chatcmpl-abc123
Time: 2025-10-30T21:12:00.500000
----------------------------------------------------------------------
2025-10-30 21:12:00,500 INFO vllm.entrypoints.logger - Received request...
2025-10-30 21:12:00,501 INFO vllm.v1.engine.async_llm - Added request...

----------------------------------------------------------------------
Request #2 (Step 2): chatcmpl-def456
Time: 2025-10-30T21:12:15.800000
----------------------------------------------------------------------
2025-10-30 21:12:15,800 INFO vllm.entrypoints.logger - Received request...
2025-10-30 21:12:15,801 INFO vllm.v1.engine.async_llm - Added request...

[... all requests from this task ...]

======================================================================
Task ended: 2025-10-30T21:15:30.000000
Total requests: 15
======================================================================
```

## Configuration

**File:** `/home/jiaheng/vllm_log/vllm_logging.json`

**Idle timeout:** 5 minutes (configurable)

Change the timeout by editing line 19:
```json
"idle_timeout_minutes": 5
```

**Logs directory:** `/home/jiaheng/vllm_log/task_logs/`

## Restart VLLM to Apply

After making changes, restart your VLLM server:

```bash
# Stop current VLLM
# Then restart with:
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
  --log-config /home/jiaheng/vllm_log/vllm_logging.json
```

Or use the script:
```bash
export PYTHONPATH="/home/jiaheng/vllm_log:$PYTHONPATH"
/home/jiaheng/vllm_log/start_vllm_with_per_request_logging.sh
```

## Viewing Task Logs

**List all tasks:**
```bash
ls -lt /home/jiaheng/vllm_log/task_logs/
```

**View latest task:**
```bash
cat /home/jiaheng/vllm_log/task_logs/task_*.log | tail -50
```

**View specific task:**
```bash
cat /home/jiaheng/vllm_log/task_logs/task_20251030_211200.log
```

**Count requests in a task:**
```bash
grep "Request #" /home/jiaheng/vllm_log/task_logs/task_20251030_211200.log | wc -l
```

## How Tasks Are Grouped

### Example: Single Browser-Use Task

You run: `python my_agent.py "Search for hello world on DuckDuckGo"`

This makes multiple VLLM requests (Step 1, Step 2, Step 3...):
- All go into **ONE file**: `task_20251030_211200.log`

### Example: Multiple Tasks

1. **Task 1** (11:00): "Search for hello world" → `task_20251030_110000.log`
2. Wait 10 minutes (idle timeout)
3. **Task 2** (11:15): "Find Python tutorials" → `task_20251030_111500.log`

Each gets its own file!

## Benefits

✅ **Easy debugging**: All requests from one task in one place
✅ **Clean organization**: No more hundreds of individual request files
✅ **Task tracking**: See exactly what happened in each agent run
✅ **Performance**: Still maintains `server.log` for reference

## Troubleshooting

### Tasks not splitting correctly

**Problem:** All requests going into same file

**Solution:** Adjust idle timeout (make it shorter):
```json
"idle_timeout_minutes": 2
```

### Too many task files

**Problem:** New file created for every request

**Solution:** Increase idle timeout:
```json
"idle_timeout_minutes": 10
```

### Handler not loading

**Problem:** Error about `browser_use_task_handler`

**Solution:** Make sure PYTHONPATH includes the log directory:
```bash
export PYTHONPATH="/home/jiaheng/vllm_log:$PYTHONPATH"
```

## Summary

- **Before:** One file per request (hundreds of files)
- **After:** One file per agent task (organized by task)
- **Automatic:** Detects task boundaries using step numbers and timeouts
- **Configurable:** Adjust timeout to match your workflow