# Benchmark.py Fixes Applied

## Issues Fixed:

### 1. **Wrong Metrics URL** (FIXED ✓)
**Problem:**
```python
METRICS_URL = "http://localhost:11435/metrics"  # Wrong port!
```

**Fix:**
```python
METRICS_URL = "http://localhost:11434/metrics"  # Same port as API
```

### 2. **Metrics Parsing Not Compatible with vLLM v1** (FIXED ✓)
**Problem:** Only looked for old metric names like `vllm_request_prefill_time_seconds_sum`

**Fix:** Updated to handle both v0 and v1 metric formats:
```python
if "prefill_time_seconds_sum" in line or "vllm:request_prefill_time_seconds_sum" in line:
    m["prefill_sum"] = float(line.strip().split()[-1])
```

### 3. **No Error Handling** (FIXED ✓)
**Added:**
- Connection error handling for metrics endpoint
- Validation warnings when metrics are None
- Debug output showing actual metric values when parsing fails
- Graceful fallback to 0.0 instead of None

### 4. **Better Debugging** (FIXED ✓)
**Added:**
- Warning messages when metrics can't be extracted
- Debug output showing before/after metric values
- Validation that server is reachable

## How to Run:

### Step 1: Make sure vLLM server is running
```bash
# Check if server is running
curl http://localhost:11434/health

# Check if metrics are available
curl http://localhost:11434/metrics | grep prefill_time_seconds
```

### Step 2: Run the benchmark
```bash
cd /home/jiaheng/vllm_log
python3 benchmark.py
```

### Expected Output:
```
[*] Base prompt tokens: 4000
[*] Warm-up …
[*] reuse=0.00  LCP=0/4000
    prefill=X.XXXXs  decode=X.XXXXs
[*] reuse=0.20  LCP=800/4000
    prefill=X.XXXXs  decode=X.XXXXs
...
Saved: kv_reuse_vs_prefill.csv
Saved plot: kv_reuse_vs_prefill.png
```

### What It Tests:
- Measures prefill time with different levels of KV cache reuse (0%, 20%, 40%, 60%, 80%, 100%)
- Shows how prefix caching reduces prefill time
- Creates a CSV with detailed results
- Generates a plot showing the relationship

## Expected Results (with prefix caching enabled):

- **0% reuse**: Longest prefill time (all tokens computed)
- **20% reuse**: Slightly faster
- **40% reuse**: Noticeably faster
- **60% reuse**: Much faster
- **80% reuse**: Very fast
- **100% reuse**: Fastest (all tokens from cache)

If you see the same prefill time for all reuse fractions, it means:
- Prefix caching is disabled (`--disable-prefix-caching` flag is set)
- Or the cache was cleared between requests

## Troubleshooting:

### If you get connection errors:
```bash
# Make sure vLLM is running on port 11434
ps aux | grep vllm
```

### If metrics are all 0.0:
This is normal at startup. Run a request first:
```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":10}'
```

### If you get import errors:
```bash
pip install transformers requests matplotlib
```

## Output Files:

1. **kv_reuse_vs_prefill.csv** - Raw benchmark data
2. **kv_reuse_vs_prefill.png** - Visualization showing prefill time vs reuse fraction
