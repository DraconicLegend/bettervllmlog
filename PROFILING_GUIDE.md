# vLLM Profiling Guide

Complete guide to profiling your vLLM server for performance analysis.

## Quick Start

### 1. Start vLLM with Profiling

```bash
# Stop current vLLM server
pkill -f vllm

# Start with profiling enabled
/home/jiaheng/vllm_log/start_vllm_with_profiling.sh
```

### 2. Run Some Requests

Make a few API calls to generate profiling data.

### 3. View Profiling Results

```bash
# Check profiler output
ls -lh /home/jiaheng/vllm_log/profiler_output/

# View in Chrome
# 1. Open chrome://tracing
# 2. Click "Load" and select the JSON file
```

## Profiling Environment Variables

### Core Profiling

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_TORCH_PROFILER_DIR` | `None` | Directory to save profiler output (.json files) |
| `VLLM_TORCH_PROFILER_RECORD_SHAPES` | `0` | Record tensor shapes (1=enable) |
| `VLLM_TORCH_PROFILER_WITH_FLOPS` | `0` | Include FLOPS calculations (1=enable) |
| `VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY` | `0` | Profile memory usage (1=enable) |
| `VLLM_TORCH_PROFILER_WITH_STACK` | `0` | Include Python stack traces (1=enable) |

### Advanced Profiling

| Variable | Options | Description |
|----------|---------|-------------|
| `VLLM_TRACE_FUNCTION` | `"all"`, `"custom"`, or function names | Enable function-level tracing |
| `VLLM_NVTX_SCOPES_FOR_PROFILING` | Comma-separated scopes | NVIDIA Nsight profiling scopes |
| `VLLM_CUSTOM_SCOPES_FOR_PROFILING` | Comma-separated scopes | Custom profiling scopes |

### CUDA Profiling

| Variable | Options | Description |
|----------|---------|-------------|
| `CUDA_LAUNCH_BLOCKING` | `0` or `1` | Synchronous CUDA calls (for debugging) |
| `KINETO_LOG_LEVEL` | `INFO`, `WARNING`, `ERROR` | PyTorch profiler log level |

## Profiling Modes

### Mode 1: Lightweight Profiling (Recommended for Production)

```bash
export VLLM_TORCH_PROFILER_DIR="/home/jiaheng/vllm_log/profiler_output"
# Only basic profiling, minimal overhead
```

### Mode 2: Detailed Profiling (For Analysis)

```bash
export VLLM_TORCH_PROFILER_DIR="/home/jiaheng/vllm_log/profiler_output"
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_WITH_FLOPS=1
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=1
export VLLM_TORCH_PROFILER_WITH_STACK=1
# More details, moderate overhead
```

### Mode 3: Deep Profiling (For Debugging)

```bash
export VLLM_TORCH_PROFILER_DIR="/home/jiaheng/vllm_log/profiler_output"
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_WITH_FLOPS=1
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=1
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TRACE_FUNCTION="all"
# Maximum detail, HIGH overhead - not for production!
```

## Viewing Profiler Data

### Option 1: Chrome Tracing (Recommended)

1. Open Chrome or Edge browser
2. Navigate to: `chrome://tracing`
3. Click "Load" button
4. Select JSON file from `/home/jiaheng/vllm_log/profiler_output/`
5. Use WASD keys to navigate the timeline

**Features:**
- Interactive timeline view
- Zoom in/out on operations
- See GPU/CPU utilization
- Identify bottlenecks visually

### Option 2: TensorBoard

```bash
# Install TensorBoard profiler plugin
pip install torch-tb-profiler

# View in TensorBoard
tensorboard --logdir=/home/jiaheng/vllm_log/profiler_output

# Open in browser
# http://localhost:6006
```

### Option 3: PyTorch Profiler Analysis

```python
import torch

# Load profiler trace
trace = torch.profiler.load('/home/jiaheng/vllm_log/profiler_output/trace.json')

# Print summary
print(trace.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export to CSV
trace.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
```

## Understanding Profiler Output

### Key Metrics

**Timeline View:**
- **Green bars**: CPU operations
- **Blue bars**: GPU kernels
- **Red bars**: Memory operations
- **Width**: Operation duration

**What to Look For:**
- ✅ Long GPU kernels (optimization opportunity)
- ✅ GPU idle time (CPU bottleneck)
- ✅ CPU idle time (GPU bottleneck)
- ✅ Memory copy operations (data transfer overhead)

### Common Bottlenecks

1. **Long Attention Computation**
   - Symptom: Large blue bars for attention ops
   - Solution: Enable FlashAttention, reduce context length

2. **CPU-GPU Data Transfer**
   - Symptom: Many red bars between CPU and GPU
   - Solution: Reduce batch size, optimize data pipeline

3. **CPU Bound**
   - Symptom: GPU idle between requests
   - Solution: Increase batch size, optimize preprocessing

4. **Memory Bound**
   - Symptom: Long memory allocation/deallocation
   - Solution: Reduce KV cache size, use quantization

## Profiling Specific Operations

### Profile Only Attention

```bash
export VLLM_NVTX_SCOPES_FOR_PROFILING="attention"
```

### Profile Forward Pass

```bash
export VLLM_NVTX_SCOPES_FOR_PROFILING="model_forward"
```

### Profile Sampling

```bash
export VLLM_NVTX_SCOPES_FOR_PROFILING="sampling"
```

### Profile Multiple Scopes

```bash
export VLLM_NVTX_SCOPES_FOR_PROFILING="model_forward,attention,sampling"
```

## Performance Tips

### Profiling Overhead

| Configuration | Overhead | Use Case |
|---------------|----------|----------|
| No profiling | 0% | Production |
| Basic profiling | ~2-5% | Continuous monitoring |
| Detailed profiling | ~10-20% | Performance analysis |
| Function tracing | ~50-100% | Deep debugging only |

### Best Practices

1. **Start Light**: Enable basic profiling first
2. **Profile Incrementally**: Add details as needed
3. **Short Sessions**: Profile 10-100 requests, not thousands
4. **Compare Baselines**: Profile before/after changes
5. **Focus on Hotspots**: Use profiler to find slow operations

## Example Workflow

### 1. Enable Basic Profiling

```bash
export VLLM_TORCH_PROFILER_DIR="/home/jiaheng/vllm_log/profiler_output"
./start_vllm_with_profiling.sh
```

### 2. Run Test Workload

```bash
# Send 10 test requests
for i in {1..10}; do
  curl -X POST http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "InternVL3_5-14B",
      "messages": [{"role": "user", "content": "Hello"}],
      "max_tokens": 100
    }'
done
```

### 3. Analyze Results

```bash
# List profiler files
ls -lh /home/jiaheng/vllm_log/profiler_output/

# Open in Chrome
# chrome://tracing → Load → select .json file
```

### 4. Identify Bottlenecks

Look for:
- Longest GPU kernels
- CPU-GPU transfer times
- Memory allocation patterns

### 5. Optimize and Re-profile

Make changes, restart with profiling, compare results.

## Integration with Existing Scripts

### Your Current Script

```bash
/home/jiaheng/vllm_log/start_vllm_with_per_request_logging.sh
```

**Features:**
- ✅ Per-request logging
- ✅ Task-based log files
- ❌ No profiling

### New Profiling Script

```bash
/home/jiaheng/vllm_log/start_vllm_with_profiling.sh
```

**Features:**
- ✅ Per-request logging
- ✅ Task-based log files
- ✅ **Torch profiling enabled**
- ✅ **Performance traces saved**

### Using Both Together

The profiling script is based on your current one, so you can:

1. Use current script for normal operation
2. Use profiling script when analyzing performance
3. Switch between them as needed

## Troubleshooting

### Profiler Not Generating Files

**Check:**
```bash
# Directory exists?
ls -la /home/jiaheng/vllm_log/profiler_output/

# Environment variable set?
echo $VLLM_TORCH_PROFILER_DIR

# Permissions OK?
touch /home/jiaheng/vllm_log/profiler_output/test.txt
```

### Large Profiler Files

**Solutions:**
- Reduce number of requests profiled
- Disable stack traces: `VLLM_TORCH_PROFILER_WITH_STACK=0`
- Disable FLOPS: `VLLM_TORCH_PROFILER_WITH_FLOPS=0`
- Profile specific scopes only

### High Overhead

**Solutions:**
- Disable function tracing
- Use basic profiling mode
- Profile fewer requests
- Disable memory profiling

## Summary

### Quick Commands

```bash
# Start with profiling
/home/jiaheng/vllm_log/start_vllm_with_profiling.sh

# View profiler output
ls -lh /home/jiaheng/vllm_log/profiler_output/

# Open in Chrome
# chrome://tracing → Load .json file

# Analyze in Python
python3 -c "
import torch
trace = torch.profiler.load('profiler_output/trace.json')
print(trace.key_averages().table(sort_by='cuda_time_total', row_limit=10))
"
```

### Files Created

- **start_vllm_with_profiling.sh** - Launch script with profiling
- **profiler_output/** - Directory for profiler traces
- **PROFILING_GUIDE.md** - This guide

### Next Steps

1. Try the profiling script
2. Run some requests
3. View traces in Chrome
4. Identify performance bottlenecks
5. Optimize and re-profile

For more details, see: https://docs.vllm.ai/en/latest/dev/profiling/
