#!/bin/bash

# vLLM Server Startup with Profiling and Per-Request Logging
# Based on: https://docs.vllm.ai/en/v0.5.5/dev/profiling/profiling_index.html

# Add the vllm_log directory to Python path so the custom handler can be imported
export PYTHONPATH="/home/jiaheng/vllm_log:$PYTHONPATH"

# ============================================================================
# PROFILING CONFIGURATION
# ============================================================================

# Enable Torch Profiler - saves profiling data to directory
export VLLM_TORCH_PROFILER_DIR="/home/jiaheng/vllm_log/profiler_output"

# Record tensor shapes in profiler (helps identify memory issues)
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1

# Include FLOPS (floating point operations) in profiler
export VLLM_TORCH_PROFILER_WITH_FLOPS=1

# Profile memory usage
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=1

# Include Python stack traces (helps identify bottlenecks)
export VLLM_TORCH_PROFILER_WITH_STACK=1

# Enable function-level tracing for detailed profiling
# Options: "all", "custom", or comma-separated function names
# export VLLM_TRACE_FUNCTION="all"  # Uncomment for detailed tracing (warning: large overhead!)

# NVTX scopes for NVIDIA Nsight profiling
# Comma-separated list of scopes to profile
# export VLLM_NVTX_SCOPES_FOR_PROFILING="model_forward,attention,sampling"

# Custom scopes for profiling (for your own instrumentation)
# export VLLM_CUSTOM_SCOPES_FOR_PROFILING="my_custom_scope"

# ============================================================================
# OPTIONAL: CUDA Profiling
# ============================================================================

# Disable CUDA kernel launch blocking for better performance
# Set to 1 only if you need synchronous CUDA calls for debugging
# export CUDA_LAUNCH_BLOCKING=0

# Kineto (PyTorch profiler backend) log level
# Options: INFO, WARNING, ERROR
# export KINETO_LOG_LEVEL=WARNING

# ============================================================================
# CREATE PROFILER OUTPUT DIRECTORY
# ============================================================================

mkdir -p /home/jiaheng/vllm_log/profiler_output

echo "======================================================================="
echo "vLLM Server with Profiling Enabled"
echo "======================================================================="
echo ""
echo "Profiling Configuration:"
echo "  - Torch Profiler: ENABLED"
echo "  - Output Directory: $VLLM_TORCH_PROFILER_DIR"
echo "  - Record Shapes: $VLLM_TORCH_PROFILER_RECORD_SHAPES"
echo "  - Include FLOPS: $VLLM_TORCH_PROFILER_WITH_FLOPS"
echo "  - Profile Memory: $VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY"
echo "  - Include Stack: $VLLM_TORCH_PROFILER_WITH_STACK"
echo ""
echo "Profiler data will be saved to:"
echo "  $VLLM_TORCH_PROFILER_DIR/*.json"
echo ""
echo "To view profiler data:"
echo "  1. Chrome/Edge: Open chrome://tracing"
echo "  2. Load the JSON file from profiler_output/"
echo "  3. Or use: python -m torch.utils.bottleneck <script>"
echo ""
echo "======================================================================="
echo ""

# ============================================================================
# START VLLM SERVER
# ============================================================================

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

# Note: Profiling has some performance overhead, especially with:
# - VLLM_TRACE_FUNCTION enabled
# - Stack traces enabled
# - Memory profiling enabled
#
# For production, consider disabling or reducing profiling scope.
