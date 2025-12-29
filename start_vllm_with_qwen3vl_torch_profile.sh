#!/bin/bash

# Add the vllm_log directory to Python path so the custom handler can be imported
export PYTHONPATH="$HOME/vllm_log:$PYTHONPATH"
export VLLM_SERVER_DEV_MODE=1  
export VLLM_TORCH_PROFILER_DIR=$HOME/vllm_log/torch_profile
export VLLM_RPC_TIMEOUT=1800000
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=1
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_WITH_FLOPS=1
# Auto-start profiling (profile first N requests)
export VLLM_TORCH_PROFILER_WAIT=1
export VLLM_TORCH_PROFILER_WARMUP=1
export VLLM_TORCH_PROFILER_ACTIVE=3
export VLLM_TORCH_PROFILER_REPEAT=1

rm -f uvicorn.log server.log
# Start VLLM server with per-request logging
vllm serve $HF_MODELS/Qwen/Qwen3-VL-30B-A3B-Instruct \
  --served-model-name Qwen3-VL-30B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 40960 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --enable-log-requests \
  --enable-log-outputs \
  --max-log-len 200000 \
  --log-config vllm_logging.json \
  --enable-prefix-caching