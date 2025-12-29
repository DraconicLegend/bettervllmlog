#!/bin/bash

# Add the vllm_log directory to Python path so the custom handler can be imported
export PYTHONPATH="~/vllm_log:$PYTHONPATH"
export VLLM_SERVER_DEV_MODE=1  
export VLLM_LOG_STATS_INTERVAL=1.0
rm -f server.log uvicorn.log
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