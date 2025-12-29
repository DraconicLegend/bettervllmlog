#!/bin/bash

# Add the vllm_log directory to Python path so the custom handler can be imported
export PYTHONPATH="/home/jiaheng/vllm_log:$PYTHONPATH"

# Start VLLM server with per-request logging
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
  --log-config /home/jiaheng/vllm_log/vllm_logging.json \
  --no-enable-prefix-caching
  # Add --disable-prefix-caching to force recomputation of all tokens (no caching)
