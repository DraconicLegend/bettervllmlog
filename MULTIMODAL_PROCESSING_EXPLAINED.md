# How VLLM Handles Multimodal Inputs (Text + Images)

## Your Setup
- **VLLM Version:** 0.11.0
- **Model:** InternVL3.5-14B (Vision-Language Model)
- **API:** OpenAI-compatible endpoint

## Processing Flow

### 1. API Request Reception

When your agent sends a request, it arrives in one of these formats:

#### Option A: OpenAI Chat Completion Format (Most Common)
```python
POST /v1/chat/completions
{
  "model": "InternVL3_5-14B",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Analyze this browser screenshot..."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANS..."
          }
        }
      ]
    }
  ]
}
```

#### Option B: VLLM Native Format
```python
{
  "prompt": "Text prompt here",
  "multi_modal_data": {
    "image": <base64_string or PIL.Image>
  }
}
```

### 2. VLLM Internal Processing

Here's what happens inside VLLM:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. API Endpoint (vllm/entrypoints/openai/api_server.py)    │
│    - Receives HTTP request                                   │
│    - Parses JSON payload                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Request Parser                                            │
│    - Extracts text from messages                             │
│    - Extracts image data from content array                  │
│    - Decodes base64 images → PIL.Image                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Tokenizer (Text Processing)                               │
│    - Tokenizes text prompt                                   │
│    - Creates input_ids: [101, 2023, 3424, ...]              │
│    - Adds special tokens for image placeholders              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Vision Encoder (Image Processing)                         │
│    - Preprocesses image (resize, normalize)                  │
│    - Runs through vision transformer (ViT)                   │
│    - Outputs: pixel_values → image_embeddings               │
│      Shape: [1, 256, 1024] (example)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Multimodal Fusion                                         │
│    - Combines text embeddings + image embeddings            │
│    - Creates unified sequence:                               │
│      [text_tok_1, IMG_EMBED_1, ..., IMG_EMBED_N, text_tok_2]│
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. LLM Engine (Generation)                                   │
│    - Processes fused embeddings                              │
│    - Generates response tokens                               │
│    - Returns text output                                     │
└─────────────────────────────────────────────────────────────┘
```

### 3. Key Components

#### A. Vision Encoder (InternVL's ViT)
```python
# Inside VLLM, approximately:
from transformers import AutoModel

vision_model = AutoModel.from_pretrained(
    "OpenGVLab/InternVL3_5-14B",
    subfolder="vision_encoder"
)

# Process image
pixel_values = preprocess_image(image)  # Shape: [3, 448, 448]
image_embeddings = vision_model(pixel_values)  # Shape: [256, 1024]
```

#### B. Multimodal Projector
```python
# Projects vision embeddings to LLM embedding space
vision_embeddings = projector(image_embeddings)
# Now compatible with text embedding dimensions
```

#### C. Combined Input to LLM
```python
# Pseudo-code showing the fusion
text_embeddings = embedding_layer(input_ids)  # [seq_len, hidden_dim]
vision_embeddings = vision_encoder(images)     # [img_tokens, hidden_dim]

# Interleave based on special tokens
combined = interleave(text_embeddings, vision_embeddings)
# Result: [total_seq_len, hidden_dim]

# Feed to transformer
output = transformer(combined)
```

## Why Images Don't Appear in Logs

### 1. **Log Entry Point**
The VLLM logger at `vllm.entrypoints.logger` only logs the **text prompt**:

```python
# In vllm/entrypoints/logger.py (simplified)
logger.info(f"Received request {request_id}: prompt: '{text_prompt}'")
```

It doesn't log image data because:
- Images are large (100KB - 10MB per image)
- Binary data is not human-readable in text logs
- Would make log files massive

### 2. **What Actually Happens to Images**
```python
# Simplified VLLM flow
request = {
    "prompt": text,           # ← Logged ✓
    "multi_modal_data": {
        "image": image_tensor  # ← NOT logged (binary data)
    }
}
```

The image tensor goes directly to the GPU for processing, bypassing text logging entirely.

## How to Verify Images Are Being Processed

### Method 1: Check Request Payload Size
```bash
# Your agent's request will be much larger if images are included
# Text-only: ~10-50KB
# With image: ~500KB - 5MB
```

### Method 2: Look for Vision Model Loading
Check your startup logs:
```bash
grep -i "vision\|image_encoder\|internvit" /home/jiaheng/vllm_log/server.log
```

You should see vision model components being loaded.

### Method 3: Monitor GPU Memory
```bash
# Image processing requires more GPU memory
# Text-only: ~2-4GB
# With images: ~8-12GB (depending on batch size)
nvidia-smi
```

### Method 4: Add Custom Logging (Advanced)

You could modify VLLM to log image metadata:

```python
# Add to vllm/entrypoints/openai/api_server.py
if hasattr(request, 'multi_modal_data') and request.multi_modal_data:
    image_data = request.multi_modal_data.get('image')
    if image_data:
        logger.info(f"Request {request_id} includes image: shape={image_data.shape}")
```

## What Your Per-Request Logger Captures

Your custom logger captures:
- ✓ Request ID (e.g., `chatcmpl-abc123`)
- ✓ Text prompt
- ✓ Request timing (received, added, completed)
- ✓ Errors and aborts
- ✗ Image pixel data (not in text logs)
- ✗ Image embeddings (processed in GPU memory)

## Understanding the Request Flow

```
Agent                    VLLM API                  VLLM Engine
  |                         |                          |
  |-- POST request -------->|                          |
  |  (text + base64 img)    |                          |
  |                         |-- Parse request -------->|
  |                         |                          |
  |                         |   Text → Tokenizer       |
  |                         |   Image → Vision Encoder |
  |                         |                          |
  |                         |<-- Combined embeddings --|
  |                         |                          |
  |                         |-- Generate response ---->|
  |<-- JSON response -------|                          |
  |  (text output only)     |                          |
```

## InternVL3.5-14B Specific Details

### Architecture
- **Vision Encoder:** InternViT-300M-448px
- **Language Model:** Qwen2.5-7B-Instruct
- **Image Resolution:** 448x448 pixels
- **Image Tokens:** ~256 tokens per image

### Special Tokens
InternVL uses special tokens to mark image positions:
```
<|im_start|>user
<image>
What's in this screenshot?
<|im_end|>
```

The `<image>` token is replaced by the actual image embeddings during processing.

## Summary

**YES, your VLLM is processing images**, but:
1. Images arrive as base64-encoded data in API requests
2. VLLM decodes and processes them through the vision encoder
3. Image embeddings are fused with text embeddings
4. Only text appears in logs (images are binary data)
5. Your per-request logger correctly captures all text-based logging

The absence of images in logs is **normal and expected** - it doesn't mean images aren't being processed!
