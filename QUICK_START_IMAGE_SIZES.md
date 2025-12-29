# Quick Start: How to See Image Sizes in VLLM Logs

## The Problem

VLLM doesn't log image dimensions by default. You can only see that images are being processed, not their sizes.

## The Solution (3 Simple Steps)

### Step 1: Install Required Package

```bash
pip install pillow requests
```

### Step 2: Start the Image Monitor

```bash
python /home/jiaheng/vllm_log/simple_request_monitor.py
```

You'll see:
```
======================================================================
VLLM Image Size Monitor (Proxy Server)
======================================================================

Proxy listening on: http://localhost:11435
Forwarding to VLLM: http://localhost:11434
Logs directory: /home/jiaheng/vllm_log/image_metadata

Configure your agent to use:
  http://localhost:11435/v1/chat/completions

Press Ctrl+C to stop
======================================================================
```

### Step 3: Configure Your Agent

Change your agent to send requests to **port 11435** instead of 11434:

```bash
# Before:
http://localhost:11434/v1/chat/completions

# After:
http://localhost:11435/v1/chat/completions
```

## Where to See the Logs

### Real-time Output (Terminal)

When a request with images comes in, you'll see:

```
✓ Intercepted request with 1 image(s)
  Image 1: 1920x1080 PNG (245.32 KB)
  Logged to: request_20251029_144709_123456.txt
```

### Log Files

**Location:** `/home/jiaheng/vllm_log/image_metadata/`

**View all logs:**
```bash
ls -lt /home/jiaheng/vllm_log/image_metadata/
```

**View latest log:**
```bash
cat /home/jiaheng/vllm_log/image_metadata/request_*.txt | tail -20
```

**Example log content:**
```
Timestamp: 2025-10-29T14:47:09.123456
Path: /v1/chat/completions
Total Images: 1

Image 1:
  Dimensions: 1920x1080 pixels
  Format: PNG
  Color Mode: RGB
  Size: 245.32 KB (251208 bytes)
  Aspect Ratio: 1.78:1
```

## How It Works

```
Agent → Port 11435 (Monitor) → Port 11434 (VLLM) → Response
         ↓
    Logs image size
    to file and console
```

The monitor:
1. Intercepts requests
2. Extracts and decodes base64 images
3. Gets dimensions with PIL
4. Logs to file
5. Forwards request to VLLM unchanged

## Troubleshooting

### "Port already in use"

```bash
# Check what's using port 11435
lsof -i :11435

# Kill it if needed
kill <PID>
```

### "Can't connect to VLLM"

Make sure VLLM is running on port 11434:
```bash
curl http://localhost:11434/v1/models
```

### "No images logged"

Check that your agent is:
1. Sending requests to port 11435
2. Including images in base64 format
3. Using the correct API format

## Alternative: Without Proxy

If you don't want to use a proxy, you can estimate image sizes:

```bash
cd /home/jiaheng/vllm_log
python image_size_detector.py
```

This gives estimates based on prompt length, not exact dimensions.

## Summary

| Method | Accuracy | Setup | Location |
|--------|----------|-------|----------|
| **Proxy Monitor** | Exact dimensions | 3 steps | `/home/jiaheng/vllm_log/image_metadata/` |
| **Estimation Tool** | Approximate | 0 steps | Run `python image_size_detector.py` |
| **Per-request logs** | No size info | Working | `/home/jiaheng/vllm_log/requests/` |

**Recommended:** Use the proxy monitor for exact image dimensions.
