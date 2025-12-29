# Where to See Image Sizes in VLLM Logs

## Current Situation

**Your current logs DON'T show exact image dimensions** because VLLM doesn't log image metadata by default. Here's what you can see and how to get more information:

## What You Can See NOW (Without Modifications)

### 1. Per-Request Logs (Already Working)

**Location:** `/home/jiaheng/vllm_log/requests/`

**Example:**
```bash
cat /home/jiaheng/vllm_log/requests/request_chatcmpl-1ce8925fb9f74424b3204cc57ad5da4d.log
```

**What you see:**
- ✅ Request ID
- ✅ Text prompt (mentions screenshots)
- ✅ Request timing
- ❌ Image dimensions (NOT included)

### 2. Image Size Estimation Tool (Run This)

**Command:**
```bash
cd /home/jiaheng/vllm_log
python image_size_detector.py
```

**What you see:**
```
Request: chatcmpl-1ce8925fb9f74424b3204cc57ad5da4d
  Prompt length: 21,338 characters
  Estimate: Small image (~448x448 or smaller)
  Indicators: screenshot, browser_vision
```

This gives you **estimates** based on prompt size, not exact dimensions.

### 3. InternVL Model Default

**Your model automatically processes images at:**
- **Input size:** 448x448 pixels (fixed)
- **Format:** RGB
- **Tokens per image:** ~256 tokens

All images sent to InternVL are resized to 448x448 regardless of original size.

## How to Get EXACT Image Dimensions

You have 3 options:

---

## Option 1: Request Interceptor (Easiest, No VLLM Modification)

Create a simple HTTP proxy to inspect requests before they reach VLLM:

**Step 1:** Create the interceptor script:

```bash
cat > /home/jiaheng/vllm_log/request_interceptor.py << 'EOF'
#!/usr/bin/env python3
"""
HTTP Request Interceptor for VLLM
Logs image metadata from API requests without modifying VLLM.

Usage:
    1. Start this proxy: python request_interceptor.py
    2. Configure your agent to send requests to http://localhost:11435
    3. This proxy forwards to VLLM at localhost:11434
    4. Image metadata is logged to image_metadata/
"""

from mitmproxy import http
from pathlib import Path
import base64
import io
import json
from PIL import Image
from datetime import datetime

log_dir = Path("/home/jiaheng/vllm_log/image_metadata")
log_dir.mkdir(exist_ok=True)

def extract_images_from_request(flow: http.HTTPFlow):
    """Extract and log image metadata from request."""

    if not flow.request.path.startswith("/v1/"):
        return

    try:
        body = json.loads(flow.request.content.decode('utf-8'))
    except:
        return

    images = []
    request_id = None

    # Extract from messages
    if 'messages' in body:
        for msg in body['messages']:
            content = msg.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'image_url':
                        url = item['image_url']['url']
                        if url.startswith('data:image'):
                            # Decode base64
                            header, data = url.split(',', 1)
                            img_bytes = base64.b64decode(data)
                            img = Image.open(io.BytesIO(img_bytes))

                            images.append({
                                'width': img.width,
                                'height': img.height,
                                'format': img.format,
                                'size_kb': len(img_bytes) / 1024
                            })

    if images:
        # Log to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"request_{timestamp}.txt"

        with open(log_file, 'w') as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Path: {flow.request.path}\n")
            f.write(f"Images: {len(images)}\n\n")

            for idx, img in enumerate(images, 1):
                f.write(f"Image {idx}:\n")
                f.write(f"  Dimensions: {img['width']}x{img['height']}\n")
                f.write(f"  Format: {img['format']}\n")
                f.write(f"  Size: {img['size_kb']:.2f} KB\n\n")

        print(f"✓ Logged {len(images)} image(s) to {log_file}")

def request(flow: http.HTTPFlow):
    extract_images_from_request(flow)

if __name__ == "__main__":
    from mitmproxy.tools.main import mitmdump
    mitmdump(["-s", __file__, "-p", "11435", "--mode", "reverse:http://localhost:11434"])
EOF
```

**Step 2:** Install mitmproxy:
```bash
pip install mitmproxy pillow
```

**Step 3:** Run the interceptor:
```bash
python /home/jiaheng/vllm_log/request_interceptor.py
```

**Step 4:** Point your agent to the proxy (port 11435 instead of 11434)

**Result:** Image dimensions logged to `/home/jiaheng/vllm_log/image_metadata/`

---

## Option 2: Modify VLLM Source (Most Accurate)

Edit VLLM's source code to log image metadata.

**File to modify:** Find your VLLM installation:
```bash
python -c "import vllm; print(vllm.__file__)"
# Example output: /home/jiaheng/.local/lib/python3.10/site-packages/vllm/__init__.py
```

**Edit:** `vllm/entrypoints/openai/api_server.py`

**Add this after request parsing:**
```python
# Around line ~200-300, after parsing images
if hasattr(request, 'images') and request.images:
    for idx, img in enumerate(request.images):
        if hasattr(img, 'size'):
            logger.info(f"Request {request_id} Image {idx}: {img.size[0]}x{img.size[1]} pixels")
```

**Restart VLLM** and check logs.

**Result:** Image dimensions appear in `/home/jiaheng/vllm_log/server.log`

---

## Option 3: Check Agent Logs (Simplest)

**If your agent saves screenshots, check where it saves them:**

```bash
# Find recent screenshot files
find /tmp -name "*.png" -mtime -1 2>/dev/null
find $HOME -name "*screenshot*.png" -mtime -1 2>/dev/null

# Check image size
identify /path/to/screenshot.png
# OR
python -c "from PIL import Image; img=Image.open('/path/to/screenshot.png'); print(f'{img.width}x{img.height}')"
```

---

## Quick Reference: Where to Look

| What | Where | Command |
|------|-------|---------|
| **Per-request logs** | `/home/jiaheng/vllm_log/requests/` | `ls -lt requests/` |
| **Size estimates** | Run analyzer | `python image_size_detector.py` |
| **Main VLLM log** | `/home/jiaheng/vllm_log/server.log` | `tail -f server.log` |
| **Image metadata (if using interceptor)** | `/home/jiaheng/vllm_log/image_metadata/` | `ls -lt image_metadata/` |
| **Example output** | `/home/jiaheng/vllm_log/image_metadata_example/` | `cat image_metadata_example/*.txt` |

---

## Example: What Image Metadata Logs Look Like

**File:** `/home/jiaheng/vllm_log/image_metadata_example/request_chatcmpl-1ce8925_images.txt`

```
Request ID: chatcmpl-1ce8925
Timestamp: 2025-10-29T14:47:09.123456
Total Images: 1

Image 1:
  Dimensions: 1920x1080 pixels
  Format: PNG
  Color Mode: RGB
  Size: 245.32 KB (251208 bytes)
  Aspect Ratio: 1.78:1
```

This is what you'll see **after** implementing one of the options above.

---

## Summary: Current vs. Future

### ❌ What You DON'T Have Now:
- Exact image dimensions in logs
- Original image sizes before InternVL processing

### ✅ What You DO Have Now:
- Per-request text logs: `/home/jiaheng/vllm_log/requests/`
- Image processing confirmation
- Size estimates based on prompt length
- Knowledge that InternVL processes at 448x448

### ✅ What You CAN Get (Choose One Option Above):
- Option 1: HTTP proxy → No VLLM changes needed
- Option 2: Modify VLLM → Most integrated
- Option 3: Check agent files → Simplest

---

## Recommended: Use Option 1 (Request Interceptor)

**Why:**
- ✅ No VLLM modification needed
- ✅ Captures original image sizes (before 448x448 resize)
- ✅ Doesn't affect VLLM performance
- ✅ Easy to enable/disable

**To use it:**
```bash
# Install dependency
pip install mitmproxy pillow

# Create the script (see Option 1 above)

# Run it
python /home/jiaheng/vllm_log/request_interceptor.py

# Configure your agent to use port 11435 instead of 11434

# Check logs
ls -lt /home/jiaheng/vllm_log/image_metadata/
```
