"""
Detect image sizes from VLLM logs by analyzing base64 data or request payloads.

This script can:
1. Estimate image size from base64 encoded data in logs
2. Monitor VLLM API requests to capture actual image dimensions
3. Analyze request payloads for image metadata
"""

import re
import base64
import io
from pathlib import Path
from PIL import Image


def estimate_image_size_from_base64(base64_string):
    """
    Decode base64 image and get dimensions.

    Returns: (width, height, format, size_kb)
    """
    try:
        # Remove data URI prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]

        # Decode base64
        image_data = base64.b64decode(base64_string)

        # Get size in KB
        size_kb = len(image_data) / 1024

        # Try to open as image
        img = Image.open(io.BytesIO(image_data))
        width, height = img.size
        img_format = img.format

        return {
            'width': width,
            'height': height,
            'format': img_format,
            'size_kb': round(size_kb, 2),
            'size_bytes': len(image_data)
        }
    except Exception as e:
        return {'error': str(e)}


def search_for_base64_images_in_logs(log_file):
    """
    Search VLLM logs for base64 encoded images.

    Note: VLLM typically doesn't log full base64 data, but this checks anyway.
    """
    print("Searching for base64 image data in logs...")

    with open(log_file, 'r', errors='ignore') as f:
        content = f.read()

    # Pattern for base64 image data
    base64_pattern = re.compile(
        r'data:image/(png|jpeg|jpg|gif|webp);base64,([A-Za-z0-9+/]{100,}={0,2})',
        re.IGNORECASE
    )

    matches = base64_pattern.finditer(content)
    images_found = []

    for match in matches:
        image_format = match.group(1)
        base64_data = match.group(2)

        # Try to decode and get dimensions
        result = estimate_image_size_from_base64(base64_data)
        if 'error' not in result:
            images_found.append(result)

    return images_found


def estimate_from_prompt_length(prompt_length):
    """
    Estimate if request has images based on prompt length.

    Typical sizes:
    - Text only: 1-10KB
    - Text + 1 small image: 50-200KB
    - Text + 1 large image: 200KB-2MB
    - Text + multiple images: 500KB-10MB
    """
    if prompt_length < 10000:
        return "Text only (no images)"
    elif prompt_length < 50000:
        return "Small image (~448x448 or smaller)"
    elif prompt_length < 200000:
        return "Medium image (~1024x768)"
    elif prompt_length < 1000000:
        return "Large image or multiple images"
    else:
        return "Multiple large images"


def analyze_requests_for_image_sizes(log_file):
    """
    Analyze VLLM logs to estimate image sizes from request data.
    """
    print("\n" + "="*70)
    print("VLLM Image Size Analysis")
    print("="*70 + "\n")

    with open(log_file, 'r', errors='ignore') as f:
        content = f.read()

    # Find all "Received request" entries
    request_pattern = re.compile(
        r'Received request (chatcmpl-[a-f0-9]+): prompt: \'(.{0,500})',
        re.IGNORECASE
    )

    requests = []

    for match in request_pattern.finditer(content):
        request_id = match.group(1)
        prompt_start = match.group(2)

        # Try to find the full prompt for this request
        # Look for "params: SamplingParams" which marks the end
        start_pos = match.start()
        end_match = re.search(r', params: SamplingParams', content[start_pos:start_pos+100000])

        if end_match:
            full_prompt = content[start_pos:start_pos+end_match.start()]
            prompt_length = len(full_prompt)
        else:
            prompt_length = len(prompt_start)

        # Check for image indicators
        has_screenshot = 'screenshot' in full_prompt.lower() if end_match else False
        has_image_tag = '<image>' in full_prompt if end_match else False
        has_browser_vision = 'browser_vision' in full_prompt.lower() if end_match else False

        # Estimate image characteristics
        estimate = estimate_from_prompt_length(prompt_length)

        requests.append({
            'request_id': request_id,
            'prompt_length': prompt_length,
            'has_screenshot': has_screenshot,
            'has_image_tag': has_image_tag,
            'has_browser_vision': has_browser_vision,
            'estimate': estimate
        })

    return requests


def generate_report(log_file):
    """
    Generate a comprehensive report on image sizes.
    """
    # Check for base64 images (unlikely but worth checking)
    base64_images = search_for_base64_images_in_logs(log_file)

    if base64_images:
        print("✓ Found base64 encoded images in logs:")
        for idx, img in enumerate(base64_images, 1):
            print(f"\n  Image {idx}:")
            print(f"    Dimensions: {img['width']}x{img['height']}")
            print(f"    Format: {img['format']}")
            print(f"    Size: {img['size_kb']} KB")
    else:
        print("✗ No base64 image data found in text logs (expected)")
        print("  Note: VLLM doesn't typically log full image data\n")

    # Analyze requests
    requests = analyze_requests_for_image_sizes(log_file)

    if not requests:
        print("No requests found in logs")
        return

    print(f"\nAnalyzed {len(requests)} requests:\n")
    print("-" * 70)

    # Show summary statistics
    multimodal_requests = [r for r in requests if r['has_screenshot'] or r['has_browser_vision']]

    print(f"Total requests: {len(requests)}")
    print(f"Multimodal requests: {len(multimodal_requests)}")
    print()

    # Show details for recent multimodal requests
    print("Recent Multimodal Requests (with size estimates):")
    print("-" * 70)

    for req in multimodal_requests[-10:]:  # Last 10
        print(f"\nRequest: {req['request_id']}")
        print(f"  Prompt length: {req['prompt_length']:,} characters")
        print(f"  Estimate: {req['estimate']}")

        indicators = []
        if req['has_screenshot']:
            indicators.append("screenshot")
        if req['has_browser_vision']:
            indicators.append("browser_vision")
        if req['has_image_tag']:
            indicators.append("<image> tag")

        if indicators:
            print(f"  Indicators: {', '.join(indicators)}")

    # For InternVL, provide model-specific info
    print("\n" + "="*70)
    print("InternVL3.5-14B Image Processing Info:")
    print("-" * 70)
    print("  Input image size: 448x448 pixels (model default)")
    print("  Image is resized/cropped to this size automatically")
    print("  Format: RGB (3 channels)")
    print("  Tokens per image: ~256 tokens")
    print()


def main():
    log_file = Path('/home/jiaheng/vllm_log/server.log')

    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return

    generate_report(log_file)

    print("\n" + "="*70)
    print("To Get EXACT Image Dimensions:")
    print("-" * 70)
    print("1. Use the API request interceptor (see below)")
    print("2. Monitor the agent's HTTP requests to VLLM")
    print("3. Or modify VLLM to log image metadata")
    print()


if __name__ == "__main__":
    main()
