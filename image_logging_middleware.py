"""
VLLM API Middleware to log image dimensions from requests.

This middleware intercepts API requests and logs image metadata
(dimensions, format, size) without logging full pixel data.

Usage:
    Add this to VLLM's API server to capture image information.
"""

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, Any


class ImageLoggingMiddleware:
    """
    Middleware that extracts and logs image metadata from VLLM API requests.
    """

    def __init__(self, log_directory="/home/jiaheng/vllm_log/image_metadata"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger('vllm.image_metadata')
        handler = logging.FileHandler(self.log_directory / 'image_sizes.log')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def extract_image_from_content(self, content_item: Dict[str, Any]) -> Optional[Dict]:
        """
        Extract image from a content item in OpenAI format.

        Args:
            content_item: Dict with 'type' and 'image_url' or 'image'

        Returns:
            Dict with image metadata or None
        """
        try:
            if content_item.get('type') == 'image_url':
                image_url = content_item.get('image_url', {})
                url = image_url.get('url', '')

                # Handle base64 data URI
                if url.startswith('data:image'):
                    # Extract base64 data
                    if ',' in url:
                        header, base64_data = url.split(',', 1)
                        image_format = header.split('/')[1].split(';')[0]

                        # Decode and get dimensions
                        image_bytes = base64.b64decode(base64_data)
                        img = Image.open(io.BytesIO(image_bytes))

                        return {
                            'width': img.width,
                            'height': img.height,
                            'format': img.format or image_format.upper(),
                            'mode': img.mode,
                            'size_bytes': len(image_bytes),
                            'size_kb': round(len(image_bytes) / 1024, 2)
                        }

            elif content_item.get('type') == 'image':
                # Direct image data
                image_data = content_item.get('image')
                if isinstance(image_data, str):
                    # Base64 string
                    image_bytes = base64.b64decode(image_data)
                    img = Image.open(io.BytesIO(image_bytes))

                    return {
                        'width': img.width,
                        'height': img.height,
                        'format': img.format,
                        'mode': img.mode,
                        'size_bytes': len(image_bytes),
                        'size_kb': round(len(image_bytes) / 1024, 2)
                    }

        except Exception as e:
            self.logger.error(f"Error extracting image: {e}")

        return None

    def log_request_images(self, request_id: str, request_data: Dict):
        """
        Extract and log image metadata from a request.

        Args:
            request_id: Unique request identifier
            request_data: The full request payload
        """
        images = []

        # Check OpenAI chat completion format
        if 'messages' in request_data:
            for message in request_data['messages']:
                content = message.get('content', [])

                # Content can be string or list
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            img_metadata = self.extract_image_from_content(item)
                            if img_metadata:
                                images.append(img_metadata)

        # Check VLLM native format
        elif 'multi_modal_data' in request_data:
            mm_data = request_data['multi_modal_data']
            if 'image' in mm_data:
                # Handle PIL Image or base64
                image_data = mm_data['image']
                # Try to extract metadata
                # This would need actual PIL Image object handling
                pass

        # Log findings
        if images:
            self.logger.info(f"Request {request_id}: {len(images)} image(s)")
            for idx, img in enumerate(images, 1):
                self.logger.info(
                    f"  Image {idx}: {img['width']}x{img['height']} "
                    f"{img['format']} {img['mode']} "
                    f"({img['size_kb']} KB)"
                )

            # Also write to per-request file
            request_file = self.log_directory / f"request_{request_id}_images.txt"
            with open(request_file, 'w') as f:
                f.write(f"Request ID: {request_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Total Images: {len(images)}\n\n")

                for idx, img in enumerate(images, 1):
                    f.write(f"Image {idx}:\n")
                    f.write(f"  Dimensions: {img['width']}x{img['height']} pixels\n")
                    f.write(f"  Format: {img['format']}\n")
                    f.write(f"  Color Mode: {img['mode']}\n")
                    f.write(f"  Size: {img['size_kb']} KB ({img['size_bytes']} bytes)\n")
                    f.write(f"  Aspect Ratio: {img['width']/img['height']:.2f}:1\n")
                    f.write("\n")

        return len(images)


# Integration instructions
INTEGRATION_CODE = """
# To integrate this into VLLM, add to vllm/entrypoints/openai/api_server.py:

from image_logging_middleware import ImageLoggingMiddleware

# Initialize middleware
image_logger = ImageLoggingMiddleware()

# In the create_chat_completion or similar endpoint, add:
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, ...):
    # Generate request_id
    request_id = f"chatcmpl-{random_uuid()}"

    # Log image metadata
    image_logger.log_request_images(request_id, request.dict())

    # Continue with normal processing...
    ...
"""


def create_example_log():
    """Create an example of what the logs would look like."""
    example_dir = Path('/home/jiaheng/vllm_log/image_metadata_example')
    example_dir.mkdir(parents=True, exist_ok=True)

    # Create example log file
    with open(example_dir / 'image_sizes.log', 'w') as f:
        f.write("2025-10-29 14:47:09 - Request chatcmpl-1ce8925: 1 image(s)\n")
        f.write("2025-10-29 14:47:09 -   Image 1: 1920x1080 PNG RGB (245.32 KB)\n")
        f.write("2025-10-29 14:48:15 - Request chatcmpl-374a908: 1 image(s)\n")
        f.write("2025-10-29 14:48:15 -   Image 1: 1280x720 JPEG RGB (156.78 KB)\n")

    # Create example per-request file
    with open(example_dir / 'request_chatcmpl-1ce8925_images.txt', 'w') as f:
        f.write("Request ID: chatcmpl-1ce8925\n")
        f.write("Timestamp: 2025-10-29T14:47:09.123456\n")
        f.write("Total Images: 1\n\n")
        f.write("Image 1:\n")
        f.write("  Dimensions: 1920x1080 pixels\n")
        f.write("  Format: PNG\n")
        f.write("  Color Mode: RGB\n")
        f.write("  Size: 245.32 KB (251208 bytes)\n")
        f.write("  Aspect Ratio: 1.78:1\n")

    print(f"✓ Created example logs in: {example_dir}")


if __name__ == "__main__":
    print("=" * 70)
    print("VLLM Image Logging Middleware")
    print("=" * 70)
    print()
    print("This middleware can be integrated into VLLM to log image dimensions.")
    print()
    print("Integration steps:")
    print("1. Copy this file to your VLLM installation")
    print("2. Modify vllm/entrypoints/openai/api_server.py")
    print("3. Add the middleware to request handling")
    print()
    print("Creating example output...")
    create_example_log()
    print()
    print("See INTEGRATION_CODE variable for detailed integration instructions.")
