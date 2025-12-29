#!/usr/bin/env python3
"""
Simple HTTP Proxy to Monitor VLLM Image Sizes

This captures image dimensions from requests to VLLM without modifying VLLM itself.

Usage:
    1. Run: python simple_request_monitor.py
    2. Point your agent to http://localhost:11435 (instead of 11434)
    3. Check logs in /home/jiaheng/vllm_log/image_metadata/
"""

import json
import base64
import io
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import requests
from PIL import Image


class VLLMProxyHandler(BaseHTTPRequestHandler):
    """Proxy handler that logs image metadata and forwards to VLLM."""

    # VLLM server location
    VLLM_HOST = "localhost"
    VLLM_PORT = 11434

    # Log directory
    LOG_DIR = Path("/home/jiaheng/vllm_log/image_metadata")

    def __init__(self, *args, **kwargs):
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        super().__init__(*args, **kwargs)

    def extract_and_log_images(self, request_body):
        """Extract image metadata from request body."""
        try:
            data = json.loads(request_body)
        except:
            return None

        images = []
        request_id = None

        # Parse OpenAI chat completion format
        if 'messages' in data:
            for message in data.get('messages', []):
                content = message.get('content', [])

                if isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'image_url':
                            url = item.get('image_url', {}).get('url', '')

                            if url.startswith('data:image'):
                                try:
                                    # Extract base64 data
                                    header, base64_data = url.split(',', 1)
                                    image_bytes = base64.b64decode(base64_data)

                                    # Open image to get dimensions
                                    img = Image.open(io.BytesIO(image_bytes))

                                    images.append({
                                        'width': img.width,
                                        'height': img.height,
                                        'format': img.format,
                                        'mode': img.mode,
                                        'size_bytes': len(image_bytes),
                                        'size_kb': round(len(image_bytes) / 1024, 2)
                                    })
                                except Exception as e:
                                    print(f"  Error decoding image: {e}")

        # Log if images found
        if images:
            timestamp = datetime.now()
            log_file = self.LOG_DIR / f"request_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.txt"

            with open(log_file, 'w') as f:
                f.write(f"Timestamp: {timestamp.isoformat()}\n")
                f.write(f"Path: {self.path}\n")
                f.write(f"Total Images: {len(images)}\n\n")

                for idx, img in enumerate(images, 1):
                    f.write(f"Image {idx}:\n")
                    f.write(f"  Dimensions: {img['width']}x{img['height']} pixels\n")
                    f.write(f"  Format: {img['format']}\n")
                    f.write(f"  Color Mode: {img['mode']}\n")
                    f.write(f"  Size: {img['size_kb']} KB ({img['size_bytes']} bytes)\n")
                    f.write(f"  Aspect Ratio: {img['width']/img['height']:.2f}:1\n")
                    f.write("\n")

            # Also log to console
            print(f"\n✓ Intercepted request with {len(images)} image(s)")
            for idx, img in enumerate(images, 1):
                print(f"  Image {idx}: {img['width']}x{img['height']} {img['format']} ({img['size_kb']} KB)")
            print(f"  Logged to: {log_file.name}")

        return len(images) if images else None

    def do_POST(self):
        """Handle POST requests."""
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        request_body = self.rfile.read(content_length)

        # Extract and log images
        self.extract_and_log_images(request_body.decode('utf-8'))

        # Forward to VLLM
        try:
            vllm_url = f"http://{self.VLLM_HOST}:{self.VLLM_PORT}{self.path}"

            # Forward headers
            headers = {
                key: value for key, value in self.headers.items()
                if key.lower() not in ['host', 'content-length']
            }

            # Make request to VLLM
            response = requests.post(
                vllm_url,
                data=request_body,
                headers=headers,
                stream=True
            )

            # Send response back to client
            self.send_response(response.status_code)

            # Forward response headers
            for key, value in response.headers.items():
                if key.lower() not in ['transfer-encoding', 'content-encoding']:
                    self.send_header(key, value)
            self.end_headers()

            # Stream response body
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    self.wfile.write(chunk)

        except Exception as e:
            print(f"✗ Error forwarding request: {e}")
            self.send_error(500, f"Proxy error: {e}")

    def do_GET(self):
        """Handle GET requests."""
        try:
            vllm_url = f"http://{self.VLLM_HOST}:{self.VLLM_PORT}{self.path}"
            response = requests.get(vllm_url, headers=dict(self.headers))

            self.send_response(response.status_code)
            for key, value in response.headers.items():
                if key.lower() not in ['transfer-encoding', 'content-encoding']:
                    self.send_header(key, value)
            self.end_headers()

            self.wfile.write(response.content)

        except Exception as e:
            self.send_error(500, f"Proxy error: {e}")

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass  # Comment this out to see all HTTP requests


def main():
    """Start the proxy server."""
    proxy_port = 8000

    print("=" * 70)
    print("VLLM Image Size Monitor (Proxy Server)")
    print("=" * 70)
    print()
    print(f"Proxy listening on: 0.0.0.0:{proxy_port} (all interfaces)")
    print(f"Forwarding to VLLM: http://localhost:{VLLMProxyHandler.VLLM_PORT}")
    print(f"Logs directory: {VLLMProxyHandler.LOG_DIR}")
    print()
    print("Configure your agent to use:")
    print(f"  http://<server-ip>:{proxy_port}/v1/chat/completions")
    print(f"  Example: http://158.130.4.155:{proxy_port}/v1/chat/completions")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()

    server = HTTPServer(('0.0.0.0', proxy_port), VLLMProxyHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down proxy...")
        server.shutdown()
        print("✓ Proxy stopped")


if __name__ == "__main__":
    main()
