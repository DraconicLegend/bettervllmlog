"""
Enhanced per-request logging handler that detects and logs multimodal indicators.

This extends the basic handler to log metadata about image processing.
"""
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from per_request_handler import PerRequestFileHandler


class EnhancedPerRequestFileHandler(PerRequestFileHandler):
    """
    Enhanced handler that logs additional metadata about multimodal requests.
    """

    def __init__(self, log_directory="/home/jiaheng/vllm_log/requests",
                 level=logging.INFO):
        super().__init__(log_directory, level)

        # Track metadata per request
        self._request_metadata = {}

    def analyze_for_images(self, message):
        """
        Analyze log message for indicators of image processing.

        Returns dict with indicators found.
        """
        indicators = {
            'has_screenshot': 'screenshot' in message.lower(),
            'has_browser_vision': 'browser_vision' in message.lower(),
            'has_image_tag': '<image>' in message,
            'has_image_url': 'image_url' in message,
            'prompt_length': len(message),
            'likely_multimodal': False
        }

        # Determine if likely multimodal
        if any([
            indicators['has_screenshot'],
            indicators['has_browser_vision'],
            indicators['has_image_tag'],
            indicators['has_image_url'],
            indicators['prompt_length'] > 20000
        ]):
            indicators['likely_multimodal'] = True

        return indicators

    def emit(self, record):
        """
        Emit a log record with enhanced metadata detection.
        """
        try:
            request_id = self.extract_request_id(record)
            message = record.getMessage()

            # Check if this is a "Received request" message
            if 'Received request' in message and request_id != 'general':
                # Analyze for image indicators
                indicators = self.analyze_for_images(message)

                # Store metadata
                self._request_metadata[request_id] = indicators

                # Log the metadata
                if indicators['likely_multimodal']:
                    metadata_msg = (
                        f"\n{'='*70}\n"
                        f"REQUEST METADATA: {request_id}\n"
                        f"{'='*70}\n"
                        f"Multimodal: YES (likely includes images)\n"
                        f"Indicators:\n"
                    )
                    if indicators['has_screenshot']:
                        metadata_msg += "  ✓ Screenshot mentioned\n"
                    if indicators['has_browser_vision']:
                        metadata_msg += "  ✓ Browser vision context\n"
                    if indicators['has_image_tag']:
                        metadata_msg += "  ✓ <image> tag present\n"
                    if indicators['has_image_url']:
                        metadata_msg += "  ✓ Image URL present\n"
                    if indicators['prompt_length'] > 20000:
                        metadata_msg += f"  ✓ Large prompt ({indicators['prompt_length']} chars)\n"
                    metadata_msg += f"{'='*70}\n"

                    # Create a fake record for metadata
                    metadata_record = logging.LogRecord(
                        name=record.name,
                        level=record.levelno,
                        pathname=record.pathname,
                        lineno=record.lineno,
                        msg=metadata_msg,
                        args=(),
                        exc_info=None
                    )
                    metadata_record.request_id = request_id

                    # Write metadata first
                    super().emit(metadata_record)

            # Then write the actual log message
            super().emit(record)

        except Exception as e:
            self.handleError(record)

    def close(self):
        """
        Close handler and write summary.
        """
        # Write summary of all requests
        summary_file = self.log_directory / "multimodal_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("VLLM Multimodal Request Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            multimodal_count = sum(1 for m in self._request_metadata.values() if m['likely_multimodal'])
            total_count = len(self._request_metadata)

            f.write(f"Total Requests: {total_count}\n")
            f.write(f"Multimodal Requests: {multimodal_count}\n")
            if total_count > 0:
                percentage = (multimodal_count / total_count) * 100
                f.write(f"Percentage: {percentage:.1f}%\n")
            f.write("\n")

            f.write("Request Details:\n")
            f.write("-" * 70 + "\n")
            for request_id, metadata in self._request_metadata.items():
                status = "MULTIMODAL" if metadata['likely_multimodal'] else "TEXT-ONLY"
                f.write(f"\n{request_id}: {status}\n")
                if metadata['likely_multimodal']:
                    f.write(f"  Prompt Length: {metadata['prompt_length']} chars\n")
                    if metadata['has_screenshot']:
                        f.write("  • Has screenshot reference\n")
                    if metadata['has_browser_vision']:
                        f.write("  • Has browser vision context\n")

        super().close()
