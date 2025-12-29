"""
Custom logging handler for VLLM that stores each request in a separate file.
"""
import logging
import os
import re
from datetime import datetime
from pathlib import Path


class PerRequestFileHandler(logging.Handler):
    """
    A logging handler that creates a separate log file for each request.
    Files are organized by request_id extracted from log messages.
    """

    def __init__(self, log_directory="/home/jiaheng/vllm_log/requests",
                 level=logging.INFO):
        """
        Initialize the handler.

        Args:
            log_directory: Directory where individual request logs will be stored
            level: Logging level
        """
        super().__init__(level)
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        # Cache for open file handlers to avoid reopening files
        self._file_handlers = {}
        # Pattern matches both "request_id=xxx" and VLLM's "request chatcmpl-xxx" format
        self._request_pattern = re.compile(r'(?:request_id[=:\s]+|request\s+)(chatcmpl-[a-f0-9]+|[a-zA-Z0-9\-_]+)')

    def extract_request_id(self, record):
        """
        Extract request_id from the log record.

        Checks multiple sources:
        1. record.request_id attribute (if set)
        2. Pattern matching in the message
        3. Falls back to 'general' for non-request logs
        """
        # Check if request_id is set as an attribute
        if hasattr(record, 'request_id'):
            return str(record.request_id)

        # Try to extract from message
        message = record.getMessage()
        match = self._request_pattern.search(message)
        if match:
            return match.group(1)

        # Check in extra fields
        if hasattr(record, 'args') and isinstance(record.args, dict):
            if 'request_id' in record.args:
                return str(record.args['request_id'])

        # Default to 'general' for logs without request_id
        return 'general'

    def get_log_filename(self, request_id):
        """
        Generate filename for a given request_id.

        Format: request_{request_id}_{timestamp}.log
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if request_id == 'general':
            return self.log_directory / f"{request_id}_{timestamp}.log"
        else:
            # For specific requests, use just the request_id
            return self.log_directory / f"request_{request_id}.log"

    def emit(self, record):
        """
        Emit a log record to the appropriate file based on request_id.
        """
        try:
            request_id = self.extract_request_id(record)

            # Get or create file handler for this request
            if request_id not in self._file_handlers:
                log_file = self.get_log_filename(request_id)
                # Open in append mode
                file_handler = open(log_file, 'a', encoding='utf-8')
                self._file_handlers[request_id] = file_handler

            file_handler = self._file_handlers[request_id]

            # Format and write the message
            msg = self.format(record)
            file_handler.write(msg + '\n')
            file_handler.flush()

        except Exception as e:
            self.handleError(record)

    def close(self):
        """
        Close all open file handlers.
        """
        for file_handler in self._file_handlers.values():
            try:
                file_handler.close()
            except:
                pass
        self._file_handlers.clear()
        super().close()


class RequestContextFilter(logging.Filter):
    """
    A filter that can be used to inject request_id into log records.
    Use this with context variables or thread-local storage.
    """

    def __init__(self):
        super().__init__()
        self._current_request_id = None

    def set_request_id(self, request_id):
        """Set the current request ID for this thread/context."""
        self._current_request_id = request_id

    def clear_request_id(self):
        """Clear the current request ID."""
        self._current_request_id = None

    def filter(self, record):
        """Add request_id to the record if available."""
        if self._current_request_id:
            record.request_id = self._current_request_id
        return True
