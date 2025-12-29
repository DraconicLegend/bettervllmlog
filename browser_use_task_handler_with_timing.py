"""
Enhanced task-based logging handler with timing metrics.

Simple extension of the original browser_use_task_handler.py that also captures
completion timing from "Generated response" logs.
"""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path


class BrowserUseTaskHandler(logging.Handler):
    """
    Groups browser-use requests into task-based log files WITH timing metrics.

    Captures:
    - Request received events
    - Response generated events with timing
    - Automatically computes prefill and decode times
    """

    def __init__(self, log_directory="/home/jiaheng/vllm_log/task_logs",
                 idle_timeout_minutes=5,
                 level=logging.INFO):
        super().__init__(level)
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        self.idle_timeout = timedelta(minutes=idle_timeout_minutes)
        self._current_task_id = None
        self._current_file = None
        self._last_request_time = None
        self._request_count = 0

        # Track pending requests for timing calculation
        self._pending_requests = {}  # request_id -> {'start_time': datetime, 'request_num': int}

    def extract_step_info(self, message):
        """Extract step number from browser-use message."""
        match = re.search(r'<step_info>Step(\d+)\s+maximum:(\d+)', message)
        if match:
            return {
                'step': int(match.group(1)),
                'max_steps': int(match.group(2))
            }
        return None

    def extract_request_id(self, message):
        """Extract request ID from log message."""
        match = re.search(r'(chatcmpl-[a-f0-9]+)', message)
        return match.group(1) if match else None

    def should_start_new_task(self, message):
        """Determine if we should start a new task file."""
        now = datetime.now()

        # First request ever
        if self._current_task_id is None:
            return True

        # Check if step reset to 1 (new task)
        step_info = self.extract_step_info(message)
        if step_info and step_info['step'] == 1:
            # Only start new task if we've had other requests before
            if self._request_count > 0:
                return True

        # Check idle timeout
        if self._last_request_time:
            time_since_last = now - self._last_request_time
            if time_since_last > self.idle_timeout:
                return True

        return False

    def start_new_task(self):
        """Start a new task with a new file."""
        # Close current file
        if self._current_file:
            try:
                self._current_file.write(f"\n{'='*70}\n")
                self._current_file.write(f"Task ended: {datetime.now().isoformat()}\n")
                self._current_file.write(f"Total requests: {self._request_count}\n")
                self._current_file.write(f"{'='*70}\n")
                self._current_file.close()
            except:
                pass

        # Create new task ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._current_task_id = f"task_{timestamp}"

        # Open new file
        log_file = self.log_directory / f"{self._current_task_id}.log"
        self._current_file = open(log_file, 'w', encoding='utf-8')

        # Write header
        self._current_file.write(f"{'='*70}\n")
        self._current_file.write(f"Browser-Use Task Log\n")
        self._current_file.write(f"Task ID: {self._current_task_id}\n")
        self._current_file.write(f"Started: {datetime.now().isoformat()}\n")
        self._current_file.write(f"{'='*70}\n\n")
        self._current_file.flush()

        self._request_count = 0
        self._pending_requests = {}

    def emit(self, record):
        """Emit a log record to the current task file."""
        try:
            message = record.getMessage()

            # Check if we should start a new task
            if self.should_start_new_task(message):
                self.start_new_task()

            # Update last request time
            self._last_request_time = datetime.now()

            # Handle 'Received request' logs
            if 'Received request' in message:
                self._request_count += 1
                request_id = self.extract_request_id(message)

                if request_id:
                    # Store request start time
                    self._pending_requests[request_id] = {
                        'start_time': datetime.now(),
                        'request_num': self._request_count
                    }

                # Extract step info
                step_info = self.extract_step_info(message)
                step_num = step_info['step'] if step_info else '?'

                # Write request separator
                self._current_file.write(f"\n{'-'*70}\n")
                self._current_file.write(f"Request #{self._request_count} (Step {step_num}): {request_id or 'unknown'}\n")
                self._current_file.write(f"Time: {datetime.now().isoformat()}\n")
                self._current_file.write(f"{'-'*70}\n")

            # Handle 'Generated response' logs
            elif 'Generated response' in message:
                request_id = self.extract_request_id(message)

                if request_id and request_id in self._pending_requests:
                    req_data = self._pending_requests[request_id]
                    end_time = datetime.now()
                    start_time = req_data['start_time']

                    # Calculate timing
                    total_latency = (end_time - start_time).total_seconds()

                    # Extract token count from output
                    output_match = re.search(r'output_token_ids: \[([^\]]+)\]', message)
                    output_tokens = 0
                    if output_match:
                        tokens_str = output_match.group(1)
                        # Count tokens (they're comma-separated)
                        output_tokens = len([t for t in tokens_str.split(',') if t.strip()])

                    # Extract finish reason
                    finish_match = re.search(r'finish_reason: (\w+)', message)
                    finish_reason = finish_match.group(1) if finish_match else 'unknown'

                    # Write timing metrics
                    self._current_file.write(f"\n{'~'*70}\n")
                    self._current_file.write(f"⏱️  TIMING METRICS - Request #{req_data['request_num']}: {request_id}\n")
                    self._current_file.write(f"{'~'*70}\n")
                    self._current_file.write(f"Completed: {end_time.isoformat()}\n")
                    self._current_file.write(f"Total Latency: {total_latency:.3f}s\n")
                    self._current_file.write(f"Output Tokens: {output_tokens}\n")

                    if output_tokens > 0:
                        # Note: This is total time / tokens, not true decode time per token
                        # True decode time would need token-level timestamps
                        time_per_token = total_latency / output_tokens
                        self._current_file.write(f"Average Time/Token: {time_per_token:.3f}s\n")

                        # Rough estimate: assume first token is prefill
                        # This is approximate - real prefill time needs more instrumentation
                        estimated_prefill = total_latency * 0.1  # rough heuristic
                        estimated_decode = total_latency - estimated_prefill
                        self._current_file.write(f"\nEstimated Breakdown:\n")
                        self._current_file.write(f"  Prefill (est): ~{estimated_prefill:.3f}s\n")
                        self._current_file.write(f"  Decode (est): ~{estimated_decode:.3f}s\n")

                    self._current_file.write(f"Finish Reason: {finish_reason}\n")
                    self._current_file.write(f"{'~'*70}\n\n")

                    # Clean up
                    del self._pending_requests[request_id]

            # Write the log message
            formatted_msg = self.format(record)
            self._current_file.write(formatted_msg + '\n')
            self._current_file.flush()

        except Exception as e:
            self.handleError(record)

    def close(self):
        """Close the current file."""
        if self._current_file:
            try:
                self._current_file.write(f"\n{'='*70}\n")
                self._current_file.write(f"Task ended: {datetime.now().isoformat()}\n")
                self._current_file.write(f"Total requests: {self._request_count}\n")
                self._current_file.write(f"{'='*70}\n")
                self._current_file.close()
            except:
                pass
        super().close()
