"""
Task-based logging handler for browser-use agent.

Groups multiple VLLM requests from the same browser-use task into one file.
Uses time-based grouping to detect when a new task starts.
"""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path


class BrowserUseTaskHandler(logging.Handler):
    """
    Groups browser-use requests into task-based log files.

    Creates a new task file when:
    1. First request of the day
    2. More than 5 minutes since last request (new task started)
    3. Step number resets to 1 (explicit new task)
    """

    def __init__(self, log_directory="task_logs",
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

    def extract_step_info(self, message):
        """Extract step number from browser-use message."""
        match = re.search(r'<step_info>Step(\d+)\s+maximum:(\d+)', message)
        if match:
            return {
                'step': int(match.group(1)),
                'max_steps': int(match.group(2))
            }
        return None

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

    def emit(self, record):
        """Emit a log record to the current task file."""
        try:
            message = record.getMessage()

            # Check if we should start a new task
            if self.should_start_new_task(message):
                self.start_new_task()

            # Update last request time
            self._last_request_time = datetime.now()

            # Check if this is a new request
            if 'Received request' in message:
                self._request_count += 1

                # Extract request ID
                req_match = re.search(r'(chatcmpl-[a-f0-9]+)', message)
                request_id = req_match.group(1) if req_match else 'unknown'

                # Extract step info
                step_info = self.extract_step_info(message)
                step_num = step_info['step'] if step_info else '?'

                # Write request separator
                self._current_file.write(f"\n{'-'*70}\n")
                self._current_file.write(f"Request #{self._request_count} (Step {step_num}): {request_id}\n")
                self._current_file.write(f"Time: {datetime.now().isoformat()}\n")
                self._current_file.write(f"{'-'*70}\n")

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
