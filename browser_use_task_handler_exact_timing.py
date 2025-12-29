"""
Task-based logging handler with EXACT timing from Prometheus.

Queries Prometheus metrics before/after each request to get exact prefill and decode times.
"""

import logging
import re
import requests
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


class BrowserUseTaskHandler(logging.Handler):
    """
    Groups browser-use requests into task-based log files with EXACT timing metrics.

    Captures exact prefill and decode times by querying Prometheus metrics.
    """

    def __init__(self, log_directory="/home/jiaheng/vllm_log/task_logs",
                 idle_timeout_minutes=5,
                 metrics_url="http://localhost:11434/metrics",
                 level=logging.INFO):
        super().__init__(level)
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.metrics_url = metrics_url

        # Metrics snapshots directory
        self.metrics_snapshots_dir = Path("/home/jiaheng/vllm_log/metrics_snapshots")
        self.metrics_snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.idle_timeout = timedelta(minutes=idle_timeout_minutes)
        self._current_task_id = None
        self._current_file = None
        self._last_request_time = None
        self._request_count = 0

        # Track metrics snapshots
        self._prev_metrics = None
        self._pending_requests = {}
        self._last_metrics_error = None
        self._session = requests.Session()

        # Thread pool for async metrics fetching
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="metrics-fetcher")
        self._lock = threading.Lock()

        # Background metrics cache
        self._metrics_cache = None
        self._metrics_cache_time = None
        self._stop_polling = threading.Event()
        self._polling_thread = threading.Thread(target=self._metrics_polling_loop, daemon=True, name="metrics-poller")
        self._polling_thread.start()

    def _metrics_polling_loop(self):
        """Background thread that polls metrics when server is idle."""
        while not self._stop_polling.is_set():
            try:
                # Only poll when server is likely idle (no pending requests)
                if not self._pending_requests:
                    metrics = self._fetch_metrics_impl(timeout=5)
                    with self._lock:
                        self._metrics_cache = metrics
                        self._metrics_cache_time = datetime.now()
                        self._last_metrics_error = None
            except Exception as e:
                # Silently fail - this is a best-effort background poller
                pass

            # Wait before next poll (adjust based on your needs)
            self._stop_polling.wait(2.0)

    def _fetch_metrics_impl(self, timeout=10):
        """Internal implementation of metrics fetching."""
        response = self._session.get(self.metrics_url, timeout=timeout)
        response.raise_for_status()
        text = response.text

        # Parse key metrics
        metrics = {}

        # Prefill time
        prefill_sum = re.search(r'vllm:request_prefill_time_seconds_sum{[^}]*} (\S+)', text)
        prefill_count = re.search(r'vllm:request_prefill_time_seconds_count{[^}]*} (\S+)', text)
        if prefill_sum and prefill_count:
            metrics['prefill_sum'] = float(prefill_sum.group(1))
            metrics['prefill_count'] = float(prefill_count.group(1))

        # Decode time
        decode_sum = re.search(r'vllm:request_decode_time_seconds_sum{[^}]*} (\S+)', text)
        decode_count = re.search(r'vllm:request_decode_time_seconds_count{[^}]*} (\S+)', text)
        if decode_sum and decode_count:
            metrics['decode_sum'] = float(decode_sum.group(1))
            metrics['decode_count'] = float(decode_count.group(1))

        # TTFT (time to first token)
        ttft_sum = re.search(r'vllm:time_to_first_token_seconds_sum{[^}]*} (\S+)', text)
        ttft_count = re.search(r'vllm:time_to_first_token_seconds_count{[^}]*} (\S+)', text)
        if ttft_sum and ttft_count:
            metrics['ttft_sum'] = float(ttft_sum.group(1))
            metrics['ttft_count'] = float(ttft_count.group(1))

        return metrics

    def fetch_metrics_summary(self, retries=1, delay=0.5, timeout=3, use_cache=True):
        """Fetch current metrics - uses cached metrics from background poller or attempts quick fetch."""
        # First, try to use cached metrics from background poller (if allowed)
        if use_cache:
            with self._lock:
                if self._metrics_cache is not None:
                    cache_age = (datetime.now() - self._metrics_cache_time).total_seconds() if self._metrics_cache_time else 999
                    # Use cache if it's less than 5 seconds old
                    if cache_age < 5:
                        return self._metrics_cache.copy()

        # If no cache or stale, try one quick fetch with short timeout
        try:
            future = self._executor.submit(self._fetch_metrics_impl, timeout)
            metrics = future.result(timeout=timeout + 2)
            self._last_metrics_error = None
            return metrics
        except Exception as exc:
            self._last_metrics_error = exc
            # Fall back to cached metrics even if stale
            with self._lock:
                if self._metrics_cache is not None:
                    return self._metrics_cache.copy()
            return None

    def fetch_metrics_after_request(self, wait_time=0.5):
        """
        Fetch metrics after a request completes.
        Waits briefly for server to be idle, then forces a fresh fetch.
        """
        # Wait a bit for the server to finish processing and become idle
        time.sleep(wait_time)
        # Force a fresh fetch (don't use cache)
        return self.fetch_metrics_summary(use_cache=False, timeout=5)

    def save_metrics_snapshot(self, request_id, stage="after"):
        """
        Save full metrics snapshot to a file for later analysis.
        Uses subprocess to run curl command asynchronously.
        """
        import subprocess

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = self.metrics_snapshots_dir / f"{request_id}_{stage}_{timestamp}.txt"

        try:
            # Run curl in background and save to file
            # Using subprocess instead of requests to avoid blocking
            cmd = f"curl -s --max-time 10 {self.metrics_url} > {filename} 2>&1"
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return str(filename)
        except Exception as e:
            return None

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

        if self._current_task_id is None:
            return True

        step_info = self.extract_step_info(message)
        if step_info and step_info['step'] == 1:
            if self._request_count > 0:
                return True

        if self._last_request_time:
            time_since_last = now - self._last_request_time
            if time_since_last > self.idle_timeout:
                return True

        return False

    def start_new_task(self):
        """Start a new task with a new file."""
        if self._current_file:
            try:
                self._current_file.write(f"\n{'='*70}\n")
                self._current_file.write(f"Task ended: {datetime.now().isoformat()}\n")
                self._current_file.write(f"Total requests: {self._request_count}\n")
                self._current_file.write(f"{'='*70}\n")
                self._current_file.close()
            except:
                pass

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._current_task_id = f"task_{timestamp}"

        log_file = self.log_directory / f"{self._current_task_id}.log"
        self._current_file = open(log_file, 'w', encoding='utf-8')

        self._current_file.write(f"{'='*70}\n")
        self._current_file.write(f"Browser-Use Task Log (with Exact Timing)\n")
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

            if self.should_start_new_task(message):
                self.start_new_task()

            self._last_request_time = datetime.now()

            # Handle 'Received request' logs
            if 'Received request' in message:
                self._request_count += 1
                request_id = self.extract_request_id(message)

                # Save metrics snapshot BEFORE request to file
                snapshot_file_before = None
                if request_id:
                    snapshot_file_before = self.save_metrics_snapshot(request_id, stage="before")

                # Also get quick metrics for immediate logging
                metrics_before = self.fetch_metrics_summary()

                if request_id:
                    self._pending_requests[request_id] = {
                        'start_time': datetime.now(),
                        'request_num': self._request_count,
                        'metrics_before': metrics_before,
                        'snapshot_before': snapshot_file_before
                    }

                step_info = self.extract_step_info(message)
                step_num = step_info['step'] if step_info else '?'

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
                    total_latency = (end_time - start_time).total_seconds()

                    # Save metrics snapshot AFTER request to file
                    snapshot_file_after = self.save_metrics_snapshot(request_id, stage="after")

                    # Snapshot metrics AFTER request (wait for server to be idle, then force fresh fetch)
                    metrics_after = self.fetch_metrics_after_request(wait_time=0.5)

                    # Extract output tokens
                    output_match = re.search(r'output_token_ids: \[([^\]]+)\]', message)
                    output_tokens = 0
                    if output_match:
                        tokens_str = output_match.group(1)
                        output_tokens = len([t for t in tokens_str.split(',') if t.strip()])

                    finish_match = re.search(r'finish_reason: (\w+)', message)
                    finish_reason = finish_match.group(1) if finish_match else 'unknown'

                    # Write timing metrics
                    self._current_file.write(f"\n{'~'*70}\n")
                    self._current_file.write(f"⏱️  EXACT TIMING - Request #{req_data['request_num']}: {request_id}\n")
                    self._current_file.write(f"{'~'*70}\n")
                    self._current_file.write(f"Completed: {end_time.isoformat()}\n")
                    self._current_file.write(f"Total Latency: {total_latency:.3f}s\n")
                    self._current_file.write(f"Output Tokens: {output_tokens}\n")

                    # Log metrics snapshot files
                    snapshot_before = req_data.get('snapshot_before')
                    if snapshot_before:
                        self._current_file.write(f"Metrics Snapshot (before): {snapshot_before}\n")
                    if snapshot_file_after:
                        self._current_file.write(f"Metrics Snapshot (after): {snapshot_file_after}\n")

                    # Calculate EXACT prefill and decode from Prometheus delta
                    metrics_before = req_data.get('metrics_before')
                    if metrics_before and metrics_after:
                        prefill_delta = metrics_after.get('prefill_sum', 0) - metrics_before.get('prefill_sum', 0)
                        decode_delta = metrics_after.get('decode_sum', 0) - metrics_before.get('decode_sum', 0)
                        ttft_delta = metrics_after.get('ttft_sum', 0) - metrics_before.get('ttft_sum', 0)

                        self._current_file.write(f"\n📊 APPROXIMATE METRICS (from Prometheus delta):\n")

                        # Warn if metrics might be inaccurate
                        if len(self._pending_requests) > 1:
                            self._current_file.write(f"  ⚠️  WARNING: {len(self._pending_requests)} concurrent requests detected - metrics may include other requests\n")

                        # Warn if all metrics are zero (likely cache issue)
                        if prefill_delta == 0 and decode_delta == 0 and ttft_delta == 0:
                            self._current_file.write(f"  ⚠️  WARNING: All metrics are zero - likely using stale cache or no delta between snapshots\n")
                            self._current_file.write(f"     Note: Metrics are cumulative. For accurate per-request timing, ensure:\n")
                            self._current_file.write(f"           1. Requests are processed sequentially (not concurrently)\n")
                            self._current_file.write(f"           2. Sufficient idle time between requests for cache refresh\n")

                        self._current_file.write(f"  Prefill Time: {prefill_delta:.3f}s\n")
                        self._current_file.write(f"  Decode Time: {decode_delta:.3f}s\n")
                        self._current_file.write(f"  Time to First Token (TTFT): {ttft_delta:.3f}s\n")

                        if output_tokens > 0 and decode_delta > 0:
                            time_per_token = decode_delta / output_tokens
                            self._current_file.write(f"  Time per Token: {time_per_token:.3f}s\n")
                    else:
                        self._current_file.write(f"\n⚠️  Could not fetch exact metrics from Prometheus\n")
                        if self._last_metrics_error:
                            self._current_file.write(f"    Last error: {self._last_metrics_error}\n")

                    self._current_file.write(f"Finish Reason: {finish_reason}\n")
                    self._current_file.write(f"{'~'*70}\n\n")

                    del self._pending_requests[request_id]

            # Write the log message
            formatted_msg = self.format(record)
            self._current_file.write(formatted_msg + '\n')
            self._current_file.flush()

        except Exception as e:
            self.handleError(record)

    def close(self):
        """Close the current file and cleanup resources."""
        if self._current_file:
            try:
                self._current_file.write(f"\n{'='*70}\n")
                self._current_file.write(f"Task ended: {datetime.now().isoformat()}\n")
                self._current_file.write(f"Total requests: {self._request_count}\n")
                self._current_file.write(f"{'='*70}\n")
                self._current_file.close()
            except:
                pass

        # Stop background polling thread
        if hasattr(self, '_stop_polling'):
            self._stop_polling.set()
        if hasattr(self, '_polling_thread') and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=2)

        # Shutdown thread pool
        if self._executor:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except:
                pass

        if self._session:
            try:
                self._session.close()
            except:
                pass
        super().close()
