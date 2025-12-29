#!/usr/bin/env python3
"""
vLLM Metrics Logger

Queries the Prometheus metrics endpoint and logs prefill/decode times to files.
Can be run standalone or integrated with existing logging.
"""

import time
import requests
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import re


class VLLMMetricsLogger:
    """
    Logs vLLM timing metrics by querying Prometheus endpoint.

    Captures:
    - Time to first token (TTFT) - prefill time
    - Time per output token (TPOT) - decode time per token
    - Inter-token latency
    - E2E request latency
    - Prefill and decode phase times
    """

    def __init__(self,
                 metrics_url: str = "http://localhost:11434/metrics",
                 log_dir: str = "/home/jiaheng/vllm_log/metrics"):
        self.metrics_url = metrics_url
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Track previous values to compute deltas
        self.prev_metrics = {}

    def fetch_metrics(self) -> Optional[str]:
        """Fetch metrics from Prometheus endpoint."""
        try:
            response = requests.get(self.metrics_url, timeout=5)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            return None

    def parse_histogram_summary(self, metrics_text: str, metric_name: str) -> Dict:
        """
        Parse histogram metrics to get count and sum.

        Returns dict with 'count' and 'sum' (total time).
        """
        count_pattern = f'{metric_name}_count{{[^}}]*}} (\\S+)'
        sum_pattern = f'{metric_name}_sum{{[^}}]*}} (\\S+)'

        count_match = re.search(count_pattern, metrics_text)
        sum_match = re.search(sum_pattern, metrics_text)

        count = float(count_match.group(1)) if count_match else 0
        total = float(sum_match.group(1)) if sum_match else 0

        avg = total / count if count > 0 else 0

        return {
            'count': count,
            'sum': total,
            'average': avg
        }

    def extract_key_metrics(self, metrics_text: str) -> Dict:
        """Extract key timing metrics from Prometheus response."""
        metrics = {}

        # Time to first token (prefill time)
        metrics['ttft'] = self.parse_histogram_summary(
            metrics_text, 'vllm:time_to_first_token_seconds'
        )

        # Time per output token (decode time per token)
        metrics['tpot'] = self.parse_histogram_summary(
            metrics_text, 'vllm:time_per_output_token_seconds'
        )

        # Inter-token latency
        metrics['inter_token_latency'] = self.parse_histogram_summary(
            metrics_text, 'vllm:inter_token_latency_seconds'
        )

        # E2E request latency
        metrics['e2e_latency'] = self.parse_histogram_summary(
            metrics_text, 'vllm:e2e_request_latency_seconds'
        )

        # Prefill time per request
        metrics['prefill_time'] = self.parse_histogram_summary(
            metrics_text, 'vllm:request_prefill_time_seconds'
        )

        # Decode time per request
        metrics['decode_time'] = self.parse_histogram_summary(
            metrics_text, 'vllm:request_decode_time_seconds'
        )

        # Token counts
        prompt_tokens_match = re.search(r'vllm:prompt_tokens_total{[^}]*} (\S+)', metrics_text)
        gen_tokens_match = re.search(r'vllm:generation_tokens_total{[^}]*} (\S+)', metrics_text)

        metrics['tokens'] = {
            'prompt_total': float(prompt_tokens_match.group(1)) if prompt_tokens_match else 0,
            'generated_total': float(gen_tokens_match.group(1)) if gen_tokens_match else 0
        }

        # Request success count
        success_match = re.search(r'vllm:request_success_total{[^}]*finished_reason="stop"[^}]*} (\S+)', metrics_text)
        metrics['requests_completed'] = float(success_match.group(1)) if success_match else 0

        return metrics

    def compute_deltas(self, current: Dict, previous: Dict) -> Dict:
        """Compute delta metrics since last measurement."""
        if not previous:
            return None

        deltas = {}

        # Requests completed since last check
        new_requests = current['requests_completed'] - previous.get('requests_completed', 0)
        deltas['new_requests'] = new_requests

        if new_requests > 0:
            # Average metrics for new requests
            for key in ['ttft', 'tpot', 'prefill_time', 'decode_time', 'e2e_latency']:
                if key in current and key in previous:
                    delta_sum = current[key]['sum'] - previous[key]['sum']
                    delta_count = current[key]['count'] - previous[key]['count']
                    avg = delta_sum / delta_count if delta_count > 0 else 0
                    deltas[key] = {
                        'sum': delta_sum,
                        'count': delta_count,
                        'average': avg
                    }

            # Token deltas
            deltas['tokens'] = {
                'prompt': current['tokens']['prompt_total'] - previous['tokens']['prompt_total'],
                'generated': current['tokens']['generated_total'] - previous['tokens']['generated_total']
            }

        return deltas

    def format_metrics_report(self, metrics: Dict, deltas: Optional[Dict] = None) -> str:
        """Format metrics into human-readable report."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"vLLM Metrics Report - {datetime.now().isoformat()}")
        lines.append("=" * 70)
        lines.append("")

        # Overall statistics
        lines.append("OVERALL STATISTICS")
        lines.append("-" * 70)
        lines.append(f"Total Requests Completed: {int(metrics['requests_completed'])}")
        lines.append(f"Total Prompt Tokens: {int(metrics['tokens']['prompt_total'])}")
        lines.append(f"Total Generated Tokens: {int(metrics['tokens']['generated_total'])}")
        lines.append("")

        # Timing metrics
        lines.append("TIMING METRICS (All Requests)")
        lines.append("-" * 70)
        lines.append(f"Prefill Time (TTFT):")
        lines.append(f"  Average: {metrics['ttft']['average']:.3f}s")
        lines.append(f"  Total: {metrics['ttft']['sum']:.3f}s")
        lines.append(f"  Count: {int(metrics['ttft']['count'])}")
        lines.append("")

        lines.append(f"Decode Time per Request:")
        lines.append(f"  Average: {metrics['decode_time']['average']:.3f}s")
        lines.append(f"  Total: {metrics['decode_time']['sum']:.3f}s")
        lines.append(f"  Count: {int(metrics['decode_time']['count'])}")
        lines.append("")

        lines.append(f"Time per Output Token:")
        lines.append(f"  Average: {metrics['tpot']['average']:.3f}s")
        lines.append("")

        lines.append(f"End-to-End Latency:")
        lines.append(f"  Average: {metrics['e2e_latency']['average']:.3f}s")
        lines.append("")

        # Delta metrics (new requests since last check)
        if deltas and deltas.get('new_requests', 0) > 0:
            lines.append("=" * 70)
            lines.append(f"NEW REQUESTS SINCE LAST CHECK: {int(deltas['new_requests'])}")
            lines.append("-" * 70)
            lines.append(f"Prefill Time (TTFT): {deltas['ttft']['average']:.3f}s avg")
            lines.append(f"Decode Time: {deltas['decode_time']['average']:.3f}s avg")
            lines.append(f"E2E Latency: {deltas['e2e_latency']['average']:.3f}s avg")
            lines.append(f"Prompt Tokens: {int(deltas['tokens']['prompt'])}")
            lines.append(f"Generated Tokens: {int(deltas['tokens']['generated'])}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def log_current_metrics(self) -> None:
        """Fetch and log current metrics."""
        metrics_text = self.fetch_metrics()
        if not metrics_text:
            return

        current = self.extract_key_metrics(metrics_text)
        deltas = self.compute_deltas(current, self.prev_metrics)

        # Generate report
        report = self.format_metrics_report(current, deltas)

        # Print to console
        print(report)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.log_dir / f"metrics_{timestamp}.txt"
        report_file.write_text(report)

        # Save JSON for programmatic access
        json_file = self.log_dir / f"metrics_{timestamp}.json"
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'current': current,
            'deltas': deltas
        }
        json_file.write_text(json.dumps(json_data, indent=2))

        # Update previous metrics
        self.prev_metrics = current

        print(f"\nMetrics saved to: {report_file}")

    def monitor_continuous(self, interval_seconds: int = 60):
        """Continuously monitor and log metrics."""
        print(f"Starting continuous monitoring (interval: {interval_seconds}s)")
        print(f"Logs will be saved to: {self.log_dir}")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                self.log_current_metrics()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")


def main():
    """CLI interface for metrics logger."""
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Metrics Logger")
    parser.add_argument(
        '--url',
        default='http://localhost:11434/metrics',
        help='Prometheus metrics endpoint URL'
    )
    parser.add_argument(
        '--log-dir',
        default='/home/jiaheng/vllm_log/metrics',
        help='Directory to save metrics logs'
    )
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Continuously monitor and log metrics'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Monitoring interval in seconds (for continuous mode)'
    )

    args = parser.parse_args()

    logger = VLLMMetricsLogger(
        metrics_url=args.url,
        log_dir=args.log_dir
    )

    if args.continuous:
        logger.monitor_continuous(args.interval)
    else:
        logger.log_current_metrics()


if __name__ == '__main__':
    main()
