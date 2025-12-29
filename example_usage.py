"""
Example usage of the per-request logging handler with VLLM.

This script shows how to:
1. Load the custom logging configuration
2. Log messages with request_id context
3. Integrate with VLLM server
"""

import logging
import logging.config
import json
import sys
from pathlib import Path

# Add the vllm_log directory to Python path so we can import per_request_handler
sys.path.insert(0, '/home/jiaheng/vllm_log')


def setup_logging():
    """Load the logging configuration from vllm_logging.json"""
    config_path = Path('/home/jiaheng/vllm_log/vllm_logging.json')

    with open(config_path, 'r') as f:
        config = json.load(f)

    logging.config.dictConfig(config)


def log_request_example(request_id, prompt, response):
    """
    Example function showing how to log a request with its ID.

    Args:
        request_id: Unique identifier for the request
        prompt: The input prompt
        response: The generated response
    """
    logger = logging.getLogger('vllm')

    # Log with request_id in the message (the handler will extract it)
    logger.info(f"request_id={request_id} Started processing request")
    logger.info(f"request_id={request_id} Prompt: {prompt}")
    logger.info(f"request_id={request_id} Response: {response}")
    logger.info(f"request_id={request_id} Request completed")


def log_request_with_extra(request_id, prompt, response):
    """
    Alternative method: Pass request_id as an extra parameter.
    This requires a minor modification to use LoggerAdapter.
    """
    logger = logging.getLogger('vllm')

    # Create a LoggerAdapter that adds request_id to all log records
    adapter = logging.LoggerAdapter(logger, {'request_id': request_id})

    adapter.info("Started processing request")
    adapter.info(f"Prompt: {prompt}")
    adapter.info(f"Response: {response}")
    adapter.info("Request completed")


def main():
    """Main example demonstrating the logging system."""
    print("Setting up logging configuration...")
    setup_logging()

    print("Simulating multiple requests...\n")

    # Simulate request 1
    log_request_example(
        request_id="req-001",
        prompt="What is the capital of France?",
        response="The capital of France is Paris."
    )

    # Simulate request 2
    log_request_example(
        request_id="req-002",
        prompt="Explain quantum computing",
        response="Quantum computing uses quantum mechanics principles..."
    )

    # Simulate request 3 using the alternative method
    log_request_with_extra(
        request_id="req-003",
        prompt="Write a Python function",
        response="def example(): pass"
    )

    print("\nLogs have been written!")
    print(f"Check /home/jiaheng/vllm_log/requests/ for individual request files")
    print(f"Check /home/jiaheng/vllm_log/server.log for combined logs")


if __name__ == "__main__":
    main()
