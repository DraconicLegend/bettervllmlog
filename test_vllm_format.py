"""
Test the per-request handler with VLLM's actual log format
"""
import logging
import logging.config
import json
import sys

sys.path.insert(0, '/home/jiaheng/vllm_log')


def test_vllm_format():
    """Test with VLLM's actual log message format"""
    # Load configuration
    with open('/home/jiaheng/vllm_log/vllm_logging.json', 'r') as f:
        config = json.load(f)
    logging.config.dictConfig(config)

    logger = logging.getLogger('vllm.entrypoints.logger')

    # Simulate VLLM log messages
    request_id1 = "chatcmpl-0ea05b11d83946dda7a0655bb10d38aa"
    request_id2 = "chatcmpl-c4bf8bee2f63475fb19f0ba21f36edb3"

    # Test request 1
    logger.info(f"Received request {request_id1}: prompt: 'Hello World'")
    logger.info(f"Added request {request_id1}.")
    logger.info(f"Processing request {request_id1}")
    logger.info(f"Request {request_id1} completed successfully")

    # Test request 2
    logger.info(f"Received request {request_id2}: prompt: 'Test prompt'")
    logger.info(f"Added request {request_id2}.")
    logger.info(f"Aborted request(s) {request_id2}.")

    print("\n✓ Test complete! Check /home/jiaheng/vllm_log/requests/ for files:")
    print(f"  - request_{request_id1}.log")
    print(f"  - request_{request_id2}.log")


if __name__ == "__main__":
    test_vllm_format()
