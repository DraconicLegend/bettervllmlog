"""
Script to analyze VLLM logs and detect if images are being processed.

This checks for indirect indicators that images were included in requests.
"""

import re
from pathlib import Path


def analyze_log_for_image_processing(log_file):
    """
    Analyze a VLLM log file for signs of image processing.

    Returns a dict with analysis results.
    """
    results = {
        'total_requests': 0,
        'likely_has_images': 0,
        'indicators': [],
        'requests_analyzed': []
    }

    with open(log_file, 'r') as f:
        content = f.read()

    # Find all request IDs
    request_pattern = re.compile(r'Received request (chatcmpl-[a-f0-9]+):')
    requests = request_pattern.finditer(content)

    for match in requests:
        request_id = match.group(1)
        results['total_requests'] += 1

        # Extract the full log entry for this request
        start_pos = match.start()
        # Find end of prompt (look for next line or end of structured data)
        end_match = re.search(r', params: SamplingParams', content[start_pos:start_pos+50000])
        if end_match:
            request_log = content[start_pos:start_pos+end_match.start()]
        else:
            request_log = content[start_pos:start_pos+10000]

        # Indicators that images might be present
        indicators = []
        has_image = False

        # 1. Check for image-related terms in prompt
        if 'screenshot' in request_log.lower():
            indicators.append('mentions_screenshot')
            has_image = True

        if 'browser_vision' in request_log.lower():
            indicators.append('has_browser_vision')
            has_image = True

        if '<image>' in request_log:
            indicators.append('has_image_tag')
            has_image = True

        if 'image_url' in request_log:
            indicators.append('has_image_url')
            has_image = True

        # 2. Check prompt length (image requests tend to have very long prompts)
        prompt_length = len(request_log)
        if prompt_length > 20000:  # Very long prompts suggest complex multimodal input
            indicators.append(f'very_long_prompt_{prompt_length}_chars')
            has_image = True

        # 3. Check for vision-related instructions
        if 'analyze' in request_log.lower() and ('image' in request_log.lower() or 'visual' in request_log.lower()):
            indicators.append('vision_analysis_task')
            has_image = True

        if has_image:
            results['likely_has_images'] += 1
            results['requests_analyzed'].append({
                'request_id': request_id,
                'indicators': indicators
            })

    return results


def check_vllm_startup_for_vision_model(log_file):
    """
    Check if VLLM loaded a vision model at startup.
    """
    vision_indicators = []

    with open(log_file, 'r') as f:
        content = f.read()

    # Check for vision model components
    if 'InternVL' in content:
        vision_indicators.append('InternVL model detected')

    if 'vision' in content.lower():
        # Count vision-related mentions
        vision_count = content.lower().count('vision')
        vision_indicators.append(f'Vision mentioned {vision_count} times')

    if 'image' in content.lower():
        image_count = content.lower().count('image')
        vision_indicators.append(f'Image mentioned {image_count} times')

    # Look for vision model loading
    if re.search(r'vision.*model|vision.*encoder|vit|clip', content, re.IGNORECASE):
        vision_indicators.append('Vision encoder/model referenced')

    return vision_indicators


def analyze_request_params(log_file):
    """
    Look at request parameters to infer multimodal data.
    """
    with open(log_file, 'r') as f:
        content = f.read()

    # Look for specific params
    findings = []

    if 'prompt_embeds' in content:
        findings.append('Found prompt_embeds parameter')

    if 'multi_modal_data' in content:
        findings.append('Found multi_modal_data parameter')

    if 'pixel_values' in content:
        findings.append('Found pixel_values parameter')

    # Check for very long token counts (images increase token count)
    token_pattern = re.compile(r'max_tokens[=:\s]+(\d+)')
    tokens = token_pattern.findall(content)
    if tokens:
        max_tokens = max(int(t) for t in tokens)
        if max_tokens > 2048:
            findings.append(f'High max_tokens={max_tokens} suggests multimodal input')

    return findings


def main():
    """Run all analyses on the VLLM logs."""
    log_file = Path('/home/jiaheng/vllm_log/server.log')

    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return

    print("=" * 70)
    print("VLLM Image Processing Detection Analysis")
    print("=" * 70)
    print()

    # Analysis 1: Startup logs
    print("📋 1. Checking startup logs for vision model...")
    print("-" * 70)
    vision_indicators = check_vllm_startup_for_vision_model(log_file)
    if vision_indicators:
        print("✓ Vision model indicators found:")
        for indicator in vision_indicators:
            print(f"  • {indicator}")
    else:
        print("✗ No vision model indicators found")
    print()

    # Analysis 2: Request parameters
    print("📋 2. Checking request parameters...")
    print("-" * 70)
    param_findings = analyze_request_params(log_file)
    if param_findings:
        print("✓ Multimodal parameter indicators:")
        for finding in param_findings:
            print(f"  • {finding}")
    else:
        print("✗ No explicit multimodal parameters found in logs")
    print()

    # Analysis 3: Individual requests
    print("📋 3. Analyzing individual requests...")
    print("-" * 70)
    results = analyze_log_for_image_processing(log_file)

    print(f"Total requests analyzed: {results['total_requests']}")
    print(f"Requests likely with images: {results['likely_has_images']}")

    if results['likely_has_images'] > 0:
        percentage = (results['likely_has_images'] / results['total_requests']) * 100
        print(f"Percentage with images: {percentage:.1f}%")
        print()
        print("Sample requests with image indicators:")
        for req in results['requests_analyzed'][:5]:  # Show first 5
            print(f"\n  Request: {req['request_id']}")
            print(f"  Indicators: {', '.join(req['indicators'])}")
    print()

    # Final verdict
    print("=" * 70)
    print("🔍 VERDICT:")
    print("-" * 70)

    confidence_score = 0
    reasons = []

    if vision_indicators:
        confidence_score += 30
        reasons.append("Vision model detected in startup logs")

    if results['likely_has_images'] > 0:
        confidence_score += 50
        reasons.append(f"{results['likely_has_images']} requests show image indicators")

    if param_findings:
        confidence_score += 20
        reasons.append("Multimodal parameters detected")

    if confidence_score >= 70:
        print("✅ HIGH CONFIDENCE: VLLM is processing images")
    elif confidence_score >= 40:
        print("⚠️  MODERATE CONFIDENCE: Likely processing images")
    else:
        print("❌ LOW CONFIDENCE: May be text-only")

    print(f"\nConfidence Score: {confidence_score}/100")
    print("\nReasons:")
    for reason in reasons:
        print(f"  • {reason}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
