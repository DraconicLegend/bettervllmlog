#!/bin/bash
# Quick script to check if VLLM is processing images

echo "========================================================================"
echo "VLLM Image Processing Check"
echo "========================================================================"
echo ""

LOG_FILE="/home/jiaheng/vllm_log/server.log"

# Check 1: Model type
echo "1. Model Type Check:"
echo "------------------------------------------------------------------------"
if grep -q "InternVL" "$LOG_FILE"; then
    echo "✅ Vision-Language Model detected: InternVL"
else
    echo "❌ No vision model detected"
fi
echo ""

# Check 2: Vision-related mentions in startup
echo "2. Vision Components in Startup Logs:"
echo "------------------------------------------------------------------------"
vision_count=$(grep -i "vision" "$LOG_FILE" | wc -l)
echo "   'vision' mentioned: $vision_count times"

image_count=$(grep -i "image" "$LOG_FILE" | wc -l)
echo "   'image' mentioned: $image_count times"
echo ""

# Check 3: Recent requests with screenshots
echo "3. Recent Requests with Screenshot Indicators:"
echo "------------------------------------------------------------------------"
screenshot_requests=$(grep -i "screenshot" "$LOG_FILE" | grep -i "received request" | wc -l)
echo "   Requests mentioning 'screenshot': $screenshot_requests"

browser_vision_requests=$(grep -i "browser_vision" "$LOG_FILE" | grep -i "received request" | wc -l)
echo "   Requests with 'browser_vision': $browser_vision_requests"
echo ""

# Check 4: Show latest request with image indicator
echo "4. Latest Multimodal Request Example:"
echo "------------------------------------------------------------------------"
latest=$(grep -i "screenshot" "$LOG_FILE" | grep "Received request" | tail -1)
if [ -n "$latest" ]; then
    request_id=$(echo "$latest" | grep -oP 'chatcmpl-[a-f0-9]+')
    echo "   Request ID: $request_id"
    echo "   ✓ This request includes image-related content"
    echo ""
    echo "   Log excerpt:"
    echo "$latest" | head -c 300
    echo "..."
else
    echo "   No recent multimodal requests found"
fi
echo ""

# Check 5: Show per-request log file for latest
echo "5. Per-Request Log Files:"
echo "------------------------------------------------------------------------"
if [ -d "/home/jiaheng/vllm_log/requests" ]; then
    file_count=$(ls -1 /home/jiaheng/vllm_log/requests/request_chatcmpl-*.log 2>/dev/null | wc -l)
    echo "   Total request log files: $file_count"

    if [ $file_count -gt 0 ]; then
        echo ""
        echo "   Latest 3 files:"
        ls -lt /home/jiaheng/vllm_log/requests/request_chatcmpl-*.log 2>/dev/null | head -3 | awk '{print "     " $9 " (" $5 " bytes)"}'
    fi
else
    echo "   No per-request directory found"
fi
echo ""

# Final verdict
echo "========================================================================"
echo "VERDICT:"
echo "------------------------------------------------------------------------"

if [ $screenshot_requests -gt 0 ] || [ $browser_vision_requests -gt 0 ]; then
    echo "✅ CONFIRMED: VLLM is processing images/screenshots"
    echo ""
    echo "   Evidence:"
    [ $screenshot_requests -gt 0 ] && echo "   • $screenshot_requests requests with screenshots"
    [ $browser_vision_requests -gt 0 ] && echo "   • $browser_vision_requests requests with browser vision"

    if grep -q "InternVL" "$LOG_FILE"; then
        echo "   • InternVL vision-language model loaded"
    fi
else
    echo "⚠️  UNCERTAIN: No clear image indicators found in recent logs"
    echo ""
    echo "   This could mean:"
    echo "   • Server recently started with no multimodal requests yet"
    echo "   • Images are being processed but not logged in text format"
fi

echo ""
echo "========================================================================"
