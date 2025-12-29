# Optional: Full Prometheus Setup

If you want fancy dashboards and historical metrics, you can set up Prometheus server.

## Quick Setup

### 1. Install Prometheus

```bash
# Download
cd /tmp
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvf prometheus-2.45.0.linux-amd64.tar.gz
cd prometheus-2.45.0.linux-amd64
```

### 2. Configure Prometheus

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:11434']
    metrics_path: '/metrics'
```

### 3. Run Prometheus

```bash
./prometheus --config.file=prometheus.yml
```

Access at: http://localhost:9090

### 4. Query Examples

In Prometheus UI, you can run queries like:

```promql
# Average prefill time over last 5 minutes
rate(vllm:request_prefill_time_seconds_sum[5m]) / rate(vllm:request_prefill_time_seconds_count[5m])

# P95 decode time
histogram_quantile(0.95, vllm:request_decode_time_seconds_bucket)

# Requests per second
rate(vllm:request_success_total[1m])
```

## But You Don't Need This!

**For your use case (logging per-request timing), you DON'T need Prometheus server.**

The enhanced task handler already queries the `/metrics` endpoint directly via HTTP, which is:
- ✅ Simpler
- ✅ No extra services to run
- ✅ Gets exact per-request timing in your logs

Only set up Prometheus if you want:
- Historical metrics storage
- Alerting
- Fancy Grafana dashboards
- Multi-server monitoring

For just logging timing in your task logs, direct HTTP queries (what we do) are perfect!
