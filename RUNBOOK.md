# Contract Analysis System - Operations Runbook

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Google Cloud SDK (for GCP deployment)
- Azure CLI (for Azure deployment)

### Local Development Setup
```bash
# 1. Clone repository
git clone <your-repo-url>
cd contract-analysis-system

# 2. Install dependencies
pip install -e .

# 3. Set up environment
cp .env.example .env
# Edit .env with your configuration

# 4. Start services
docker-compose up -d

# 5. Access applications
# API: http://localhost:8000
# UI: http://localhost:8501
# Metrics: http://localhost:9090
```

## 📊 Monitoring & Health Checks

### Health Endpoints
- **API Health**: `GET /health`
- **Detailed Health**: `GET /health/detailed`
- **Metrics**: `GET /metrics`

### Key Metrics to Monitor
- **API Response Time**: p95 < 1.5s
- **Error Rate**: < 1%
- **Memory Usage**: < 80%
- **CPU Usage**: < 70%

### Alerts Configuration
```yaml
# Prometheus alerts
groups:
  - name: contract-analysis
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. API Not Starting
```bash
# Check logs
docker-compose logs api

# Common fixes
- Check port 8000 is available
- Verify .env configuration
- Ensure all dependencies installed
```

#### 2. Model Loading Errors
```bash
# Check model files exist
ls -la app/artifacts/snapshot_20250909/models/

# Verify model paths in .env
MODEL_SNAPSHOT=snapshot_20250909
ARTIFACTS_DIR=app/artifacts
```

#### 3. RAG Index Issues
```bash
# Rebuild index
python app/jobs/run_jobs.py --job rebuild_index

# Check index status
python app/jobs/run_jobs.py --job health_check
```

#### 4. High Memory Usage
```bash
# Check memory usage
docker stats

# Optimize model cache
# Reduce MODEL_CACHE_SIZE in .env
MODEL_CACHE_SIZE=50
```

### Log Analysis
```bash
# View structured logs
docker-compose logs -f api | jq '.'

# Filter by level
docker-compose logs api | grep "ERROR"

# Search for specific patterns
docker-compose logs api | grep "contract_analysis"
```

## 🚀 Deployment

### Google Cloud Run
```bash
# Deploy API
./scripts/deploy-gcp-enhanced.sh --service api --environment production

# Deploy UI
./scripts/deploy-gcp-enhanced.sh --service ui --environment production
```

### Azure Container Apps
```bash
# Deploy API
./scripts/deploy-azure-enhanced.sh --service api --environment production

# Deploy UI
./scripts/deploy-azure-enhanced.sh --service ui --environment production
```

### Streamlit Share
```bash
# Deploy UI
./scripts/deploy-streamlit.sh --environment production
```

## 🔄 Maintenance

### Daily Tasks
- [ ] Check health endpoints
- [ ] Review error logs
- [ ] Monitor resource usage
- [ ] Check backup status

### Weekly Tasks
- [ ] Run model drift checks
- [ ] Review performance metrics
- [ ] Update dependencies
- [ ] Clean up old logs

### Monthly Tasks
- [ ] Security audit
- [ ] Performance optimization review
- [ ] Capacity planning
- [ ] Disaster recovery test

## 📞 Emergency Contacts

### Escalation Matrix
1. **Level 1**: Development Team
2. **Level 2**: DevOps Team
3. **Level 3**: Security Team
4. **Level 4**: Management

### Emergency Procedures
1. **Service Down**: Check health endpoints, restart services
2. **Security Incident**: Follow security runbook
3. **Data Loss**: Restore from backup
4. **Performance Degradation**: Scale resources, check logs

## 🔐 Security

### Access Control
- API requires authentication
- UI behind login screen
- Admin endpoints protected
- Rate limiting enabled

### Data Protection
- PII scrubbing enabled
- Encryption at rest
- Secure communication (HTTPS)
- Regular security scans

### Compliance
- GDPR compliance
- Data retention policies
- Audit logging
- Regular security reviews
