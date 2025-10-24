# LLM Monitoring - Quick Reference

## 🚀 Quick Start

### 1. Check Configuration Status
```bash
curl http://localhost:5555/api/patterns/llm/status
```

### 2. View Usage Statistics
```bash
curl http://localhost:5555/api/patterns/llm/stats
```

### 3. Environment Variables
```bash
# Rate Limits
MAX_REQUESTS_PER_MINUTE=60
MAX_TOKENS_PER_MINUTE=90000

# Cost Limits
MAX_COST_PER_REQUEST=0.50
DAILY_SPENDING_LIMIT=10.00
```

---

## 📊 Pricing Reference

| Provider | Model | Input/1M | Output/1M | Cheapest |
|----------|-------|----------|-----------|----------|
| OpenAI | GPT-4o | $2.50 | $10.00 | |
| OpenAI | GPT-3.5 Turbo | $0.50 | $1.50 | |
| Anthropic | Claude 3.5 Sonnet | $3.00 | $15.00 | |
| Anthropic | Claude 3 Haiku | $0.25 | $1.25 | |
| Google | **Gemini 2.5 Flash** | **$0.075** | **$0.30** | ✅ |
| Google | Gemini Pro | $0.50 | $1.50 | |

**💡 Tip**: Gemini Flash is 33x cheaper than GPT-4o for input tokens!

---

## 🔧 Common Tasks

### Enable Monitoring
Already enabled by default! All LLM calls are automatically monitored.

### Check if Rate Limited
```bash
# Returns HTTP 429 if limited
curl -w "%{http_code}" http://localhost:5555/api/patterns/reflection
```

### View Today's Spending
```bash
curl http://localhost:5555/api/patterns/llm/stats | jq '.daily_costs'
```

### Check Rate Limit Status
```bash
curl http://localhost:5555/api/patterns/llm/stats | jq '.rate_limits'
```

### Estimate Request Cost
```python
from backend.services.llm_monitoring_service import CostCalculator

cost = CostCalculator.calculate_cost(
    provider="openai",
    model="gpt-4o",
    input_tokens=1000,
    output_tokens=500
)
print(f"Estimated cost: ${cost:.4f}")
# Output: Estimated cost: $0.0075
```

---

## 🚨 Error Responses

### Rate Limit Exceeded
```json
{
  "detail": "Rate limit exceeded: requests_per_minute"
}
```
**HTTP Status**: 429  
**Headers**: `Retry-After: 5` (seconds)

### No API Key Configured
```json
{
  "detail": "LLM service not configured. Please set API keys in .env file."
}
```
**HTTP Status**: 500

---

## 📈 Monitoring Patterns

### Check Usage After Pattern Execution
```python
# Execute pattern
result = await pattern_service.reflection_example(...)

# Check stats
stats = monitoring_service.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Total cost: ${stats['total_cost']:.4f}")
print(f"Average cost: ${stats['average_cost']:.4f}")
```

### Monitor Daily Spending
```python
from datetime import datetime

today = datetime.now().date().isoformat()
daily_cost = monitoring_service.daily_costs.get(today, 0.0)

if daily_cost > 5.0:
    print(f"⚠️  High daily spending: ${daily_cost:.2f}")
```

### Get Recent Activity (Last Hour)
```python
stats = monitoring_service.get_stats()
recent = stats['recent_hour']

print(f"Last hour:")
print(f"  Requests: {recent['requests']}")
print(f"  Tokens: {recent['tokens']:,}")
print(f"  Cost: ${recent['cost']:.4f}")
```

---

## 🧪 Testing

### Run All Tests
```bash
cd showcase_app
pytest tests/test_llm_monitoring.py -v
pytest tests/test_pattern_monitoring_integration.py -v
```

### Run Specific Test
```bash
pytest tests/test_llm_monitoring.py::TestCostCalculator::test_openai_gpt4o_cost -v
```

### Test with Coverage
```bash
pytest tests/test_llm_monitoring.py --cov=backend/services/llm_monitoring_service
```

---

## 🎯 Best Practices

### 1. Set Appropriate Limits
```bash
# Development (generous limits)
MAX_REQUESTS_PER_MINUTE=60
MAX_TOKENS_PER_MINUTE=90000
MAX_COST_PER_REQUEST=1.00
DAILY_SPENDING_LIMIT=50.00

# Production (conservative)
MAX_REQUESTS_PER_MINUTE=30
MAX_TOKENS_PER_MINUTE=45000
MAX_COST_PER_REQUEST=0.50
DAILY_SPENDING_LIMIT=10.00
```

### 2. Choose Cheapest Provider
For development/testing, use **Gemini Flash** (33x cheaper than GPT-4o).

### 3. Monitor Daily Costs
Check stats endpoint regularly:
```bash
# Add to monitoring dashboard
curl http://localhost:5555/api/patterns/llm/stats | jq '.daily_costs'
```

### 4. Handle Rate Limits Gracefully
```python
from fastapi import HTTPException

try:
    result = await pattern_service.reflection_example(...)
except HTTPException as e:
    if e.status_code == 429:
        retry_after = e.headers.get("Retry-After", 60)
        print(f"Rate limited. Retry after {retry_after}s")
        time.sleep(int(retry_after))
        # Retry
```

### 5. Track Cost per Feature
```python
# Before feature execution
start_cost = monitoring_service.daily_costs.get(today, 0.0)

# Execute feature
result = await some_pattern(...)

# After
end_cost = monitoring_service.daily_costs.get(today, 0.0)
feature_cost = end_cost - start_cost
print(f"Feature cost: ${feature_cost:.4f}")
```

---

## 🐛 Troubleshooting

### Issue: No stats showing
**Cause**: No LLM calls made yet  
**Solution**: Execute a pattern first

### Issue: HTTP 429 errors
**Cause**: Rate limit exceeded  
**Solution**: Wait for `Retry-After` seconds or increase limits

### Issue: High costs
**Cause**: Large prompts, many iterations, or expensive model  
**Solution**:
- Use cheaper model (Gemini Flash)
- Reduce `max_tokens`
- Reduce `max_iterations`
- Set lower `MAX_COST_PER_REQUEST`

### Issue: "LLM service not configured"
**Cause**: No API keys in `.env`  
**Solution**: Add at least one provider's API key

---

## 📚 Code Examples

### Custom Monitoring
```python
from backend.services.llm_monitoring_service import LLMMonitoringService

# Create custom service
monitor = LLMMonitoringService()

# Check before making request
rate_check = monitor.check_rate_limits(estimated_tokens=2000)
if not rate_check["allowed"]:
    print(f"Rate limited: {rate_check['reason']}")
    print(f"Retry after: {rate_check['retry_after']}s")
    
# Track custom usage
monitor.track_usage(
    provider="openai",
    model="gpt-4o",
    input_tokens=1500,
    output_tokens=800,
    duration_seconds=2.3
)

# Get stats
stats = monitor.get_stats()
print(stats)
```

### Calculate Request Cost
```python
from backend.services.llm_monitoring_service import CostCalculator

# OpenAI GPT-4o
cost_openai = CostCalculator.calculate_cost(
    "openai", "gpt-4o", 
    input_tokens=1000, 
    output_tokens=500
)

# Anthropic Claude
cost_anthropic = CostCalculator.calculate_cost(
    "anthropic", "claude-3-5-sonnet-20241022",
    input_tokens=1000,
    output_tokens=500
)

# Google Gemini
cost_gemini = CostCalculator.calculate_cost(
    "google", "gemini-2.5-flash",
    input_tokens=1000,
    output_tokens=500
)

print(f"OpenAI:    ${cost_openai:.4f}")
print(f"Anthropic: ${cost_anthropic:.4f}")
print(f"Gemini:    ${cost_gemini:.4f}")

# Output:
# OpenAI:    $0.0075
# Anthropic: $0.0105
# Gemini:    $0.0002  ← Cheapest!
```

---

## 🔍 Key Files

| File | Purpose |
|------|---------|
| `backend/services/llm_monitoring_service.py` | Monitoring logic |
| `backend/services/pattern_service.py` | Pattern integration |
| `backend/api/patterns.py` | API endpoints |
| `tests/test_llm_monitoring.py` | Unit tests |
| `tests/test_pattern_monitoring_integration.py` | Integration tests |
| `.env.example` | Configuration template |
| `LLM_MONITORING.md` | Full documentation |
| `TEST_SUMMARY.md` | Test results |

---

## 📞 Support

**Issues**: Check `LLM_MONITORING.md` for detailed docs  
**Tests**: Run `pytest tests/test_llm_monitoring.py -v`  
**Logs**: Monitor console output for warnings

---

**Last Updated**: 2025-10-24  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
