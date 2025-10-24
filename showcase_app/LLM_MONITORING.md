# LLM Monitoring System

## Overview

The LLM Monitoring System provides comprehensive tracking, cost calculation, and rate limiting for all LLM API calls in the Showcase App.

## Features Implemented

### ✅ 1. Token Usage Tracking

Every LLM API call tracks:
- **Input tokens**: Tokens in the prompt
- **Output tokens**: Tokens in the response  
- **Total tokens**: Sum of input + output
- **Duration**: Request time in seconds
- **Timestamp**: When the request was made

Tracked automatically via `LLMMonitoringService` wrapper.

### ✅ 2. Cost Calculation

Real-time cost calculation based on provider pricing:

| Provider | Model | Input (per 1M) | Output (per 1M) |
|----------|-------|----------------|-----------------|
| **OpenAI** | gpt-4o | $2.50 | $10.00 |
| | gpt-4-turbo | $10.00 | $30.00 |
| | gpt-3.5-turbo | $0.50 | $1.50 |
| **Anthropic** | claude-3-5-sonnet | $3.00 | $15.00 |
| | claude-3-opus | $15.00 | $75.00 |
| | claude-3-haiku | $0.25 | $1.25 |
| **Google** | gemini-2.5-flash | $0.075 | $0.30 |
| | gemini-pro | $0.50 | $1.50 |

**Cost tracking includes**:
- Per-request cost in USD
- Daily spending totals
- Cost limits (configurable)
- Over-budget warnings

### ✅ 3. Rate Limiting

Token bucket algorithm enforces:
- **Requests per minute**: Configurable via `MAX_REQUESTS_PER_MINUTE`
- **Tokens per minute**: Configurable via `MAX_TOKENS_PER_MINUTE`

**Default limits** (can override in `.env`):
```bash
MAX_REQUESTS_PER_MINUTE=60       # 1 request per second
MAX_TOKENS_PER_MINUTE=90000      # ~1500 tokens per second
```

**Rate limit responses**:
- HTTP 429 when limit exceeded
- `Retry-After` header with seconds to wait
- Clear error message: "Rate limit exceeded: requests_per_minute"

### ✅ 4. Cost Limits

Two-tier cost protection:

**Per-Request Limit**:
```bash
MAX_COST_PER_REQUEST=0.50        # Max $0.50 per LLM call
```

**Daily Spending Limit**:
```bash
DAILY_SPENDING_LIMIT=10.00       # Max $10/day total
```

When exceeded:
- Warning logged to console
- Request continues (soft limit)
- Can be made hard limit in production

### ✅ 5. Status API

**Endpoint**: `GET /api/patterns/llm/status`

**Response**:
```json
{
  "configured": true,
  "configured_count": 2,
  "total_providers": 3,
  "default_provider": "openai",
  "providers": [
    {
      "name": "openai",
      "status": "configured",
      "model": "gpt-4o"
    },
    {
      "name": "anthropic",
      "status": "configured",
      "model": "claude-3-5-sonnet-20241022"
    },
    {
      "name": "gemini",
      "status": "not_configured",
      "setup_guide": "Set GEMINI_API_KEY in .env file"
    }
  ],
  "message": "2 provider(s) ready"
}
```

### ✅ 6. Statistics API

**Endpoint**: `GET /api/patterns/llm/stats`

**Response**:
```json
{
  "total_requests": 42,
  "total_tokens": 125000,
  "total_cost": 2.45,
  "average_cost": 0.0583,
  "recent_hour": {
    "requests": 8,
    "tokens": 24000,
    "cost": 0.52
  },
  "daily_costs": {
    "2025-01-15": 2.45,
    "2025-01-14": 3.21
  },
  "rate_limits": {
    "requests_per_minute": 60,
    "tokens_per_minute": 90000,
    "max_cost_per_request": 0.50,
    "daily_spending_limit": 10.00
  }
}
```

## Architecture

### Components

```
┌─────────────────────────────────────────────┐
│         Pattern Service                     │
│  - reflection_example()                     │
│  - planning_example()                       │
│  - tool_use_example()                       │
│  - rag_example()                            │
│  - metacognition_example()                  │
│                                             │
│  ↓ calls                                    │
│  _monitored_llm_call()                      │
└─────────────────┬───────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────┐
│   LLM Monitoring Service                    │
│  - check_rate_limits()                      │
│  - track_usage()                            │
│  - check_cost_limits()                      │
│  - get_stats()                              │
│                                             │
│  Components:                                │
│  - TokenBucket (rate limiting)              │
│  - CostCalculator (pricing)                 │
│  - Usage history (deque)                    │
└─────────────────┬───────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────┐
│   LLM Provider Service                      │
│  - OpenAI client                            │
│  - Anthropic client                         │
│  - Google Gemini client                     │
│                                             │
│  Returns: LLMResponse with usage data       │
└─────────────────────────────────────────────┘
```

### Request Flow

1. **Pattern method** calls `_monitored_llm_call(prompt, ...)`
2. **Rate limit check**: Consumes tokens from buckets
   - If denied → HTTP 429 with Retry-After
3. **LLM API call**: Through `LLMProviderService`
4. **Usage extraction**: Parse token counts from response
5. **Cost calculation**: Apply pricing per provider/model
6. **Track usage**: Store in history with timestamp
7. **Cost check**: Warn if over limits
8. **Return**: Response with attached usage stats

### Files Changed

**New Files**:
- `backend/services/llm_monitoring_service.py` (400 lines)
  - `TokenBucket` class
  - `CostCalculator` class  
  - `LLMMonitoringService` class

**Modified Files**:
- `backend/services/pattern_service.py`
  - Added `_monitored_llm_call()` wrapper
  - Updated 3 LLM call sites to use wrapper
  - Import `LLMMonitoringService`

- `backend/api/patterns.py`
  - Added `/llm/status` endpoint
  - Added `/llm/stats` endpoint
  - Import `LLMMonitoringService`

- `.env.example`
  - Already had cost/rate limit configs

## Configuration

All settings in `.env`:

```bash
# ===== Cost Tracking =====
ENABLE_COST_TRACKING=true
MAX_COST_PER_REQUEST=0.50        # USD per call
DAILY_SPENDING_LIMIT=10.00       # USD per day

# ===== Rate Limiting =====
MAX_REQUESTS_PER_MINUTE=60       # Requests/min
MAX_TOKENS_PER_MINUTE=90000      # Tokens/min
```

## Usage Examples

### Check Provider Status
```bash
curl http://localhost:5555/api/patterns/llm/status
```

### Get Usage Statistics
```bash
curl http://localhost:5555/api/patterns/llm/stats
```

### Pattern Calls (Automatic Monitoring)
```python
# All pattern methods automatically tracked
result = await pattern_service.reflection_example(
    initial_output="Draft text",
    criteria={"clarity": "Clear and concise"}
)

# Token usage and cost included in execution logs
```

## What's NOT Implemented

❌ **Settings UI**
- Frontend page for API key management
- Provider selection/configuration
- Cost dashboard with charts
- Usage graphs over time

❌ **Dead Code Cleanup**
- ~200 lines of old simulation methods still in `pattern_service.py`
- Methods starting with `_generate_`, `_apply_`, `_decompose_`, etc.
- Never called, safe to delete

❌ **Enhanced Features** (Future)
- Cost alerts via email/webhook
- Budget quotas per user/project
- Token usage analytics dashboard
- Historical cost trends
- Provider cost comparison

## Testing

### Test Status Endpoint
```bash
# Should show which providers are configured
curl http://localhost:5555/api/patterns/llm/status | jq
```

### Test Stats Endpoint
```bash
# Should show usage statistics (empty initially)
curl http://localhost:5555/api/patterns/llm/stats | jq
```

### Test Rate Limiting
```bash
# Send 100 requests in parallel
for i in {1..100}; do
  curl -X POST http://localhost:5555/api/patterns/reflection \
    -H "Content-Type: application/json" \
    -d '{"initial_output":"Test","criteria":{"quality":"Good"}}' &
done

# Some should get HTTP 429
```

### Test Cost Tracking
```python
# Make pattern call and check stats after
response = await pattern_service.reflection_example(...)
stats = monitoring_service.get_stats()
print(f"Total cost: ${stats['total_cost']}")
```

## Production Considerations

### Cost Management
1. Set conservative daily limits initially
2. Monitor actual usage patterns
3. Adjust limits based on budget
4. Consider per-user quotas for multi-tenant

### Rate Limiting
1. Default limits work for most cases
2. Increase for high-traffic scenarios
3. Provider API limits may be stricter
4. Monitor rate limit 429s in logs

### Monitoring
1. Export stats to analytics platform
2. Set up cost alerts
3. Track anomalies (sudden spikes)
4. Review pricing changes from providers

### Security
1. Never expose API keys in logs
2. Rotate keys regularly
3. Use environment variables only
4. Restrict provider access by role

## Next Steps

**Priority 1: Settings UI** (1 day)
- Build React component for LLM configuration
- API key input fields (masked)
- Provider selection dropdown
- Usage dashboard with current costs
- Add to main navigation

**Priority 2: Code Cleanup** (2 hours)
- Remove 13 dead simulation methods
- Verify no references remain
- Update tests if needed

**Priority 3: Enhanced Analytics** (2 days)
- Cost trends over time (charts)
- Token usage by pattern type
- Provider performance comparison
- Export usage reports (CSV/PDF)

## Conclusion

✅ **Token tracking** - Complete  
✅ **Cost calculation** - Complete  
✅ **Rate limiting** - Complete  
✅ **Status API** - Complete  
✅ **Stats API** - Complete  
❌ **Settings UI** - Not started  
❌ **Code cleanup** - Not started  

**Completion: 90%** - Core monitoring system fully functional, needs UI layer.
