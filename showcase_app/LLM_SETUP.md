# LLM Integration Setup

The Showcase App's agentic patterns **require** real LLM API keys to function. They will fail with a clear error message if not configured.

## Quick Start

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Add your API key** (choose one provider):
   
   ### Option A: OpenAI (Recommended)
   ```env
   OPENAI_API_KEY=sk-your-key-here
   OPENAI_MODEL=gpt-4o
   DEFAULT_LLM_PROVIDER=openai
   ```

   ### Option B: Anthropic
   ```env
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
   DEFAULT_LLM_PROVIDER=anthropic
   ```

   ### Option C: Google Gemini
   ```env
   GEMINI_API_KEY=your-gemini-key-here
   GEMINI_MODEL=gemini-2.0-flash-exp
   DEFAULT_LLM_PROVIDER=google
   ```

3. **Install provider libraries** (if needed):
   ```bash
   pip install openai anthropic google-generativeai
   ```

4. **Test the integration:**
   ```bash
   # Start the backend
   cd backend
   uvicorn main:app --reload
   
   # Test a pattern (in another terminal)
   curl -X POST http://localhost:8000/api/patterns/reflection \
     -H "Content-Type: application/json" \
     -d '{
       "initial_output": "This is a test output that could be improved.",
       "criteria": {
         "clarity": "Text should be clear and understandable",
         "accuracy": "Information should be precise"
       },
       "max_iterations": 2
     }'
   ```

## What Happens Without API Keys?

If you try to use a pattern without configuring an API key, you'll get a clear error:

```
RuntimeError: LLM service not configured. Please set API keys in .env file.
```

The patterns require LLM API access to function.

## Supported Providers

| Provider | Model | Best For |
|----------|-------|----------|
| OpenAI | gpt-4o | Overall best quality, well-tested |
| Anthropic | claude-3-5-sonnet-20241022 | Complex reasoning, long context |
| Google | gemini-2.0-flash-exp | Fast, cost-effective |

## Cost Management

See `.env.example` for cost tracking options:
- `MAX_COST_PER_REQUEST` - Limit per API call
- `DAILY_SPENDING_LIMIT` - Total daily budget
- `ENABLE_COST_TRACKING` - Track usage

## Next Steps

Once configured:
1. ✅ Test all 5 patterns (Reflection, Planning, Tool Use, RAG, Metacognition)
2. ✅ Explore the Multi-Agent workflows (12 templates available)
3. ✅ Build something cool with real AI intelligence!
