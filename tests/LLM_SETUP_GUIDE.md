# LLM Provider Setup Guide

## Quick Start with Google Gemini

### 1. Get Your Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key (keep it secure!)

### 2. Set Environment Variable

**Windows (Command Prompt):**
```bash
set GOOGLE_API_KEY=your_actual_api_key_here
```

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY="your_actual_api_key_here"
```

**Linux/Mac:**
```bash
export GOOGLE_API_KEY="your_actual_api_key_here"
```

### 3. Install Required Package

```bash
pip install google-generativeai
```

### 4. Test the Setup

```bash
cd ia_modules
python tests/llm_config_example.py
```

### 5. Run Agent Pipeline with LLM

```bash
python tests/pipeline_runner_with_llm.py tests/pipelines/agent_pipeline/pipeline.json --input '{"task": "sentiment_analysis", "text": "I love this product!"}'
```

## API Key Locations (in order of preference)

### Option 1: Environment Variables (RECOMMENDED)
```bash
export GOOGLE_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
```

### Option 2: .env File
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

### Option 3: Configuration File
```json
{
  "llm_providers": {
    "google": {
      "api_key": "your_key_here",
      "model": "gemini-2.5-flash"
    }
  }
}
```

## Supported Providers

| Provider | Environment Variable | Models | Notes |
|----------|---------------------|---------|-------|
| Google Gemini | `GOOGLE_API_KEY` | gemini-2.5-flash, gemini-2.5-pro | Fast and cost-effective |
| OpenAI | `OPENAI_API_KEY` | gpt-3.5-turbo, gpt-4 | Widely compatible |
| Anthropic | `ANTHROPIC_API_KEY` | claude-3-haiku, claude-3-sonnet | Great reasoning |
| Groq | `GROQ_API_KEY` | llama3-70b, mixtral-8x7b | Ultra-fast inference |
| Ollama | N/A (local) | llama3, mistral, codellama | Local/private |

## Usage Examples

### Basic LLM Pipeline Run
```bash
# Set your API key
export GOOGLE_API_KEY="your_key_here"

# Run with sentiment analysis
python tests/pipeline_runner_with_llm.py tests/pipelines/agent_pipeline/pipeline.json \
  --input '{"task": "sentiment_analysis", "text": "This is amazing!"}' \
  --output ./results
```

### Custom Configuration
```bash
# Run with specific task
python tests/pipeline_runner_with_llm.py tests/pipelines/agent_pipeline/pipeline.json \
  --input '{"task": "summarize", "text": "Long text to summarize..."}' \
  --output ./my_results
```

## Testing Your Setup

1. **Test LLM Service Only:**
   ```bash
   python tests/llm_config_example.py
   ```

2. **Test Agent Pipeline without LLM:**
   ```bash
   python tests/pipeline_runner.py tests/pipelines/agent_pipeline/pipeline.json \
     --input '{"task": "test", "text": "hello"}'
   ```

3. **Test Agent Pipeline with LLM:**
   ```bash
   python tests/pipeline_runner_with_llm.py tests/pipelines/agent_pipeline/pipeline.json \
     --input '{"task": "test", "text": "hello"}'
   ```

## What Changes with LLM Integration?

### Without LLM (fallback mode):
```json
{
  "agent_name": "step1",
  "simple_result": "Processed test with basic transformation",
  "metadata": {
    "llm_used": false,
    "processing_type": "simple_agent"
  }
}
```

### With LLM (enhanced mode):
```json
{
  "agent_name": "step1",
  "llm_response": {
    "analysis": "This appears to be a test input with minimal content...",
    "result": "Successfully analyzed the test data",
    "confidence": 0.85
  },
  "metadata": {
    "llm_used": true,
    "processing_type": "llm_agent"
  }
}
```

## Troubleshooting

### Common Issues:

1. **"No LLM providers configured"**
   - Check if environment variable is set: `echo $GOOGLE_API_KEY`
   - Make sure the API key is valid

2. **"Google Generative AI package not available"**
   - Install the package: `pip install google-generativeai`

3. **"API key invalid"**
   - Verify your API key at [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Make sure billing is enabled if required

4. **"Quota exceeded"**
   - Check your API usage limits
   - Wait for quota reset or upgrade plan

### Getting Help:
- Test with simple examples first
- Check the log files in the output directory
- Ensure your API key has the necessary permissions