"""
LLM Configuration Example

This file shows how to configure and use the LLM Provider Service with your API keys.
"""

import os
import asyncio
from ia_modules.pipeline.llm_provider_service import LLMProviderService, LLMProvider


async def setup_llm_service():
    """
    Example of how to configure the LLM Provider Service

    API Keys Configuration:
    You can set API keys in several ways:

    1. Environment Variables (RECOMMENDED):
       - GOOGLE_API_KEY=your_google_api_key
       - OPENAI_API_KEY=your_openai_api_key
       - ANTHROPIC_API_KEY=your_anthropic_api_key

    2. Direct Configuration (for testing):
       - Pass api_key parameter directly to register_provider()

    3. Configuration file:
       - Create a config file and load from there
    """

    # Initialize the LLM service
    llm_service = LLMProviderService()

    # Method 1: Using Environment Variables (RECOMMENDED)
    # Set these in your environment or .env file:
    # export GOOGLE_API_KEY="your_actual_google_api_key_here"
    # export OPENAI_API_KEY="your_actual_openai_api_key_here"

    # Register Google/Gemini as default provider
    if os.getenv('GOOGLE_API_KEY'):
        llm_service.register_provider(
            name="google_default",
            provider=LLMProvider.GOOGLE,
            api_key=os.getenv('GOOGLE_API_KEY'),
            model="gemini-2.5-flash",  # Fast and cost-effective
            temperature=0.7,
            max_tokens=1000,
            is_default=True
        )
        print("✅ Google/Gemini configured as default provider")
    else:
        print("❌ GOOGLE_API_KEY environment variable not set")

    # Register OpenAI as backup provider
    if os.getenv('OPENAI_API_KEY'):
        llm_service.register_provider(
            name="openai_backup",
            provider=LLMProvider.OPENAI,
            api_key=os.getenv('OPENAI_API_KEY'),
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        print("✅ OpenAI configured as backup provider")

    # Register Anthropic provider
    if os.getenv('ANTHROPIC_API_KEY'):
        llm_service.register_provider(
            name="claude",
            provider=LLMProvider.ANTHROPIC,
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            model="claude-3-haiku-20240307",
            temperature=0.7,
            max_tokens=1000
        )
        print("✅ Anthropic/Claude configured")

    # Method 2: Direct API Key Configuration (for testing only)
    # NEVER commit actual API keys to source code!
    """
    llm_service.register_provider(
        name="google_direct",
        provider=LLMProvider.GOOGLE,
        api_key="your_actual_google_api_key_here",  # NEVER do this in production!
        model="gemini-2.5-flash",
        is_default=True
    )
    """

    return llm_service


async def test_llm_service():
    """Test the LLM service"""
    llm_service = await setup_llm_service()

    if not llm_service.providers:
        print("❌ No LLM providers configured. Please set your API keys.")
        return

    try:
        # Test basic completion
        response = await llm_service.generate_completion(
            "What is artificial intelligence? Answer in one sentence."
        )
        print(f"✅ LLM Test Response: {response.content}")
        print(f"   Provider: {response.provider.value}, Model: {response.model}")
        print(f"   Usage: {response.usage}")

        # Test structured output
        structured = await llm_service.generate_structured_output(
            prompt="Analyze the sentiment of this text: 'I love this product!'",
            schema={
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reasoning": {"type": "string"}
                }
            }
        )
        print(f"✅ Structured Output: {structured}")

    except Exception as e:
        print(f"❌ LLM service test failed: {e}")

    # List all providers
    providers = llm_service.list_providers()
    print(f"\n📋 Configured Providers: {providers}")


if __name__ == "__main__":
    asyncio.run(test_llm_service())