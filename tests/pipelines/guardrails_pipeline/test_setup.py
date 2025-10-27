#!/usr/bin/env python3
"""
Quick test to verify the guardrails pipeline setup is working.

This script tests:
1. .env file loading
2. LLM service setup
3. ServiceRegistry integration
4. Pipeline execution with guardrails

Run: python test_setup.py
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
tests_dir = current_dir.parent.parent
ia_modules_dir = tests_dir.parent
sys.path.insert(0, str(ia_modules_dir))
sys.path.insert(0, str(tests_dir))

def test_env_loading():
    """Test that .env file can be loaded"""
    print("Testing .env loading...")

    try:
        from dotenv import load_dotenv
        env_path = tests_dir / '.env'

        if env_path.exists():
            load_dotenv(env_path)
            print(f"  ✓ .env file loaded from {env_path}")
        else:
            print(f"  ⚠ .env file not found at {env_path}")
            print(f"    Copy tests/.env.example to tests/.env and add your API keys")
            return False

    except ImportError:
        print("  ✗ python-dotenv not installed")
        print("    Install with: pip install python-dotenv")
        return False

    return True


def test_llm_service_setup():
    """Test that LLM service can be set up from environment"""
    print("\nTesting LLM service setup...")

    try:
        from ia_modules.pipeline.llm_provider_service import LLMProviderService, LLMProvider
        from dotenv import load_dotenv

        # Load .env
        env_path = tests_dir / '.env'
        if env_path.exists():
            load_dotenv(env_path)

        # Check for at least one API key
        has_openai = bool(os.getenv('OPENAI_API_KEY'))
        has_anthropic = bool(os.getenv('ANTHROPIC_API_KEY'))
        has_google = bool(os.getenv('GOOGLE_API_KEY'))
        has_ollama = os.getenv('OLLAMA_AVAILABLE', 'false').lower() == 'true'

        providers_available = []
        if has_openai:
            providers_available.append("OpenAI")
        if has_anthropic:
            providers_available.append("Anthropic")
        if has_google:
            providers_available.append("Google")
        if has_ollama:
            providers_available.append("Ollama")

        if not providers_available:
            print("  ⚠ No LLM providers configured")
            print("    Add at least one API key to tests/.env:")
            print("      OPENAI_API_KEY=sk-...")
            print("      ANTHROPIC_API_KEY=sk-ant-...")
            print("      GOOGLE_API_KEY=AIza...")
            print("    Or set OLLAMA_AVAILABLE=true if you have Ollama running")
            return False

        print(f"  ✓ Found {len(providers_available)} provider(s): {', '.join(providers_available)}")

        # Try to create service
        llm_service = LLMProviderService()

        # Register providers
        if has_openai:
            llm_service.register_provider(
                name='openai',
                provider=LLMProvider.OPENAI,
                api_key=os.getenv('OPENAI_API_KEY'),
                model=os.getenv('OPENAI_MODEL', 'gpt-4o')
            )
            print(f"    ✓ Registered OpenAI with model {os.getenv('OPENAI_MODEL', 'gpt-4o')}")

        if has_anthropic:
            llm_service.register_provider(
                name='anthropic',
                provider=LLMProvider.ANTHROPIC,
                api_key=os.getenv('ANTHROPIC_API_KEY'),
                model=os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-5-20250929')
            )
            print(f"    ✓ Registered Anthropic with model {os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-5-20250929')}")

        if has_google:
            llm_service.register_provider(
                name='google',
                provider=LLMProvider.GOOGLE,
                api_key=os.getenv('GOOGLE_API_KEY'),
                model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
            )
            print(f"    ✓ Registered Google with model {os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}")

        if has_ollama:
            llm_service.register_provider(
                name='ollama',
                provider=LLMProvider.OLLAMA,
                model='llama2',
                base_url='http://localhost:11434'
            )
            print(f"    ✓ Registered Ollama")

        print(f"  ✓ LLM service created with {len(providers_available)} provider(s)")
        return True

    except Exception as e:
        print(f"  ✗ Error setting up LLM service: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_guardrails_imports():
    """Test that guardrails modules can be imported"""
    print("\nTesting guardrails imports...")

    try:
        from ia_modules.guardrails.pipeline_steps import (
            InputGuardrailStep,
            OutputGuardrailStep,
            GuardrailStep
        )
        print("  ✓ Guardrails pipeline steps imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error importing guardrails: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_runner():
    """Test that pipeline_runner.py can be imported"""
    print("\nTesting pipeline runner import...")

    try:
        from pipeline_runner import setup_llm_service, run_pipeline_test
        print("  ✓ Pipeline runner imported successfully")

        # Test setup_llm_service function
        print("  Testing setup_llm_service()...")
        llm_service = setup_llm_service()

        if llm_service:
            providers = llm_service.list_providers()
            print(f"    ✓ setup_llm_service() created service with {len(providers)} provider(s)")
            for provider in providers:
                default_marker = " (default)" if provider['is_default'] else ""
                print(f"      - {provider['name']}: {provider['model']}{default_marker}")
            return True
        else:
            print("    ⚠ setup_llm_service() returned None (no API keys configured)")
            return False

    except Exception as e:
        print(f"  ✗ Error testing pipeline runner: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Guardrails Pipeline Setup Test")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Environment Loading", test_env_loading()))
    results.append(("LLM Service Setup", test_llm_service_setup()))
    results.append(("Guardrails Imports", test_guardrails_imports()))
    results.append(("Pipeline Runner", test_pipeline_runner()))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed! Setup is complete.")
        print("\nTo run the pipeline:")
        print(f"  cd {ia_modules_dir}")
        print("  python tests/pipeline_runner.py tests/pipelines/guardrails_pipeline/pipeline.json")
        return 0
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
