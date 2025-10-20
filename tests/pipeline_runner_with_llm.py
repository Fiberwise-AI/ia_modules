"""
Pipeline Runner with LLM Integration

This is an enhanced version of the pipeline runner that includes LLM provider service integration.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.pipeline_runner import run_pipeline_test, create_pipeline_from_json, run_with_new_schema
from ia_modules.pipeline.llm_provider_service import LLMProviderService, LLMProvider


async def setup_llm_service():
    """Setup and configure the LLM Provider Service"""
    llm_service = LLMProviderService()

    # Configure Google/Gemini if API key is available
    google_api_key = os.getenv('GEMINI_API_KEY')
    if google_api_key:
        llm_service.register_provider(
            name="google_default",
            provider=LLMProvider.GOOGLE,
            api_key=google_api_key,
            model="gemini-2.5-flash",
            temperature=0.7,
            max_tokens=1000,
            is_default=True
        )
        logging.info("Google/Gemini LLM provider configured")
    else:
        logging.warning("GEMINI_API_KEY not found in environment variables")

    # Configure OpenAI if API key is available
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        llm_service.register_provider(
            name="openai_backup",
            provider=LLMProvider.OPENAI,
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        logging.info("OpenAI LLM provider configured")

    # Configure Anthropic if API key is available
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if anthropic_api_key:
        llm_service.register_provider(
            name="claude",
            provider=LLMProvider.ANTHROPIC,
            api_key=anthropic_api_key,
            model="claude-3-haiku-20240307",
            temperature=0.7,
            max_tokens=1000
        )
        logging.info("Anthropic/Claude LLM provider configured")

    await llm_service.initialize()
    return llm_service


async def run_pipeline_with_llm(
    pipeline_file: str = None,
    slug: str = None,
    input_data: dict = None,
    parameter_values: dict = None,
    db_url: str = None,
    output_file: str = None
):
    """
    Run pipeline with LLM service integration

    This function sets up the LLM service and runs the pipeline with it registered
    in the service registry.
    """
    from ia_modules.pipeline.services import ServiceRegistry

    # Setup LLM service
    llm_service = await setup_llm_service()

    if not llm_service.providers:
        logging.warning("No LLM providers configured. Pipeline will run without LLM capabilities.")
        print("⚠️  No LLM providers configured. Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY environment variables.")

    # Create service registry and register LLM service
    service_registry = ServiceRegistry()
    service_registry.register('llm_provider', llm_service)

    # Set up timestamped output directory and logging
    base_output_dir = output_file if output_file else "."
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"pipeline_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Configure logging
    log_file_path = os.path.join(run_dir, "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ],
        force=True
    )

    try:
        # Load pipeline configuration
        if slug:
            # TODO: Implement database loading
            raise NotImplementedError("Database loading not implemented yet")
        else:
            if not pipeline_file or not os.path.exists(pipeline_file):
                raise FileNotFoundError(f"Pipeline file '{pipeline_file}' not found")

            with open(pipeline_file, 'r') as f:
                pipeline_config = json.load(f)

        # Create pipeline with service registry
        pipeline = create_pipeline_from_json(pipeline_config, service_registry)

        # Run the pipeline
        result = await run_with_new_schema(pipeline, pipeline_config, input_data, None)

        print("✅ Pipeline execution completed successfully!")

        # Write result file
        result_file = os.path.join(run_dir, "pipeline_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n📁 Results saved to directory: {run_dir}")
        print(f"   📄 Result file: {result_file}")
        print(f"   📋 Log file: {log_file_path}")

        # Show LLM usage summary
        if llm_service.providers:
            providers = llm_service.list_providers()
            print(f"\n🤖 LLM Providers used: {[p['name'] for p in providers]}")

        # Display results
        print("\n📊 Step-by-step results:")
        if 'steps' in result:
            for step_name, step_result in result['steps'].items():
                print(f"  {step_name}:")
                # Show whether LLM was used
                metadata = step_result.get('metadata', {})
                if metadata.get('llm_used'):
                    print(f"    🤖 LLM-powered: {metadata.get('processing_type', 'unknown')}")
                else:
                    print(f"    🔧 Simple processing: {metadata.get('processing_type', 'unknown')}")

                print(json.dumps(step_result, indent=2, default=str)[:300] + "..." if len(str(step_result)) > 300 else json.dumps(step_result, indent=2, default=str))
                print()

        return result

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        print(f"❌ Pipeline execution failed: {e}")
        raise
    finally:
        await llm_service.cleanup()


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pipeline_runner_with_llm.py <pipeline_file> [options]")
        print()
        print("Options:")
        print("  --input <json>        Input data as JSON string")
        print("  --output <path>       Output folder for results")
        print()
        print("Environment Variables:")
        print("  GEMINI_API_KEY        Google/Gemini API key")
        print("  OPENAI_API_KEY        OpenAI API key")
        print("  ANTHROPIC_API_KEY     Anthropic/Claude API key")
        print()
        print("Examples:")
        print("  export GOOGLE_API_KEY='your_key_here'")
        print("  python pipeline_runner_with_llm.py tests/pipelines/agent_pipeline/pipeline.json --input '{\"task\": \"sentiment_analysis\", \"text\": \"I love this!\"}'")
        return

    # Parse arguments
    args = sys.argv[1:]
    pipeline_file = None
    input_data = {}
    output_file = None

    i = 0
    while i < len(args):
        if args[i] == "--input":
            try:
                input_data = json.loads(args[i + 1])
            except json.JSONDecodeError:
                print("ERROR: Invalid input JSON format")
                return
            i += 2
        elif args[i] == "--output":
            output_file = args[i + 1]
            i += 2
        elif not args[i].startswith("--"):
            if pipeline_file is None:
                pipeline_file = args[i]
            i += 1
        else:
            print(f"ERROR: Unknown option: {args[i]}")
            return

    if not pipeline_file:
        print("ERROR: Pipeline file must be provided")
        return

    # Run the pipeline
    asyncio.run(run_pipeline_with_llm(
        pipeline_file=pipeline_file,
        input_data=input_data,
        output_file=output_file
    ))


if __name__ == "__main__":
    main()