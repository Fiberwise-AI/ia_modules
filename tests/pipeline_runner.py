#!/usr/bin/env python3
"""
General Purpose Pipeline Test Runner

A simple runner for executing pipeline configurations from JSON files
or database with parameter support.
"""

import json
import sys
import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime
from pydantic import BaseModel
from enum import Enum

try:
    from pipeline.graph_runner import Pipeline
except ImportError:
    Pipeline = None

# Add the IA modules to the path
current_dir = Path(__file__).parent
ia_modules_path = current_dir.parent
sys.path.insert(0, str(ia_modules_path))

# Also add the project root to make test modules importable
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from ia_modules.pipeline.services import ServiceRegistry  # noqa: E402
from nexusql import DatabaseManager  # noqa: E402
from ia_modules.pipeline.graph_pipeline_runner import GraphPipelineRunner  # noqa: E402
from ia_modules.pipeline.llm_provider_service import LLMProviderService, LLMProvider  # noqa: E402

# Load environment variables
try:
    from dotenv import load_dotenv
    # Try to load .env from tests directory
    env_path = current_dir / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"Loaded environment variables from {env_path}")
    else:
        # Try .env.example as fallback
        env_example_path = current_dir / '.env.example'
        if env_example_path.exists():
            logging.info("Note: .env not found, using .env.example. Copy .env.example to .env and add your API keys.")
except ImportError:
    logging.warning("python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    logging.warning(f"Could not load .env file: {e}")


class PipelineLoader:
    """Handles loading pipelines from files or database"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager

    def load_from_file(self, pipeline_file: str) -> Dict[str, Any]:
        """Load pipeline from JSON file"""
        json_path = Path(pipeline_file)
        if not json_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")

        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_from_database(self, slug: str) -> Dict[str, Any]:
        """Load pipeline from database by slug"""
        if not self.db_manager:
            raise ValueError("Database manager required for database loading")

        result = self.db_manager.fetch_one(
            "SELECT pipeline_config FROM pipelines WHERE slug = :slug",
            {'slug': slug}
        )

        if not result:
            raise ValueError(f"Pipeline not found with slug: {slug}")

        return json.loads(result['pipeline_config'])


def resolve_parameters(config: Dict[str, Any], parameter_values: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve parameter templates in pipeline configuration"""

    def resolve_value(obj):
        if isinstance(obj, dict):
            return {k: resolve_value(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_value(item) for item in obj]
        elif isinstance(obj, str):
            # Replace {{ parameters.name }} with actual values
            def replace_param(match):
                param_path = match.group(1).strip()
                if param_path.startswith('parameters.'):
                    param_name = param_path[11:]  # Remove 'parameters.'
                    return str(parameter_values.get(param_name, match.group(0)))
                return match.group(0)

            return re.sub(r'\{\{\s*([^}]+)\s*\}\}', replace_param, obj)
        else:
            return obj

    return resolve_value(config)


def setup_llm_service() -> Optional[LLMProviderService]:
    """
    Create and configure LLMProviderService from environment variables.

    Looks for the following environment variables:
    - OPENAI_API_KEY: OpenAI API key
    - ANTHROPIC_API_KEY: Anthropic API key
    - GOOGLE_API_KEY: Google AI API key
    - OPENAI_MODEL: OpenAI model name (default: gpt-4o)
    - ANTHROPIC_MODEL: Anthropic model name (default: claude-sonnet-4-5-20250929)
    - GEMINI_MODEL: Google model name (default: gemini-2.5-flash)
    - OLLAMA_AVAILABLE: Set to 'true' if Ollama is available

    Returns:
        LLMProviderService with registered providers, or None if no keys found
    """
    llm_service = LLMProviderService()
    providers_registered = 0
    logger = logging.getLogger(__name__)

    # OpenAI
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o')
        llm_service.register_provider(
            name='openai',
            provider=LLMProvider.OPENAI,
            api_key=openai_key,
            model=openai_model,
            is_default=(providers_registered == 0)
        )
        providers_registered += 1
        logger.info(f"Registered OpenAI provider with model: {openai_model}")

    # Anthropic
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if anthropic_key:
        anthropic_model = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-5-20250929')
        llm_service.register_provider(
            name='anthropic',
            provider=LLMProvider.ANTHROPIC,
            api_key=anthropic_key,
            model=anthropic_model,
            is_default=(providers_registered == 0)
        )
        providers_registered += 1
        logger.info(f"Registered Anthropic provider with model: {anthropic_model}")

    # Google AI
    google_key = os.getenv('GOOGLE_API_KEY')
    if google_key:
        gemini_model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        llm_service.register_provider(
            name='google',
            provider=LLMProvider.GOOGLE,
            api_key=google_key,
            model=gemini_model,
            is_default=(providers_registered == 0)
        )
        providers_registered += 1
        logger.info(f"Registered Google provider with model: {gemini_model}")

    # Ollama (local, no API key needed)
    ollama_available = os.getenv('OLLAMA_AVAILABLE', 'false').lower() == 'true'
    if ollama_available:
        llm_service.register_provider(
            name='ollama',
            provider=LLMProvider.OLLAMA,
            model='llama2',  # Default model
            base_url='http://localhost:11434',
            is_default=(providers_registered == 0)
        )
        providers_registered += 1
        logger.info("Registered Ollama provider")

    if providers_registered == 0:
        logger.info("No LLM providers configured. Set API keys in .env file to use LLM steps.")
        return None

    logger.info(f"LLM Service initialized with {providers_registered} provider(s)")
    return llm_service


def run_pipeline_test(
    pipeline_file: str = None,
    slug: str = None,
    input_data: Dict[str, Any] = None,
    parameter_values: Dict[str, Any] = None,
    db_url: str = None
) -> Dict[str, Any]:
    """Run a pipeline test with minimal setup"""

    # Setup database if needed
    db_manager = None
    if slug or db_url:
        if not db_url:
            raise ValueError("Database URL required when using slug")

        db_manager = DatabaseManager(db_url)
        if not db_manager.connect():
            raise RuntimeError(f"Failed to connect to database: {db_url}")

    # Load pipeline configuration
    loader = PipelineLoader(db_manager)

    if pipeline_file:
        pipeline_config = loader.load_from_file(pipeline_file)
        # Add pipeline directory to Python path for imports
        pipeline_dir = Path(pipeline_file).parent
        if str(pipeline_dir) not in sys.path:
            sys.path.insert(0, str(pipeline_dir))
    elif slug:
        pipeline_config = loader.load_from_database(slug)
    else:
        raise ValueError("Either pipeline_file or slug must be provided")

    # Resolve parameters
    if parameter_values:
        pipeline_config = resolve_parameters(pipeline_config, parameter_values)

    # Ensure input data exists - provide default if none given
    if input_data is None:
        input_data = {
            "topic": "artificial intelligence",
            "data": [
                {"name": "example1", "value": 10},
                {"name": "example2", "value": 20}
            ]
        }

    # Create services registry
    services = ServiceRegistry()
    if db_manager:
        services.register('database', db_manager)

    # Set up LLM service if API keys are available
    llm_service = setup_llm_service()
    if llm_service:
        services.register('llm_provider', llm_service)

    # Create GraphPipelineRunner with services
    runner = GraphPipelineRunner(services)
    
    # Run pipeline using unified execution engine
    result = asyncio.run(runner.run_pipeline_from_json(
        pipeline_config=pipeline_config,
        input_data=input_data,
        use_enhanced_features=True
    ))

    # Cleanup
    if db_manager:
        db_manager.disconnect()

    return result


def run_with_new_schema(
    pipeline: Pipeline,
    pipeline_config: Dict[str, Any],
    input_data: Dict[str, Any],
    db_manager: Optional[DatabaseManager] = None
) -> Dict[str, Any]:
    """Run a pipeline using the new execution schema with ExecutionContext"""
    import uuid
    from ia_modules.pipeline.core import ExecutionContext

    # Create execution context
    execution_context = ExecutionContext(
        execution_id=str(uuid.uuid4()),
        pipeline_id=pipeline_config.get('name', 'unknown'),
        metadata={'source': 'test_runner'}
    )

    # Run the pipeline
    result = asyncio.run(pipeline.run(input_data, execution_context))

    return result


def main():
    """Main entry point for pipeline testing"""

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pipeline_runner.py <pipeline_file> [options]")
        print("  python pipeline_runner.py --slug <pipeline_slug> --db-url <url> [options]")
        print()
        print("Options:")
        print("  --input <json>        Input data as JSON string")
        print("  --output <path>       Output folder or file path for results")
        print("  --db-url <url>        Database URL (required for --slug)")
        print()
        print("LLM Service:")
        print("  The runner automatically loads LLM API keys from tests/.env file")
        print("  Copy tests/.env.example to tests/.env and add your API keys:")
        print("    OPENAI_API_KEY=sk-...")
        print("    ANTHROPIC_API_KEY=sk-ant-...")
        print("    GOOGLE_API_KEY=AIza...")
        print()
        print("Examples:")
        print("  python pipeline_runner.py tests/pipelines/simple_pipeline/pipeline.json")
        print("  python pipeline_runner.py tests/pipelines/simple_pipeline/pipeline.json --input '{\"topic\": \"AI\"}'")
        print("  python pipeline_runner.py tests/pipelines/guardrails_pipeline/pipeline.json")
        print("  python pipeline_runner.py --slug my-pipeline --db-url sqlite:///db.sqlite")
        return

    # Parse arguments
    args = sys.argv[1:]
    pipeline_file = None
    slug = None
    input_data = {}
    db_url = None
    output_file = None

    i = 0
    while i < len(args):
        if args[i] == "--slug":
            slug = args[i + 1]
            i += 2
        elif args[i] == "--db-url":
            db_url = args[i + 1]
            i += 2
        elif args[i] == "--input":
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

    # Validate arguments
    if not pipeline_file and not slug:
        print("ERROR: Either pipeline file or --slug must be provided")
        return

    if slug and not db_url:
        print("ERROR: --db-url required when using --slug")
        return

    # Set up timestamped output directory and logging
    # Determine base directory based on pipeline file location or current dir
    if pipeline_file:
        pipeline_dir = os.path.dirname(os.path.abspath(pipeline_file))
        base_output_dir = os.path.join(pipeline_dir, "runs")
    else:
        base_output_dir = os.path.join(".", "runs")

    # Allow user to override with --output
    if output_file:
        base_output_dir = output_file

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Configure logging to write to the run directory
    log_file_path = os.path.join(run_dir, "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ],
        force=True  # Override any existing logging configuration
    )

    # Run the pipeline
    try:
        result = run_pipeline_test(
            pipeline_file=pipeline_file,
            slug=slug,
            input_data=input_data if input_data else None,
            parameter_values=None,
            db_url=db_url
        )

        print("Pipeline execution completed successfully!")

        # Write result file to the already created run directory
        final_filename = os.path.join(run_dir, "pipeline_result.json")

        # Create a clean copy without circular references
        def clean_result(obj, seen=None):
            """Remove circular references from result for JSON serialization"""
            if seen is None:
                seen = set()

            obj_id = id(obj)
            if obj_id in seen:
                return "[Circular Reference]"

            # Handle Pydantic models
            if isinstance(obj, BaseModel):
                return clean_result(obj.model_dump(), seen)

            # Handle Enums
            if isinstance(obj, Enum):
                return obj.value

            if isinstance(obj, dict):
                seen.add(obj_id)
                cleaned = {}
                for k, v in obj.items():
                    # Skip internal objects that cause circular refs
                    if k in ['services', 'logger', '_logger', 'engine']:
                        continue
                    cleaned[k] = clean_result(v, seen.copy())
                return cleaned
            elif isinstance(obj, (list, tuple)):
                seen.add(obj_id)
                return [clean_result(item, seen.copy()) for item in obj]
            elif hasattr(obj, '__dict__') and not isinstance(obj, type):
                return f"<{type(obj).__name__} object>"
            else:
                return obj

        clean_result_data = clean_result(result)

        with open(final_filename, 'w', encoding='utf-8') as f:
            json.dump(clean_result_data, f, indent=2, default=str)

        print(f"\nResults saved to directory: {run_dir}")
        print(f"  - Result file: {final_filename}")
        print(f"  - Log file: {log_file_path}")

        print("\nStep-by-step results:")
        if 'steps' in result:
            steps = result['steps']
            if isinstance(steps, dict):
                for step_name, step_result in steps.items():
                    print(f"  {step_name}:")
                    print(json.dumps(clean_result(step_result), indent=2, default=str))
                    print()

        print("Final result:")
        print(json.dumps(clean_result_data, indent=2, default=str))

    except Exception as e:
        print(f"ERROR: Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()