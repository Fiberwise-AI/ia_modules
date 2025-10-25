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

# Add the IA modules to the path
current_dir = Path(__file__).parent
ia_modules_path = current_dir.parent
sys.path.insert(0, str(ia_modules_path))

# Also add the project root to make test modules importable
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.database.manager import DatabaseManager
from ia_modules.pipeline.graph_pipeline_runner import GraphPipelineRunner


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

    # Create minimal services registry
    services = ServiceRegistry()
    if db_manager:
        services.register('database', db_manager)

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
        print("Examples:")
        print("  python pipeline_runner.py tests/pipelines/simple_pipeline/pipeline.json")
        print("  python pipeline_runner.py tests/pipelines/simple_pipeline/pipeline.json --input '{\"topic\": \"AI\"}'")
        print("  python pipeline_runner.py --slug my-pipeline --db-url sqlite:///db.sqlite")
        return

    # Parse arguments
    args = sys.argv[1:]
    pipeline_file = None
    slug = None
    input_data = {}
    parameter_values = {}
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
    import os
    from datetime import datetime

    base_output_dir = output_file if output_file else "."
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"pipeline_run_{timestamp}")
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
        with open(final_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\nResults saved to directory: {run_dir}")
        print(f"  - Result file: {final_filename}")
        print(f"  - Log file: {log_file_path}")

        print("\nStep-by-step results:")
        if 'steps' in result:
            for step_name, step_result in result['steps'].items():
                print(f"  {step_name}:")
                print(json.dumps(step_result, indent=2, default=str))
                print()

        print("Final result:")
        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(f"ERROR: Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()