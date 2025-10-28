"""Run agentic RAG with execution hooks for monitoring."""
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add paths
current_dir = Path(__file__).parent
tests_dir = current_dir.parent.parent
ia_modules_dir = tests_dir.parent
sys.path.insert(0, str(ia_modules_dir))
sys.path.insert(0, str(tests_dir))

from dotenv import load_dotenv
from ia_modules.agents.orchestrator import AgentOrchestrator
from ia_modules.agents.state import StateManager
from ia_modules.pipeline.llm_provider_service import LLMProviderService, LLMProvider
from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.guardrails.pipeline_steps import InputGuardrailStep, OutputGuardrailStep

# Import agents
sys.path.insert(0, str(current_dir))
from agents.query_analyzer_agent import QueryAnalyzerAgent
from agents.retriever_agent import RetrieverAgent
from agents.answer_generator_agent import AnswerGeneratorAgent


class AgentRAGRunner:
    """Runner for agentic RAG with execution hooks."""

    def __init__(self):
        self.execution_log = []
        self.agent_stats = {}

    async def run(self, query: str, thread_id: str = None):
        """Run agentic RAG pipeline."""
        if thread_id is None:
            thread_id = f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Load environment
        env_path = tests_dir / '.env'
        if env_path.exists():
            load_dotenv(env_path)

        # Setup LLM service
        llm_service = self._setup_llm_service()

        # Create services registry
        services = ServiceRegistry()
        if llm_service:
            services.register('llm_provider', llm_service)

        # Create state manager
        state = StateManager(thread_id=thread_id)

        # Create orchestrator
        orchestrator = AgentOrchestrator(state)

        # Create agents
        query_analyzer = QueryAnalyzerAgent(
            "query_analyzer",
            {"use_llm": True}
        )
        query_analyzer.services = services

        retriever = RetrieverAgent(
            "retriever",
            {
                "docs_dir": "../rag_with_guardrails_pipeline/sample_docs",
                "top_k": 3
            }
        )
        retriever.services = services

        answer_generator = AnswerGeneratorAgent(
            "answer_generator",
            {
                "temperature": 0.7,
                "max_tokens": 500,
                "max_context_length": 3000
            }
        )
        answer_generator.services = services

        # Register agents
        orchestrator.add_agent("query_analyzer", query_analyzer)
        orchestrator.add_agent("retriever", retriever)
        orchestrator.add_agent("answer_generator", answer_generator)

        # Define routing
        orchestrator.add_transition("query_analyzer", "retriever")
        orchestrator.add_transition("retriever", "answer_generator")

        # Register execution hooks
        orchestrator.add_hook("agent_start", self._on_agent_start)
        orchestrator.add_hook("agent_complete", self._on_agent_complete)
        orchestrator.add_hook("agent_error", self._on_agent_error)

        print(f"\n=== Agentic RAG Pipeline ===")
        print(f"Thread ID: {thread_id}")
        print(f"Query: {query}\n")

        # Apply input guardrails
        print("Applying input guardrails...")
        input_step = InputGuardrailStep("input_guard", {
            "content_field": "query",
            "fail_on_block": True
        })
        input_step.services = services

        guarded_input = await input_step.run({"query": query})
        print(f"Input guardrails: {guarded_input['input_guardrails_result']['action']}\n")

        # Run agent workflow
        print("Starting agent workflow...\n")
        result = await orchestrator.run("query_analyzer", {"query": query})

        # Apply output guardrails
        print("\nApplying output guardrails...")
        output_step = OutputGuardrailStep("output_guard", {
            "content_field": "answer",
            "fail_on_block": True,
            "max_length": 1000
        })
        output_step.services = services

        guarded_output = await output_step.run(result)
        print(f"Output guardrails: {guarded_output['output_guardrails_result']['action']}\n")

        # Print results
        self._print_results(result)

        # Save results
        self._save_results(thread_id, query, result)

        return result

    async def _on_agent_start(self, agent_id: str, input_data: Dict[str, Any]):
        """Hook called when agent starts."""
        self.execution_log.append({
            "event": "start",
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "input_keys": list(input_data.keys())
        })
        print(f"[AGENT START] {agent_id}")

    async def _on_agent_complete(self, agent_id: str, output_data: Dict[str, Any], duration: float):
        """Hook called when agent completes."""
        self.execution_log.append({
            "event": "complete",
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "output_keys": list(output_data.keys())
        })

        # Track stats
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = {"executions": 0, "total_time": 0}
        self.agent_stats[agent_id]["executions"] += 1
        self.agent_stats[agent_id]["total_time"] += duration

        print(f"[AGENT COMPLETE] {agent_id} - {duration:.3f}s")

    async def _on_agent_error(self, agent_id: str, error: Exception):
        """Hook called when agent errors."""
        self.execution_log.append({
            "event": "error",
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "error": str(error)
        })
        print(f"[AGENT ERROR] {agent_id}: {error}")

    def _setup_llm_service(self) -> LLMProviderService:
        """Setup LLM service from environment."""
        import os

        llm_service = LLMProviderService()
        providers_registered = 0

        if os.getenv('OPENAI_API_KEY'):
            llm_service.register_provider(
                name='openai',
                provider=LLMProvider.OPENAI,
                api_key=os.getenv('OPENAI_API_KEY'),
                model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
                is_default=True
            )
            providers_registered += 1
            print(f"Registered OpenAI provider")

        if os.getenv('ANTHROPIC_API_KEY'):
            llm_service.register_provider(
                name='anthropic',
                provider=LLMProvider.ANTHROPIC,
                api_key=os.getenv('ANTHROPIC_API_KEY'),
                model=os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-5-20250929')
            )
            providers_registered += 1
            print(f"Registered Anthropic provider")

        if providers_registered == 0:
            print("WARNING: No LLM providers configured")
            return None

        return llm_service

    def _print_results(self, result: Dict[str, Any]):
        """Print formatted results."""
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)

        print(f"\nQuery Analysis:")
        if "query_analysis" in result:
            analysis = result["query_analysis"]
            print(f"  Intent: {analysis.get('intent')}")
            print(f"  Keywords: {', '.join(analysis.get('keywords', []))}")
            print(f"  Analysis Type: {analysis.get('analysis_type')}")

        print(f"\nRetrieval:")
        print(f"  Retrieved: {result.get('num_retrieved', 0)} documents")
        print(f"  Total Available: {result.get('total_docs', 0)} documents")

        print(f"\nAnswer:")
        print(f"  {result.get('answer', 'No answer generated')}")

        if "llm_metadata" in result:
            meta = result["llm_metadata"]
            print(f"\nLLM Metadata:")
            print(f"  Provider: {meta.get('provider')}")
            print(f"  Model: {meta.get('model')}")
            if "usage" in meta:
                print(f"  Tokens: {meta['usage'].get('total_tokens', 'N/A')}")

        print(f"\nExecution Statistics:")
        for agent_id, stats in self.agent_stats.items():
            avg_time = stats["total_time"] / stats["executions"]
            print(f"  {agent_id}: {stats['executions']} exec, avg {avg_time:.3f}s")

        print()

    def _save_results(self, thread_id: str, query: str, result: Dict[str, Any]):
        """Save results to file."""
        output_dir = current_dir / "runs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full result
        result_file = output_dir / "result.json"
        with open(result_file, 'w') as f:
            json.dump({
                "thread_id": thread_id,
                "query": query,
                "result": result,
                "execution_log": self.execution_log,
                "agent_stats": self.agent_stats
            }, f, indent=2, default=str)

        print(f"Results saved to: {output_dir}")


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_agent_rag.py <query>")
        print("\nExample:")
        print('  python run_agent_rag.py "What is attention mechanism?"')
        return

    query = " ".join(sys.argv[1:])

    runner = AgentRAGRunner()
    await runner.run(query)


if __name__ == "__main__":
    asyncio.run(main())
