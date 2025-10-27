# Agentic RAG Pipeline

Multi-agent RAG system using AgentOrchestrator with execution hooks for monitoring.

## Architecture

**Agents:**
1. **Query Analyzer** - Analyzes user query, extracts intent and keywords
2. **Retriever Agent** - Searches documents based on analyzed query
3. **Context Synthesizer** - Combines and ranks retrieved documents
4. **Answer Generator** - Generates final answer using LLM with context

**Execution Hooks:**
- Track agent execution times
- Log agent communication
- Monitor retrieval quality
- Capture errors

## Usage

```bash
python tests/pipeline_runner.py tests/pipelines/agent_rag_pipeline/pipeline.json \
  --input '{"query": "What is attention mechanism?"}'
```

## Features

- ✅ Multi-agent collaboration
- ✅ State management between agents
- ✅ Execution hooks for observability
- ✅ Guardrails on input/output
- ✅ LLM integration
- ✅ Database or file-based retrieval
