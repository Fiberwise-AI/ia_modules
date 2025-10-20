# Pipeline Examples & Feature Showcase

This guide showcases the key features of the IA Modules framework through practical, runnable examples. Each pipeline demonstrates specific capabilities and patterns you can use in your own applications.

## Table of Contents

- [Framework Features Overview](#framework-features-overview)
- [Feature Showcase Pipelines](#feature-showcase-pipelines)
  - [🔗 Sequential Processing](#-sequential-processing---simple-pipeline)
  - [🤖 AI/LLM Integration](#-aillm-integration---agent-pipeline)
  - [⚡ Parallel Execution](#-parallel-execution---parallel-pipeline)
  - [🎯 Conditional Routing](#-conditional-routing---conditional-pipeline)
- [Running the Examples](#running-the-examples)
- [Advanced Features Deep Dive](#advanced-features-deep-dive)
- [Building Your Own Pipelines](#building-your-own-pipelines)

## Framework Features Overview

The IA Modules pipeline framework provides powerful capabilities for building complex, intelligent workflows. This guide demonstrates each major feature through practical, runnable examples located in `tests/pipelines/`.

### 🎯 **What You'll Learn**

| Feature | Example Pipeline | Key Capability |
|---------|------------------|----------------|
| **Sequential Processing** | Simple Pipeline | Basic step chaining with data flow |
| **AI/LLM Integration** | Agent Pipeline | AI-powered processing with multiple providers |
| **Parallel Execution** | Parallel Pipeline | Concurrent processing and result merging |
| **Conditional Routing** | Conditional Pipeline | Dynamic flow control based on data |

### 🚀 **Why These Examples Matter**

Each pipeline is a **complete, working application** that you can:
- ✅ **Run immediately** with the provided commands
- 🔧 **Modify and extend** for your use cases  
- 📖 **Learn from** to understand framework patterns
- 🎯 **Use as templates** for your own pipelines

## Feature Showcase Pipelines

### 🔗 Sequential Processing - Simple Pipeline

> **🎯 Showcases**: Step chaining, data flow, parameter templating, schema validation

**Why This Matters**: Most business workflows follow sequential patterns. This example shows how to build reliable, maintainable pipelines where each step builds on the previous one's output.

**Key Framework Features Demonstrated**:
- ✅ **Step Chaining**: Linear execution flow (step1 → step2 → step3)
- ✅ **Data Flow**: Seamless data passing between steps  
- ✅ **Parameter Templates**: `{parameters.topic}` and `{steps.step1.output.topic}` syntax
- ✅ **Schema Validation**: Input/output type checking and validation

**Configuration**:
```json
{
  "name": "Simple Three-Step Pipeline",
  "description": "A simple pipeline with three steps passing data between them",
  "version": "1.0.0"
}
```

**Steps**:
1. **Data Preparation** - Processes initial topic data
2. **Data Enrichment** - Enhances the processed data
3. **Data Finalization** - Produces final output

**Usage**:
```bash
# Basic execution
python tests/pipeline_runner.py tests/pipelines/simple_pipeline/pipeline.json

# With custom input
python tests/pipeline_runner.py tests/pipelines/simple_pipeline/pipeline.json --input '{"topic": "machine learning"}'
```

---

### 🤖 AI/LLM Integration - Agent Pipeline

> **🎯 Showcases**: AI integration, multi-provider LLM support, service injection

**Why This Matters**: Modern applications need AI capabilities. This example shows how to integrate multiple LLM providers seamlessly, making your pipelines both intelligent and reliable.

**Key Framework Features Demonstrated**:
- 🤖 **Multi-Provider LLM Support**: Google Gemini, OpenAI, Anthropic with automatic switching
-  **Structured AI Output**: JSON schema-based responses with automatic parsing
- 🔌 **Service Injection**: Clean access to LLM services via `self.services.get('llm_provider')`
- ⚡ **Production-Ready**: Real error handling, logging, and monitoring

**Configuration**:
```json
{
  "name": "Agent-Based Processing Pipeline",
  "description": "Pipeline that uses agent-based processing steps",
  "version": "1.0.0"
}
```

**Steps**:
1. **Data Ingestor** (`SimpleAgentStep`) - LLM-powered data analysis and preparation
2. **Decision Agent** (`DecisionAgentStep`) - AI-driven validation and decision making
3. **Final Processor** (`SimpleAgentStep`) - LLM-enhanced final processing

**LLM Integration**:
- Supports multiple providers (Google Gemini, OpenAI, Anthropic)
- Automatic JSON parsing with markdown code block handling
- Graceful degradation to simple processing if LLM unavailable

**Usage**:
```bash
# Set up environment variable
export GEMINI_API_KEY="your_api_key_here"

# Run with LLM integration
python tests/pipeline_runner_with_llm.py tests/pipelines/agent_pipeline/pipeline.json --input '{"task": "sentiment_analysis", "text": "I love this product!"}'

# Alternative providers
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

**Example Output**:
```json
{
  "agent_name": "step1",
  "processing_type": "ingestion",
  "llm_response": {
    "summary": "Positive sentiment analysis of product feedback",
    "entities": [{"entity": "product", "type": "object", "confidence": 0.95}],
    "quality_score": 1.0
  },
  "metadata": {
    "llm_used": true,
    "processing_type": "llm_agent"
  }
}
```

---

### ⚡ Parallel Execution - Parallel Pipeline

> **🎯 Showcases**: Concurrent execution, data splitting, result merging, complex flow patterns

**Why This Matters**: Large datasets and compute-intensive tasks need parallel processing. This example shows how to build high-performance pipelines that scale horizontally.

**Key Framework Features Demonstrated**:
- 🚀 **Concurrent Execution**: Multiple steps running simultaneously
- 📊 **Data Splitting**: Intelligent partitioning for parallel processing
- 🔄 **Result Merging**: Automatic aggregation of parallel results  
- 📈 **Performance Scaling**: Handle large datasets efficiently
- 🎯 **Complex Flow Control**: Multi-path execution with synchronization

#### How Parallel Execution Works

The framework automatically detects when steps can run in parallel based on the flow configuration:

**1. Dependency Analysis**
```json
{
  "flow": {
    "start_at": "step1",
    "paths": [
      {"from_step": "step1", "to_step": "step2"},
      {"from_step": "step1", "to_step": "step3"}, 
      {"from_step": "step1", "to_step": "step4"}
    ]
  }
}
```

**2. Parallel Execution**
- `step1` completes first
- `step2`, `step3`, `step4` **run concurrently** (they all depend only on step1)
- `step5` waits for all three to complete before starting

**3. Synchronization**
The framework automatically:
- ✅ **Starts steps** as soon as their dependencies are satisfied
- ⚡ **Runs independent steps** concurrently using async/await
- 🔄 **Waits for completion** before proceeding to dependent steps
- 📊 **Merges results** when multiple parallel steps feed into one step

**Configuration**:
```json
{
  "name": "Parallel Data Processing Pipeline",
  "description": "Pipeline that processes data in parallel streams",
  "version": "1.0.0"
}
```

**Steps**:
1. **Data Splitter** - Divides input data into processable chunks
2. **Stream Processor 1-3** - Parallel processing streams for analytics
3. **Results Merger** - Combines results from parallel streams
4. **Stats Collector** - Generates final statistics and metrics

**Flow Pattern**:
```
step1 (splitter)
  ├── step2 (stream1) ──┐
  ├── step3 (stream2) ──┼── step5 (merger) ── step6 (stats)
  └── step4 (stream3) ──┘
```

**Usage**:
```bash
# Run with array data
python tests/pipeline_runner.py tests/pipelines/parallel_pipeline/pipeline.json --input '{"loaded_data": [{"id": 1, "value": 100}, {"id": 2, "value": 200}]}'
```

**Real Pipeline JSON (Parallel Execution)**:
```json
{
  "flow": {
    "start_at": "step1",
    "paths": [
      {"from_step": "step1", "to_step": "step2"},
      {"from_step": "step1", "to_step": "step3"},
      {"from_step": "step1", "to_step": "step4"},
      {"from_step": "step2", "to_step": "step5"},
      {"from_step": "step3", "to_step": "step5"}, 
      {"from_step": "step4", "to_step": "step5"}
    ]
  }
}
```

**Execution Timeline**:
```
Time 0: step1 starts
Time 1: step1 completes
Time 1: step2, step3, step4 start SIMULTANEOUSLY  
Time 3: step2 completes
Time 4: step3 completes  
Time 5: step4 completes
Time 5: step5 starts (waits for ALL to complete)
Time 6: step5 completes
```

🔥 **Key Point**: Steps 2, 3, and 4 run **concurrently**, not sequentially!

---

### 🎯 Conditional Routing - Conditional Pipeline

> **🎯 Showcases**: Dynamic routing, conditional logic, threshold-based decisions, multi-path processing

**Why This Matters**: Real-world data varies in quality and type. This example shows how to build adaptive pipelines that automatically choose the right processing path based on data characteristics.

**Key Framework Features Demonstrated**:
- 🎯 **Dynamic Routing**: Automatic path selection based on data conditions
- 📊 **Threshold Logic**: Quality-based decision making (`>= 0.8` vs `< 0.8`)
- 🔀 **Multi-Path Processing**: Different algorithms for different data types
- 🔧 **Template Parameters**: Runtime configuration via `{{ parameters.threshold }}`  
- 🎭 **Adaptive Behavior**: Same pipeline, different outcomes based on input

**Configuration**:
```json
{
  "name": "Conditional Processing Pipeline",
  "description": "Pipeline with conditional logic based on data quality",
  "version": "1.0.0"
}
```

**Steps**:
1. **Data Ingestor** - Loads test data with quality scores
2. **Quality Checker** - Evaluates data against threshold
3. **High Quality Processor** - Full processing for high-quality data
4. **Low Quality Processor** - Basic processing for low-quality data
5. **Results Aggregator** - Combines results from both processing paths

**Conditional Flow**:
```
step1 (ingest) → step2 (quality check)
                     ├── step3 (high quality) ──┐
                     └── step4 (low quality) ───┼── step5 (aggregator)
```

**Usage**:
```bash
# Run with default threshold (0.8)
python tests/pipeline_runner.py tests/pipelines/conditional_pipeline/pipeline.json

# Custom threshold
python tests/pipeline_runner.py tests/pipelines/conditional_pipeline/pipeline.json --input '{"threshold": 0.9}'
```

## Running the Examples

### 🚀 Quick Start Commands

**Try Sequential Processing:**
```bash
cd ia_modules
python tests/pipeline_runner.py tests/pipelines/simple_pipeline/pipeline.json --input '{"topic": "machine learning"}'
```

**Try AI-Powered Processing:**
```bash
# Set up AI provider
export GEMINI_API_KEY="your_api_key"

# Run intelligent pipeline  
python tests/pipeline_runner_with_llm.py tests/pipelines/agent_pipeline/pipeline.json --input '{"task": "sentiment_analysis", "text": "This framework is amazing!"}'
```

**Try Parallel Processing:**
```bash
python tests/pipeline_runner.py tests/pipelines/parallel_pipeline/pipeline.json --input '{"loaded_data": [{"id": 1, "value": 100}, {"id": 2, "value": 200}, {"id": 3, "value": 300}]}'
```

**Try Conditional Logic:**
```bash
python tests/pipeline_runner.py tests/pipelines/conditional_pipeline/pipeline.json
```

### 🔧 Runner Options

| Runner | Use Case | Command |
|--------|----------|---------|
| **Standard** | Basic pipelines | `python tests/pipeline_runner.py <pipeline.json>` |
| **AI-Enhanced** | LLM-powered pipelines | `python tests/pipeline_runner_with_llm.py <pipeline.json>` |

**Common Options:**
- `--input <json>` - Custom input data
- `--output <path>` - Results directory  
- `--slug <name> --db-url <url>` - Database execution

### Output Structure

All pipeline runs generate timestamped output directories:

```
pipeline_run_YYYYMMDD_HHMMSS/
├── pipeline_result.json    # Final execution results
└── pipeline.log           # Detailed execution logs
```

## Advanced Features Deep Dive

### 🤖 AI/LLM Integration

The Agent Pipeline showcases production-ready AI integration:

#### Supported Providers

| Provider | Environment Variable | Models | Notes |
|----------|---------------------|---------|-------|
| Google Gemini | `GEMINI_API_KEY` | gemini-2.5-flash, gemini-2.5-pro | Fast and cost-effective |
| OpenAI | `OPENAI_API_KEY` | gpt-3.5-turbo, gpt-4 | Widely compatible |
| Anthropic | `ANTHROPIC_API_KEY` | claude-3-haiku, claude-3-sonnet | Great reasoning |

#### Production-Ready AI Features

- ** Structured Output**: JSON schema-enforced responses  
- **🔀 Provider Switching**: Seamless failover between OpenAI, Anthropic, Google
- **⚡ Performance**: Async processing with proper error handling
- **🛡️ Reliability**: Production logging and monitoring built-in

#### See It In Action

```bash
# Run with AI-powered processing
export GEMINI_API_KEY="your_key"
python tests/pipeline_runner_with_llm.py tests/pipelines/agent_pipeline/pipeline.json --input '{"task": "analyze", "text": "sample"}'

# Alternative providers
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### ⚡ Parallel Execution Patterns

The framework supports several parallel execution patterns:

#### **1. Fan-Out Pattern** (1 → Many)
```json
{
  "paths": [
    {"from_step": "input", "to_step": "process_a"},
    {"from_step": "input", "to_step": "process_b"},
    {"from_step": "input", "to_step": "process_c"}
  ]
}
```
✅ `process_a`, `process_b`, `process_c` run **simultaneously** after `input` completes

#### **2. Fan-In Pattern** (Many → 1)  
```json
{
  "paths": [
    {"from_step": "process_a", "to_step": "merge"},
    {"from_step": "process_b", "to_step": "merge"},
    {"from_step": "process_c", "to_step": "merge"}
  ]
}
```
✅ `merge` waits for **all three** parallel steps to complete

#### **3. Pipeline Parallelism** (Concurrent Chains)
```json
{
  "paths": [
    {"from_step": "split", "to_step": "chain1_step1"},
    {"from_step": "chain1_step1", "to_step": "chain1_step2"},
    {"from_step": "split", "to_step": "chain2_step1"}, 
    {"from_step": "chain2_step1", "to_step": "chain2_step2"}
  ]
}
```
✅ Two processing chains run **independently and concurrently**

#### **Performance Benefits**

- **🚀 Speed**: CPU-bound tasks utilize multiple cores  
- **⚡ Throughput**: I/O-bound tasks don't block each other
- **📈 Scalability**: Add more parallel paths to increase capacity
- **🔧 Resource Efficiency**: Better utilization of available hardware

### 🎯 Business Logic Flexibility  

The Conditional Pipeline shows how to build adaptive systems:

- **Runtime Decisions**: Change behavior based on live data
- **Quality Gates**: Automatic quality control and routing
- **Multi-Algorithm Support**: Different processing for different data types
- **Configuration-Driven**: Change logic without code changes

## Building Your Own Pipelines

### 🎯 Choose Your Pattern

Based on the examples above, pick the pattern that fits your use case:

| **Your Need** | **Use This Example** | **Key Features** |
|---------------|---------------------|------------------|
| **Simple data processing** | Simple Pipeline | Sequential steps, parameter passing |
| **AI-enhanced workflows** | Agent Pipeline | LLM integration, intelligent processing |  
| **High-performance processing** | Parallel Pipeline | Concurrent execution, data splitting |
| **Adaptive business logic** | Conditional Pipeline | Dynamic routing, quality gates |

### 🏗️ Project Structure

```bash
your_pipeline_project/
├── pipeline.json          # Your pipeline definition  
├── steps/                 # Your step implementations
│   ├── __init__.py
│   └── your_step.py
└── main.py               # Your application entry point
```

### Step Implementation Template

```python
from ia_modules.pipeline.core import Step
from typing import Dict, Any

class YourCustomStep(Step):
    """Your custom step implementation"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
    
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement your step logic here"""
        
        # Access services if needed
        # llm_service = self.services.get('llm_provider')
        # db_service = self.services.get('database')
        
        # Process data
        result = {
            "step_name": self.name,
            "processed_data": data,
            "timestamp": "2025-09-25T12:00:00"
        }
        
        return result
```

### Pipeline Configuration Template

```json
{
  "name": "Your Pipeline Name",
  "description": "Description of what your pipeline does",
  "version": "1.0.0",
  "parameters": {
    "your_parameter": "default_value"
  },
  "steps": [
    {
      "id": "step1",
      "name": "Your Step",
      "step_class": "YourCustomStep",
      "module": "tests.pipelines.your_pipeline_name.steps.your_step",
      "config": {
        "setting1": "value1"
      }
    }
  ],
  "flow": {
    "start_at": "step1",
    "paths": []
  }
}
```

### Testing Your Pipeline

1. **Create the directory structure**
2. **Implement your steps**
3. **Define the pipeline JSON**
4. **Test execution**:

```bash
# Test your new pipeline
python tests/pipeline_runner.py tests/pipelines/your_pipeline_name/pipeline.json

# With LLM support (if applicable)
python tests/pipeline_runner_with_llm.py tests/pipelines/your_pipeline_name/pipeline.json
```

---

## Troubleshooting

### Common Issues

1. **"Module not found"**: Ensure step module paths are correct in pipeline JSON
2. **"LLM providers not configured"**: Set appropriate environment variables
3. **"Step class not found"**: Verify class names match between JSON and Python files
4. **JSON parsing errors**: Check LLM responses for markdown code blocks

### Getting Help

- Check the log files in output directories for detailed error information
- Test individual components before full pipeline execution
- Ensure all dependencies are installed (`pip install -r requirements.txt`)
- Verify API keys have necessary permissions and quota

---

*For more information about the IA Modules framework, see the [main README](README.md).*