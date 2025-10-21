# Cyclic Graph Support in IA Modules

## Overview

IA Modules now supports **cyclic graphs** (pipelines with loops) allowing you to create iterative workflows that can loop back to previous steps based on conditions. This enables powerful patterns like:

- Iterative refinement (e.g., content generation with review cycles)
- Quality assurance loops (retry until quality threshold met)
- AI agent workflows (reasoning loops, tool calling iterations)
- Batch processing with retries
- State machines with cyclic transitions

## When to Use Loops vs DAG

### Use Cyclic Graphs (Loops) When:
- ✅ You need iterative refinement (generate → review → regenerate)
- ✅ Quality gates require retry logic (process → validate → retry if failed)
- ✅ Implementing AI agents with reasoning loops
- ✅ State machines with repeating states
- ✅ Batch processing with retry on partial failure

### Use DAG (Directed Acyclic Graph) When:
- ✅ One-time sequential processing (ETL pipelines)
- ✅ Data transformations without feedback
- ✅ Simple linear workflows
- ✅ Fan-out/fan-in parallelization without cycles
- ✅ No iteration or retry logic needed

## Basic Example

Here's a simple iterative content generation pipeline with a review loop:

```json
{
  "name": "Iterative Content Generation",
  "steps": [
    {
      "id": "draft_content",
      "name": "Draft Content",
      "module": "my_steps.content",
      "step_class": "DraftContentStep",
      "config": {}
    },
    {
      "id": "review_content",
      "name": "Review Content",
      "module": "my_steps.content",
      "step_class": "ReviewContentStep",
      "config": {
        "quality_threshold": 0.8
      }
    },
    {
      "id": "publish_content",
      "name": "Publish Content",
      "module": "my_steps.content",
      "step_class": "PublishContentStep",
      "config": {}
    }
  ],
  "flow": {
    "start_at": "draft_content",
    "transitions": [
      {
        "from": "draft_content",
        "to": "review_content"
      },
      {
        "from": "review_content",
        "to": "draft_content",
        "condition": {
          "type": "expression",
          "config": {
            "source": "review_content.approved",
            "operator": "equals",
            "value": false
          }
        }
      },
      {
        "from": "review_content",
        "to": "publish_content",
        "condition": {
          "type": "expression",
          "config": {
            "source": "review_content.approved",
            "operator": "equals",
            "value": true
          }
        }
      }
    ]
  },
  "loop_config": {
    "max_iterations": 5,
    "max_loop_time_seconds": 300,
    "iteration_delay_seconds": 1
  }
}
```

## Loop Configuration

The `loop_config` section defines safety limits and behavior for loops:

```json
{
  "loop_config": {
    "max_iterations": 100,              // Maximum iterations per step (default: 100)
    "max_loop_time_seconds": 3600,      // Maximum loop execution time (default: 3600 = 1 hour)
    "iteration_delay_seconds": 0        // Delay between iterations (default: 0)
  }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_iterations` | int | 100 | Maximum number of times a single step can be executed in a loop |
| `max_loop_time_seconds` | int | 3600 | Maximum time (seconds) a loop can run before stopping |
| `iteration_delay_seconds` | float | 0 | Delay (seconds) between loop iterations (useful for rate limiting) |

## Loop Safety Features

IA Modules includes comprehensive safety mechanisms to prevent infinite loops:

### 1. Maximum Iterations
Each step tracks how many times it's been executed. Once `max_iterations` is reached, the loop terminates with an error:

```
Loop safety limit reached: Max iterations (5) reached for step 'review_content'
```

### 2. Maximum Loop Time
Each loop tracks total execution time. If a loop runs longer than `max_loop_time_seconds`, it terminates:

```
Loop safety limit reached: Max loop time (300s) exceeded for loop 'review_loop'
```

### 3. Exit Condition Validation
The CLI validation tool checks that all loops have at least one exit condition:

```bash
ia-modules validate pipeline.json --strict
```

Output:
```
❌ ERROR: Loop ['step_a', 'step_b'] has no exit condition - infinite loop!
```

### 4. Loop Detection
Automatically detects all loops in your pipeline during initialization:

```python
pipeline = Pipeline("my_pipeline", steps, flow, services, loop_config=loop_config)

if pipeline.has_loops():
    loops = pipeline.get_loops()
    print(f"Detected {len(loops)} loop(s)")
    for loop in loops:
        print(f"  Loop: {' -> '.join(loop.steps)}")
        print(f"  Exit conditions: {len(loop.exit_conditions)}")
```

## Advanced Examples

### Example 1: Quality Assurance Loop

A pipeline that processes data and retries if quality is below threshold:

```json
{
  "name": "Quality Assurance Pipeline",
  "steps": [
    {
      "id": "process_data",
      "name": "Process Data",
      "module": "my_steps.processing",
      "step_class": "DataProcessingStep",
      "config": {}
    },
    {
      "id": "quality_check",
      "name": "Quality Check",
      "module": "my_steps.validation",
      "step_class": "QualityCheckStep",
      "config": {
        "min_quality_score": 0.9
      }
    },
    {
      "id": "finalize",
      "name": "Finalize",
      "module": "my_steps.processing",
      "step_class": "FinalizeStep",
      "config": {}
    }
  ],
  "flow": {
    "start_at": "process_data",
    "transitions": [
      {
        "from": "process_data",
        "to": "quality_check"
      },
      {
        "from": "quality_check",
        "to": "process_data",
        "condition": {
          "type": "expression",
          "config": {
            "source": "quality_check.quality_score",
            "operator": "less_than",
            "value": 0.9
          }
        }
      },
      {
        "from": "quality_check",
        "to": "finalize",
        "condition": {
          "type": "expression",
          "config": {
            "source": "quality_check.quality_score",
            "operator": "greater_than_or_equal",
            "value": 0.9
          }
        }
      }
    ]
  },
  "loop_config": {
    "max_iterations": 10,
    "max_loop_time_seconds": 600
  }
}
```

### Example 2: AI Agent Reasoning Loop

An AI agent that reasons iteratively until reaching a conclusion:

```json
{
  "name": "AI Reasoning Agent",
  "steps": [
    {
      "id": "analyze_problem",
      "name": "Analyze Problem",
      "module": "my_steps.ai",
      "step_class": "ProblemAnalysisStep",
      "config": {}
    },
    {
      "id": "generate_hypothesis",
      "name": "Generate Hypothesis",
      "module": "my_steps.ai",
      "step_class": "HypothesisGenerationStep",
      "config": {}
    },
    {
      "id": "test_hypothesis",
      "name": "Test Hypothesis",
      "module": "my_steps.ai",
      "step_class": "HypothesisTestingStep",
      "config": {}
    },
    {
      "id": "evaluate_results",
      "name": "Evaluate Results",
      "module": "my_steps.ai",
      "step_class": "ResultEvaluationStep",
      "config": {
        "confidence_threshold": 0.95
      }
    },
    {
      "id": "report_conclusion",
      "name": "Report Conclusion",
      "module": "my_steps.ai",
      "step_class": "ConclusionReportStep",
      "config": {}
    }
  ],
  "flow": {
    "start_at": "analyze_problem",
    "transitions": [
      {
        "from": "analyze_problem",
        "to": "generate_hypothesis"
      },
      {
        "from": "generate_hypothesis",
        "to": "test_hypothesis"
      },
      {
        "from": "test_hypothesis",
        "to": "evaluate_results"
      },
      {
        "from": "evaluate_results",
        "to": "generate_hypothesis",
        "condition": {
          "type": "expression",
          "config": {
            "source": "evaluate_results.confidence",
            "operator": "less_than",
            "value": 0.95
          }
        }
      },
      {
        "from": "evaluate_results",
        "to": "report_conclusion",
        "condition": {
          "type": "expression",
          "config": {
            "source": "evaluate_results.confidence",
            "operator": "greater_than_or_equal",
            "value": 0.95
          }
        }
      }
    ]
  },
  "loop_config": {
    "max_iterations": 20,
    "max_loop_time_seconds": 1800,
    "iteration_delay_seconds": 2
  }
}
```

### Example 3: Nested Loops

A pipeline with multiple nested loops (outer loop for batches, inner loop for retries):

```json
{
  "name": "Batch Processing with Retries",
  "steps": [
    {
      "id": "get_next_batch",
      "name": "Get Next Batch",
      "module": "my_steps.batch",
      "step_class": "BatchFetchStep",
      "config": {}
    },
    {
      "id": "process_batch",
      "name": "Process Batch",
      "module": "my_steps.batch",
      "step_class": "BatchProcessStep",
      "config": {}
    },
    {
      "id": "validate_batch",
      "name": "Validate Batch",
      "module": "my_steps.batch",
      "step_class": "BatchValidationStep",
      "config": {}
    },
    {
      "id": "check_more_batches",
      "name": "Check More Batches",
      "module": "my_steps.batch",
      "step_class": "BatchCheckStep",
      "config": {}
    },
    {
      "id": "complete",
      "name": "Complete Processing",
      "module": "my_steps.batch",
      "step_class": "CompletionStep",
      "config": {}
    }
  ],
  "flow": {
    "start_at": "get_next_batch",
    "transitions": [
      {
        "from": "get_next_batch",
        "to": "process_batch"
      },
      {
        "from": "process_batch",
        "to": "validate_batch"
      },
      {
        "from": "validate_batch",
        "to": "process_batch",
        "condition": {
          "type": "expression",
          "config": {
            "source": "validate_batch.is_valid",
            "operator": "equals",
            "value": false
          }
        }
      },
      {
        "from": "validate_batch",
        "to": "check_more_batches",
        "condition": {
          "type": "expression",
          "config": {
            "source": "validate_batch.is_valid",
            "operator": "equals",
            "value": true
          }
        }
      },
      {
        "from": "check_more_batches",
        "to": "get_next_batch",
        "condition": {
          "type": "expression",
          "config": {
            "source": "check_more_batches.has_more",
            "operator": "equals",
            "value": true
          }
        }
      },
      {
        "from": "check_more_batches",
        "to": "complete",
        "condition": {
          "type": "expression",
          "config": {
            "source": "check_more_batches.has_more",
            "operator": "equals",
            "value": false
          }
        }
      }
    ]
  },
  "loop_config": {
    "max_iterations": 50,
    "max_loop_time_seconds": 7200
  }
}
```

## Implementing Loop-Compatible Steps

Steps in loops should be designed to:

1. **Track iteration state** (optional but useful):
```python
from ia_modules.pipeline.core import Step
from typing import Dict, Any

class IterativeContentStep(Step):
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Get current iteration (if provided by loop executor)
        iteration = data.get('_loop_iteration', 0)

        # Your step logic here
        content = await self.generate_content(data, iteration)

        return {
            "content": content,
            "iteration": iteration,
            "metadata": {"attempted_at_iteration": iteration}
        }
```

2. **Provide clear exit conditions**:
```python
class QualityCheckStep(Step):
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        content = data.get('content')

        # Calculate quality score
        quality_score = await self.assess_quality(content)

        # Return clear boolean for routing
        return {
            "quality_score": quality_score,
            "passed": quality_score >= self.config['min_quality_score'],
            "approved": quality_score >= self.config['min_quality_score']  # For readability
        }
```

3. **Handle state properly** (data from previous iterations):
```python
class ReviewStep(Step):
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Access previous iteration's output
        previous_content = data.get('content')
        previous_score = data.get('quality_score', 0)

        # Make improvements based on previous attempt
        improved_content = await self.improve(previous_content, previous_score)

        return {
            "content": improved_content,
            "improvement_made": True
        }
```

## Debugging Loops

### Enable Loop Logging

Set log level to INFO to see loop execution details:

```python
import logging
logging.basicConfig(level=logging.INFO)

pipeline = Pipeline("my_pipeline", steps, flow, services, loop_config=loop_config)
result = await pipeline.run(input_data)
```

Output:
```
INFO:Pipeline.my_pipeline:Detected 1 loop(s) in pipeline 'my_pipeline'
INFO:Pipeline.my_pipeline:Loop 1: draft_content -> review_content -> draft_content
INFO:Pipeline.my_pipeline:  Exit conditions: 1
INFO:Pipeline.my_pipeline:Executing step: draft_content (iteration 1)
INFO:Pipeline.my_pipeline:Executing step: review_content (iteration 1)
INFO:Pipeline.my_pipeline:Loop condition met, returning to: draft_content
INFO:Pipeline.my_pipeline:Executing step: draft_content (iteration 2)
...
```

### Use CLI Validation

Validate your pipeline before running:

```bash
# Validate loop structure
ia-modules validate pipeline.json

# Strict mode (warnings become errors)
ia-modules validate pipeline.json --strict

# JSON output for CI/CD
ia-modules validate pipeline.json --output json
```

### Visualize Loop Flow

Generate visual diagrams to understand loop structure:

```bash
# Generate PNG diagram
ia-modules visualize pipeline.json --output pipeline.png

# Generate SVG (better for large pipelines)
ia-modules visualize pipeline.json --output pipeline.svg --format svg
```

Loops are color-coded in the visualization with special edge labels showing exit conditions.

### Inspect Loop History

After execution, inspect the loop history:

```python
pipeline = Pipeline("my_pipeline", steps, flow, services, loop_config=loop_config)
result = await pipeline.run(input_data)

# Access loop execution history (if loop_executor available)
if pipeline.loop_executor:
    history = pipeline.loop_executor.loop_context.loop_history

    for entry in history:
        print(f"Loop: {entry['loop_id']}")
        print(f"  Iteration: {entry['iteration']}")
        print(f"  Step: {entry['step_id']}")
        print(f"  Timestamp: {entry['timestamp']}")
```

## Performance Considerations

### Loop Overhead

Loop detection and execution add minimal overhead:
- Loop detection: O(V + E) where V = steps, E = transitions (runs once at init)
- Iteration tracking: O(1) per step execution
- Safety checks: O(1) per step execution

**Total overhead**: <1% for typical pipelines

### Optimizing Loop Performance

1. **Set realistic iteration limits**:
```json
{
  "loop_config": {
    "max_iterations": 10  // Don't set to 1000 if you expect 3-5 iterations
  }
}
```

2. **Add iteration delays for API rate limiting**:
```json
{
  "loop_config": {
    "iteration_delay_seconds": 1  // Prevents API throttling
  }
}
```

3. **Use early exit conditions**:
```python
# Good: Exit as soon as threshold met
{
  "condition": {
    "type": "expression",
    "config": {
      "source": "quality_check.score",
      "operator": "greater_than",
      "value": 0.9  // Exit immediately when reached
    }
  }
}
```

4. **Monitor loop metrics with telemetry**:
```python
from ia_modules.telemetry import get_telemetry

telemetry = get_telemetry()
pipeline = Pipeline("my_pipeline", steps, flow, services,
                   enable_telemetry=True, loop_config=loop_config)

result = await pipeline.run(input_data)

# Check loop metrics
metrics = telemetry.get_metrics()
for metric in metrics:
    if 'loop' in metric.name:
        print(f"{metric.name}: {metric.value}")
```

## Best Practices

### ✅ DO:
- Always configure `max_iterations` and `max_loop_time_seconds`
- Provide clear exit conditions for every loop
- Log iteration count and state in your steps
- Use CLI validation before deploying
- Test with realistic data to determine iteration counts
- Monitor loop execution with telemetry in production

### ❌ DON'T:
- Create loops without exit conditions (infinite loops)
- Set `max_iterations` to extremely high values (>1000) without good reason
- Rely solely on timeouts (use iteration limits too)
- Forget to handle state between iterations
- Skip validation with `ia-modules validate`
- Ignore loop warnings in logs

## Comparison with LangGraph

IA Modules' cyclic graph support is designed to match and extend LangGraph's capabilities:

| Feature | IA Modules | LangGraph |
|---------|-----------|-----------|
| Cyclic graphs | ✅ Yes | ✅ Yes |
| Safety limits (max iterations) | ✅ Yes | ✅ Yes |
| Safety limits (timeout) | ✅ Yes | ✅ Yes |
| Loop detection | ✅ Automatic | ⚠️ Manual |
| Exit condition validation | ✅ Automatic | ⚠️ Manual |
| Visualization | ✅ CLI tool | ✅ Built-in |
| Checkpointing | 🚧 Coming in v0.0.3 | ✅ Yes |
| General workflows (non-AI) | ✅ Yes | ❌ AI-only |
| Cost tracking in loops | ✅ Yes | ❌ No |

## Migration from DAG to Cyclic Graphs

If you have an existing DAG pipeline and want to add loops:

**Before (DAG)**:
```json
{
  "flow": {
    "start_at": "process",
    "transitions": [
      {"from": "process", "to": "validate"},
      {"from": "validate", "to": "finish"}
    ]
  }
}
```

**After (with retry loop)**:
```json
{
  "flow": {
    "start_at": "process",
    "transitions": [
      {"from": "process", "to": "validate"},
      {
        "from": "validate",
        "to": "process",
        "condition": {
          "type": "expression",
          "config": {
            "source": "validate.is_valid",
            "operator": "equals",
            "value": false
          }
        }
      },
      {
        "from": "validate",
        "to": "finish",
        "condition": {
          "type": "expression",
          "config": {
            "source": "validate.is_valid",
            "operator": "equals",
            "value": true
          }
        }
      }
    ]
  },
  "loop_config": {
    "max_iterations": 5,
    "max_loop_time_seconds": 300
  }
}
```

**Key changes**:
1. Add backward transition (`validate` → `process`)
2. Add exit condition on forward transition
3. Add `loop_config` section

## Troubleshooting

### Problem: "Loop validation: Loop [...] has no exit condition - infinite loop!"

**Solution**: Add a conditional transition that exits the loop:
```json
{
  "from": "step_in_loop",
  "to": "step_outside_loop",
  "condition": {
    "type": "expression",
    "config": {"source": "step_in_loop.done", "operator": "equals", "value": true}
  }
}
```

### Problem: "Max iterations (100) reached for step 'my_step'"

**Solution**: Either:
1. Increase `max_iterations` if legitimately needed:
   ```json
   {"loop_config": {"max_iterations": 200}}
   ```
2. Or fix your exit condition logic to ensure loop terminates earlier

### Problem: "Max loop time (3600s) exceeded"

**Solution**: Either:
1. Optimize your steps to run faster
2. Increase timeout if legitimately needed:
   ```json
   {"loop_config": {"max_loop_time_seconds": 7200}}
   ```

### Problem: Loop not detected by CLI validation

**Solution**: Ensure you're using `transitions` (not `paths`) format, or update to latest version.

## Further Reading

- [Pipeline System Documentation](../PIPELINE_SYSTEM.md)
- [CLI Tool Documentation](./CLI_TOOL_DOCUMENTATION.md)
- [Checkpointing Design](./CHECKPOINTING_DESIGN.md) (Coming in v0.0.3)
- [IMPLEMENTATION_PLAN_V0.0.3.md](../IMPLEMENTATION_PLAN_V0.0.3.md)

## Support

For issues or questions:
- File a GitHub issue: https://github.com/yourusername/ia_modules/issues
- Check examples: `ia_modules/tests/pipelines/loop_pipeline/`
- Run validation: `ia-modules validate your_pipeline.json`
