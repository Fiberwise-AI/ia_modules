# Benchmark Metrics Guide

## Overview

The benchmarking framework now tracks multiple types of metrics beyond basic timing:

1. **Timing Statistics** - Mean, median, P95, P99, etc.
2. **Throughput** - Operations per second, items per second
3. **Cost Tracking** - API calls, estimated costs
4. **Resource Efficiency** - Memory and CPU per operation

## Basic Usage

### Automatic Metrics

Operations per second is calculated automatically:

```python
from ia_modules.benchmarking import BenchmarkRunner, BenchmarkConfig

async def process_data(data):
    # Your processing logic
    await asyncio.sleep(0.01)

config = BenchmarkConfig(iterations=100)
runner = BenchmarkRunner(config)
result = await runner.run("my_pipeline", process_data, my_data)

# Automatically includes operations per second
print(result.get_summary())
# Output:
#   Iterations: 100
#   Mean: 10.00ms
#   Throughput: 100.00 ops/sec
```

## Cost Tracking

Track API calls and estimated costs for cloud services:

```python
# Run your benchmark
result = await runner.run("api_processor", process_with_api, data)

# Track API usage and costs
result.set_cost_tracking(
    api_calls=500,        # Total API calls made
    cost_usd=2.50         # Estimated cost in USD
)

print(result.get_summary())
# Output includes:
#   API Calls: 500
#   Est. Cost: $2.5000 ($0.005000/op)
```

### Example: OpenAI API Cost Tracking

```python
async def process_with_openai(texts):
    total_tokens = 0
    for text in texts:
        response = await openai_client.complete(text)
        total_tokens += response['usage']['total_tokens']
    return total_tokens

result = await runner.run("openai_processing", process_with_openai, batch_texts)

# Calculate cost (example: GPT-4 pricing)
total_tokens = result.metadata.get('total_tokens', 0)
cost_per_1k_tokens = 0.03  # $0.03 per 1K tokens
estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens

result.set_cost_tracking(
    api_calls=len(batch_texts),
    cost_usd=estimated_cost
)
```

## Throughput Metrics

Track items/records processed for data processing pipelines:

```python
async def process_records(records):
    processed = []
    for record in records:
        processed.append(await transform_record(record))
    return processed

records = load_records()  # 10,000 records
result = await runner.run("record_processor", process_records, records)

# Set throughput based on items processed
result.set_throughput(items_processed=len(records))

print(result.get_summary())
# Output includes:
#   Throughput: 10.00 ops/sec
#   Items/sec: 2500.00  # Processing 2,500 records per second
```

## Resource Efficiency

Track memory and CPU usage per operation:

```python
config = BenchmarkConfig(
    iterations=50,
    profile_memory=True,  # Enable memory profiling
    profile_cpu=True      # Enable CPU profiling
)

runner = BenchmarkRunner(config)
result = await runner.run("resource_test", heavy_computation, data)

print(result.get_summary())
# Output includes:
#   Memory/op: 15.50MB
#   CPU/op: 65.25%
```

## Complete Example

Combining all metrics for a comprehensive benchmark:

```python
import asyncio
from ia_modules.benchmarking import BenchmarkRunner, BenchmarkConfig

async def pipeline_with_api(batch):
    """Process a batch of data with API calls"""
    api_calls = 0
    processed_items = 0

    for item in batch:
        # Simulate API call
        result = await call_external_api(item)
        api_calls += 1
        processed_items += 1

    return {
        'items': processed_items,
        'api_calls': api_calls
    }

async def main():
    # Configure benchmark
    config = BenchmarkConfig(
        iterations=20,
        warmup_iterations=5,
        profile_memory=True,
        profile_cpu=True
    )

    runner = BenchmarkRunner(config)
    batch = create_test_batch(size=100)

    # Run benchmark
    result = await runner.run("production_pipeline", pipeline_with_api, batch)

    # Add all metrics
    result.set_cost_tracking(
        api_calls=20 * 100,  # 100 calls per iteration, 20 iterations
        cost_usd=0.20        # $0.01 per 100 calls
    )

    result.set_throughput(
        items_processed=20 * 100  # 100 items per iteration
    )

    # Display complete summary
    print(result.get_summary())

    # Export to JSON for CI/CD
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(result.to_dict(), f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
```

## Output Example

```
production_pipeline:
  Iterations: 20
  Mean: 125.50ms
  Median: 124.00ms
  Std Dev: 12.35ms
  Min: 110.00ms
  Max: 145.00ms
  P95: 138.50ms
  P99: 142.00ms
  Total: 2.51s
  Throughput: 7.97 ops/sec
  Items/sec: 796.81
  API Calls: 2000
  Est. Cost: $0.2000 ($0.010000/op)
  Memory/op: 12.45MB
  CPU/op: 42.30%
```

## Method Chaining

All metric-setting methods return `self` for easy chaining:

```python
result = await runner.run("my_pipeline", process, data)

result\
    .set_cost_tracking(api_calls=100, cost_usd=1.50)\
    .set_throughput(items_processed=5000)

print(result.get_summary())
```

## CI/CD Integration

Use these metrics in CI/CD to fail builds on cost or resource thresholds:

```python
# Check if cost per operation exceeds budget
if result.cost_per_operation > 0.05:
    raise Exception(f"Cost per operation ${result.cost_per_operation:.4f} exceeds budget!")

# Check if throughput meets requirements
if result.operations_per_second < 10.0:
    raise Exception(f"Throughput {result.operations_per_second:.2f} ops/sec below minimum!")

# Check resource efficiency
if result.memory_per_operation_mb > 100.0:
    raise Exception(f"Memory usage {result.memory_per_operation_mb:.2f}MB too high!")
```

## Best Practices

1. **Always set baselines** - Run benchmarks on known-good code to establish baselines
2. **Track trends** - Use historical data to identify gradual regressions
3. **Set budgets** - Define acceptable ranges for cost, memory, and CPU
4. **Test in production-like environments** - Ensure benchmarks reflect real usage
5. **Document assumptions** - Note pricing models, API tiers, etc. in benchmark code

## API Reference

### BenchmarkResult Methods

- `set_cost_tracking(api_calls: int, cost_usd: float)` - Set cost metrics
- `set_throughput(items_processed: int)` - Set throughput metrics
- `get_summary()` - Get human-readable summary with all metrics
- `to_dict()` - Export all data as dictionary

### Metrics Available

**Timing:**
- `mean_time`, `median_time`, `std_dev`
- `min_time`, `max_time`
- `p95_time`, `p99_time`
- `total_time`

**Throughput:**
- `operations_per_second` (automatic)
- `items_processed` (via `set_throughput()`)
- `items_per_second` (calculated)

**Cost:**
- `api_calls_count`
- `estimated_cost_usd`
- `cost_per_operation` (calculated)

**Resources:**
- `memory_per_operation_mb` (requires `profile_memory=True`)
- `cpu_per_operation_percent` (requires `profile_cpu=True`)
