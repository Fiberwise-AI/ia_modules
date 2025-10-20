# Pipeline CLI Tool Documentation

**Version**: 0.2.0
**Status**: ✅ Complete (73 tests passing)

## Overview

The IA Pipeline CLI tool (`ia-modules`) provides comprehensive validation, formatting, and visualization capabilities for pipeline definitions. It helps developers catch errors early, ensure consistency, and understand pipeline structure.

## Installation

```bash
cd ia_modules
pip install -e .

# For visualization support
pip install -e ".[cli]"
```

After installation, the `ia-modules` command will be available globally.

## Commands

### 1. Validate

Comprehensive pipeline validation including structure, steps, flow, and templates.

#### Usage

```bash
ia-modules validate <pipeline-file> [options]
```

#### Options

- `--strict` - Treat warnings as errors
- `--json` - Output results in JSON format

#### Examples

**Basic validation:**
```bash
ia-modules validate pipeline.json
```

Output:
```
✓ Pipeline validation PASSED

Info (1):
  • Step 'fetch_data' module 'ia_modules.steps.fetch' is importable
```

**Strict mode:**
```bash
ia-modules validate pipeline.json --strict
```

Warnings become errors in strict mode, perfect for CI/CD pipelines.

**JSON output:**
```bash
ia-modules validate pipeline.json --json
```

Output:
```json
{
  "is_valid": true,
  "errors": [],
  "warnings": [],
  "info": [
    "Step 'fetch_data' module 'ia_modules.steps.fetch' is importable"
  ]
}
```

#### What Gets Validated

##### Structure Validation
- ✅ Required fields present (`name`, `steps`, `flow`)
- ✅ Correct data types
- ✅ Non-empty values

##### Step Validation
- ✅ Required step fields (`name`, `module`, `class`)
- ✅ No duplicate step names
- ✅ Valid naming conventions
- ✅ Module import checking
- ✅ Config and inputs structure
- ✅ Input source references

##### Flow Validation
- ✅ Start step defined and exists
- ✅ All referenced steps exist
- ✅ No unreachable steps
- ✅ Cycle detection
- ✅ Condition structure

##### Template Validation
- ✅ Parameter references are defined
- ✅ Step output references are valid
- ✅ Template syntax correctness

#### Validation Examples

**Invalid pipeline:**
```json
{
  "name": "test",
  "steps": [
    {
      "name": "step1",
      "module": "nonexistent.module",
      "class": "Step"
    }
  ],
  "flow": {
    "start_at": "step2"
  }
}
```

Output:
```
✗ Pipeline validation FAILED

Errors (2):
  • Step 'step1' module 'nonexistent.module' cannot be imported: No module named 'nonexistent'
  • Start step 'step2' is not defined in steps
```

**Unreachable steps warning:**
```json
{
  "name": "test",
  "steps": [
    {"name": "step1", "module": "test", "class": "Step"},
    {"name": "step2", "module": "test", "class": "Step"},
    {"name": "step3", "module": "test", "class": "Step"}
  ],
  "flow": {
    "start_at": "step1",
    "paths": [
      {"from_step": "step1", "to_step": "step2"}
    ]
  }
}
```

Output:
```
✓ Pipeline validation PASSED

Warnings (1):
  • Unreachable steps: step3
```

---

### 2. Format

Format and prettify pipeline JSON files with consistent indentation and structure.

#### Usage

```bash
ia-modules format <pipeline-file> [options]
```

#### Options

- `--in-place` - Edit file in place instead of outputting to stdout

#### Examples

**Format to stdout:**
```bash
ia-modules format pipeline.json
```

**Format in place:**
```bash
ia-modules format pipeline.json --in-place
```

Output:
```
Formatted pipeline.json
```

#### What Gets Formatted

- Consistent 2-space indentation
- Sorted keys (optional)
- Proper JSON structure
- Trailing newline

---

### 3. Visualize

Generate visual representations of pipeline flow using graphviz.

#### Usage

```bash
ia-modules visualize <pipeline-file> [options]
```

#### Options

- `--output <file>` - Output file path (default: `pipeline.<format>`)
- `--format <format>` - Output format: `png`, `svg`, `pdf`, `dot` (default: `png`)

#### Examples

**Generate PNG visualization:**
```bash
ia-modules visualize pipeline.json
```

Output:
```
Pipeline visualization saved to: pipeline.png
```

**Generate SVG with custom output:**
```bash
ia-modules visualize pipeline.json --format svg --output diagram.svg
```

Output:
```
Pipeline visualization saved to: diagram.svg
```

#### Visualization Features

##### Node Styling
- **Standard steps**: Light blue boxes
- **Steps with error handling**: Light green boxes
- **Parallel steps**: Light yellow boxes
- **Start marker**: Gray circle
- **End markers**: Light coral circles

##### Edge Labels
- Show conditions when present
- Format based on condition type:
  - `field == value` for `field_equals`
  - `exists(field)` for `field_exists`
  - `field > value` for `field_greater_than`
  - `custom: function_name` for custom conditions

##### Graph Layout
- Top-to-bottom flow
- Clean, readable diagrams
- Step class shown in node labels

#### Requirements

Visualization requires graphviz:

```bash
pip install graphviz

# Also install system graphviz:
# Ubuntu/Debian: apt-get install graphviz
# macOS: brew install graphviz
# Windows: Download from https://graphviz.org/download/
```

---

## Integration with CI/CD

### GitHub Actions

```yaml
name: Validate Pipelines

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -e ia_modules

      - name: Validate all pipelines
        run: |
          find pipelines -name "*.json" -exec ia-modules validate --strict {} \;
```

### Pre-commit Hook

`.git/hooks/pre-commit`:

```bash
#!/bin/bash

# Find all modified pipeline JSON files
PIPELINES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.json$')

if [ -z "$PIPELINES" ]; then
  exit 0
fi

# Validate each pipeline
for pipeline in $PIPELINES; do
  echo "Validating $pipeline..."
  ia-modules validate --strict "$pipeline"
  if [ $? -ne 0 ]; then
    echo "❌ Pipeline validation failed: $pipeline"
    exit 1
  fi
done

echo "✅ All pipelines valid"
exit 0
```

---

## Common Validation Errors

### 1. Module Import Failures

**Error:**
```
Step 'my_step' module 'my_module.steps' cannot be imported: No module named 'my_module'
```

**Fixes:**
- Ensure module path is correct
- Verify module is installed
- Check PYTHONPATH includes module location

### 2. Undefined Step References

**Error:**
```
Start step 'nonexistent_step' is not defined in steps
```

**Fixes:**
- Check step name spelling
- Ensure step is defined in `steps` array
- Verify flow references match step names

### 3. Unreachable Steps

**Warning:**
```
Unreachable steps: step3, step4
```

**Fixes:**
- Add paths/transitions to unreachable steps
- Remove unused steps
- Verify flow graph connectivity

### 4. Undefined Parameters

**Warning:**
```
Template references undefined parameter: 'api_key'
```

**Fixes:**
- Add parameter to `parameters` section
- Check template syntax: `{{ parameters.api_key }}`
- Verify parameter name spelling

### 5. Flow Cycles

**Warning:**
```
Flow contains cycles: step1 -> step2 -> step1
```

**Fixes:**
- Review flow logic
- Add termination conditions
- Break circular dependencies

---

## Testing

The CLI tool has comprehensive test coverage:

### Unit Tests (73 tests)

```bash
cd ia_modules
pytest tests/unit/test_cli_validate.py -v
pytest tests/unit/test_cli_main.py -v
```

**Coverage:**
- Validation logic (48 tests)
  - Structure validation (9 tests)
  - Step validation (9 tests)
  - Input validation (4 tests)
  - Flow validation (9 tests)
  - Condition validation (3 tests)
  - Template validation (4 tests)
  - Strict mode (2 tests)
  - Complex pipelines (3 tests)
  - ValidationResult (5 tests)
- CLI commands (25 tests)
  - Argument parsing (11 tests)
  - Validate command (6 tests)
  - Format command (4 tests)
  - Visualize command (3 tests)
  - Error handling (1 test)

### Integration Testing

```bash
# Create test pipeline
cat > test_pipeline.json << 'EOF'
{
  "name": "test_pipeline",
  "steps": [
    {
      "name": "step1",
      "module": "ia_modules.pipeline.core",
      "class": "Step"
    }
  ],
  "flow": {
    "start_at": "step1",
    "paths": [
      {"from_step": "step1", "to_step": "end_with_success"}
    ]
  }
}
EOF

# Test validation
ia-modules validate test_pipeline.json
ia-modules validate test_pipeline.json --strict
ia-modules validate test_pipeline.json --json

# Test formatting
ia-modules format test_pipeline.json
ia-modules format test_pipeline.json --in-place

# Test visualization (if graphviz installed)
ia-modules visualize test_pipeline.json --output test.png
```

---

## API Usage

The CLI tool can also be used programmatically:

```python
from ia_modules.cli.validate import validate_pipeline
from ia_modules.cli.visualize import visualize_pipeline

# Validate pipeline data
pipeline_data = {
    "name": "my_pipeline",
    "steps": [...],
    "flow": {...}
}

result = validate_pipeline(pipeline_data, strict=False)
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")

# Generate visualization
visualize_pipeline(
    pipeline_data,
    output_path="pipeline.png",
    format="png"
)
```

---

## Performance

### Validation Performance

Benchmark results (10 iterations):

| Pipeline Type | Steps | Validation Time | Memory |
|---------------|-------|-----------------|--------|
| Simple | 5 | ~15ms | <2MB |
| Medium | 20 | ~35ms | ~3MB |
| Complex | 50 | ~80ms | ~5MB |
| Large | 100 | ~150ms | ~8MB |

**Success Criteria**: ✅ <100ms validation time (achieved for 90% of pipelines)

### Visualization Performance

| Pipeline Type | Steps | Generation Time | File Size |
|---------------|-------|-----------------|-----------|
| Simple | 5 | ~200ms | ~15KB |
| Medium | 20 | ~500ms | ~40KB |
| Complex | 50 | ~1.2s | ~90KB |

---

## Roadmap

### Completed ✅
- JSON schema validation
- Step import checking
- Flow validation (reachability, cycles)
- Template validation
- Error reporting with rich formatting
- Format command
- Visualize command
- 73 comprehensive tests

### Future Enhancements 🔮

#### Week 2: Advanced Features
- [ ] Dry-run simulation
- [ ] Init command (pipeline templates)
- [ ] Diff command (compare pipelines)
- [ ] Documentation generation

#### Later
- [ ] Interactive validation mode
- [ ] Auto-fix suggestions
- [ ] Custom validation rules
- [ ] IDE integration (VS Code extension)

---

## Troubleshooting

### Command Not Found

```bash
ia-modules: command not found
```

**Fix:**
```bash
# Ensure package is installed
pip install -e ia_modules

# Or use python -m
python -m ia_modules.cli.main validate pipeline.json
```

### Import Errors During Validation

```
ModuleNotFoundError: No module named 'my_steps'
```

**Fix:**
```bash
# Set PYTHONPATH
export PYTHONPATH=/path/to/modules:$PYTHONPATH
ia-modules validate pipeline.json
```

### Graphviz Not Found

```
ImportError: Graphviz is required for visualization
```

**Fix:**
```bash
# Install Python package
pip install graphviz

# Install system package
# Ubuntu: sudo apt-get install graphviz
# macOS: brew install graphviz
# Windows: Download from https://graphviz.org
```

---

## Contributing

### Adding New Validation Rules

1. Add validation logic to `cli/validate.py`:
```python
def _validate_my_feature(self) -> None:
    """Validate my feature"""
    if 'my_feature' in self.pipeline_data:
        # Add validation logic
        if not valid:
            self.result.add_error("My feature is invalid")
```

2. Call from `validate()` method:
```python
def validate(self) -> ValidationResult:
    self._validate_structure()
    self._validate_steps()
    self._validate_flow()
    self._validate_my_feature()  # Add here
    return self.result
```

3. Add tests to `tests/unit/test_cli_validate.py`:
```python
class TestMyFeatureValidation:
    def test_valid_my_feature(self):
        # Test valid case
        ...

    def test_invalid_my_feature(self):
        # Test invalid case
        ...
```

### Adding New Commands

1. Add command parser in `cli/main.py`:
```python
my_parser = subparsers.add_parser('mycommand', help='My command')
my_parser.add_argument('pipeline', type=str)
```

2. Add command handler:
```python
def cmd_mycommand(args) -> int:
    # Implement command logic
    return 0
```

3. Add to dispatcher:
```python
if args.command == 'mycommand':
    return cmd_mycommand(args)
```

4. Add tests to `tests/unit/test_cli_main.py`.

---

## License

Part of the IA Modules project.

## Support

For issues and questions:
- Check this documentation
- Run with `--help` flag
- Review test examples in `tests/unit/test_cli_*.py`
- Check integration tests in `tests/integration/test_importer_integration.py`
