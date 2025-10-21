# Plugin System Documentation

**Version**: 0.3.0
**Status**: ✅ Production Ready
**Tests**: 18/18 passing (100%)

---

## 🎯 Overview

The IA Modules Plugin System provides an extensible architecture for adding custom functionality to pipelines. Plugins can implement conditions, steps, validators, hooks, and more.

### Key Features

✅ **Type-Safe Plugin Interface** - Base classes for all plugin types
✅ **Automatic Discovery** - Load plugins from directories
✅ **Dependency Management** - Handle plugin dependencies
✅ **Registry System** - Central plugin management
✅ **Decorator Support** - Simple plugin creation with `@plugin`
✅ **Built-in Plugins** - 15+ ready-to-use plugins

---

## 📦 Plugin Types

### 1. Condition Plugins

Custom conditions for pipeline routing.

```python
from ia_modules.plugins import ConditionPlugin, PluginMetadata, PluginType

class MyCondition(ConditionPlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="my_condition",
            version="1.0.0",
            description="My custom condition",
            plugin_type=PluginType.CONDITION
        )

    async def evaluate(self, data: dict) -> bool:
        # Your logic here
        return data.get('value', 0) > 10
```

### 2. Step Plugins

Custom processing steps.

```python
from ia_modules.plugins import StepPlugin

class MyStep(StepPlugin):
    async def execute(self, data: dict) -> dict:
        # Process data
        data['processed'] = True
        return data
```

### 3. Validator Plugins

Data validation logic.

```python
from ia_modules.plugins import ValidatorPlugin

class MyValidator(ValidatorPlugin):
    async def validate(self, data: dict) -> tuple[bool, str]:
        if 'required_field' not in data:
            return False, "Missing required_field"
        return True, None
```

### 4. Transform Plugins

Data transformers.

```python
from ia_modules.plugins import TransformPlugin

class MyTransform(TransformPlugin):
    async def transform(self, data: dict) -> dict:
        # Transform data
        data['transformed'] = True
        return data
```

### 5. Hook Plugins

Lifecycle event handlers.

```python
from ia_modules.plugins import HookPlugin

class MyHook(HookPlugin):
    async def on_pipeline_start(self, pipeline_name: str, data: dict):
        print(f"Pipeline {pipeline_name} starting")

    async def on_step_end(self, step_name: str, result: dict):
        print(f"Step {step_name} completed")
```

---

## 🚀 Quick Start

### Using Decorators

The simplest way to create plugins:

```python
from ia_modules.plugins import condition_plugin, ConditionPlugin

@condition_plugin(
    name="temperature_check",
    description="Check if temperature is comfortable",
    version="1.0.0",
    tags=["weather", "temperature"]
)
class TemperatureCheck(ConditionPlugin):
    async def evaluate(self, data: dict) -> bool:
        temp = data.get('temperature', 0)
        return 18 <= temp <= 26
```

### Function-Based Plugins

Even simpler for basic conditions:

```python
from ia_modules.plugins import function_plugin, PluginType

@function_plugin(
    name="is_weekend",
    description="Check if it's the weekend",
    plugin_type=PluginType.CONDITION
)
async def is_weekend(data: dict) -> bool:
    from datetime import datetime
    return datetime.now().weekday() >= 5  # Saturday or Sunday
```

---

## 📚 Plugin Registry

### Registering Plugins

```python
from ia_modules.plugins import get_registry

# Get global registry
registry = get_registry()

# Register plugin class
registry.register(MyPlugin)

# Register with custom config
registry.register(MyPlugin, config={'key': 'value'})
```

### Using Plugins

```python
# Get plugin instance
plugin = registry.get("my_plugin")

# Create new instance with custom config
instance = registry.create_instance("my_plugin", {'custom': 'config'})

# List all plugins
all_plugins = registry.list_plugins()

# List by type
from ia_modules.plugins import PluginType
conditions = registry.list_plugins(PluginType.CONDITION)
```

### Plugin Information

```python
# Get info about specific plugin
info = registry.get_info("my_plugin")
print(f"Name: {info['name']}")
print(f"Version: {info['version']}")
print(f"Type: {info['type']}")

# Get all plugin info
all_info = registry.get_all_info()
```

---

## 🔍 Plugin Discovery

### Auto-Discovery

Load plugins from default locations:

```python
from ia_modules.plugins import PluginLoader

loader = PluginLoader()
count = loader.auto_discover()
print(f"Loaded {count} plugins")
```

### Load from Directory

```python
from pathlib import Path

# Load all plugins from directory
loader.load_from_directory(Path("/path/to/plugins"))

# Non-recursive
loader.load_from_directory(Path("/path/to/plugins"), recursive=False)
```

### Load from Module

```python
# Load from Python module
loader.load_from_module("my_app.plugins")
```

### Default Plugin Locations

1. Built-in: `ia_modules/plugins/builtin/`
2. User directory: `~/.ia_modules/plugins/`
3. Current directory: `./plugins/`
4. Environment: `$IA_PLUGIN_PATH`

---

## 🔧 Plugin Configuration

### Config Schema

```python
class MyPlugin(ConditionPlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            config_schema={
                "type": "object",
                "properties": {
                    "threshold": {"type": "number"},
                    "enabled": {"type": "boolean"}
                },
                "required": ["threshold"]
            }
        )
```

### Accessing Config

```python
class MyPlugin(ConditionPlugin):
    async def evaluate(self, data: dict) -> bool:
        threshold = self.config.get('threshold', 0)
        enabled = self.config.get('enabled', True)

        if not enabled:
            return True

        return data.get('value', 0) > threshold
```

---

## 📦 Built-in Plugins

### Weather Plugins

**weather_condition**: Route based on weather
**is_good_weather**: Check if weather is suitable for activities

```python
# Usage in pipeline
{
    "condition": {
        "type": "plugin",
        "plugin": "weather_condition",
        "config": {
            "condition": "temperature_above",
            "threshold": 20
        }
    }
}
```

### Database Plugins

**database_record_exists**: Check if record exists
**database_value_condition**: Compare database values

```python
{
    "condition": {
        "type": "plugin",
        "plugin": "database_record_exists",
        "config": {
            "table": "users",
            "id_field": "user_id",
            "id_key": "id"
        }
    }
}
```

### API Plugins

**api_status_condition**: Check API response status
**api_data_condition**: Validate API response data
**api_call_step**: Make HTTP API calls

```python
{
    "condition": {
        "type": "plugin",
        "plugin": "api_status_condition",
        "config": {
            "url": "https://api.example.com/health",
            "expected_status": 200
        }
    }
}
```

### Time Plugins

**business_hours**: Check if within business hours
**time_range**: Check if within time range
**day_of_week**: Check specific day

```python
{
    "condition": {
        "type": "plugin",
        "plugin": "business_hours",
        "config": {
            "start_hour": 9,
            "end_hour": 17,
            "weekdays_only": true
        }
    }
}
```

### Validation Plugins

**email_validator**: Validate email format
**range_validator**: Validate numeric range
**regex_validator**: Validate using regex
**schema_validator**: Validate against JSON schema

```python
{
    "condition": {
        "type": "plugin",
        "plugin": "email_validator",
        "config": {
            "field": "email"
        }
    }
}
```

---

## 🔗 Plugin Dependencies

### Declaring Dependencies

```python
@condition_plugin(
    name="advanced_plugin",
    version="1.0.0",
    dependencies=["basic_plugin", "helper_plugin"]
)
class AdvancedPlugin(ConditionPlugin):
    async def evaluate(self, data: dict) -> bool:
        # Can safely use other plugins
        pass
```

### Checking Dependencies

```python
from ia_modules.plugins import get_registry

registry = get_registry()

# Check if dependencies satisfied
satisfied, missing = registry.check_dependencies("advanced_plugin")
if not satisfied:
    print(f"Missing dependencies: {missing}")

# Get plugins in dependency order
ordered = registry.get_dependency_order()
```

---

## 🏗️ Creating Custom Plugins

### Step 1: Choose Plugin Type

```python
from ia_modules.plugins import ConditionPlugin, StepPlugin, ValidatorPlugin
```

### Step 2: Implement Interface

```python
class WeatherCondition(ConditionPlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="weather_condition",
            version="1.0.0",
            description="Check weather conditions",
            tags=["weather"]
        )

    async def evaluate(self, data: dict) -> bool:
        # Implementation
        return True
```

### Step 3: Add Initialization (Optional)

```python
class APIPlugin(ConditionPlugin):
    async def _initialize(self):
        # Setup connections, load resources
        self.api_client = APIClient(self.config['api_key'])

    async def _shutdown(self):
        # Cleanup
        await self.api_client.close()
```

### Step 4: Register Plugin

```python
from ia_modules.plugins import get_registry

registry = get_registry()
registry.register(WeatherCondition)
```

---

## 📝 Best Practices

### 1. Naming Conventions

- Use descriptive, lowercase names with underscores
- Include plugin type in name: `*_condition`, `*_step`, `*_validator`
- Example: `temperature_check_condition`

### 2. Error Handling

```python
class SafePlugin(ConditionPlugin):
    async def evaluate(self, data: dict) -> bool:
        try:
            return self._do_evaluation(data)
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return False  # Safe default
```

### 3. Configuration Validation

```python
class ConfiguredPlugin(ConditionPlugin):
    def validate_config(self) -> bool:
        required = ['threshold', 'field']
        for key in required:
            if key not in self.config:
                self.logger.error(f"Missing required config: {key}")
                return False
        return True
```

### 4. Logging

```python
class LoggingPlugin(ConditionPlugin):
    async def evaluate(self, data: dict) -> bool:
        self.logger.debug(f"Evaluating with data: {data}")
        result = data.get('value', 0) > 10
        self.logger.info(f"Evaluation result: {result}")
        return result
```

### 5. Testing

```python
import pytest

@pytest.mark.asyncio
async def test_my_plugin():
    plugin = MyPlugin({'threshold': 10})

    # Test positive case
    assert await plugin.evaluate({'value': 15}) is True

    # Test negative case
    assert await plugin.evaluate({'value': 5}) is False
```

---

## 🔌 Integration with Pipelines

### Using Plugins in Pipeline JSON

```json
{
  "name": "weather_pipeline",
  "steps": [
    {
      "name": "check_weather",
      "module": "my_steps",
      "class": "WeatherFetchStep"
    },
    {
      "name": "good_weather_path",
      "module": "my_steps",
      "class": "OutdoorActivityStep"
    },
    {
      "name": "bad_weather_path",
      "module": "my_steps",
      "class": "IndoorActivityStep"
    }
  ],
  "flow": {
    "start_at": "check_weather",
    "transitions": [
      {
        "from": "check_weather",
        "to": "good_weather_path",
        "condition": {
          "type": "plugin",
          "plugin": "is_good_weather"
        }
      },
      {
        "from": "check_weather",
        "to": "bad_weather_path",
        "condition": {
          "type": "always"
        }
      }
    ]
  }
}
```

### Programmatic Usage

```python
from ia_modules.plugins import get_registry

# Get plugin
registry = get_registry()
condition = registry.get("is_good_weather")

# Evaluate
data = {
    'weather': {
        'temperature': 22,
        'condition': 'sunny'
    }
}

result = await condition.evaluate(data)
print(f"Good weather: {result}")
```

---

## 🧪 Testing Plugins

### Unit Tests

```python
import pytest
from ia_modules.plugins import ConditionPlugin

class TestMyPlugin:
    @pytest.mark.asyncio
    async def test_positive_case(self):
        plugin = MyPlugin({'threshold': 10})
        result = await plugin.evaluate({'value': 15})
        assert result is True

    @pytest.mark.asyncio
    async def test_negative_case(self):
        plugin = MyPlugin({'threshold': 10})
        result = await plugin.evaluate({'value': 5})
        assert result is False

    @pytest.mark.asyncio
    async def test_edge_case(self):
        plugin = MyPlugin({'threshold': 10})
        result = await plugin.evaluate({'value': 10})
        assert result is False  # Not greater than threshold
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_plugin_in_pipeline():
    from ia_modules.pipeline import GraphPipelineRunner

    # Load plugin
    registry.register(MyPlugin)

    # Create pipeline using plugin
    pipeline_data = {
        # ... pipeline definition with plugin condition
    }

    runner = GraphPipelineRunner(pipeline_data)
    result = await runner.run({'value': 15})

    assert result['success'] is True
```

---

## 📊 Plugin Statistics

### Project Status

| Metric | Value |
|--------|-------|
| Total Plugin Types | 6 |
| Built-in Plugins | 15+ |
| Test Coverage | 18/18 (100%) |
| Lines of Code | ~2,000 |

### Plugin Types Distribution

```
Condition Plugins: 11
Step Plugins: 1
Validator Plugins: 4
Transform Plugins: 0
Hook Plugins: 0
Reporter Plugins: 0
```

---

## 🔮 Future Enhancements

### Planned Features

- [ ] Plugin versioning and updates
- [ ] Plugin marketplace/registry
- [ ] Hot-reloading of plugins
- [ ] Plugin sandboxing
- [ ] Performance metrics per plugin
- [ ] Plugin documentation generator

---

## 💡 Examples

### Example 1: Custom Business Logic

```python
@condition_plugin(
    name="customer_tier_check",
    description="Check customer tier for premium features"
)
class CustomerTierCheck(ConditionPlugin):
    async def evaluate(self, data: dict) -> bool:
        customer = data.get('customer', {})
        required_tier = self.config.get('tier', 'gold')

        tier_hierarchy = ['bronze', 'silver', 'gold', 'platinum']
        customer_tier = customer.get('tier', 'bronze')

        return (
            tier_hierarchy.index(customer_tier) >=
            tier_hierarchy.index(required_tier)
        )
```

### Example 2: External API Integration

```python
@step_plugin(
    name="enrich_with_crm",
    description="Enrich data with CRM information"
)
class CRMEnrichmentStep(StepPlugin):
    async def _initialize(self):
        self.crm_client = CRMClient(self.config['api_key'])

    async def execute(self, data: dict) -> dict:
        customer_id = data.get('customer_id')

        if customer_id:
            crm_data = await self.crm_client.get_customer(customer_id)
            data['crm_data'] = crm_data

        return data

    async def _shutdown(self):
        await self.crm_client.close()
```

### Example 3: Validation Chain

```python
class ValidationChain(ValidatorPlugin):
    async def validate(self, data: dict) -> tuple[bool, str]:
        # Check required fields
        required = ['email', 'name', 'age']
        for field in required:
            if field not in data:
                return False, f"Missing {field}"

        # Validate email
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(email_pattern, data['email']):
            return False, "Invalid email format"

        # Validate age
        if not (18 <= data['age'] <= 120):
            return False, "Age must be between 18 and 120"

        return True, None
```

---

## 📄 License

Part of the IA Modules project.

---

## 🤝 Contributing

To contribute a plugin:

1. Create plugin class
2. Add tests
3. Update documentation
4. Submit PR

---

**Status**: ✅ Production Ready
**Version**: 0.3.0
**Tests**: 18/18 passing (100%)
