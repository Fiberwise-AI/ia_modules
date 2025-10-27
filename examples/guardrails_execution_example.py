"""
Execution Rails Example

Demonstrates tool and code execution safety with execution rails.
"""
import asyncio
from ia_modules.guardrails import GuardrailConfig, RailType, RailAction
from ia_modules.guardrails.execution_rails import (
    ToolValidationRail,
    CodeExecutionSafetyRail,
    ParameterValidationRail,
    ResourceLimitRail
)


async def test_tool_validation():
    """Test tool validation rail."""
    print("\n=== Tool Validation Rail ===")

    config = GuardrailConfig(
        name="tool_validation",
        type=RailType.EXECUTION
    )

    rail = ToolValidationRail(
        config,
        allowed_tools=["search", "calculator", "weather"],
        blocked_tools=["delete_database", "send_money"],
        require_confirmation=["send_email"]
    )

    # Test 1: Allowed tool
    result1 = await rail.execute(
        {"tool_name": "search", "query": "Python tutorials"},
        context={"tool_name": "search"}
    )
    print(f"\nTest 1 - Allowed tool (search):")
    print(f"  Action: {result1.action.value}")
    print(f"  Tool: {result1.metadata.get('tool_name')}")

    # Test 2: Blocked tool
    result2 = await rail.execute(
        {"tool_name": "delete_database"},
        context={"tool_name": "delete_database"}
    )
    print(f"\nTest 2 - Blocked tool:")
    print(f"  Action: {result2.action.value}")
    print(f"  Triggered: {result2.triggered}")
    print(f"  Reason: {result2.reason}")

    # Test 3: Requires confirmation
    result3 = await rail.execute(
        {"tool_name": "send_email", "to": "user@example.com"},
        context={"tool_name": "send_email"}
    )
    print(f"\nTest 3 - Requires confirmation:")
    print(f"  Action: {result3.action.value}")
    print(f"  Triggered: {result3.triggered}")
    print(f"  Reason: {result3.reason}")

    # Test 4: Not in allowed list
    result4 = await rail.execute(
        {"tool_name": "unknown_tool"},
        context={"tool_name": "unknown_tool"}
    )
    print(f"\nTest 4 - Not in allowed list:")
    print(f"  Action: {result4.action.value}")
    print(f"  Triggered: {result4.triggered}")

    print(f"\nStatistics: {rail.get_stats()}")


async def test_code_execution_safety():
    """Test code execution safety rail."""
    print("\n=== Code Execution Safety Rail ===")

    config = GuardrailConfig(
        name="code_safety",
        type=RailType.EXECUTION
    )

    rail = CodeExecutionSafetyRail(
        config,
        allow_file_read=True,
        allow_network=False
    )

    # Test 1: Safe code
    safe_code = """
import math

def calculate_circle_area(radius):
    return math.pi * radius ** 2

result = calculate_circle_area(5)
print(result)
"""
    result1 = await rail.execute(safe_code)
    print(f"\nTest 1 - Safe code:")
    print(f"  Action: {result1.action.value}")
    print(f"  Imports: {result1.metadata.get('imports', [])}")

    # Test 2: Dangerous code (eval)
    dangerous_code1 = """
user_input = input("Enter expression: ")
result = eval(user_input)
"""
    result2 = await rail.execute(dangerous_code1)
    print(f"\nTest 2 - Dangerous code (eval):")
    print(f"  Action: {result2.action.value}")
    print(f"  Triggered: {result2.triggered}")
    print(f"  Reason: {result2.reason}")

    # Test 3: Dangerous code (system call)
    dangerous_code2 = """
import os
os.system('rm -rf /')
"""
    result3 = await rail.execute(dangerous_code2)
    print(f"\nTest 3 - Dangerous code (system call):")
    print(f"  Action: {result3.action.value}")
    print(f"  Triggered: {result3.triggered}")
    print(f"  Total matches: {result3.metadata.get('total_matches', 0)}")

    # Test 4: Unsafe imports
    unsafe_import_code = """
import subprocess
import requests

subprocess.run(['ls', '-la'])
"""
    result4 = await rail.execute(unsafe_import_code)
    print(f"\nTest 4 - Unsafe imports:")
    print(f"  Action: {result4.action.value}")
    print(f"  Triggered: {result4.triggered}")
    print(f"  Reason: {result4.reason}")

    print(f"\nStatistics: {rail.get_stats()}")


async def test_parameter_validation():
    """Test parameter validation rail."""
    print("\n=== Parameter Validation Rail ===")

    config = GuardrailConfig(
        name="param_validation",
        type=RailType.EXECUTION
    )

    schemas = {
        "send_email": {
            "to": {"type": "email", "required": True},
            "subject": {"type": "string", "required": True, "max_length": 100},
            "body": {"type": "string", "max_length": 5000}
        },
        "transfer_money": {
            "amount": {"type": "number", "required": True, "min": 0, "max": 10000},
            "to_account": {"type": "string", "required": True}
        }
    }

    rail = ParameterValidationRail(config, parameter_schemas=schemas)

    # Test 1: Valid parameters
    params1 = {
        "to": "user@example.com",
        "subject": "Hello",
        "body": "Test message"
    }
    result1 = await rail.execute(params1, context={"tool_name": "send_email"})
    print(f"\nTest 1 - Valid parameters:")
    print(f"  Action: {result1.action.value}")
    print(f"  Validated params: {result1.metadata.get('validated_params', [])}")

    # Test 2: Missing required parameter
    params2 = {
        "subject": "Hello"
        # Missing 'to'
    }
    result2 = await rail.execute(params2, context={"tool_name": "send_email"})
    print(f"\nTest 2 - Missing required parameter:")
    print(f"  Action: {result2.action.value}")
    print(f"  Triggered: {result2.triggered}")
    print(f"  Errors: {result2.metadata.get('errors', [])}")

    # Test 3: Invalid email format
    params3 = {
        "to": "invalid-email",
        "subject": "Test"
    }
    result3 = await rail.execute(params3, context={"tool_name": "send_email"})
    print(f"\nTest 3 - Invalid email format:")
    print(f"  Action: {result3.action.value}")
    print(f"  Triggered: {result3.triggered}")
    print(f"  Errors: {result3.metadata.get('errors', [])}")

    # Test 4: Number out of range
    params4 = {
        "amount": 50000,  # Exceeds max of 10000
        "to_account": "123456"
    }
    result4 = await rail.execute(params4, context={"tool_name": "transfer_money"})
    print(f"\nTest 4 - Number out of range:")
    print(f"  Action: {result4.action.value}")
    print(f"  Triggered: {result4.triggered}")
    print(f"  Errors: {result4.metadata.get('errors', [])}")

    print(f"\nStatistics: {rail.get_stats()}")


async def test_resource_limits():
    """Test resource limit rail."""
    print("\n=== Resource Limit Rail ===")

    config = GuardrailConfig(
        name="resource_limits",
        type=RailType.EXECUTION
    )

    rail = ResourceLimitRail(
        config,
        max_execution_time=5.0,
        max_iterations=1000
    )

    # Test 1: Safe code
    safe_code = """
for i in range(10):
    print(i)
"""
    result1 = await rail.execute(safe_code)
    print(f"\nTest 1 - Safe code:")
    print(f"  Action: {result1.action.value}")

    # Test 2: Potential infinite loop
    infinite_loop = """
while True:
    print("Running...")
"""
    result2 = await rail.execute(infinite_loop)
    print(f"\nTest 2 - Potential infinite loop:")
    print(f"  Action: {result2.action.value}")
    print(f"  Triggered: {result2.triggered}")
    print(f"  Reason: {result2.reason}")

    # Test 3: Too many iterations
    large_range = """
for i in range(10000):
    process(i)
"""
    result3 = await rail.execute(large_range)
    print(f"\nTest 3 - Too many iterations:")
    print(f"  Action: {result3.action.value}")
    print(f"  Triggered: {result3.triggered}")
    print(f"  Warnings: {result3.metadata.get('warnings', [])}")

    # Test 4: Execution time exceeded (from context)
    result4 = await rail.execute(
        "some_code",
        context={"execution_time": 10.5}
    )
    print(f"\nTest 4 - Execution time exceeded:")
    print(f"  Action: {result4.action.value}")
    print(f"  Triggered: {result4.triggered}")
    print(f"  Reason: {result4.reason}")

    print(f"\nStatistics: {rail.get_stats()}")


async def test_execution_pipeline():
    """Test complete execution pipeline with all rails."""
    print("\n=== Complete Execution Pipeline ===")

    # Create all execution rails
    tool_rail = ToolValidationRail(
        GuardrailConfig(name="tool", type=RailType.EXECUTION),
        allowed_tools=["execute_code", "run_query"],
        blocked_tools=["delete_all"]
    )

    code_rail = CodeExecutionSafetyRail(
        GuardrailConfig(name="code", type=RailType.EXECUTION),
        allow_file_read=True,
        allow_network=False
    )

    param_rail = ParameterValidationRail(
        GuardrailConfig(name="params", type=RailType.EXECUTION),
        parameter_schemas={
            "execute_code": {
                "code": {"type": "string", "required": True, "max_length": 1000}
            }
        }
    )

    resource_rail = ResourceLimitRail(
        GuardrailConfig(name="resources", type=RailType.EXECUTION),
        max_iterations=100
    )

    # Test execution request
    code_to_execute = """
import math

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
"""

    print("\nProcessing execution request through pipeline...")

    # Step 1: Tool validation
    tool_result = await tool_rail.execute(
        {"tool_name": "execute_code"},
        context={"tool_name": "execute_code"}
    )
    print(f"\n  Step 1 - Tool validation: {tool_result.action.value}")

    if tool_result.action == RailAction.BLOCK:
        print("  Pipeline stopped: Tool not allowed")
        return

    # Step 2: Parameter validation
    param_result = await param_rail.execute(
        {"code": code_to_execute},
        context={"tool_name": "execute_code"}
    )
    print(f"  Step 2 - Parameter validation: {param_result.action.value}")

    if param_result.action == RailAction.BLOCK:
        print(f"  Pipeline stopped: {param_result.reason}")
        return

    # Step 3: Code safety
    code_result = await code_rail.execute(code_to_execute)
    print(f"  Step 3 - Code safety: {code_result.action.value}")

    if code_result.action == RailAction.BLOCK:
        print(f"  Pipeline stopped: {code_result.reason}")
        return

    # Step 4: Resource limits
    resource_result = await resource_rail.execute(code_to_execute)
    print(f"  Step 4 - Resource limits: {resource_result.action.value}")

    if resource_result.action == RailAction.BLOCK:
        print(f"  Pipeline stopped: {resource_result.reason}")
        return

    print("\n  Pipeline result: All checks passed - SAFE TO EXECUTE")

    # Print statistics
    print("\n  Rail Statistics:")
    print(f"    Tool validation: {tool_rail.get_stats()['trigger_rate']:.1%} trigger rate")
    print(f"    Code safety: {code_rail.get_stats()['trigger_rate']:.1%} trigger rate")
    print(f"    Parameter validation: {param_rail.get_stats()['trigger_rate']:.1%} trigger rate")
    print(f"    Resource limits: {resource_rail.get_stats()['trigger_rate']:.1%} trigger rate")


async def main():
    """Run all execution rail examples."""
    print("Execution Rails Example")
    print("=" * 50)

    await test_tool_validation()
    await test_code_execution_safety()
    await test_parameter_validation()
    await test_resource_limits()
    await test_execution_pipeline()

    print("\n" + "=" * 50)
    print("All execution rail tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
