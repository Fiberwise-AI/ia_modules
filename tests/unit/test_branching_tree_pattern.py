"""
Unit tests for branching tree execution pattern.

This tests the specific pattern where one step branches to multiple terminal steps,
some of which execute conditionally. Example from ia_chat_app simple_chat pipeline:

    fetch_history
        ├─→ chat (always)
        └─→ generate_title (if is_first_message)

Both 'chat' and 'generate_title' are terminal branches with no further steps.
"""

import pytest
from ia_modules.pipeline.graph_pipeline_runner import GraphPipelineRunner
from ia_modules.pipeline.services import ServiceRegistry


@pytest.mark.asyncio
class TestBranchingTreePattern:
    """Test suite for branching tree execution patterns"""

    async def test_terminal_branches_both_execute(self):
        """Test that both terminal branches execute when condition is met"""

        config = {
            "name": "Branching Tree Test",
            "steps": [
                {"id": "step1", "name": "step1", "step_class": "RootStep",
                 "module": "tests.pipelines.branching_tree_pipeline.steps.test_steps"},
                {"id": "step2", "name": "step2", "step_class": "AlwaysBranchStep",
                 "module": "tests.pipelines.branching_tree_pipeline.steps.test_steps"},
                {"id": "step3", "name": "step3", "step_class": "ConditionalBranchStep",
                 "module": "tests.pipelines.branching_tree_pipeline.steps.test_steps"}
            ],
            "flow": {
                "start_at": "step1",
                "paths": [
                    {"from": "step1", "to": "step2", "condition": {"type": "always"}},
                    {"from": "step1", "to": "step3", "condition": {
                        "type": "expression",
                        "config": {"source": "step1.should_branch", "operator": "equals", "value": True}
                    }}
                ]
            }
        }

        services = ServiceRegistry()
        runner = GraphPipelineRunner(services)

        # Run with should_branch=True to trigger conditional branch
        result = await runner.run_pipeline_from_json(config, {"value": "test", "should_branch": True})

        # Verify all three steps executed
        assert "steps" in result
        step_names = [step["step_name"] for step in result["steps"]]
        assert "step1" in step_names
        assert "step2" in step_names
        assert "step3" in step_names
        assert len(result["steps"]) == 3

    async def test_conditional_branch_skipped_when_condition_false(self):
        """Test that conditional branch is skipped when condition is not met"""

        config = {
            "name": "Branching Tree Test",
            "steps": [
                {"id": "step1", "name": "step1", "step_class": "RootStep",
                 "module": "tests.pipelines.branching_tree_pipeline.steps.test_steps"},
                {"id": "step2", "name": "step2", "step_class": "AlwaysBranchStep",
                 "module": "tests.pipelines.branching_tree_pipeline.steps.test_steps"},
                {"id": "step3", "name": "step3", "step_class": "ConditionalBranchStep",
                 "module": "tests.pipelines.branching_tree_pipeline.steps.test_steps"}
            ],
            "flow": {
                "start_at": "step1",
                "paths": [
                    {"from": "step1", "to": "step2", "condition": {"type": "always"}},
                    {"from": "step1", "to": "step3", "condition": {
                        "type": "expression",
                        "config": {"source": "step1.should_branch", "operator": "equals", "value": True}
                    }}
                ]
            }
        }

        services = ServiceRegistry()
        runner = GraphPipelineRunner(services)

        # Run with should_branch=False to skip conditional branch
        result = await runner.run_pipeline_from_json(config, {"value": "test", "should_branch": False})

        # Verify step1 and step2 executed, but not step3
        assert "steps" in result
        step_names = [step["step_name"] for step in result["steps"]]
        assert "step1" in step_names
        assert "step2" in step_names

        # step3 should NOT have executed
        assert "step3" not in step_names
        assert len(result["steps"]) == 2

    async def test_exact_step_count_with_condition_met(self):
        """Test that exactly 3 steps execute when condition is met"""

        config = {
            "name": "Branching Tree Test",
            "steps": [
                {"id": "step1", "name": "step1", "step_class": "RootStep",
                 "module": "tests.pipelines.branching_tree_pipeline.steps.test_steps"},
                {"id": "step2", "name": "step2", "step_class": "AlwaysBranchStep",
                 "module": "tests.pipelines.branching_tree_pipeline.steps.test_steps"},
                {"id": "step3", "name": "step3", "step_class": "ConditionalBranchStep",
                 "module": "tests.pipelines.branching_tree_pipeline.steps.test_steps"}
            ],
            "flow": {
                "start_at": "step1",
                "paths": [
                    {"from": "step1", "to": "step2", "condition": {"type": "always"}},
                    {"from": "step1", "to": "step3", "condition": {
                        "type": "expression",
                        "config": {"source": "step1.should_branch", "operator": "equals", "value": True}
                    }}
                ]
            }
        }

        services = ServiceRegistry()
        runner = GraphPipelineRunner(services)

        result = await runner.run_pipeline_from_json(config, {"value": "test", "should_branch": True})

        # Assert exactly 3 steps executed
        assert len(result["steps"]) == 3
        step_names = {step["step_name"] for step in result["steps"]}
        assert step_names == {"step1", "step2", "step3"}

    async def test_exact_step_count_with_condition_not_met(self):
        """Test that exactly 2 steps execute when condition is not met"""

        config = {
            "name": "Branching Tree Test",
            "steps": [
                {"id": "step1", "name": "step1", "step_class": "RootStep",
                 "module": "tests.pipelines.branching_tree_pipeline.steps.test_steps"},
                {"id": "step2", "name": "step2", "step_class": "AlwaysBranchStep",
                 "module": "tests.pipelines.branching_tree_pipeline.steps.test_steps"},
                {"id": "step3", "name": "step3", "step_class": "ConditionalBranchStep",
                 "module": "tests.pipelines.branching_tree_pipeline.steps.test_steps"}
            ],
            "flow": {
                "start_at": "step1",
                "paths": [
                    {"from": "step1", "to": "step2", "condition": {"type": "always"}},
                    {"from": "step1", "to": "step3", "condition": {
                        "type": "expression",
                        "config": {"source": "step1.should_branch", "operator": "equals", "value": True}
                    }}
                ]
            }
        }

        services = ServiceRegistry()
        runner = GraphPipelineRunner(services)

        result = await runner.run_pipeline_from_json(config, {"value": "test", "should_branch": False})

        # Assert exactly 2 steps executed
        assert len(result["steps"]) == 2
        step_names = {step["step_name"] for step in result["steps"]}
        assert step_names == {"step1", "step2"}
