"""
Comprehensive unit tests for the prompt_optimization module.

Covers evaluators, reinforcement learning, templates, optimizer base classes,
A/B testing, and genetic optimizer with focus on increasing coverage.
"""

import asyncio
import math
import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ia_modules.prompt_optimization.evaluators import (
    AccuracyEvaluator,
    CoherenceEvaluator,
    CompositeEvaluator,
    EvaluationMetrics,
    LengthEvaluator,
    PromptEvaluator,
    RelevanceEvaluator,
    SpecificityEvaluator,
)
from ia_modules.prompt_optimization.reinforcement import (
    PromptAction,
    RLConfig,
    RLOptimizer,
)
from ia_modules.prompt_optimization.templates import (
    PromptTemplate,
    TemplateComposer,
    TemplateLibrary,
    TemplateVariable,
    VariableType,
)
from ia_modules.prompt_optimization.optimizer import (
    GridSearchOptimizer,
    OptimizationResult,
    OptimizationStrategy,
    PromptCandidate,
    PromptOptimizer,
    RandomSearchOptimizer,
)
from ia_modules.prompt_optimization.ab_testing import (
    ABTestConfig,
    ABTester,
    PromptVariant,
)
from ia_modules.prompt_optimization.genetic import (
    GeneticConfig,
    GeneticOptimizer,
)


# ---------------------------------------------------------------------------
# Helpers / mock evaluators
# ---------------------------------------------------------------------------

class SimpleEvaluator(PromptEvaluator):
    """Returns a fixed score for any prompt."""

    def __init__(self, score: float = 0.5, weight: float = 1.0):
        super().__init__(weight=weight)
        self._score = score

    async def evaluate(self, prompt, context):
        return self._score


class LengthBasedEvaluator(PromptEvaluator):
    """Scores based on prompt length (longer = higher, capped at 1.0)."""

    async def evaluate(self, prompt, context):
        return min(1.0, len(prompt) / 200)


# ===========================================================================
# EVALUATORS
# ===========================================================================


class TestEvaluationMetrics:

    def test_creation(self):
        m = EvaluationMetrics(score=0.9, metrics={"a": 0.8}, feedback="good")
        assert m.score == 0.9
        assert m.feedback == "good"

    def test_to_dict(self):
        m = EvaluationMetrics(score=0.7, metrics={"x": 1.0}, feedback="ok")
        d = m.to_dict()
        assert d["score"] == 0.7
        assert d["metrics"] == {"x": 1.0}
        assert d["feedback"] == "ok"

    def test_default_feedback(self):
        m = EvaluationMetrics(score=0.5, metrics={})
        assert m.feedback == ""


class TestPromptEvaluatorBase:

    async def test_evaluate_detailed_default(self):
        ev = SimpleEvaluator(score=0.6)
        result = await ev.evaluate_detailed("hello", {})
        assert isinstance(result, EvaluationMetrics)
        assert result.score == 0.6
        assert result.metrics == {"overall": 0.6}


class TestAccuracyEvaluator:

    async def test_empty_test_cases(self):
        ev = AccuracyEvaluator(llm_fn=lambda p: "x", test_cases=[], weight=1.0)
        score = await ev.evaluate("prompt", {})
        assert score == 0.0

    async def test_sync_llm_fn_matching(self):
        test_cases = [
            {"input": "hello", "expected": "world"},
            {"input": "foo", "expected": "bar"},
        ]
        ev = AccuracyEvaluator(
            llm_fn=lambda p: "world bar",
            test_cases=test_cases,
        )
        score = await ev.evaluate("{input}", {})
        # Both match because "world" and "bar" are substrings of "world bar"
        assert score == 1.0

    async def test_async_llm_fn(self):
        async def async_llm(prompt):
            return "expected_output"

        test_cases = [{"input": "test", "expected": "expected_output"}]
        ev = AccuracyEvaluator(llm_fn=async_llm, test_cases=test_cases)
        score = await ev.evaluate("{input}", {})
        assert score == 1.0

    async def test_partial_match(self):
        test_cases = [
            {"input": "a", "expected": "yes"},
            {"input": "b", "expected": "no"},
        ]
        ev = AccuracyEvaluator(
            llm_fn=lambda p: "yes",
            test_cases=test_cases,
        )
        score = await ev.evaluate("{input}", {})
        assert score == 0.5

    async def test_no_match(self):
        test_cases = [{"input": "a", "expected": "xyz"}]
        ev = AccuracyEvaluator(llm_fn=lambda p: "abc", test_cases=test_cases)
        score = await ev.evaluate("{input}", {})
        assert score == 0.0

    def test_check_match_case_insensitive(self):
        ev = AccuracyEvaluator(llm_fn=lambda p: "", test_cases=[])
        assert ev._check_match("Hello World", "hello") is True
        assert ev._check_match("Hello World", "xyz") is False


class TestCoherenceEvaluator:

    async def test_good_coherent_prompt(self):
        ev = CoherenceEvaluator()
        prompt = "Please explain the differences between Python and Java. Provide detailed examples."
        score = await ev.evaluate(prompt, {})
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Should score reasonably well

    async def test_very_short_prompt(self):
        ev = CoherenceEvaluator()
        score = await ev.evaluate("Hi", {})
        assert 0.0 <= score <= 1.0

    async def test_medium_length_prompt(self):
        ev = CoherenceEvaluator()
        prompt = "Hello there friend."
        score = await ev.evaluate(prompt, {})
        assert 0.0 <= score <= 1.0

    async def test_prompt_with_instruction_words(self):
        ev = CoherenceEvaluator()
        prompt = "Please provide a detailed analysis of the market trends."
        score = await ev.evaluate(prompt, {})
        assert score > 0.2

    async def test_prompt_with_clarity_words(self):
        ev = CoherenceEvaluator()
        prompt = "Give a clear and specific description of the process."
        score = await ev.evaluate(prompt, {})
        assert score > 0.2

    async def test_prompt_ending_with_punctuation(self):
        ev = CoherenceEvaluator()
        score1 = await ev.evaluate("Describe the process.", {})
        score2 = await ev.evaluate("Describe the process", {})
        # Punctuation ending should help
        assert score1 >= score2

    async def test_prompt_with_repetition(self):
        ev = CoherenceEvaluator()
        prompt = "the the the the the the the the the the the the"
        score = await ev.evaluate(prompt, {})
        assert 0.0 <= score <= 1.0

    async def test_prompt_with_question_mark(self):
        ev = CoherenceEvaluator()
        prompt = "What is the best way to explain this concept?"
        score = await ev.evaluate(prompt, {})
        assert score > 0.0

    async def test_empty_sentences_filtered(self):
        ev = CoherenceEvaluator()
        prompt = "Hello. . . World."
        score = await ev.evaluate(prompt, {})
        assert 0.0 <= score <= 1.0

    async def test_long_prompt(self):
        ev = CoherenceEvaluator()
        prompt = " ".join(["word"] * 250)
        score = await ev.evaluate(prompt, {})
        assert 0.0 <= score <= 1.0


class TestRelevanceEvaluator:

    async def test_all_keywords_present(self):
        ev = RelevanceEvaluator(task_keywords=["python", "code"], weight=1.0)
        score = await ev.evaluate("Write python code", {})
        assert score == 1.0

    async def test_no_keywords_present(self):
        ev = RelevanceEvaluator(task_keywords=["python", "code"])
        score = await ev.evaluate("Hello world", {})
        assert score == 0.0

    async def test_partial_keywords(self):
        ev = RelevanceEvaluator(task_keywords=["python", "code", "test", "debug"])
        score = await ev.evaluate("Write python code", {})
        assert score == 0.5

    async def test_empty_task_keywords(self):
        ev = RelevanceEvaluator(task_keywords=[])
        score = await ev.evaluate("anything", {})
        assert score == 1.0

    async def test_required_keywords_all_present(self):
        ev = RelevanceEvaluator(
            task_keywords=["python"],
            required_keywords=["must", "include"],
        )
        score = await ev.evaluate("must include python", {})
        assert score == 1.0

    async def test_required_keywords_missing(self):
        ev = RelevanceEvaluator(
            task_keywords=["python"],
            required_keywords=["must", "include"],
        )
        score = await ev.evaluate("python only must", {})
        # "include" missing -> penalized
        assert score < 0.5

    async def test_required_keywords_none_present(self):
        ev = RelevanceEvaluator(
            task_keywords=["python"],
            required_keywords=["must", "include"],
        )
        score = await ev.evaluate("python only", {})
        assert score == 0.0


class TestLengthEvaluator:

    async def test_optimal_length(self):
        ev = LengthEvaluator(min_length=10, max_length=100, optimal_length=50)
        prompt = " ".join(["word"] * 50)
        score = await ev.evaluate(prompt, {})
        assert score == 1.0

    async def test_too_short(self):
        ev = LengthEvaluator(min_length=10, max_length=100, optimal_length=50)
        prompt = " ".join(["word"] * 5)
        score = await ev.evaluate(prompt, {})
        assert score < 0.5

    async def test_too_long(self):
        ev = LengthEvaluator(min_length=10, max_length=100, optimal_length=50)
        prompt = " ".join(["word"] * 150)
        score = await ev.evaluate(prompt, {})
        assert score < 1.0

    async def test_within_range(self):
        ev = LengthEvaluator(min_length=10, max_length=100, optimal_length=50)
        prompt = " ".join(["word"] * 30)
        score = await ev.evaluate(prompt, {})
        assert 0.5 < score <= 1.0

    async def test_very_long(self):
        ev = LengthEvaluator(min_length=10, max_length=100, optimal_length=50)
        prompt = " ".join(["word"] * 500)
        score = await ev.evaluate(prompt, {})
        assert score >= 0.0

    async def test_custom_params(self):
        ev = LengthEvaluator(min_length=5, max_length=20, optimal_length=10, weight=2.0)
        assert ev.weight == 2.0


class TestSpecificityEvaluator:

    async def test_specific_prompt(self):
        ev = SpecificityEvaluator()
        prompt = "Provide exactly 5 specifically detailed concrete examples"
        score = await ev.evaluate(prompt, {})
        assert score > 0.5

    async def test_vague_prompt(self):
        ev = SpecificityEvaluator()
        prompt = "something might maybe possibly do some stuff"
        score = await ev.evaluate(prompt, {})
        assert score < 0.9

    async def test_empty_prompt(self):
        ev = SpecificityEvaluator()
        score = await ev.evaluate("", {})
        assert score == 0.0

    async def test_prompt_with_numbers(self):
        ev = SpecificityEvaluator()
        prompt = "List 5 items in 3 categories"
        score = await ev.evaluate(prompt, {})
        assert score > 0.5

    async def test_no_vague_or_specific_words(self):
        ev = SpecificityEvaluator()
        prompt = "Write a function that sorts an array"
        score = await ev.evaluate(prompt, {})
        assert 0.8 <= score <= 1.0


class TestCompositeEvaluator:

    async def test_single_evaluator(self):
        ev = CompositeEvaluator([SimpleEvaluator(score=0.8)])
        score = await ev.evaluate("test", {})
        assert abs(score - 0.8) < 0.01

    async def test_multiple_evaluators_equal_weight(self):
        ev = CompositeEvaluator([
            SimpleEvaluator(score=0.6, weight=1.0),
            SimpleEvaluator(score=0.8, weight=1.0),
        ])
        score = await ev.evaluate("test", {})
        assert abs(score - 0.7) < 0.01

    async def test_weighted_evaluators(self):
        ev = CompositeEvaluator([
            SimpleEvaluator(score=1.0, weight=3.0),
            SimpleEvaluator(score=0.0, weight=1.0),
        ])
        score = await ev.evaluate("test", {})
        assert abs(score - 0.75) < 0.01

    def test_empty_evaluators_raises(self):
        with pytest.raises(ValueError, match="At least one evaluator"):
            CompositeEvaluator([])

    async def test_evaluate_detailed(self):
        e1 = SimpleEvaluator(score=0.6, weight=1.0)
        e2 = SimpleEvaluator(score=0.8, weight=1.0)
        ev = CompositeEvaluator([e1, e2])
        result = await ev.evaluate_detailed("test", {})
        assert isinstance(result, EvaluationMetrics)
        assert "SimpleEvaluator" in result.metrics
        assert "overall" in result.metrics
        assert abs(result.score - 0.7) < 0.01

    async def test_composite_weight_is_sum(self):
        ev = CompositeEvaluator([
            SimpleEvaluator(score=0.5, weight=2.0),
            SimpleEvaluator(score=0.5, weight=3.0),
        ])
        assert ev.weight == 5.0


# ===========================================================================
# REINFORCEMENT LEARNING
# ===========================================================================


class TestRLConfig:

    def test_defaults(self):
        cfg = RLConfig()
        assert cfg.learning_rate == 0.1
        assert cfg.discount_factor == 0.95
        assert cfg.epsilon == 0.1
        assert cfg.max_episodes == 100

    def test_validate_success(self):
        RLConfig().validate()

    def test_validate_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate"):
            RLConfig(learning_rate=1.5).validate()
        with pytest.raises(ValueError, match="learning_rate"):
            RLConfig(learning_rate=-0.1).validate()

    def test_validate_discount_factor(self):
        with pytest.raises(ValueError, match="discount_factor"):
            RLConfig(discount_factor=1.5).validate()

    def test_validate_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            RLConfig(epsilon=2.0).validate()

    def test_validate_min_epsilon(self):
        with pytest.raises(ValueError, match="min_epsilon"):
            RLConfig(epsilon=0.1, min_epsilon=0.5).validate()


class TestPromptAction:

    def test_creation(self):
        action = PromptAction("test", lambda p: p + "!")
        assert action.name == "test"

    def test_apply(self):
        action = PromptAction("add_excl", lambda p: p + "!")
        assert action.apply("hello") == "hello!"

    def test_repr(self):
        action = PromptAction("test_action", lambda p: p)
        assert "test_action" in repr(action)

    def test_hash(self):
        a1 = PromptAction("same", lambda p: p)
        a2 = PromptAction("same", lambda p: p + "x")
        assert hash(a1) == hash(a2)

    def test_eq(self):
        a1 = PromptAction("same", lambda p: p)
        a2 = PromptAction("same", lambda p: p)
        a3 = PromptAction("diff", lambda p: p)
        assert a1 == a2
        assert a1 != a3

    def test_eq_non_action(self):
        a = PromptAction("x", lambda p: p)
        assert a != "not an action"
        assert a != 42


class TestRLOptimizer:

    @pytest.fixture
    def evaluator(self):
        return SimpleEvaluator(score=0.5)

    @pytest.fixture
    def rl_optimizer(self, evaluator):
        config = RLConfig(
            max_episodes=3,
            max_steps_per_episode=2,
            epsilon=0.5,
            epsilon_decay=0.9,
            min_epsilon=0.01,
        )
        return RLOptimizer(evaluator=evaluator, config=config)

    def test_get_strategy(self, rl_optimizer):
        assert rl_optimizer.get_strategy() == OptimizationStrategy.REINFORCEMENT_LEARNING

    def test_default_actions_created(self, rl_optimizer):
        assert len(rl_optimizer.actions) == 8

    def test_custom_actions(self, evaluator):
        actions = [PromptAction("noop", lambda p: p)]
        opt = RLOptimizer(evaluator=evaluator, actions=actions)
        assert len(opt.actions) == 1

    def test_get_state_key_short(self, rl_optimizer):
        key = rl_optimizer._get_state_key("Hi?")
        assert "short" in key
        assert "question" in key

    def test_get_state_key_medium_statement(self, rl_optimizer):
        key = rl_optimizer._get_state_key("x" * 60)
        assert "medium" in key
        assert "statement" in key

    def test_get_state_key_long(self, rl_optimizer):
        key = rl_optimizer._get_state_key("x" * 200)
        assert "long" in key

    def test_get_state_key_polite(self, rl_optimizer):
        key = rl_optimizer._get_state_key("Please do something")
        assert "polite" in key

    def test_get_state_key_direct(self, rl_optimizer):
        key = rl_optimizer._get_state_key("Do something")
        assert "direct" in key

    def test_select_action_exploration(self, rl_optimizer):
        rl_optimizer.current_epsilon = 1.0  # always explore
        action = rl_optimizer._select_action("test_state")
        assert isinstance(action, PromptAction)

    def test_select_action_exploitation(self, rl_optimizer):
        rl_optimizer.current_epsilon = 0.0  # always exploit
        state = "test_state"
        # Set one action to have high Q-value
        actions = rl_optimizer.actions
        rl_optimizer.q_table[state][actions[0]] = 10.0
        action = rl_optimizer._select_action(state)
        assert action == actions[0]

    def test_update_q_value(self, rl_optimizer):
        state = "s1"
        next_state = "s2"
        action = rl_optimizer.actions[0]
        rl_optimizer._update_q_value(state, action, reward=1.0, next_state=next_state)
        assert rl_optimizer.q_table[state][action] != 0.0

    def test_decay_epsilon(self, rl_optimizer):
        initial = rl_optimizer.current_epsilon
        rl_optimizer._decay_epsilon()
        assert rl_optimizer.current_epsilon < initial
        assert rl_optimizer.current_epsilon >= rl_optimizer.config.min_epsilon

    def test_decay_epsilon_respects_min(self, rl_optimizer):
        rl_optimizer.current_epsilon = 0.01
        rl_optimizer.config.min_epsilon = 0.01
        rl_optimizer._decay_epsilon()
        assert rl_optimizer.current_epsilon == 0.01

    async def test_run_episode(self, rl_optimizer):
        reward = await rl_optimizer._run_episode("Test prompt", {})
        assert isinstance(reward, float)

    async def test_optimize(self, rl_optimizer):
        result = await rl_optimizer.optimize("Explain this topic")
        assert isinstance(result, OptimizationResult)
        assert result.best_prompt is not None
        assert result.best_score >= 0
        assert result.strategy == OptimizationStrategy.REINFORCEMENT_LEARNING
        assert "q_table_size" in result.metadata
        assert "final_epsilon" in result.metadata
        assert "total_episodes" in result.metadata
        assert "average_reward" in result.metadata

    async def test_optimize_verbose(self, evaluator):
        config = RLConfig(max_episodes=2, max_steps_per_episode=2)
        opt = RLOptimizer(evaluator=evaluator, config=config, verbose=True)
        result = await opt.optimize("Test")
        assert result is not None

    def test_get_action_statistics_empty(self, rl_optimizer):
        stats = rl_optimizer.get_action_statistics()
        assert isinstance(stats, dict)

    async def test_get_action_statistics_after_optimize(self, rl_optimizer):
        await rl_optimizer.optimize("Test prompt")
        stats = rl_optimizer.get_action_statistics()
        assert len(stats) > 0

    async def test_get_best_action_per_state(self, rl_optimizer):
        await rl_optimizer.optimize("Test prompt")
        best = rl_optimizer.get_best_action_per_state()
        assert isinstance(best, dict)
        for state, action_name in best.items():
            assert isinstance(action_name, str)

    def test_default_actions_apply(self, rl_optimizer):
        """Test that all default actions can be applied without error."""
        prompt = "Test prompt"
        for action in rl_optimizer.actions:
            result = action.apply(prompt)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_default_action_add_please_idempotent(self, rl_optimizer):
        add_please = next(a for a in rl_optimizer.actions if a.name == "add_please")
        result = add_please.apply("Please do something")
        assert not result.startswith("Please Please")

    def test_default_action_make_question_idempotent(self, rl_optimizer):
        make_q = next(a for a in rl_optimizer.actions if a.name == "make_question")
        result = make_q.apply("Is this a question?")
        assert result.endswith("?")
        assert not result.endswith("??")

    async def test_optimize_convergence(self, evaluator):
        """Test that convergence detection works (all scores same -> converge)."""
        config = RLConfig(
            max_episodes=200,
            max_steps_per_episode=2,
            convergence_threshold=0.001,
        )
        opt = RLOptimizer(evaluator=evaluator, config=config)
        result = await opt.optimize("Test prompt")
        # Should converge early since evaluator returns constant score
        assert result.metadata["total_episodes"] < 200


# ===========================================================================
# TEMPLATES
# ===========================================================================


class TestVariableType:

    def test_enum_values(self):
        assert VariableType.STRING.value == "string"
        assert VariableType.INTEGER.value == "integer"
        assert VariableType.FLOAT.value == "float"
        assert VariableType.BOOLEAN.value == "boolean"
        assert VariableType.LIST.value == "list"
        assert VariableType.DICT.value == "dict"


class TestTemplateVariable:

    def test_string_validation(self):
        v = TemplateVariable(name="x", var_type=VariableType.STRING)
        assert v.validate("hello") is True
        assert v.validate(123) is False

    def test_integer_validation(self):
        v = TemplateVariable(name="x", var_type=VariableType.INTEGER)
        assert v.validate(42) is True
        assert v.validate("42") is False

    def test_float_validation(self):
        v = TemplateVariable(name="x", var_type=VariableType.FLOAT)
        assert v.validate(3.14) is True
        assert v.validate(42) is True  # int accepted as float
        assert v.validate("3.14") is False

    def test_boolean_validation(self):
        v = TemplateVariable(name="x", var_type=VariableType.BOOLEAN)
        assert v.validate(True) is True
        assert v.validate(False) is True
        assert v.validate(1) is False

    def test_list_validation(self):
        v = TemplateVariable(name="x", var_type=VariableType.LIST)
        assert v.validate([1, 2]) is True
        assert v.validate("not a list") is False

    def test_dict_validation(self):
        v = TemplateVariable(name="x", var_type=VariableType.DICT)
        assert v.validate({"a": 1}) is True
        assert v.validate([1]) is False

    def test_custom_validator(self):
        v = TemplateVariable(
            name="x",
            var_type=VariableType.STRING,
            validator=lambda val: len(val) > 3,
        )
        assert v.validate("long enough") is True
        assert v.validate("ab") is False

    def test_defaults(self):
        v = TemplateVariable(name="x")
        assert v.var_type == VariableType.STRING
        assert v.default is None
        assert v.required is True
        assert v.description == ""
        assert v.validator is None


class TestPromptTemplate:

    def test_auto_extract_variables(self):
        t = PromptTemplate(name="t", template="Hello {name}, your age is {age}")
        names = t.get_variable_names()
        assert "name" in names
        assert "age" in names

    def test_explicit_variables(self):
        t = PromptTemplate(
            name="t",
            template="{x}",
            variables=[TemplateVariable(name="x")],
        )
        assert t.get_variable_names() == {"x"}

    def test_render_simple(self):
        t = PromptTemplate(name="t", template="Hello {name}!")
        result = t.render(name="World")
        assert result == "Hello World!"

    def test_render_missing_required(self):
        t = PromptTemplate(name="t", template="Hello {name}!")
        with pytest.raises(ValueError, match="Missing required"):
            t.render()

    def test_render_invalid_value(self):
        t = PromptTemplate(
            name="t",
            template="{count} items",
            variables=[TemplateVariable(name="count", var_type=VariableType.INTEGER)],
        )
        with pytest.raises(ValueError, match="Invalid value"):
            t.render(count="not_int")

    def test_render_with_default(self):
        t = PromptTemplate(
            name="t",
            template="{greeting} {name}",
            variables=[
                TemplateVariable(name="greeting", default="Hi", required=False),
                TemplateVariable(name="name"),
            ],
        )
        result = t.render(name="Alice")
        assert result == "Hi Alice"

    def test_render_optional_no_default(self):
        t = PromptTemplate(
            name="t",
            template="{greeting}{name}",
            variables=[
                TemplateVariable(name="greeting", required=False),
                TemplateVariable(name="name"),
            ],
        )
        result = t.render(name="Bob")
        assert result == "Bob"

    def test_to_dict(self):
        t = PromptTemplate(
            name="t",
            template="{x}",
            variables=[TemplateVariable(name="x", description="the x var")],
            description="test template",
            tags=["tag1"],
            examples=[{"x": "val"}],
        )
        d = t.to_dict()
        assert d["name"] == "t"
        assert d["template"] == "{x}"
        assert d["description"] == "test template"
        assert d["tags"] == ["tag1"]
        assert len(d["variables"]) == 1
        assert d["variables"][0]["name"] == "x"
        assert d["examples"] == [{"x": "val"}]

    def test_render_key_error(self):
        """Template with a variable not covered by TemplateVariable list."""
        t = PromptTemplate(
            name="t",
            template="{a} {b}",
            variables=[TemplateVariable(name="a")],
        )
        # "b" is not in variables, so template.format will raise KeyError
        with pytest.raises(ValueError, match="Template rendering failed"):
            t.render(a="hello")

    def test_get_variable_names(self):
        t = PromptTemplate(
            name="t",
            template="{x} {y}",
            variables=[
                TemplateVariable(name="x"),
                TemplateVariable(name="y"),
            ],
        )
        assert t.get_variable_names() == {"x", "y"}


class TestTemplateLibrary:

    def test_default_templates_loaded(self):
        lib = TemplateLibrary()
        assert lib.get("task_instruction") is not None
        assert lib.get("question_answering") is not None
        assert lib.get("code_generation") is not None
        assert lib.get("analysis") is not None
        assert lib.get("comparison") is not None
        assert lib.get("chain_of_thought") is not None

    def test_add_and_get(self):
        lib = TemplateLibrary()
        t = PromptTemplate(name="custom", template="Hello {who}")
        lib.add(t)
        assert lib.get("custom") is not None
        assert lib.get("custom").template == "Hello {who}"

    def test_get_nonexistent(self):
        lib = TemplateLibrary()
        assert lib.get("nonexistent") is None

    def test_remove_existing(self):
        lib = TemplateLibrary()
        assert lib.remove("task_instruction") is True
        assert lib.get("task_instruction") is None

    def test_remove_nonexistent(self):
        lib = TemplateLibrary()
        assert lib.remove("nonexistent") is False

    def test_list_all(self):
        lib = TemplateLibrary()
        templates = lib.list()
        assert len(templates) >= 6

    def test_list_filtered_by_tags(self):
        lib = TemplateLibrary()
        result = lib.list(tags=["code"])
        assert any(t.name == "code_generation" for t in result)

    def test_list_filtered_no_match(self):
        lib = TemplateLibrary()
        result = lib.list(tags=["nonexistent_tag"])
        assert len(result) == 0

    def test_search_by_name(self):
        lib = TemplateLibrary()
        result = lib.search("code")
        assert any(t.name == "code_generation" for t in result)

    def test_search_by_description(self):
        lib = TemplateLibrary()
        result = lib.search("chain of thought")
        assert any(t.name == "chain_of_thought" for t in result)

    def test_search_no_results(self):
        lib = TemplateLibrary()
        result = lib.search("xyznonexistent")
        assert len(result) == 0

    def test_render_task_instruction(self):
        lib = TemplateLibrary()
        t = lib.get("task_instruction")
        result = t.render(action="analyze", subject="market trends")
        assert "analyze" in result
        assert "market trends" in result


class TestTemplateComposer:

    def test_compose_single(self):
        lib = TemplateLibrary()
        composer = TemplateComposer(library=lib)
        result = composer.compose(
            ["task_instruction"],
            {"action": "summarize", "subject": "the report"},
        )
        assert "summarize" in result

    def test_compose_multiple(self):
        lib = TemplateLibrary()
        lib.add(PromptTemplate(name="intro", template="Introduction: {topic}"))
        lib.add(PromptTemplate(name="body", template="Details: {topic}"))
        composer = TemplateComposer(library=lib)
        result = composer.compose(
            ["intro", "body"],
            {"topic": "AI"},
            separator="\n",
        )
        assert "Introduction: AI" in result
        assert "Details: AI" in result

    def test_compose_template_not_found(self):
        composer = TemplateComposer()
        with pytest.raises(ValueError, match="Template not found"):
            composer.compose(["nonexistent"], {})

    def test_compose_with_context(self):
        lib = TemplateLibrary()
        lib.add(PromptTemplate(name="ctx", template="Context: {info}"))
        lib.add(PromptTemplate(name="main", template="Main: {info}"))
        composer = TemplateComposer(library=lib)
        result = composer.compose_with_context(
            "main", ["ctx"], {"info": "test data"}
        )
        assert "Context: test data" in result
        assert "Main: test data" in result

    def test_compose_conditional_true(self):
        lib = TemplateLibrary()
        lib.add(PromptTemplate(name="a", template="Part A: {x}"))
        lib.add(PromptTemplate(name="b", template="Part B: {x}"))
        composer = TemplateComposer(library=lib)
        result = composer.compose_conditional(
            [
                ("a", lambda vals: True),
                ("b", lambda vals: vals.get("include_b", False)),
            ],
            {"x": "data", "include_b": False},
        )
        assert "Part A" in result
        assert "Part B" not in result

    def test_compose_conditional_all_true(self):
        lib = TemplateLibrary()
        lib.add(PromptTemplate(name="a", template="A: {x}"))
        lib.add(PromptTemplate(name="b", template="B: {x}"))
        composer = TemplateComposer(library=lib)
        result = composer.compose_conditional(
            [
                ("a", lambda vals: True),
                ("b", lambda vals: True),
            ],
            {"x": "val"},
        )
        assert "A: val" in result
        assert "B: val" in result

    def test_compose_conditional_template_not_found(self):
        composer = TemplateComposer()
        with pytest.raises(ValueError, match="Template not found"):
            composer.compose_conditional(
                [("nonexistent", lambda v: True)], {}
            )

    def test_create_few_shot(self):
        lib = TemplateLibrary()
        lib.add(PromptTemplate(name="simple", template="Input: {x}, Output: {y}"))
        composer = TemplateComposer(library=lib)
        result = composer.create_few_shot(
            "simple",
            examples=[
                {"x": "a", "y": "1"},
                {"x": "b", "y": "2"},
            ],
            query={"x": "c", "y": "?"},
        )
        assert "Example 1:" in result
        assert "Example 2:" in result
        assert "Now, for the following:" in result

    def test_create_few_shot_not_found(self):
        composer = TemplateComposer()
        with pytest.raises(ValueError, match="Template not found"):
            composer.create_few_shot("nonexistent", [], {})

    def test_create_chain(self):
        lib = TemplateLibrary()
        lib.add(PromptTemplate(name="step", template="Do: {task}"))
        composer = TemplateComposer(library=lib)
        result = composer.create_chain(
            "step",
            [{"task": "first"}, {"task": "second"}],
        )
        assert "Step 1:" in result
        assert "Step 2:" in result
        assert "Do: first" in result
        assert "Do: second" in result

    def test_create_chain_not_found(self):
        composer = TemplateComposer()
        with pytest.raises(ValueError, match="Template not found"):
            composer.create_chain("nonexistent", [{}])

    def test_default_library_created(self):
        composer = TemplateComposer()
        # Should have a library with default templates
        assert composer.library.get("task_instruction") is not None


# ===========================================================================
# OPTIMIZER BASE CLASSES
# ===========================================================================


class TestOptimizationResult:

    def test_to_dict(self):
        result = OptimizationResult(
            best_prompt="Best",
            best_score=0.95,
            history=[{"score": 0.9}],
            iterations=5,
            strategy=OptimizationStrategy.GENETIC,
        )
        d = result.to_dict()
        assert d["best_prompt"] == "Best"
        assert d["best_score"] == 0.95
        assert d["strategy"] == "genetic"
        assert "timestamp" in d

    def test_metadata_default(self):
        result = OptimizationResult(
            best_prompt="x", best_score=0.5, history=[], iterations=1,
            strategy=OptimizationStrategy.RANDOM_SEARCH,
        )
        assert result.metadata == {}


class TestPromptCandidate:

    def test_to_dict(self):
        c = PromptCandidate(prompt="test", score=0.8, iteration=3, metadata={"k": "v"})
        d = c.to_dict()
        assert d["prompt"] == "test"
        assert d["score"] == 0.8
        assert d["iteration"] == 3
        assert d["metadata"] == {"k": "v"}


class TestPromptOptimizerBase:
    """Test the base PromptOptimizer through concrete subclasses."""

    async def test_evaluate_prompt(self):
        ev = SimpleEvaluator(score=0.7)
        opt = RandomSearchOptimizer(ev, variation_fn=lambda p: p + "x", max_iterations=1)
        score = await opt.evaluate_prompt("test")
        assert score == 0.7

    async def test_evaluate_batch(self):
        ev = SimpleEvaluator(score=0.5)
        opt = RandomSearchOptimizer(ev, variation_fn=lambda p: p, max_iterations=1)
        scores = await opt.evaluate_batch(["a", "b", "c"])
        assert len(scores) == 3
        assert all(s == 0.5 for s in scores)

    def test_track_candidate(self):
        ev = SimpleEvaluator()
        opt = RandomSearchOptimizer(ev, variation_fn=lambda p: p, max_iterations=1)
        c = opt.track_candidate("p1", 0.5)
        assert c.prompt == "p1"
        assert opt.best_candidate == c
        # Better candidate replaces
        c2 = opt.track_candidate("p2", 0.9)
        assert opt.best_candidate == c2

    def test_track_candidate_worse(self):
        ev = SimpleEvaluator()
        opt = RandomSearchOptimizer(ev, variation_fn=lambda p: p, max_iterations=1)
        opt.track_candidate("p1", 0.9)
        opt.track_candidate("p2", 0.5)
        assert opt.best_candidate.prompt == "p1"

    def test_has_converged_not_enough_history(self):
        ev = SimpleEvaluator()
        opt = RandomSearchOptimizer(ev, variation_fn=lambda p: p, max_iterations=1)
        assert opt.has_converged(window_size=10) is False

    def test_has_converged_true(self):
        ev = SimpleEvaluator()
        opt = RandomSearchOptimizer(ev, variation_fn=lambda p: p, max_iterations=1)
        for i in range(20):
            opt.track_candidate(f"p{i}", 0.5)
        assert opt.has_converged(window_size=10) is True

    def test_has_converged_false(self):
        ev = SimpleEvaluator()
        opt = RandomSearchOptimizer(ev, variation_fn=lambda p: p, max_iterations=1)
        for i in range(20):
            opt.track_candidate(f"p{i}", float(i) / 20)
        assert opt.has_converged(window_size=10) is False

    def test_get_optimization_result_no_candidates(self):
        ev = SimpleEvaluator()
        opt = RandomSearchOptimizer(ev, variation_fn=lambda p: p, max_iterations=1)
        with pytest.raises(ValueError, match="No candidates"):
            opt.get_optimization_result()

    def test_reset(self):
        ev = SimpleEvaluator()
        opt = RandomSearchOptimizer(ev, variation_fn=lambda p: p, max_iterations=1)
        opt.track_candidate("p", 0.5)
        opt.current_iteration = 10
        opt.reset()
        assert len(opt.history) == 0
        assert opt.best_candidate is None
        assert opt.current_iteration == 0


class TestRandomSearchOptimizer:

    async def test_optimize(self):
        ev = SimpleEvaluator(score=0.5)
        opt = RandomSearchOptimizer(
            ev, variation_fn=lambda p: p + " modified", max_iterations=5
        )
        result = await opt.optimize("base")
        assert isinstance(result, OptimizationResult)
        assert result.strategy == OptimizationStrategy.RANDOM_SEARCH

    async def test_optimize_converges(self):
        ev = SimpleEvaluator(score=0.5)
        opt = RandomSearchOptimizer(
            ev, variation_fn=lambda p: p,
            max_iterations=100, convergence_threshold=0.001,
        )
        result = await opt.optimize("base")
        assert result.iterations < 100

    async def test_optimize_verbose(self):
        ev = SimpleEvaluator(score=0.5)
        opt = RandomSearchOptimizer(
            ev, variation_fn=lambda p: p,
            max_iterations=5, verbose=True,
        )
        result = await opt.optimize("base")
        assert result is not None


class TestGridSearchOptimizer:

    async def test_optimize(self):
        ev = SimpleEvaluator(score=0.5)
        grid = {"tone": ["formal", "casual"], "length": ["short", "long"]}
        opt = GridSearchOptimizer(
            ev,
            parameter_grid=grid,
            template_fn=lambda params: f"{params['tone']} {params['length']} prompt",
        )
        result = await opt.optimize("")
        assert isinstance(result, OptimizationResult)
        assert result.strategy == OptimizationStrategy.GRID_SEARCH
        assert len(opt.history) == 4  # 2x2 grid

    async def test_optimize_verbose(self):
        ev = SimpleEvaluator(score=0.5)
        grid = {"a": [1, 2]}
        opt = GridSearchOptimizer(
            ev,
            parameter_grid=grid,
            template_fn=lambda p: f"prompt {p['a']}",
            verbose=True,
        )
        result = await opt.optimize("")
        assert result is not None

    def test_generate_grid_points(self):
        ev = SimpleEvaluator()
        grid = {"x": [1, 2], "y": ["a", "b", "c"]}
        opt = GridSearchOptimizer(ev, parameter_grid=grid, template_fn=lambda p: "")
        points = opt._generate_grid_points()
        assert len(points) == 6


# ===========================================================================
# A/B TESTING (additional coverage)
# ===========================================================================


class TestABTestConfigValidation:

    def test_validate_explore_probability_invalid(self):
        with pytest.raises(ValueError, match="explore_probability"):
            ABTestConfig(explore_probability=1.5).validate()


class TestABTesterAdditional:

    @pytest.fixture
    def evaluator(self):
        return SimpleEvaluator(score=0.6)

    async def test_two_variant_statistical_test(self, evaluator):
        """With exactly 2 variants, statistical significance is calculated."""
        config = ABTestConfig(min_samples=5, max_iterations=30)
        tester = ABTester(
            evaluator=evaluator,
            variants=[("a", "Prompt A"), ("b", "Prompt B")],
            config=config,
            selection_strategy="epsilon_greedy",
        )
        result = await tester.optimize("")
        assert "statistical_test" in result.metadata

    async def test_get_variant_statistics(self, evaluator):
        config = ABTestConfig(min_samples=3, max_iterations=10)
        tester = ABTester(
            evaluator=evaluator,
            variants=[("a", "Prompt A"), ("b", "Prompt B")],
            config=config,
        )
        await tester.optimize("")
        stats = tester.get_variant_statistics()
        assert len(stats) == 2
        assert all("name" in s for s in stats)

    async def test_get_winner_two_variants(self, evaluator):
        config = ABTestConfig(min_samples=3, max_iterations=10)
        tester = ABTester(
            evaluator=evaluator,
            variants=[("a", "Prompt A"), ("b", "Prompt B")],
            config=config,
        )
        await tester.optimize("")
        winner = tester.get_winner(min_confidence=0.0)
        # With very low confidence, might return a winner
        # but with constant evaluator, p-value may be high
        # just check it doesn't error
        assert winner is None or isinstance(winner, PromptVariant)

    async def test_get_winner_multiple_variants(self, evaluator):
        config = ABTestConfig(min_samples=3, max_iterations=15)
        tester = ABTester(
            evaluator=evaluator,
            variants=[("a", "A"), ("b", "B"), ("c", "C")],
            config=config,
        )
        await tester.optimize("")
        winner = tester.get_winner()
        assert isinstance(winner, PromptVariant)

    async def test_thompson_sampling(self, evaluator):
        config = ABTestConfig(min_samples=3, max_iterations=10)
        tester = ABTester(
            evaluator=evaluator,
            variants=[("a", "A"), ("b", "B")],
            config=config,
            selection_strategy="thompson",
        )
        result = await tester.optimize("")
        assert result is not None

    async def test_ucb_all_variants_tried(self, evaluator):
        config = ABTestConfig(min_samples=3, max_iterations=10)
        tester = ABTester(
            evaluator=evaluator,
            variants=[("a", "A"), ("b", "B"), ("c", "C")],
            config=config,
            selection_strategy="ucb",
        )
        await tester.optimize("")
        for v in tester.variants:
            assert v.num_selections > 0

    def test_select_variant_ucb_zero_total(self, evaluator):
        config = ABTestConfig(min_samples=3, max_iterations=10)
        tester = ABTester(
            evaluator=evaluator,
            variants=[("a", "A"), ("b", "B")],
            config=config,
            selection_strategy="ucb",
        )
        variant = tester._select_variant_ucb(0)
        assert isinstance(variant, PromptVariant)

    def test_select_variant_ucb_unselected(self, evaluator):
        config = ABTestConfig(min_samples=3, max_iterations=10)
        tester = ABTester(
            evaluator=evaluator,
            variants=[("a", "A"), ("b", "B")],
            config=config,
            selection_strategy="ucb",
        )
        # Give one variant selections, leave other at 0
        tester.variants[0].add_sample(0.5)
        variant = tester._select_variant_ucb(1)
        # Should pick the unselected one (infinity UCB)
        assert variant == tester.variants[1]

    def test_t_cdf_large_df(self, evaluator):
        config = ABTestConfig(min_samples=3, max_iterations=10)
        tester = ABTester(
            evaluator=evaluator,
            variants=[("a", "A"), ("b", "B")],
            config=config,
        )
        result = tester._t_cdf(0.0, 50)
        assert abs(result - 0.5) < 0.01

    def test_t_cdf_small_df(self, evaluator):
        config = ABTestConfig(min_samples=3, max_iterations=10)
        tester = ABTester(
            evaluator=evaluator,
            variants=[("a", "A"), ("b", "B")],
            config=config,
        )
        result = tester._t_cdf(1.0, 5)
        assert result == 0.5  # placeholder

    def test_statistical_significance_insufficient_samples(self, evaluator):
        config = ABTestConfig(min_samples=3, max_iterations=10)
        tester = ABTester(
            evaluator=evaluator,
            variants=[("a", "A"), ("b", "B")],
            config=config,
        )
        p_value, is_sig = tester._calculate_statistical_significance(
            tester.variants[0], tester.variants[1]
        )
        assert p_value == 1.0
        assert is_sig is False

    def test_statistical_significance_with_samples(self, evaluator):
        config = ABTestConfig(min_samples=3, max_iterations=10)
        tester = ABTester(
            evaluator=evaluator,
            variants=[("a", "A"), ("b", "B")],
            config=config,
        )
        # Add samples with variance so we don't get division by zero
        for v in [0.85, 0.90, 0.95, 0.88, 0.92, 0.87, 0.91, 0.93, 0.89, 0.86]:
            tester.variants[0].add_sample(v)
        for v in [0.10, 0.15, 0.12, 0.08, 0.11, 0.14, 0.09, 0.13, 0.07, 0.16]:
            tester.variants[1].add_sample(v)
        p_value, is_sig = tester._calculate_statistical_significance(
            tester.variants[0], tester.variants[1]
        )
        assert isinstance(p_value, float)
        assert isinstance(is_sig, bool)


class TestPromptVariantAdditional:

    def test_variance_multiple_samples(self):
        v = PromptVariant(name="t", prompt="p")
        v.samples = [0.2, 0.4, 0.6, 0.8, 1.0]
        assert v.variance > 0

    def test_std_dev_is_sqrt_variance(self):
        v = PromptVariant(name="t", prompt="p")
        v.samples = [0.2, 0.4, 0.6, 0.8, 1.0]
        assert abs(v.std_dev - math.sqrt(v.variance)) < 1e-10


# ===========================================================================
# GENETIC OPTIMIZER (additional coverage)
# ===========================================================================


class TestGeneticOptimizerAdditional:

    async def test_population_diversity(self):
        ev = SimpleEvaluator(score=0.5)
        config = GeneticConfig(population_size=5, max_generations=2, elite_size=1)
        opt = GeneticOptimizer(evaluator=ev, config=config)
        await opt.optimize("Test prompt for diversity")
        diversity = opt.get_population_diversity()
        assert 0.0 <= diversity <= 1.0

    async def test_fitness_statistics(self):
        ev = SimpleEvaluator(score=0.5)
        config = GeneticConfig(population_size=5, max_generations=2, elite_size=1)
        opt = GeneticOptimizer(evaluator=ev, config=config)
        await opt.optimize("Test prompt")
        stats = opt.get_fitness_statistics()
        assert "mean" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" in stats

    def test_fitness_statistics_empty_population(self):
        ev = SimpleEvaluator()
        config = GeneticConfig(population_size=5, max_generations=2, elite_size=1)
        opt = GeneticOptimizer(evaluator=ev, config=config)
        stats = opt.get_fitness_statistics()
        assert stats == {}

    async def test_population_diversity_single(self):
        ev = SimpleEvaluator(score=0.5)
        config = GeneticConfig(population_size=5, max_generations=2, elite_size=1)
        opt = GeneticOptimizer(evaluator=ev, config=config)
        opt.population = [("only one", 0.5)]
        assert opt.get_population_diversity() == 0.0

    def test_default_mutation_empty(self):
        ev = SimpleEvaluator()
        config = GeneticConfig(population_size=5, max_generations=2, elite_size=1)
        opt = GeneticOptimizer(evaluator=ev, config=config)
        result = opt._default_mutation("")
        assert result == ""

    def test_default_crossover_empty(self):
        ev = SimpleEvaluator()
        config = GeneticConfig(population_size=5, max_generations=2, elite_size=1)
        opt = GeneticOptimizer(evaluator=ev, config=config)
        c1, c2 = opt._default_crossover("", "test")
        assert c1 == ""
        assert c2 == "test"

    def test_tournament_selection(self):
        ev = SimpleEvaluator()
        config = GeneticConfig(population_size=5, max_generations=2, elite_size=1, tournament_size=2)
        opt = GeneticOptimizer(evaluator=ev, config=config)
        pop = [("a", 0.1), ("b", 0.9), ("c", 0.5)]
        # Run many times; best should win often
        winners = set()
        for _ in range(50):
            w = opt._tournament_selection(pop)
            winners.add(w)
        assert "b" in winners  # "b" has highest score, should win sometimes

    async def test_with_seed_prompts(self):
        ev = SimpleEvaluator(score=0.5)
        config = GeneticConfig(population_size=5, max_generations=2, elite_size=1)
        opt = GeneticOptimizer(
            evaluator=ev, config=config,
            seed_prompts=["Seed 1", "Seed 2", "Seed 3"],
        )
        result = await opt.optimize("Base")
        assert result is not None

    def test_config_validate_tournament_size(self):
        with pytest.raises(ValueError, match="tournament_size"):
            GeneticConfig(tournament_size=0).validate()


# ===========================================================================
# OPTIMIZATION STRATEGY ENUM
# ===========================================================================


class TestOptimizationStrategy:

    def test_values(self):
        assert OptimizationStrategy.GENETIC.value == "genetic"
        assert OptimizationStrategy.REINFORCEMENT_LEARNING.value == "reinforcement_learning"
        assert OptimizationStrategy.AB_TESTING.value == "ab_testing"
        assert OptimizationStrategy.RANDOM_SEARCH.value == "random_search"
        assert OptimizationStrategy.GRID_SEARCH.value == "grid_search"


# ===========================================================================
# MODULE __init__
# ===========================================================================


class TestModuleInit:

    def test_version(self):
        from ia_modules.prompt_optimization import __version__
        assert __version__ == "1.0.0"

    def test_all_exports(self):
        from ia_modules.prompt_optimization import __all__
        expected = [
            "PromptOptimizer", "OptimizationStrategy",
            "GeneticOptimizer", "GeneticConfig",
            "RLOptimizer", "RLConfig",
            "ABTester", "ABTestConfig",
            "PromptEvaluator", "AccuracyEvaluator",
            "CoherenceEvaluator", "RelevanceEvaluator", "CompositeEvaluator",
            "PromptTemplate", "TemplateLibrary",
            "TemplateComposer", "TemplateVariable",
        ]
        for name in expected:
            assert name in __all__
