"""
Prompt template library and composition system.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable
from enum import Enum


class VariableType(Enum):
    """Types of template variables."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"


@dataclass
class TemplateVariable:
    """A variable in a prompt template."""

    name: str
    var_type: VariableType = VariableType.STRING
    default: Optional[Any] = None
    required: bool = True
    description: str = ""
    validator: Optional[Callable[[Any], bool]] = None

    def validate(self, value: Any) -> bool:
        """
        Validate a value for this variable.

        Args:
            value: Value to validate

        Returns:
            True if valid
        """
        # Type validation
        if self.var_type == VariableType.STRING and not isinstance(value, str):
            return False
        elif self.var_type == VariableType.INTEGER and not isinstance(value, int):
            return False
        elif self.var_type == VariableType.FLOAT and not isinstance(value, (int, float)):
            return False
        elif self.var_type == VariableType.BOOLEAN and not isinstance(value, bool):
            return False
        elif self.var_type == VariableType.LIST and not isinstance(value, list):
            return False
        elif self.var_type == VariableType.DICT and not isinstance(value, dict):
            return False

        # Custom validation
        if self.validator and not self.validator(value):
            return False

        return True


@dataclass
class PromptTemplate:
    """
    A prompt template with variables.

    Templates use {variable_name} syntax for substitution.
    """

    name: str
    template: str
    variables: List[TemplateVariable] = field(default_factory=list)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Extract variables from template if not provided."""
        if not self.variables:
            self.variables = self._extract_variables()

    def _extract_variables(self) -> List[TemplateVariable]:
        """
        Extract variable names from template.

        Returns:
            List of TemplateVariable objects
        """
        pattern = r'\{([^}]+)\}'
        var_names = re.findall(pattern, self.template)

        return [
            TemplateVariable(name=name)
            for name in set(var_names)
        ]

    def get_variable_names(self) -> Set[str]:
        """
        Get all variable names in template.

        Returns:
            Set of variable names
        """
        return {v.name for v in self.variables}

    def render(self, **kwargs) -> str:
        """
        Render template with provided values.

        Args:
            **kwargs: Variable values

        Returns:
            Rendered prompt

        Raises:
            ValueError: If required variables are missing or invalid
        """
        # Check required variables
        provided = set(kwargs.keys())
        required = {v.name for v in self.variables if v.required}
        missing = required - provided

        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Validate and prepare values
        values = {}
        for var in self.variables:
            if var.name in kwargs:
                value = kwargs[var.name]
                if not var.validate(value):
                    raise ValueError(
                        f"Invalid value for variable '{var.name}': {value}"
                    )
                values[var.name] = value
            elif var.default is not None:
                values[var.name] = var.default
            elif not var.required:
                values[var.name] = ""

        # Render template
        try:
            return self.template.format(**values)
        except KeyError as e:
            raise ValueError(f"Template rendering failed: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "variables": [
                {
                    "name": v.name,
                    "type": v.var_type.value,
                    "default": v.default,
                    "required": v.required,
                    "description": v.description,
                }
                for v in self.variables
            ],
            "description": self.description,
            "tags": self.tags,
            "examples": self.examples,
        }


class TemplateLibrary:
    """
    Library of reusable prompt templates.

    Provides storage, retrieval, and search of templates.
    """

    def __init__(self):
        """Initialize template library."""
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_default_templates()

    def _initialize_default_templates(self):
        """Add default templates to library."""
        # Task instruction template
        self.add(PromptTemplate(
            name="task_instruction",
            template="Please {action} the following: {subject}. {additional_instructions}",
            variables=[
                TemplateVariable("action", description="Action verb (e.g., analyze, summarize)"),
                TemplateVariable("subject", description="Subject of the task"),
                TemplateVariable("additional_instructions", required=False, default=""),
            ],
            description="Basic task instruction template",
            tags=["instruction", "task"],
            examples=[
                {
                    "action": "analyze",
                    "subject": "market trends in AI technology",
                    "additional_instructions": "Focus on the last 6 months.",
                }
            ],
        ))

        # Question answering template
        self.add(PromptTemplate(
            name="question_answering",
            template="Given the context: {context}\n\nQuestion: {question}\n\nProvide a {detail_level} answer.",
            variables=[
                TemplateVariable("context", description="Background context"),
                TemplateVariable("question", description="Question to answer"),
                TemplateVariable("detail_level", default="detailed", description="Level of detail"),
            ],
            description="Question answering with context",
            tags=["qa", "question"],
        ))

        # Code generation template
        self.add(PromptTemplate(
            name="code_generation",
            template="Write {language} code to {task}. Requirements:\n{requirements}\n\nProvide clean, well-commented code.",
            variables=[
                TemplateVariable("language", description="Programming language"),
                TemplateVariable("task", description="Task description"),
                TemplateVariable("requirements", description="Specific requirements"),
            ],
            description="Code generation template",
            tags=["code", "generation"],
        ))

        # Analysis template
        self.add(PromptTemplate(
            name="analysis",
            template="Analyze the following {subject} and provide:\n1. {aspect1}\n2. {aspect2}\n3. {aspect3}\n\n{subject_content}",
            variables=[
                TemplateVariable("subject", description="Subject to analyze"),
                TemplateVariable("aspect1", description="First aspect to cover"),
                TemplateVariable("aspect2", description="Second aspect to cover"),
                TemplateVariable("aspect3", description="Third aspect to cover"),
                TemplateVariable("subject_content", description="Content to analyze"),
            ],
            description="Structured analysis template",
            tags=["analysis", "structured"],
        ))

        # Comparison template
        self.add(PromptTemplate(
            name="comparison",
            template="Compare {item1} and {item2} based on:\n- {criterion1}\n- {criterion2}\n- {criterion3}\n\nProvide a balanced comparison.",
            variables=[
                TemplateVariable("item1", description="First item"),
                TemplateVariable("item2", description="Second item"),
                TemplateVariable("criterion1", description="First comparison criterion"),
                TemplateVariable("criterion2", description="Second comparison criterion"),
                TemplateVariable("criterion3", description="Third comparison criterion"),
            ],
            description="Comparison template",
            tags=["comparison", "analysis"],
        ))

        # Chain of thought template
        self.add(PromptTemplate(
            name="chain_of_thought",
            template="{task}\n\nLet's approach this step by step:\n1. {step1}\n2. {step2}\n3. {step3}\n\nProvide your reasoning at each step.",
            variables=[
                TemplateVariable("task", description="Task to solve"),
                TemplateVariable("step1", description="First step"),
                TemplateVariable("step2", description="Second step"),
                TemplateVariable("step3", description="Third step"),
            ],
            description="Chain of thought reasoning template",
            tags=["reasoning", "cot"],
        ))

    def add(self, template: PromptTemplate) -> None:
        """
        Add template to library.

        Args:
            template: Template to add
        """
        self.templates[template.name] = template

    def get(self, name: str) -> Optional[PromptTemplate]:
        """
        Get template by name.

        Args:
            name: Template name

        Returns:
            PromptTemplate or None
        """
        return self.templates.get(name)

    def remove(self, name: str) -> bool:
        """
        Remove template from library.

        Args:
            name: Template name

        Returns:
            True if removed, False if not found
        """
        if name in self.templates:
            del self.templates[name]
            return True
        return False

    def list(self, tags: Optional[List[str]] = None) -> List[PromptTemplate]:
        """
        List all templates, optionally filtered by tags.

        Args:
            tags: Filter by tags

        Returns:
            List of templates
        """
        templates = list(self.templates.values())

        if tags:
            templates = [
                t for t in templates
                if any(tag in t.tags for tag in tags)
            ]

        return templates

    def search(self, query: str) -> List[PromptTemplate]:
        """
        Search templates by name or description.

        Args:
            query: Search query

        Returns:
            List of matching templates
        """
        query_lower = query.lower()

        return [
            t for t in self.templates.values()
            if query_lower in t.name.lower() or query_lower in t.description.lower()
        ]


class TemplateComposer:
    """
    Composes complex prompts from multiple templates.

    Supports template chaining, nesting, and conditional inclusion.
    """

    def __init__(self, library: Optional[TemplateLibrary] = None):
        """
        Initialize template composer.

        Args:
            library: Template library to use
        """
        self.library = library or TemplateLibrary()

    def compose(
        self,
        template_names: List[str],
        values: Dict[str, Any],
        separator: str = "\n\n",
    ) -> str:
        """
        Compose multiple templates into one prompt.

        Args:
            template_names: List of template names to compose
            values: Variable values for all templates
            separator: String to join templates

        Returns:
            Composed prompt
        """
        rendered_parts = []

        for name in template_names:
            template = self.library.get(name)
            if template is None:
                raise ValueError(f"Template not found: {name}")

            # Filter values for this template
            template_values = {
                k: v for k, v in values.items()
                if k in template.get_variable_names()
            }

            rendered = template.render(**template_values)
            rendered_parts.append(rendered)

        return separator.join(rendered_parts)

    def compose_with_context(
        self,
        main_template: str,
        context_templates: List[str],
        values: Dict[str, Any],
    ) -> str:
        """
        Compose with context templates preceding main template.

        Args:
            main_template: Name of main template
            context_templates: Names of context templates
            values: Variable values

        Returns:
            Composed prompt
        """
        all_templates = context_templates + [main_template]
        return self.compose(all_templates, values)

    def compose_conditional(
        self,
        template_conditions: List[tuple[str, Callable[[Dict[str, Any]], bool]]],
        values: Dict[str, Any],
        separator: str = "\n\n",
    ) -> str:
        """
        Compose templates conditionally based on values.

        Args:
            template_conditions: List of (template_name, condition_fn) tuples
            values: Variable values
            separator: String to join templates

        Returns:
            Composed prompt
        """
        rendered_parts = []

        for template_name, condition in template_conditions:
            if condition(values):
                template = self.library.get(template_name)
                if template is None:
                    raise ValueError(f"Template not found: {template_name}")

                template_values = {
                    k: v for k, v in values.items()
                    if k in template.get_variable_names()
                }

                rendered = template.render(**template_values)
                rendered_parts.append(rendered)

        return separator.join(rendered_parts)

    def create_few_shot(
        self,
        template_name: str,
        examples: List[Dict[str, Any]],
        query: Dict[str, Any],
    ) -> str:
        """
        Create few-shot prompt with examples.

        Args:
            template_name: Template to use
            examples: List of example values
            query: Query values

        Returns:
            Few-shot prompt
        """
        template = self.library.get(template_name)
        if template is None:
            raise ValueError(f"Template not found: {template_name}")

        parts = []

        # Add examples
        for i, example in enumerate(examples, 1):
            parts.append(f"Example {i}:")
            parts.append(template.render(**example))

        # Add query
        parts.append("\nNow, for the following:")
        parts.append(template.render(**query))

        return "\n\n".join(parts)

    def create_chain(
        self,
        template_name: str,
        chain_values: List[Dict[str, Any]],
    ) -> str:
        """
        Create chain of prompts with same template.

        Args:
            template_name: Template to use
            chain_values: List of value dicts for each step

        Returns:
            Chained prompt
        """
        template = self.library.get(template_name)
        if template is None:
            raise ValueError(f"Template not found: {template_name}")

        parts = []

        for i, values in enumerate(chain_values, 1):
            parts.append(f"Step {i}:")
            parts.append(template.render(**values))

        return "\n\n".join(parts)
