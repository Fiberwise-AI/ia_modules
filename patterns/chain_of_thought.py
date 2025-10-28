"""
Chain-of-Thought (CoT) pattern implementation.

Enables explicit step-by-step reasoning for improved accuracy on complex tasks.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import re


@dataclass
class CoTConfig:
    """Configuration for Chain-of-Thought reasoning."""
    show_reasoning: bool = True
    reasoning_depth: int = 3
    validation_step: bool = True
    format: str = "numbered"  # "numbered", "bullet", "freeform"
    temperature: float = 0.7
    retry_on_validation_failure: bool = True


class ChainOfThoughtStep:
    """
    Execute an LLM step with explicit chain-of-thought reasoning.
    
    Improves accuracy by 20-40% on tasks requiring multi-step reasoning.
    
    Example:
        step = ChainOfThoughtStep(
            name="solve_math",
            prompt="What is 15% of 80?",
            model="gpt-4",
            config=CoTConfig(show_reasoning=True, reasoning_depth=3)
        )
    
    Research: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
              (Wei et al., 2022)
    """
    
    def __init__(
        self,
        name: str,
        prompt: str,
        model: str = "gpt-4",
        config: Optional[CoTConfig] = None,
        **kwargs
    ):
        self.name = name
        self.prompt = prompt
        self.model = model
        self.config = config or CoTConfig()
        self.kwargs = kwargs
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with chain-of-thought reasoning."""
        
        # Build CoT prompt
        cot_prompt = self._build_cot_prompt(context)
        
        # Get LLM response
        llm_service = context.get('services', {}).get('llm')
        if not llm_service:
            raise ValueError("LLM service not found in context")
        
        response = await llm_service.generate(
            prompt=cot_prompt,
            model=self.model,
            temperature=self.config.temperature
        )
        
        # Parse reasoning steps and final answer
        reasoning_steps, final_answer = self._parse_response(response)
        
        # Optional validation step
        if self.config.validation_step:
            is_valid, validation_msg = self._validate_reasoning(
                reasoning_steps, final_answer
            )
            
            if not is_valid and self.config.retry_on_validation_failure:
                # Retry with validation feedback
                response = await self._retry_with_feedback(
                    cot_prompt, validation_msg, llm_service
                )
                reasoning_steps, final_answer = self._parse_response(response)
        
        result = {
            "answer": final_answer,
            "model_used": self.model,
            "raw_response": response,
        }
        
        if self.config.show_reasoning:
            result["reasoning"] = reasoning_steps
        
        return result
    
    def _build_cot_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt that encourages step-by-step reasoning."""
        
        # Format base prompt with context variables
        try:
            base_prompt = self.prompt.format(**context)
        except KeyError:
            base_prompt = self.prompt
        
        if self.config.format == "numbered":
            cot_instruction = """

Please solve this step-by-step:
1. First, identify what we know
2. Then, determine what we need to find
3. Next, work through the calculation/logic
4. Finally, state the answer clearly

Let's begin:
"""
        elif self.config.format == "bullet":
            cot_instruction = """

Let's think through this step by step:
• What information do we have?
• What are we trying to find?
• How can we get there?
• What is the final answer?
"""
        else:  # freeform
            cot_instruction = "\n\nLet's think step by step:\n"
        
        return f"{base_prompt}{cot_instruction}"
    
    def _parse_response(self, response: str) -> tuple[List[str], str]:
        """Extract reasoning steps and final answer from response."""
        
        lines = response.strip().split('\n')
        reasoning_steps = []
        final_answer = ""
        
        # Look for numbered steps or bullet points
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if it's a reasoning step
            if re.match(r'^(\d+\.|\d+\)|\•|\-|\*)', line):
                reasoning_steps.append(line)
            elif any(keyword in line.lower() for keyword in [
                'therefore', 'final answer', 'conclusion', 'result is', 'answer is'
            ]):
                final_answer = line
        
        # If no clear final answer, use last meaningful line
        if not final_answer and lines:
            # Get last non-empty line
            for line in reversed(lines):
                if line.strip():
                    final_answer = line.strip()
                    break
        
        return reasoning_steps, final_answer
    
    def _validate_reasoning(
        self,
        reasoning_steps: List[str],
        final_answer: str
    ) -> tuple[bool, str]:
        """Validate the reasoning chain for logical consistency."""
        
        if not reasoning_steps:
            return False, "No reasoning steps provided"
        
        if not final_answer:
            return False, "No final answer provided"
        
        # Check for common reasoning errors
        combined = ' '.join(reasoning_steps + [final_answer]).lower()
        
        error_patterns = [
            ("i don't know", "Response indicates uncertainty"),
            ("cannot be determined", "Response indicates inability to solve"),
            ("not enough information", "Response indicates missing information"),
            ("unclear", "Response indicates lack of clarity"),
        ]
        
        for pattern, message in error_patterns:
            if pattern in combined:
                return False, message
        
        return True, "Reasoning appears valid"
    
    async def _retry_with_feedback(
        self,
        original_prompt: str,
        feedback: str,
        llm_service: Any
    ) -> str:
        """Retry with validation feedback."""
        
        retry_prompt = f"""{original_prompt}

Previous attempt had issues: {feedback}

Please try again, being more careful with your reasoning:
"""
        
        return await llm_service.generate(
            prompt=retry_prompt,
            model=self.model,
            temperature=self.config.temperature * 0.7  # Lower temperature for retry
        )
