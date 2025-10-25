# Advanced AI Patterns Implementation Guide

**Date:** October 25, 2025  
**Target Version:** v0.0.4+  
**Status:** Design Document for Future Implementation

---

## Overview

This guide provides detailed specifications for implementing advanced AI agent patterns in IA Modules. Each pattern includes conceptual explanation, code examples, and integration strategies.

---

## 1. Chain-of-Thought (CoT) Prompting

### What It Is
Chain-of-Thought prompting encourages LLMs to break down complex reasoning into explicit intermediate steps, significantly improving accuracy on tasks requiring multi-step reasoning.

### Why It Matters
- **Accuracy Boost:** 20-40% improvement on math, logic, and reasoning tasks
- **Transparency:** Users can see the reasoning process
- **Debuggability:** Easier to identify where reasoning fails
- **Trust:** Explanations build user confidence

### Research Background
- **Paper:** "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)
- **Key Finding:** Simply adding "Let's think step by step" improves reasoning

### Implementation Design

```python
# File: ia_modules/patterns/chain_of_thought.py

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from ia_modules.steps.base import BaseStep

@dataclass
class CoTConfig:
    """Configuration for Chain-of-Thought reasoning."""
    show_reasoning: bool = True
    reasoning_depth: int = 3
    validation_step: bool = True
    format: str = "numbered"  # "numbered", "bullet", "freeform"

class ChainOfThoughtStep(BaseStep):
    """
    Execute an LLM step with explicit chain-of-thought reasoning.
    
    Example:
        step = ChainOfThoughtStep(
            name="solve_math",
            prompt="What is 15% of 80?",
            model="gpt-4",
            config=CoTConfig(show_reasoning=True, reasoning_depth=3)
        )
    """
    
    def __init__(
        self,
        name: str,
        prompt: str,
        model: str = "gpt-4",
        config: Optional[CoTConfig] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.prompt = prompt
        self.model = model
        self.config = config or CoTConfig()
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with chain-of-thought reasoning."""
        
        # Build CoT prompt
        cot_prompt = self._build_cot_prompt(context)
        
        # Get LLM response
        llm_service = context.get('services').get('llm')
        response = await llm_service.generate(
            prompt=cot_prompt,
            model=self.model,
            temperature=0.7  # Some creativity for reasoning
        )
        
        # Parse reasoning steps and final answer
        reasoning_steps, final_answer = self._parse_response(response)
        
        # Optional validation step
        if self.config.validation_step:
            is_valid, validation_msg = await self._validate_reasoning(
                reasoning_steps, final_answer, context
            )
            if not is_valid:
                # Retry with validation feedback
                response = await self._retry_with_feedback(
                    cot_prompt, validation_msg, llm_service
                )
                reasoning_steps, final_answer = self._parse_response(response)
        
        result = {
            "answer": final_answer,
            "reasoning": reasoning_steps if self.config.show_reasoning else None,
            "raw_response": response,
            "model_used": self.model
        }
        
        return result
    
    def _build_cot_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt that encourages step-by-step reasoning."""
        
        base_prompt = self.prompt.format(**context)
        
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
            if line.startswith(('1.', '2.', '3.', '•', '-')):
                reasoning_steps.append(line)
            elif any(keyword in line.lower() for keyword in ['therefore', 'final answer', 'conclusion']):
                final_answer = line
        
        # If no clear final answer, use last line
        if not final_answer and lines:
            final_answer = lines[-1].strip()
        
        return reasoning_steps, final_answer
    
    async def _validate_reasoning(
        self,
        reasoning_steps: List[str],
        final_answer: str,
        context: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Validate the reasoning chain for logical consistency."""
        
        # Simple validation: check if answer follows from steps
        # In production, this could be another LLM call
        
        if not reasoning_steps:
            return False, "No reasoning steps provided"
        
        if not final_answer:
            return False, "No final answer provided"
        
        # Check for common reasoning errors
        combined = ' '.join(reasoning_steps + [final_answer]).lower()
        
        error_patterns = [
            "i don't know",
            "cannot be determined",
            "not enough information"
        ]
        
        for pattern in error_patterns:
            if pattern in combined:
                return False, f"Reasoning contains uncertainty: {pattern}"
        
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
            temperature=0.5  # Lower temperature for retry
        )

```

### Usage Example

```python
# In a pipeline definition
{
    "name": "math_reasoning",
    "steps": [
        {
            "name": "solve_problem",
            "type": "chain_of_thought",
            "config": {
                "prompt": "A store has 120 apples. They sell 30% in the morning and 25% of the remainder in the afternoon. How many apples are left?",
                "model": "gpt-4",
                "show_reasoning": True,
                "reasoning_depth": 4,
                "validation_step": True
            }
        }
    ]
}
```

### Expected Output

```json
{
    "answer": "63 apples remain",
    "reasoning": [
        "1. Initial apples: 120",
        "2. Morning sales: 30% of 120 = 0.30 × 120 = 36 apples",
        "3. Remaining after morning: 120 - 36 = 84 apples",
        "4. Afternoon sales: 25% of 84 = 0.25 × 84 = 21 apples",
        "5. Final remaining: 84 - 21 = 63 apples"
    ]
}
```

---

## 2. Tree of Thoughts (ToT)

### What It Is
Tree of Thoughts extends CoT by exploring multiple reasoning paths simultaneously, evaluating each branch, and pruning unpromising paths. Think of it as "breadth-first search" for reasoning.

### Why It Matters
- **Better Solutions:** Explores multiple approaches, finds optimal path
- **Complex Problems:** Excels at puzzles, planning, creative tasks
- **Adaptability:** Can backtrack from dead ends
- **Quality:** 40-70% improvement over CoT on hard problems

### Research Background
- **Paper:** "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023)
- **Key Innovation:** Systematic exploration of reasoning space

### Implementation Design

```python
# File: ia_modules/patterns/tree_of_thoughts.py

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio

class PruningStrategy(Enum):
    """Strategy for pruning unpromising branches."""
    BEST_FIRST = "best_first"  # Keep only top-k at each level
    THRESHOLD = "threshold"     # Keep all above threshold
    BEAM_SEARCH = "beam_search" # Fixed beam width

@dataclass
class ToTNode:
    """Node in the Tree of Thoughts."""
    thought: str
    parent: Optional['ToTNode'] = None
    children: List['ToTNode'] = field(default_factory=list)
    score: float = 0.0
    depth: int = 0
    is_solution: bool = False

@dataclass
class ToTConfig:
    """Configuration for Tree of Thoughts reasoning."""
    branching_factor: int = 3      # Number of branches per node
    max_depth: int = 4             # Maximum tree depth
    pruning_strategy: PruningStrategy = PruningStrategy.BEST_FIRST
    beam_width: int = 2            # For beam search
    threshold: float = 0.6         # For threshold pruning
    evaluation_model: str = "gpt-4"  # Model for evaluating thoughts

class TreeOfThoughtsStep(BaseStep):
    """
    Execute reasoning using Tree of Thoughts pattern.
    
    Example:
        step = TreeOfThoughtsStep(
            name="solve_puzzle",
            prompt="How can we fit 10 items in 3 boxes with constraints...",
            evaluation_fn=score_solution,
            config=ToTConfig(branching_factor=3, max_depth=4)
        )
    """
    
    def __init__(
        self,
        name: str,
        prompt: str,
        evaluation_fn: Optional[Callable] = None,
        config: Optional[ToTConfig] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.prompt = prompt
        self.evaluation_fn = evaluation_fn
        self.config = config or ToTConfig()
        self.tree_root: Optional[ToTNode] = None
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Tree of Thoughts reasoning."""
        
        llm_service = context.get('services').get('llm')
        
        # Initialize root node
        self.tree_root = ToTNode(
            thought="Initial problem state",
            depth=0
        )
        
        # Build tree level by level
        current_level = [self.tree_root]
        
        for depth in range(self.config.max_depth):
            next_level = []
            
            # Generate branches for each node in current level
            for node in current_level:
                if node.is_solution:
                    continue
                
                # Generate candidate thoughts
                branches = await self._generate_branches(node, llm_service, context)
                
                # Evaluate each branch
                for branch_text in branches:
                    branch_node = ToTNode(
                        thought=branch_text,
                        parent=node,
                        depth=depth + 1
                    )
                    
                    # Score this thought
                    branch_node.score = await self._evaluate_thought(
                        branch_node, llm_service, context
                    )
                    
                    # Check if it's a solution
                    branch_node.is_solution = await self._is_solution(
                        branch_node, llm_service, context
                    )
                    
                    node.children.append(branch_node)
                    next_level.append(branch_node)
            
            # Prune unpromising branches
            current_level = self._prune_branches(next_level)
            
            # Early exit if we found a solution
            solutions = [n for n in current_level if n.is_solution]
            if solutions:
                best_solution = max(solutions, key=lambda n: n.score)
                return self._build_result(best_solution)
        
        # No perfect solution found, return best path
        best_leaf = max(current_level, key=lambda n: n.score)
        return self._build_result(best_leaf)
    
    async def _generate_branches(
        self,
        node: ToTNode,
        llm_service: Any,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate branching thoughts from current node."""
        
        # Build prompt with current reasoning path
        path = self._get_path_to_root(node)
        
        prompt = f"""Given the problem:
{self.prompt}

Current reasoning path:
{self._format_path(path)}

Generate {self.config.branching_factor} different next steps to explore.
Each should be a distinct approach or consideration.
Format as numbered list:
"""
        
        response = await llm_service.generate(
            prompt=prompt,
            model=self.config.evaluation_model,
            temperature=0.9  # High creativity for diversity
        )
        
        # Parse branches
        branches = []
        for line in response.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove number prefix
                branch = line.split('.', 1)[1].strip() if '.' in line else line
                branches.append(branch)
        
        return branches[:self.config.branching_factor]
    
    async def _evaluate_thought(
        self,
        node: ToTNode,
        llm_service: Any,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate how promising this thought is (0-1 score)."""
        
        if self.evaluation_fn:
            # Use custom evaluation function
            return self.evaluation_fn(node, context)
        
        # Default: Ask LLM to evaluate
        path = self._get_path_to_root(node)
        
        eval_prompt = f"""Problem: {self.prompt}

Reasoning path:
{self._format_path(path)}

On a scale of 0-10, how likely is this reasoning path to lead to a correct solution?
Consider:
- Logical consistency
- Progress toward solution
- Potential dead ends

Score (0-10):"""
        
        response = await llm_service.generate(
            prompt=eval_prompt,
            model=self.config.evaluation_model,
            temperature=0.3  # Low temperature for consistent scoring
        )
        
        # Extract numeric score
        try:
            score_str = response.strip().split('\n')[0]
            score = float(''.join(c for c in score_str if c.isdigit() or c == '.'))
            return min(score / 10.0, 1.0)  # Normalize to 0-1
        except:
            return 0.5  # Default middle score if parsing fails
    
    async def _is_solution(
        self,
        node: ToTNode,
        llm_service: Any,
        context: Dict[str, Any]
    ) -> bool:
        """Check if this thought represents a complete solution."""
        
        # Check if thought contains solution indicators
        thought_lower = node.thought.lower()
        solution_keywords = ['therefore', 'final answer', 'solution is', 'result is']
        
        return any(keyword in thought_lower for keyword in solution_keywords)
    
    def _prune_branches(self, nodes: List[ToTNode]) -> List[ToTNode]:
        """Prune unpromising branches based on strategy."""
        
        if not nodes:
            return nodes
        
        strategy = self.config.pruning_strategy
        
        if strategy == PruningStrategy.BEST_FIRST:
            # Keep only top-k nodes
            sorted_nodes = sorted(nodes, key=lambda n: n.score, reverse=True)
            return sorted_nodes[:self.config.beam_width]
        
        elif strategy == PruningStrategy.THRESHOLD:
            # Keep all nodes above threshold
            return [n for n in nodes if n.score >= self.config.threshold]
        
        elif strategy == PruningStrategy.BEAM_SEARCH:
            # Fixed beam width
            sorted_nodes = sorted(nodes, key=lambda n: n.score, reverse=True)
            return sorted_nodes[:self.config.beam_width]
        
        return nodes
    
    def _get_path_to_root(self, node: ToTNode) -> List[ToTNode]:
        """Get path from node to root."""
        path = []
        current = node
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def _format_path(self, path: List[ToTNode]) -> str:
        """Format reasoning path as string."""
        return '\n'.join(f"{i}. {node.thought}" for i, node in enumerate(path, 1))
    
    def _build_result(self, solution_node: ToTNode) -> Dict[str, Any]:
        """Build final result from solution node."""
        
        path = self._get_path_to_root(solution_node)
        
        return {
            "solution": solution_node.thought,
            "reasoning_path": [n.thought for n in path],
            "confidence": solution_node.score,
            "tree_depth": solution_node.depth,
            "is_complete_solution": solution_node.is_solution
        }

```

### Usage Example

```python
# Example: Game of 24 puzzle
step = TreeOfThoughtsStep(
    name="solve_24_puzzle",
    prompt="Use the numbers 4, 9, 10, 13 with operations +, -, *, / to get 24",
    config=ToTConfig(
        branching_factor=4,  # Try 4 different operations
        max_depth=3,         # 3 operations needed (4 numbers → 1)
        pruning_strategy=PruningStrategy.BEST_FIRST,
        beam_width=3
    )
)
```

### Expected Output

```json
{
    "solution": "(10 - 4) * (13 - 9) = 6 * 4 = 24",
    "reasoning_path": [
        "Consider operations between 10 and 4",
        "10 - 4 = 6, which is a factor of 24",
        "Look for way to make 4 from remaining numbers",
        "13 - 9 = 4",
        "Multiply results: 6 * 4 = 24"
    ],
    "confidence": 0.95,
    "tree_depth": 3,
    "is_complete_solution": true
}
```

---

## 3. ReAct Pattern (Reasoning + Acting)

### What It Is
ReAct interleaves reasoning (thinking) and acting (using tools) in a loop. The agent reasons about what to do, takes an action, observes the result, and repeats until the task is complete.

### Why It Matters
- **Industry Standard:** Used by LangChain, AutoGPT, BabyAGI
- **Tool Use:** Essential for agents that interact with external systems
- **Adaptability:** Agent adjusts based on action results
- **Grounding:** Actions provide factual grounding for reasoning

### Research Background
- **Paper:** "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
- **Key Innovation:** Explicit reasoning traces improve tool use by 40%

### Implementation Design

```python
# File: ia_modules/patterns/react_agent.py

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class AgentState(Enum):
    """Current state of the ReAct agent."""
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"

@dataclass
class ReActStep:
    """Single step in ReAct loop."""
    state: AgentState
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None

@dataclass
class ReActConfig:
    """Configuration for ReAct agent."""
    max_iterations: int = 10
    reasoning_model: str = "gpt-4"
    action_model: str = "gpt-3.5-turbo"  # Can use cheaper model for actions
    tools: List[str] = None
    verbose: bool = True

class ReActAgent(BaseStep):
    """
    ReAct agent that interleaves reasoning and acting.
    
    Example:
        agent = ReActAgent(
            name="research_assistant",
            task="Find the population of Tokyo and compare to NYC",
            tools=["search", "calculator", "wikipedia"],
            config=ReActConfig(max_iterations=10)
        )
    """
    
    def __init__(
        self,
        name: str,
        task: str,
        tools: Optional[List[str]] = None,
        config: Optional[ReActConfig] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.task = task
        self.config = config or ReActConfig()
        self.config.tools = tools or []
        self.trajectory: List[ReActStep] = []
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ReAct reasoning-acting loop."""
        
        llm_service = context.get('services').get('llm')
        tool_service = context.get('services').get('tools')
        
        for iteration in range(self.config.max_iterations):
            # THINK: Reason about what to do next
            thought, action, action_input = await self._think(
                llm_service, context
            )
            
            if action == "finish":
                # Agent decided it's done
                return self._build_final_answer(thought, context)
            
            # Record thought
            step = ReActStep(
                state=AgentState.THINKING,
                thought=thought,
                action=action,
                action_input=action_input
            )
            self.trajectory.append(step)
            
            # ACT: Execute the chosen action
            observation = await self._act(
                action, action_input, tool_service, context
            )
            
            # Record observation
            step.observation = observation
            
            if self.config.verbose:
                self._log_step(step)
        
        # Max iterations reached
        return {
            "answer": "Task incomplete - reached maximum iterations",
            "trajectory": [self._format_step(s) for s in self.trajectory],
            "iterations": self.config.max_iterations
        }
    
    async def _think(
        self,
        llm_service: Any,
        context: Dict[str, Any]
    ) -> tuple[str, str, Dict[str, Any]]:
        """Reason about what action to take next."""
        
        # Build prompt with task and trajectory
        prompt = self._build_thinking_prompt()
        
        response = await llm_service.generate(
            prompt=prompt,
            model=self.config.reasoning_model,
            temperature=0.7
        )
        
        # Parse: Thought, Action, Action Input
        thought, action, action_input = self._parse_thinking_response(response)
        
        return thought, action, action_input
    
    def _build_thinking_prompt(self) -> str:
        """Build prompt for reasoning step."""
        
        tools_desc = self._format_tools()
        trajectory_str = self._format_trajectory()
        
        prompt = f"""You are an assistant that solves tasks by reasoning and taking actions.

Task: {self.task}

Available tools:
{tools_desc}

You can use these tools by outputting:
Thought: [your reasoning about what to do]
Action: [tool name]
Action Input: {{"param": "value"}}

Or finish with:
Thought: [final reasoning]
Action: finish
Action Input: {{"answer": "your final answer"}}

{trajectory_str}

Now, what should you do next?
"""
        return prompt
    
    def _format_tools(self) -> str:
        """Format available tools for prompt."""
        
        # In production, get descriptions from tool registry
        tool_descriptions = {
            "search": "Search the internet for information. Input: {\"query\": \"search terms\"}",
            "calculator": "Perform calculations. Input: {\"expression\": \"1 + 2 * 3\"}",
            "wikipedia": "Get Wikipedia article. Input: {\"page\": \"Tokyo\"}",
            "finish": "Complete the task. Input: {\"answer\": \"final answer\"}"
        }
        
        lines = []
        for tool in self.config.tools + ["finish"]:
            if tool in tool_descriptions:
                lines.append(f"- {tool}: {tool_descriptions[tool]}")
        
        return '\n'.join(lines)
    
    def _format_trajectory(self) -> str:
        """Format previous steps for context."""
        
        if not self.trajectory:
            return ""
        
        lines = ["Previous steps:"]
        for i, step in enumerate(self.trajectory, 1):
            lines.append(f"\nStep {i}:")
            lines.append(f"Thought: {step.thought}")
            lines.append(f"Action: {step.action}")
            if step.action_input:
                lines.append(f"Action Input: {step.action_input}")
            if step.observation:
                lines.append(f"Observation: {step.observation}")
        
        lines.append("")  # Empty line before next step
        return '\n'.join(lines)
    
    def _parse_thinking_response(self, response: str) -> tuple[str, str, Dict]:
        """Parse LLM response into thought, action, action_input."""
        
        lines = response.strip().split('\n')
        thought = ""
        action = ""
        action_input = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                input_str = line.replace("Action Input:", "").strip()
                try:
                    import json
                    action_input = json.loads(input_str)
                except:
                    action_input = {"raw": input_str}
        
        return thought, action, action_input
    
    async def _act(
        self,
        action: str,
        action_input: Dict[str, Any],
        tool_service: Any,
        context: Dict[str, Any]
    ) -> str:
        """Execute the chosen action and return observation."""
        
        if action not in self.config.tools:
            return f"Error: Unknown tool '{action}'"
        
        try:
            result = await tool_service.execute(action, action_input, context)
            return str(result)
        except Exception as e:
            return f"Error executing {action}: {str(e)}"
    
    def _build_final_answer(
        self,
        final_thought: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build final result."""
        
        return {
            "answer": final_thought,
            "trajectory": [self._format_step(s) for s in self.trajectory],
            "iterations": len(self.trajectory)
        }
    
    def _format_step(self, step: ReActStep) -> Dict[str, Any]:
        """Format step for output."""
        
        return {
            "thought": step.thought,
            "action": step.action,
            "action_input": step.action_input,
            "observation": step.observation
        }
    
    def _log_step(self, step: ReActStep):
        """Log step for debugging."""
        
        print(f"\nThought: {step.thought}")
        print(f"Action: {step.action}")
        if step.action_input:
            print(f"Action Input: {step.action_input}")
        if step.observation:
            print(f"Observation: {step.observation}")

```

### Usage Example

```python
# Research assistant example
agent = ReActAgent(
    name="research_assistant",
    task="What is the population of Tokyo and how does it compare to New York City?",
    tools=["search", "wikipedia", "calculator"],
    config=ReActConfig(
        max_iterations=10,
        reasoning_model="gpt-4",
        verbose=True
    )
)

result = await agent.execute(context)
```

### Expected Output

```json
{
    "answer": "Tokyo has approximately 14 million people (37 million in greater metro area), while NYC has approximately 8.3 million (20 million in metro area). Tokyo's city proper is 1.7x larger, and its metro area is 1.85x larger than NYC's.",
    "trajectory": [
        {
            "thought": "I need to find the current population of Tokyo",
            "action": "search",
            "action_input": {"query": "Tokyo population 2025"},
            "observation": "Tokyo's population is approximately 14 million..."
        },
        {
            "thought": "Now I need NYC's population for comparison",
            "action": "search",
            "action_input": {"query": "New York City population 2025"},
            "observation": "NYC has a population of about 8.3 million..."
        },
        {
            "thought": "Let me calculate the ratio",
            "action": "calculator",
            "action_input": {"expression": "14 / 8.3"},
            "observation": "1.686746..."
        }
    ],
    "iterations": 3
}
```

---

## 4. Self-Consistency

### What It Is
Self-Consistency samples multiple independent reasoning paths (with higher temperature) and uses majority voting or weighted consensus to select the final answer.

### Why It Matters
- **Reduces Hallucinations:** 30-50% reduction through consensus
- **Confidence Estimation:** Agreement level indicates confidence
- **Robustness:** Less sensitive to prompt variations
- **Better Accuracy:** 10-20% improvement on reasoning tasks

### Implementation Design

```python
# File: ia_modules/patterns/self_consistency.py

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import Counter
import asyncio

class VotingStrategy(Enum):
    """Strategy for combining multiple answers."""
    MAJORITY = "majority"           # Most common answer wins
    WEIGHTED = "weighted"           # Weight by confidence scores
    CONFIDENCE_THRESHOLD = "threshold"  # Require minimum agreement
    UNANIMOUS = "unanimous"         # All must agree

@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency pattern."""
    num_samples: int = 5
    temperature: float = 0.8
    voting_strategy: VotingStrategy = VotingStrategy.MAJORITY
    confidence_threshold: float = 0.6
    model: str = "gpt-4"

class SelfConsistencyStep(BaseStep):
    """
    Generate multiple reasoning paths and use consensus for answer.
    
    Example:
        step = SelfConsistencyStep(
            name="robust_answer",
            prompt="What is the capital of Australia?",
            config=SelfConsistencyConfig(
                num_samples=5,
                voting_strategy=VotingStrategy.MAJORITY
            )
        )
    """
    
    def __init__(
        self,
        name: str,
        prompt: str,
        config: Optional[SelfConsistencyConfig] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.prompt = prompt
        self.config = config or SelfConsistencyConfig()
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute self-consistency reasoning."""
        
        llm_service = context.get('services').get('llm')
        
        # Generate multiple independent samples
        samples = await self._generate_samples(llm_service, context)
        
        # Extract answers from samples
        answers = [self._extract_answer(s) for s in samples]
        
        # Apply voting strategy
        final_answer, confidence = self._vote(answers, samples)
        
        return {
            "answer": final_answer,
            "confidence": confidence,
            "num_samples": len(samples),
            "all_answers": answers,
            "agreement_rate": self._calculate_agreement(answers),
            "voting_strategy": self.config.voting_strategy.value
        }
    
    async def _generate_samples(
        self,
        llm_service: Any,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate multiple independent reasoning paths."""
        
        # Generate all samples concurrently
        tasks = [
            llm_service.generate(
                prompt=self.prompt,
                model=self.config.model,
                temperature=self.config.temperature
            )
            for _ in range(self.config.num_samples)
        ]
        
        samples = await asyncio.gather(*tasks)
        return samples
    
    def _extract_answer(self, sample: str) -> str:
        """Extract the final answer from a reasoning path."""
        
        # Look for common answer patterns
        lines = sample.strip().split('\n')
        
        for line in reversed(lines):  # Check from bottom
            line_lower = line.lower().strip()
            
            # Check for answer indicators
            if any(indicator in line_lower for indicator in [
                'answer:', 'therefore', 'conclusion:', 'result:',
                'final answer', 'the answer is'
            ]):
                # Extract text after indicator
                for indicator in ['answer:', 'therefore', 'conclusion:', 
                                'result:', 'final answer', 'the answer is']:
                    if indicator in line_lower:
                        answer = line_lower.split(indicator, 1)[1].strip()
                        # Clean up
                        answer = answer.rstrip('.!,;')
                        return answer
        
        # Fallback: use last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        
        return sample.strip()
    
    def _vote(
        self,
        answers: List[str],
        samples: List[str]
    ) -> tuple[str, float]:
        """Apply voting strategy to select final answer."""
        
        strategy = self.config.voting_strategy
        
        if strategy == VotingStrategy.MAJORITY:
            return self._majority_vote(answers)
        
        elif strategy == VotingStrategy.WEIGHTED:
            return self._weighted_vote(answers, samples)
        
        elif strategy == VotingStrategy.CONFIDENCE_THRESHOLD:
            return self._threshold_vote(answers)
        
        elif strategy == VotingStrategy.UNANIMOUS:
            return self._unanimous_vote(answers)
        
        return answers[0], 1.0 / len(answers)
    
    def _majority_vote(self, answers: List[str]) -> tuple[str, float]:
        """Simple majority voting."""
        
        # Normalize answers for comparison (lowercase, strip)
        normalized = [a.lower().strip() for a in answers]
        
        # Count votes
        counter = Counter(normalized)
        winner_normalized, count = counter.most_common(1)[0]
        
        # Find original answer (preserve case)
        winner = next(a for a, n in zip(answers, normalized) if n == winner_normalized)
        
        confidence = count / len(answers)
        
        return winner, confidence
    
    def _weighted_vote(
        self,
        answers: List[str],
        samples: List[str]
    ) -> tuple[str, float]:
        """Weight answers by their reasoning quality."""
        
        # Score each sample's reasoning
        scores = [self._score_reasoning(sample) for sample in samples]
        
        # Normalize answers
        normalized = [a.lower().strip() for a in answers]
        
        # Weight votes by scores
        weighted_counts = {}
        for answer, score in zip(normalized, scores):
            weighted_counts[answer] = weighted_counts.get(answer, 0) + score
        
        # Find winner
        winner_normalized = max(weighted_counts, key=weighted_counts.get)
        winner = next(a for a, n in zip(answers, normalized) if n == winner_normalized)
        
        # Calculate confidence
        total_weight = sum(weighted_counts.values())
        confidence = weighted_counts[winner_normalized] / total_weight if total_weight > 0 else 0
        
        return winner, confidence
    
    def _threshold_vote(self, answers: List[str]) -> tuple[str, float]:
        """Require minimum agreement threshold."""
        
        winner, confidence = self._majority_vote(answers)
        
        if confidence < self.config.confidence_threshold:
            return (
                f"Low confidence ({confidence:.2f} < {self.config.confidence_threshold}). "
                f"Top answer: {winner}",
                confidence
            )
        
        return winner, confidence
    
    def _unanimous_vote(self, answers: List[str]) -> tuple[str, float]:
        """Require all answers to agree."""
        
        normalized = [a.lower().strip() for a in answers]
        
        if len(set(normalized)) == 1:
            return answers[0], 1.0
        else:
            # Not unanimous
            counter = Counter(normalized)
            most_common = counter.most_common(1)[0]
            return (
                f"No consensus. Most common answer ({most_common[1]}/{len(answers)}): "
                f"{next(a for a, n in zip(answers, normalized) if n == most_common[0])}",
                most_common[1] / len(answers)
            )
    
    def _score_reasoning(self, sample: str) -> float:
        """Score the quality of reasoning in a sample."""
        
        # Simple heuristic: longer, more structured reasoning = better
        lines = [l.strip() for l in sample.split('\n') if l.strip()]
        
        score = 0.0
        
        # More steps = better
        score += min(len(lines) * 0.1, 0.5)
        
        # Structured reasoning indicators
        indicators = ['first', 'second', 'then', 'therefore', 'because']
        for indicator in indicators:
            if indicator in sample.lower():
                score += 0.1
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _calculate_agreement(self, answers: List[str]) -> float:
        """Calculate overall agreement rate."""
        
        normalized = [a.lower().strip() for a in answers]
        counter = Counter(normalized)
        max_count = counter.most_common(1)[0][1] if counter else 0
        
        return max_count / len(answers) if answers else 0.0

```

### Usage Example

```python
step = SelfConsistencyStep(
    name="robust_capital_answer",
    prompt="What is the capital of Australia? Think step by step.",
    config=SelfConsistencyConfig(
        num_samples=5,
        temperature=0.8,
        voting_strategy=VotingStrategy.MAJORITY,
        confidence_threshold=0.6
    )
)
```

### Expected Output

```json
{
    "answer": "Canberra",
    "confidence": 1.0,
    "num_samples": 5,
    "all_answers": [
        "Canberra",
        "Canberra",
        "Canberra",
        "Canberra",
        "Canberra"
    ],
    "agreement_rate": 1.0,
    "voting_strategy": "majority"
}
```

---

## Summary & Integration Roadmap

### Implementation Priority

| Pattern | Complexity | Value | Priority | Estimated Effort |
|---------|-----------|-------|----------|-----------------|
| **Chain-of-Thought** | Low | High | 🔴 P0 | 1 week |
| **Self-Consistency** | Low | High | 🔴 P0 | 1 week |
| **ReAct** | Medium | Very High | 🔴 P0 | 2 weeks |
| **Tree of Thoughts** | High | Medium | 🟠 P1 | 2 weeks |
| **Constitutional AI** | Medium | Medium | 🟠 P1 | 1.5 weeks |

### Next Steps

1. **Week 1:** Implement Chain-of-Thought and Self-Consistency (foundational patterns)
2. **Week 2-3:** Implement ReAct (most requested pattern)
3. **Week 4-5:** Implement Tree of Thoughts (for complex problems)
4. **Week 6:** Integration testing and documentation

### Integration with Existing Framework

All patterns integrate as:
```python
# As pipeline steps
{
    "steps": [
        {"name": "reasoning", "type": "chain_of_thought", "config": {...}},
        {"name": "agent", "type": "react", "config": {...}},
        {"name": "consensus", "type": "self_consistency", "config": {...}}
    ]
}

# Or as standalone agents
from ia_modules.patterns import ReActAgent, ChainOfThoughtStep

agent = ReActAgent(name="assistant", task="...", tools=[...])
```

---

**Status:** Design document complete, ready for implementation in v0.0.4+

