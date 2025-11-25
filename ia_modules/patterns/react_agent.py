"""
ReAct (Reasoning + Acting) pattern implementation.

Interleaves reasoning and acting in a loop for tool-using agents.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import json
import re


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
    action_model: str = "gpt-3.5-turbo"
    tools: Optional[List[str]] = None
    verbose: bool = True
    temperature: float = 0.7


class ReActAgent:
    """
    ReAct agent that interleaves reasoning and acting.
    
    Industry standard pattern used by LangChain, AutoGPT, BabyAGI.
    
    Example:
        agent = ReActAgent(
            name="research_assistant",
            task="Find the population of Tokyo and compare to NYC",
            tools=["search", "calculator", "wikipedia"],
            config=ReActConfig(max_iterations=10)
        )
    
    Research: "ReAct: Synergizing Reasoning and Acting in Language Models"
              (Yao et al., 2022)
    """
    
    def __init__(
        self,
        name: str,
        task: str,
        tools: Optional[List[str]] = None,
        config: Optional[ReActConfig] = None,
        **kwargs
    ):
        self.name = name
        self.task = task
        self.config = config or ReActConfig()
        self.config.tools = tools or []
        self.trajectory: List[ReActStep] = []
        self.kwargs = kwargs
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ReAct reasoning-acting loop."""
        
        llm_service = context.get('services', {}).get('llm')
        tool_service = context.get('services', {}).get('tools')
        
        if not llm_service:
            raise ValueError("LLM service not found in context")
        if not tool_service:
            raise ValueError("Tool service not found in context")
        
        for iteration in range(self.config.max_iterations):
            # THINK: Reason about what to do next
            thought, action, action_input = await self._think(llm_service, context)
            
            if action == "finish":
                # Agent decided it's done
                return self._build_final_answer(thought, action_input)
            
            # Record thought and planned action
            step = ReActStep(
                state=AgentState.THINKING,
                thought=thought,
                action=action,
                action_input=action_input
            )
            
            # ACT: Execute the chosen action
            try:
                observation = await self._act(action, action_input, tool_service, context)
            except Exception as e:
                observation = f"Error executing {action}: {str(e)}"
            
            # Record observation
            step.observation = observation
            step.state = AgentState.OBSERVING
            self.trajectory.append(step)
            
            if self.config.verbose:
                self._log_step(step)
        
        # Max iterations reached
        return {
            "answer": "Task incomplete - reached maximum iterations",
            "trajectory": [self._format_step(s) for s in self.trajectory],
            "iterations": self.config.max_iterations,
            "success": False
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
            temperature=self.config.temperature
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

Format your response as:
Thought: [your reasoning about what to do]
Action: [tool name]
Action Input: {{"param": "value"}}

Or to finish:
Thought: [final reasoning]
Action: finish
Action Input: {{"answer": "your final answer"}}

{trajectory_str}

What should you do next?
"""
        return prompt
    
    def _format_tools(self) -> str:
        """Format available tools for prompt."""
        
        # Default tool descriptions
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
            else:
                lines.append(f"- {tool}: Available tool")
        
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
                lines.append(f"Action Input: {json.dumps(step.action_input)}")
            if step.observation:
                lines.append(f"Observation: {step.observation}")
        
        lines.append("")
        return '\n'.join(lines)
    
    def _parse_thinking_response(self, response: str) -> tuple[str, str, Dict]:
        """Parse LLM response into thought, action, action_input."""
        
        thought = ""
        action = ""
        action_input = {}
        
        # Parse line by line
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip().lower()
            
            elif line.startswith("Action Input:"):
                input_str = line.replace("Action Input:", "").strip()
                try:
                    action_input = json.loads(input_str)
                except json.JSONDecodeError:
                    # Try to extract key-value pairs
                    match = re.search(r'\{(.+)\}', input_str)
                    if match:
                        try:
                            action_input = json.loads('{' + match.group(1) + '}')
                        except json.JSONDecodeError:
                            action_input = {"raw": input_str}
                    else:
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
        
        if action not in self.config.tools and action != "finish":
            return f"Error: Unknown tool '{action}'. Available tools: {', '.join(self.config.tools)}"
        
        if action == "finish":
            return "Task completed"
        
        # Execute tool
        result = await tool_service.execute(action, action_input, context)
        return str(result)
    
    def _build_final_answer(
        self,
        final_thought: str,
        action_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build final result."""
        
        answer = action_input.get("answer", final_thought)
        
        return {
            "answer": answer,
            "trajectory": [self._format_step(s) for s in self.trajectory],
            "iterations": len(self.trajectory),
            "success": True
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
        
        print(f"\n{'='*60}")
        print(f"Thought: {step.thought}")
        print(f"Action: {step.action}")
        if step.action_input:
            print(f"Action Input: {json.dumps(step.action_input, indent=2)}")
        if step.observation:
            print(f"Observation: {step.observation}")
        print(f"{'='*60}")
