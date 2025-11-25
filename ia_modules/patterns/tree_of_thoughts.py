"""
Tree of Thoughts (ToT) pattern implementation.

Explores multiple reasoning paths systematically using tree search.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import re


class PruningStrategy(Enum):
    """Strategy for pruning unpromising branches."""
    BEST_FIRST = "best_first"
    THRESHOLD = "threshold"
    BEAM_SEARCH = "beam_search"


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
    branching_factor: int = 3
    max_depth: int = 4
    pruning_strategy: PruningStrategy = PruningStrategy.BEST_FIRST
    beam_width: int = 2
    threshold: float = 0.6
    evaluation_model: str = "gpt-4"
    generation_temperature: float = 0.9
    evaluation_temperature: float = 0.3


class TreeOfThoughtsStep:
    """
    Execute reasoning using Tree of Thoughts pattern.
    
    Explores multiple reasoning paths and prunes unpromising branches.
    Achieves 40-70% improvement over Chain-of-Thought on complex problems.
    
    Example:
        step = TreeOfThoughtsStep(
            name="solve_puzzle",
            prompt="Use 4, 9, 10, 13 to make 24",
            evaluation_fn=score_solution,
            config=ToTConfig(branching_factor=3, max_depth=4)
        )
    
    Research: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
              (Yao et al., 2023)
    """
    
    def __init__(
        self,
        name: str,
        prompt: str,
        evaluation_fn: Optional[Callable] = None,
        config: Optional[ToTConfig] = None,
        **kwargs
    ):
        self.name = name
        self.prompt = prompt
        self.evaluation_fn = evaluation_fn
        self.config = config or ToTConfig()
        self.tree_root: Optional[ToTNode] = None
        self.kwargs = kwargs
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Tree of Thoughts reasoning."""
        
        llm_service = context.get('services', {}).get('llm')
        if not llm_service:
            raise ValueError("LLM service not found in context")
        
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
                    branch_node.is_solution = self._is_solution(branch_node)
                    
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
        if current_level:
            best_leaf = max(current_level, key=lambda n: n.score)
            return self._build_result(best_leaf)
        
        # Fallback if no nodes remain
        return {
            "solution": "No solution found",
            "reasoning_path": [],
            "confidence": 0.0,
            "tree_depth": 0,
            "is_complete_solution": False
        }
    
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
Format as numbered list (1., 2., 3., etc.):
"""
        
        response = await llm_service.generate(
            prompt=prompt,
            model=self.config.evaluation_model,
            temperature=self.config.generation_temperature
        )
        
        # Parse branches
        branches = []
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Match numbered items
            match = re.match(r'^(\d+)[\.\)]\s*(.+)', line)
            if match:
                branch = match.group(2).strip()
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

Respond with just a number from 0-10:"""
        
        response = await llm_service.generate(
            prompt=eval_prompt,
            model=self.config.evaluation_model,
            temperature=self.config.evaluation_temperature
        )
        
        # Extract numeric score
        try:
            # Find first number in response
            match = re.search(r'(\d+\.?\d*)', response.strip())
            if match:
                score = float(match.group(1))
                return min(score / 10.0, 1.0)
        except (ValueError, AttributeError):
            pass
        
        return 0.5  # Default middle score if parsing fails
    
    def _is_solution(self, node: ToTNode) -> bool:
        """Check if this thought represents a complete solution."""
        
        thought_lower = node.thought.lower()
        
        solution_keywords = [
            'therefore', 'final answer', 'solution is', 'result is',
            'answer is', 'equals', '=', 'conclusion'
        ]
        
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
