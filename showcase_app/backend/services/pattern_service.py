"""
Pattern Service

Demonstrates agentic design patterns built from first principles.

Core Concept:
AI Agent = LLM + System Prompt + Tools + Memory + Reasoning Pattern

Patterns Implemented:
1. Reflection: Self-critique and iterative improvement
2. Planning: Multi-step goal decomposition  
3. Tool Use: Dynamic capability selection
4. Agentic RAG: Query refinement and relevance evaluation
5. Metacognition: Self-monitoring and strategy adjustment

Each pattern demonstrates a fundamental building block of autonomous agents.
These are the patterns that frameworks like LangChain/CrewAI implement under the hood.

Reference: Building agents from scratch to understand what frameworks abstract away.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os
import time
from ia_modules.pipeline.llm_provider_service import LLMProviderService, LLMProvider
from .llm_monitoring_service import LLMMonitoringService


class PatternService:
    """Service for demonstrating agentic design patterns"""
    
    def __init__(self):
        self.pattern_history: Dict[str, List[Dict]] = {}
        self.llm_service = None
        self.monitoring_service = LLMMonitoringService()
        
        # Initialize LLM provider service
        try:
            self.llm_service = LLMProviderService()
            
            # Register providers based on available API keys
            if os.getenv("OPENAI_API_KEY"):
                self.llm_service.register_provider(
                    "openai",
                    LLMProvider.OPENAI,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
                    is_default=True
                )
            
            if os.getenv("ANTHROPIC_API_KEY"):
                self.llm_service.register_provider(
                    "anthropic",
                    LLMProvider.ANTHROPIC,
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                    model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
                )
            
            if os.getenv("GEMINI_API_KEY"):
                self.llm_service.register_provider(
                    "google",
                    LLMProvider.GOOGLE,
                    api_key=os.getenv("GEMINI_API_KEY"),
                    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
                )
            
            if not self.llm_service.providers:
                self.llm_service = None
                    
        except Exception as e:
            print(f"⚠ Failed to initialize LLM provider service: {e}")
            self.llm_service = None
    
    async def _monitored_llm_call(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Any:
        """
        Wrapper for LLM calls with rate limiting and usage tracking
        
        Args:
            prompt: The prompt to send
            temperature: Temperature parameter
            max_tokens: Max tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with usage data
            
        Raises:
            HTTPException: If rate limited or cost limit exceeded
        """
        from fastapi import HTTPException
        
        if self.llm_service is None:
            raise RuntimeError("LLM service not configured. Please set API keys in .env file.")
        
        # Check rate limits
        rate_check = self.monitoring_service.check_rate_limits(max_tokens)
        if not rate_check["allowed"]:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {rate_check['reason']}",
                headers={"Retry-After": str(int(rate_check["retry_after"]))}
            )
        
        # Make LLM call and track time
        start_time = time.time()
        response = await self.llm_service.generate_completion(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        duration = time.time() - start_time
        
        # Extract token counts from response
        usage = response.usage
        input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or usage.get("prompt_token_count", 0)
        output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or usage.get("candidates_token_count", 0)
        
        # Track usage and calculate cost
        usage_stats = self.monitoring_service.track_usage(
            provider=response.provider.value,
            model=response.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_seconds=duration
        )
        
        # Check if we exceeded cost limits (warn but don't block)
        if usage_stats.get("over_request_limit"):
            print(f"⚠ Warning: Request cost ${usage_stats['cost_usd']:.4f} exceeded limit ${self.monitoring_service.max_cost_per_request}")
        
        if usage_stats.get("over_daily_limit"):
            print(f"⚠ Warning: Daily spending ${usage_stats['daily_total_cost']:.2f} exceeded limit ${self.monitoring_service.daily_spending_limit}")
        
        # Attach usage stats to response
        response.usage_stats = usage_stats
        
        return response
    
    # ==================== REFLECTION PATTERN ====================
    
    async def reflection_example(
        self,
        initial_output: str,
        criteria: Dict[str, str],
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Demonstrate reflection pattern with self-critique
        
        Args:
            initial_output: Initial agent output
            criteria: Quality criteria to evaluate against
            max_iterations: Max improvement iterations
            
        Returns:
            Reflection history with improvements
        """
        iterations = []
        current_output = initial_output
        
        for i in range(max_iterations):
            critique = await self._llm_generate_critique(current_output, criteria)
            quality_score = self._calculate_quality_score(critique, criteria)
            improvements = self._extract_improvements(critique)
            
            iteration_data = {
                "iteration": i + 1,
                "output": current_output,
                "critique": critique,
                "quality_score": quality_score,
                "improvements_suggested": improvements,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            iterations.append(iteration_data)
            
            if quality_score >= 0.85:
                iteration_data["improved_output"] = current_output
                break
            
            current_output = await self._llm_apply_improvements(current_output, improvements, criteria)
        
        return {
            "pattern": "reflection",
            "initial_output": initial_output,
            "final_output": current_output,
            "iterations": iterations,
            "total_iterations": len(iterations),
            "final_quality_score": iterations[-1]["quality_score"]
        }
    
    async def _llm_generate_critique(self, output: str, criteria: Dict[str, str]) -> str:
        """Generate self-critique using real LLM"""
        if self.llm_service is None:
            raise RuntimeError("LLM service not configured. Please set API keys in .env file.")
            
        criteria_text = "\n".join([f"- {name}: {desc}" for name, desc in criteria.items()])
        
        prompt = f"""You are a critical evaluator analyzing text quality.
Provide honest, constructive critique based on the given criteria.
Focus on specific issues and be direct about weaknesses.

Evaluate this output against the criteria:

OUTPUT TO EVALUATE:
{output}

CRITERIA:
{criteria_text}

Provide a detailed critique addressing each criterion. Be specific about what works and what doesn't."""
        
        response = await self._monitored_llm_call(
            prompt=prompt,
            temperature=0.3,
            max_tokens=500
        )
        
        return response.content
    
    async def _llm_apply_improvements(self, output: str, improvements: List[str], criteria: Dict[str, str]) -> str:
        """Apply improvements to output using real LLM"""
        if self.llm_service is None:
            raise RuntimeError("LLM service not configured. Please set API keys in .env file.")
            
        improvements_text = "\n".join([f"- {imp}" for imp in improvements])
        criteria_text = "\n".join([f"- {name}: {desc}" for name, desc in criteria.items()])
        
        prompt = f"""You are an expert editor improving text quality.
Apply the suggested improvements while maintaining the core message.
Make specific, measurable improvements.

Improve this output by applying the suggested improvements:

CURRENT OUTPUT:
{output}

IMPROVEMENTS TO APPLY:
{improvements_text}

CRITERIA TO MEET:
{criteria_text}

Provide the improved version directly, without explanations."""
        
        response = await self._monitored_llm_call(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.content.strip()
    
    def _generate_critique(self, output: str, criteria: Dict[str, str]) -> str:
        """Simulate LLM generating self-critique"""
        critiques = []
        
        if "clarity" in criteria:
            if len(output) < 50:
                critiques.append("Output is too brief, needs more detail")
            elif "unclear" in output.lower():
                critiques.append("Language could be clearer")
            else:
                critiques.append("Clarity is acceptable")
        
        if "accuracy" in criteria:
            if "approximately" in output.lower() or "maybe" in output.lower():
                critiques.append("Contains uncertainty, needs verification")
            else:
                critiques.append("Appears accurate")
        
        if "completeness" in criteria:
            if len(output.split()) < 20:
                critiques.append("Response incomplete, missing key information")
            else:
                critiques.append("Reasonably complete")
        
        return ". ".join(critiques)
    
    def _calculate_quality_score(self, critique: str, criteria: Dict) -> float:
        """Calculate quality score from critique"""
        negative_words = ["too brief", "unclear", "incomplete", "uncertainty", "missing"]
        positive_words = ["acceptable", "accurate", "complete", "clear"]
        
        negative_count = sum(1 for word in negative_words if word in critique.lower())
        positive_count = sum(1 for word in positive_words if word in critique.lower())
        
        total_criteria = len(criteria)
        score = (positive_count - negative_count) / total_criteria if total_criteria > 0 else 0.5
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, 0.5 + score * 0.25))
    
    def _extract_improvements(self, critique: str) -> List[str]:
        """Extract improvement suggestions from critique"""
        improvements = []
        
        if "too brief" in critique.lower():
            improvements.append("Add more detailed explanations")
        if "unclear" in critique.lower():
            improvements.append("Use simpler, clearer language")
        if "incomplete" in critique.lower():
            improvements.append("Cover all aspects of the topic")
        if "uncertainty" in critique.lower():
            improvements.append("Verify facts and remove hedging language")
        
        return improvements
    
    def _apply_improvements(self, output: str, improvements: List[str]) -> str:
        """Simulate applying improvements to output"""
        improved = output
        
        # Simulate adding detail
        if "Add more detailed" in str(improvements):
            improved += " This provides comprehensive coverage of the key aspects."
        
        # Simulate clarification
        if "clearer language" in str(improvements):
            improved = improved.replace("might", "will").replace("maybe", "definitely")
        
        # Simulate adding completeness
        if "Cover all aspects" in str(improvements):
            improved += " Additionally, this addresses all relevant considerations."
        
        return improved
    
    # ==================== PLANNING PATTERN ====================
    
    async def planning_example(
        self,
        goal: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Demonstrate planning pattern with goal decomposition
        
        Args:
            goal: High-level goal to achieve
            constraints: Optional constraints (budget, time, etc.)
            
        Returns:
            Multi-step plan with reasoning
        """
        subgoals = await self._llm_decompose_goal(goal, constraints)
        
        steps = []
        for i, subgoal in enumerate(subgoals):
            step = {
                "step_number": i + 1,
                "subgoal": subgoal["description"],
                "reasoning": subgoal["reasoning"],
                "estimated_duration": subgoal["duration"],
                "dependencies": subgoal.get("dependencies", []),
                "success_criteria": subgoal["success_criteria"]
            }
            steps.append(step)
        
        return {
            "pattern": "planning",
            "goal": goal,
            "constraints": constraints or {},
            "total_steps": len(steps),
            "estimated_total_time": sum(s["estimated_duration"] for s in steps),
            "plan": steps,
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def _llm_decompose_goal(self, goal: str, constraints: Optional[Dict]) -> List[Dict]:
        """Decompose goal into actionable steps using real LLM"""
        if self.llm_service is None:
            raise RuntimeError("LLM service not configured. Please set API keys in .env file.")
            
        constraints_text = ""
        if constraints:
            constraints_text = "\n".join([f"- {k}: {v}" for k, v in constraints.items()])
        
        prompt = f"""You are an expert planner who breaks down complex goals into actionable steps.
Create a detailed, realistic plan with clear dependencies and success criteria.

Break down this goal into a step-by-step plan:

GOAL: {goal}

{f"CONSTRAINTS:{constraints_text}" if constraints_text else ""}

Create a plan with 3-5 steps. For each step, provide:
1. description: What needs to be done
2. reasoning: Why this step is important
3. duration: Estimated time in minutes
4. dependencies: Which previous steps must complete first (use step numbers, e.g. [1, 2])
5. success_criteria: How to know this step is complete (list of 2-3 criteria)

Return valid JSON in this format:
[
  {{
    "description": "Step description",
    "reasoning": "Why this step matters",
    "duration": 30,
    "dependencies": [],
    "success_criteria": ["Criterion 1", "Criterion 2"]
  }}
]"""
        
        result = await self.llm_service.generate_structured_output(
            prompt=prompt,
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "duration": {"type": "number"},
                        "dependencies": {"type": "array"},
                        "success_criteria": {"type": "array"}
                    }
                }
            }
        )
        
        if isinstance(result, dict) and "steps" in result:
            return result["steps"]
        elif isinstance(result, dict) and "plan" in result:
            return result["plan"]
        elif isinstance(result, list):
            return result
        else:
            return []
    
    def _decompose_goal(self, goal: str, constraints: Optional[Dict]) -> List[Dict]:
        """Break down high-level goal into subgoals"""
        if "research" in goal.lower():
            return [
                {
                    "description": "Define research question and scope",
                    "reasoning": "Clear scope prevents wasted effort",
                    "duration": 15,
                    "success_criteria": ["Well-defined question", "Clear boundaries"]
                },
                {
                    "description": "Gather relevant sources",
                    "reasoning": "Quality sources ensure accurate results",
                    "duration": 30,
                    "dependencies": [1],
                    "success_criteria": ["5+ credible sources", "Diverse perspectives"]
                },
                {
                    "description": "Analyze and synthesize information",
                    "reasoning": "Synthesis creates new insights",
                    "duration": 45,
                    "dependencies": [2],
                    "success_criteria": ["Key themes identified", "Contradictions noted"]
                },
                {
                    "description": "Draft findings report",
                    "reasoning": "Documentation enables sharing",
                    "duration": 30,
                    "dependencies": [3],
                    "success_criteria": ["Clear structure", "Evidence-based conclusions"]
                }
            ]
        
        elif "build" in goal.lower() or "create" in goal.lower():
            return [
                {
                    "description": "Requirements analysis",
                    "reasoning": "Understanding needs prevents rework",
                    "duration": 20,
                    "success_criteria": ["Requirements documented", "Stakeholders aligned"]
                },
                {
                    "description": "Design architecture",
                    "reasoning": "Good design enables scalability",
                    "duration": 40,
                    "dependencies": [1],
                    "success_criteria": ["Architecture diagram", "Technology choices justified"]
                },
                {
                    "description": "Implement core features",
                    "reasoning": "Core features deliver primary value",
                    "duration": 120,
                    "dependencies": [2],
                    "success_criteria": ["Core functionality works", "Tests passing"]
                },
                {
                    "description": "Testing and refinement",
                    "reasoning": "Quality assurance builds trust",
                    "duration": 40,
                    "dependencies": [3],
                    "success_criteria": ["All tests pass", "Edge cases handled"]
                }
            ]
        
        else:
            # Generic decomposition
            return [
                {
                    "description": "Analyze requirements",
                    "reasoning": "Understanding the need",
                    "duration": 15,
                    "success_criteria": ["Clear objectives"]
                },
                {
                    "description": "Plan approach",
                    "reasoning": "Strategy before execution",
                    "duration": 20,
                    "dependencies": [1],
                    "success_criteria": ["Approach documented"]
                },
                {
                    "description": "Execute plan",
                    "reasoning": "Take action",
                    "duration": 60,
                    "dependencies": [2],
                    "success_criteria": ["Objectives met"]
                }
            ]
    
    # ==================== TOOL USE PATTERN ====================
    
    async def tool_use_example(
        self,
        task: str,
        available_tools: List[str]
    ) -> Dict[str, Any]:
        """
        Demonstrate tool use pattern with dynamic tool selection
        
        Args:
            task: Task to accomplish
            available_tools: List of available tool names
            
        Returns:
            Tool selection reasoning and execution plan
        """
        tool_analysis = await self._llm_analyze_and_select_tools(task, available_tools)
        
        return {
            "pattern": "tool_use",
            "task": task,
            "available_tools": available_tools,
            "analysis": tool_analysis["analysis"],
            "selected_tools": tool_analysis["selected_tools"],
            "execution_plan": tool_analysis["execution_plan"],
            "reasoning": tool_analysis["reasoning"]
        }
    
    async def _llm_analyze_and_select_tools(
        self,
        task: str,
        available_tools: List[str]
    ) -> Dict[str, Any]:
        """Analyze task and select appropriate tools using real LLM"""
        if self.llm_service is None:
            raise RuntimeError("LLM service not configured. Please set API keys in .env file.")
            
        tools_list = ", ".join(available_tools)
        
        prompt = f"""You are an expert task analyzer selecting the right tools for a job.

TASK: {task}

AVAILABLE TOOLS: {tools_list}

Analyze the task and select the most appropriate tools. Provide:
1. Task analysis: What capabilities are needed?
2. Tool selection: Which tools from the available list should be used and why?
3. Execution plan: In what order should tools be used?

Return valid JSON in this format:
{{
  "analysis": {{
    "task_type": "category of task",
    "required_capabilities": ["capability1", "capability2"],
    "complexity": "low/medium/high"
  }},
  "selected_tools": [
    {{
      "tool": "tool_name",
      "reasoning": "why this tool is needed",
      "priority": 1
    }}
  ],
  "execution_plan": [
    {{
      "step": 1,
      "tool": "tool_name",
      "action": "what to do",
      "input": "expected input",
      "output": "expected output"
    }}
  ],
  "reasoning": "overall strategy explanation"
}}"""
        
        result = await self.llm_service.generate_structured_output(
            prompt=prompt,
            schema={
                "type": "object",
                "properties": {
                    "analysis": {"type": "object"},
                    "selected_tools": {"type": "array"},
                    "execution_plan": {"type": "array"},
                    "reasoning": {"type": "string"}
                }
            }
        )
        
        return result
    
    def _analyze_task_requirements(self, task: str) -> List[str]:
        """Analyze what capabilities are needed for task"""
        requirements = []
        
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["search", "find", "lookup"]):
            requirements.append("information_retrieval")
        if any(word in task_lower for word in ["calculate", "compute", "sum"]):
            requirements.append("computation")
        if any(word in task_lower for word in ["write", "generate", "create"]):
            requirements.append("generation")
        if any(word in task_lower for word in ["analyze", "evaluate", "assess"]):
            requirements.append("analysis")
        if any(word in task_lower for word in ["save", "store", "database"]):
            requirements.append("storage")
        
        return requirements if requirements else ["general_purpose"]
    
    def _select_tools(self, requirements: List[str], available: List[str]) -> List[Dict]:
        """Select appropriate tools based on requirements"""
        tool_mapping = {
            "information_retrieval": {
                "tools": ["web_search", "database_query", "knowledge_base"],
                "input": "search query",
                "output": "relevant information"
            },
            "computation": {
                "tools": ["calculator", "python_executor", "excel"],
                "input": "mathematical expression",
                "output": "computed result"
            },
            "generation": {
                "tools": ["llm_generator", "template_engine", "code_generator"],
                "input": "generation prompt",
                "output": "generated content"
            },
            "analysis": {
                "tools": ["data_analyzer", "llm_analyzer", "statistics"],
                "input": "data to analyze",
                "output": "analysis results"
            },
            "storage": {
                "tools": ["database", "file_system", "cloud_storage"],
                "input": "data to store",
                "output": "storage confirmation"
            }
        }
        
        selections = []
        for req in requirements:
            if req in tool_mapping:
                mapping = tool_mapping[req]
                # Find first available tool for this requirement
                for tool in mapping["tools"]:
                    if tool in available:
                        selections.append({
                            "tool": tool,
                            "reasoning": f"Selected {tool} for {req} capability",
                            "input": mapping["input"],
                            "expected_output": mapping["output"]
                        })
                        break
        
        return selections
    
    def _create_execution_plan(self, selected_tools: List[Dict]) -> List[Dict]:
        """Create execution plan from selected tools"""
        plan = []
        for i, tool in enumerate(selected_tools, 1):
            plan.append({
                "step": i,
                "tool": tool.get("tool", "unknown"),
                "action": f"Execute {tool.get('tool', 'tool')}",
                "input": tool.get("input", "data"),
                "output": tool.get("expected_output", "result")
            })
        return plan
    
    def _estimate_success_rate(self, usage_plan: List[Dict]) -> float:
        """Estimate likelihood of success"""
        if not usage_plan:
            return 0.3
        
        # More tools = higher complexity = slightly lower success rate
        base_rate = 0.9
        complexity_penalty = len(usage_plan) * 0.05
        
        return max(0.5, base_rate - complexity_penalty)
    
    # ==================== AGENTIC RAG PATTERN ====================
    
    async def agentic_rag_example(
        self,
        initial_query: str,
        max_refinements: int = 3
    ) -> Dict[str, Any]:
        """
        Demonstrate agentic RAG with query refinement
        
        Args:
            initial_query: Original user query
            max_refinements: Maximum query refinement iterations
            
        Returns:
            RAG iterations with refinements
        """
        iterations = []
        current_query = initial_query
        
        for i in range(max_refinements):
            documents = self._retrieve_documents(current_query)
            
            evaluation = await self._llm_evaluate_documents(current_query, documents)
            
            iteration_data = {
                "iteration": i + 1,
                "query": current_query,
                "documents_retrieved": len(documents),
                "average_relevance": evaluation["average_relevance"],
                "documents": evaluation["document_scores"],
                "evaluation_reasoning": evaluation["reasoning"]
            }
            
            if evaluation["average_relevance"] >= 0.75:
                iterations.append(iteration_data)
                break
            
            refined_query = await self._llm_refine_query(
                current_query,
                documents,
                evaluation
            )
            iteration_data["refined_query"] = refined_query
            iteration_data["refinement_reasoning"] = evaluation.get("refinement_suggestion", "")
            
            iterations.append(iteration_data)
            current_query = refined_query
        
        return {
            "pattern": "agentic_rag",
            "initial_query": initial_query,
            "final_query": current_query,
            "total_iterations": len(iterations),
            "iterations": iterations,
            "final_relevance": iterations[-1]["average_relevance"] if iterations else 0
        }
    
    async def _llm_evaluate_documents(
        self,
        query: str,
        documents: List[Dict]
    ) -> Dict[str, Any]:
        """Use real LLM to evaluate document relevance"""
        if self.llm_service is None:
            raise RuntimeError("LLM service not configured. Please set API keys in .env file.")
            
        docs_text = "\n\n".join([
            f"DOCUMENT {i+1}:\nTitle: {doc['title']}\nContent: {doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""You are an expert at evaluating document relevance for search queries.

QUERY: {query}

RETRIEVED DOCUMENTS:
{docs_text}

Evaluate each document's relevance to the query. For each document, provide:
1. A relevance score from 0.0 to 1.0
2. Brief reasoning for the score

Also provide:
- Overall assessment of retrieval quality
- Suggestion for query refinement if needed

Return valid JSON:
{{
  "document_scores": [
    {{
      "document_number": 1,
      "title": "document title",
      "relevance_score": 0.0-1.0,
      "reasoning": "why this score"
    }}
  ],
  "average_relevance": 0.0-1.0,
  "reasoning": "overall assessment",
  "refinement_suggestion": "how to improve query if needed"
}}"""
        
        result = await self.llm_service.generate_structured_output(
            prompt=prompt,
            schema={
                "type": "object",
                "properties": {
                    "document_scores": {"type": "array"},
                    "average_relevance": {"type": "number"},
                    "reasoning": {"type": "string"},
                    "refinement_suggestion": {"type": "string"}
                }
            }
        )
        
        return result
    
    async def _llm_refine_query(
        self,
        original_query: str,
        documents: List[Dict],
        evaluation: Dict[str, Any]
    ) -> str:
        """Use real LLM to refine search query based on results"""
        if self.llm_service is None:
            raise RuntimeError("LLM service not configured. Please set API keys in .env file.")
            
        prompt = f"""You are an expert at refining search queries to improve results.

ORIGINAL QUERY: {original_query}

EVALUATION: {evaluation.get('reasoning', 'Results not satisfactory')}

REFINEMENT SUGGESTION: {evaluation.get('refinement_suggestion', 'Make query more specific')}

Create an improved search query that will retrieve more relevant documents.
The refined query should be more specific, use better keywords, or focus on a particular aspect.

Return only the refined query text, no explanation."""
        
        response = await self._monitored_llm_call(
            prompt=prompt,
            temperature=0.5,
            max_tokens=100
        )
        
        return response.content.strip()
    
    def _retrieve_documents(self, query: str) -> List[Dict]:
        """Retrieve documents based on query"""
        docs = [
            {
                "title": f"Document about {query.split()[0] if query.split() else 'topic'}",
                "content": f"This document discusses {query}. It provides detailed information about the subject matter."
            },
            {
                "title": f"Research on {query.split()[-1] if len(query.split()) > 1 else 'subject'}",
                "content": f"A comprehensive study examining {query} from multiple perspectives."
            },
            {
                "title": "Related Topic Overview",
                "content": f"While not directly about {query}, this document covers related concepts."
            }
        ]
        return docs[:3]
    
    def _evaluate_relevance(self, query: str, documents: List[Dict]) -> Dict[str, Any]:
        """Evaluate document relevance to query"""
        scores = []
        query_words = set(query.lower().split())
        
        document_scores = []
        for i, doc in enumerate(documents):
            content = (doc["title"] + " " + doc["content"]).lower()
            content_words = set(content.split())
            
            overlap = len(query_words & content_words)
            score = min(1.0, overlap / len(query_words) if query_words else 0)
            scores.append(score)
            
            document_scores.append({
                "document_number": i + 1,
                "title": doc["title"],
                "relevance_score": score,
                "reasoning": f"Word overlap: {overlap}/{len(query_words)}"
            })
        
        avg_relevance = sum(scores) / len(scores) if scores else 0
        
        return {
            "document_scores": document_scores,
            "average_relevance": avg_relevance,
            "reasoning": f"Average relevance: {avg_relevance:.2f}",
            "refinement_suggestion": "Make query more specific" if avg_relevance < 0.7 else "Query is working well"
        }
    
    def _refine_query(
        self,
        original_query: str,
        documents: List[Dict],
        relevance_scores: List[float]
    ) -> str:
        """Refine query based on retrieval results"""
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        if avg_relevance < 0.5:
            # Add specificity
            return f"{original_query} detailed explanation with examples"
        else:
            # Slight refinement
            return f"{original_query} comprehensive overview"
    
    def _refine_query_heuristic(
        self,
        original_query: str,
        evaluation: Dict[str, Any]
    ) -> str:
        """Heuristic-based query refinement when LLM is unavailable"""
        avg_relevance = evaluation.get('average_relevance', 0.5)
        suggestion = evaluation.get('refinement_suggestion', '')
        
        if avg_relevance < 0.5:
            # Poor results - make more specific
            if 'specific' in suggestion.lower():
                return f"{original_query} detailed"
            elif 'different' in suggestion.lower():
                words = original_query.split()
                if len(words) > 1:
                    # Try rephrasing
                    return f"{words[-1]} {' '.join(words[:-1])}"
            return f"{original_query} comprehensive overview"
        else:
            # Good results - minor refinement
            return f"{original_query} examples"
    
    # ==================== METACOGNITION PATTERN ====================
    
    async def metacognition_example(
        self,
        execution_trace: List[Dict],
        performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Demonstrate metacognition with self-monitoring
        
        Args:
            execution_trace: History of agent actions
            performance_metrics: Performance measurements
            
        Returns:
            Self-assessment and strategy adjustments
        """
        analysis = await self._llm_analyze_performance(execution_trace, performance_metrics)
        
        return {
            "pattern": "metacognition",
            "performance_assessment": analysis["assessment"],
            "patterns_detected": analysis["patterns"],
            "issues_identified": analysis["issues"],
            "strategy_adjustments": analysis["adjustments"],
            "confidence_level": analysis.get("confidence", 0.7),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _llm_analyze_performance(
        self,
        execution_trace: List[Dict],
        performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Use real LLM to analyze performance and suggest improvements"""
        if self.llm_service is None:
            raise RuntimeError("LLM service not configured. Please set API keys in .env file.")
            
        trace_text = "\n".join([
            f"Step {i+1}: {step}"
            for i, step in enumerate(execution_trace)
        ])
        
        metrics_text = "\n".join([
            f"- {metric}: {value:.2f}"
            for metric, value in performance_metrics.items()
        ])
        
        prompt = f"""You are an AI agent analyzing your own performance to improve future execution.

EXECUTION TRACE:
{trace_text}

PERFORMANCE METRICS:
{metrics_text}

Perform metacognitive analysis:

1. ASSESSMENT: How well did you perform overall?
2. PATTERNS: What patterns do you notice in your execution?
3. ISSUES: What specific problems occurred?
4. ADJUSTMENTS: What strategy changes would improve performance?

Return valid JSON:
{{
  "assessment": {{
    "overall_score": 0.0-1.0,
    "summary": "brief assessment",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"]
  }},
  "patterns": [
    "pattern description 1",
    "pattern description 2"
  ],
  "issues": [
    {{
      "issue": "problem description",
      "severity": "low/medium/high",
      "impact": "impact description"
    }}
  ],
  "adjustments": [
    "adjustment recommendation 1",
    "adjustment recommendation 2"
  ],
  "confidence": 0.0-1.0
}}"""
        
        result = await self.llm_service.generate_structured_output(
            prompt=prompt,
            schema={
                "type": "object",
                "properties": {
                    "assessment": {"type": "object"},
                    "patterns": {"type": "array"},
                    "issues": {"type": "array"},
                    "adjustments": {"type": "array"},
                    "confidence": {"type": "number"}
                }
            }
        )
        
        return result
    
    def _assess_performance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess own performance"""
        assessment = {
            "overall_score": sum(metrics.values()) / len(metrics) if metrics else 0,
            "strengths": [],
            "weaknesses": []
        }
        
        for metric, value in metrics.items():
            if value >= 0.8:
                assessment["strengths"].append(f"{metric}: {value:.2f}")
            elif value < 0.5:
                assessment["weaknesses"].append(f"{metric}: {value:.2f}")
        
        return assessment
    
    def _detect_patterns(self, trace: List[Dict]) -> List[str]:
        """Detect patterns in execution history"""
        patterns = []
        
        if len(trace) > 5:
            patterns.append("Long execution sequence detected")
        
        error_count = sum(1 for step in trace if step.get("status") == "error")
        if error_count > 0:
            patterns.append(f"Encountered {error_count} errors during execution")
        
        return patterns
    
    def _detect_issues(
        self,
        trace: List[Dict],
        metrics: Dict[str, float]
    ) -> List[str]:
        """Detect potential issues"""
        issues = []
        
        # Check for low performance
        if metrics.get("accuracy", 1.0) < 0.7:
            issues.append("Low accuracy detected")
        
        # Check for errors
        if any(step.get("status") == "error" for step in trace):
            issues.append("Error recovery needed")
        
        # Check for inefficiency
        if len(trace) > 10:
            issues.append("Execution may be inefficient")
        
        return issues
    
    def _suggest_adjustments(
        self,
        assessment: Dict,
        patterns: List[str],
        issues: List[str]
    ) -> List[Dict]:
        """Suggest strategy adjustments"""
        adjustments = []
        
        if "Low accuracy" in str(issues):
            adjustments.append({
                "aspect": "accuracy",
                "suggestion": "Increase validation steps",
                "expected_impact": "15-20% improvement"
            })
        
        if "inefficient" in str(issues):
            adjustments.append({
                "aspect": "efficiency",
                "suggestion": "Reduce redundant steps",
                "expected_impact": "30% faster execution"
            })
        
        if "error" in str(issues):
            adjustments.append({
                "aspect": "reliability",
                "suggestion": "Add error handling and retries",
                "expected_impact": "95% success rate"
            })
        
        return adjustments
    
    def _calculate_confidence(
        self,
        assessment: Dict,
        issues: List[str]
    ) -> float:
        """Calculate confidence in current strategy"""
        base_confidence = assessment.get("overall_score", 0.5)
        issue_penalty = len(issues) * 0.1
        
        return max(0.0, min(1.0, base_confidence - issue_penalty))
