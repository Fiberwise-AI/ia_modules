# Microsoft AI Agents Course - Inspired Features

**Date**: October 23, 2025  
**Inspired by**: [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners)  
**Integration Target**: ia_modules showcase_app

---

## 🎓 Overview

This document outlines features and modules inspired by Microsoft's comprehensive AI Agents course that should be added to ia_modules and showcased in the showcase_app.

---

## 🆕 New Modules to Add to ia_modules

### 1. **Metacognition Module** ⭐⭐⭐ (HIGH PRIORITY)

**Location**: `ia_modules/agents/metacognition.py`

**Why**: Core to advanced agent behavior - agents that can reflect on their own performance and self-improve

```python
"""
Metacognition Module

Enables agents to:
- Self-evaluate outputs
- Critique and iteratively improve
- Adjust strategies based on feedback
- Detect and self-correct errors
- Monitor their own performance
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ReflectionStrategy(Enum):
    """Strategies for agent self-reflection"""
    SELF_CRITIQUE = "self_critique"
    PEER_REVIEW = "peer_review"
    GOAL_BASED = "goal_based"
    ERROR_ANALYSIS = "error_analysis"

@dataclass
class Reflection:
    """Result of agent reflection"""
    original_output: Any
    critique: str
    quality_score: float  # 0.0 to 1.0
    improvements_suggested: List[str]
    improved_output: Optional[Any] = None

class MetacognitiveAgent:
    """
    Agent with metacognitive capabilities
    
    Features:
    - Self-reflection on outputs
    - Iterative improvement loops
    - Strategy adjustment
    - Error detection and correction
    - Performance monitoring
    """
    
    def __init__(
        self,
        llm_provider,
        reflection_strategy: ReflectionStrategy = ReflectionStrategy.SELF_CRITIQUE,
        quality_threshold: float = 0.8
    ):
        self.llm_provider = llm_provider
        self.reflection_strategy = reflection_strategy
        self.quality_threshold = quality_threshold
        self.reflection_history: List[Reflection] = []
    
    async def reflect_on_output(
        self,
        output: Any,
        criteria: Dict[str, str],
        context: Optional[Dict] = None
    ) -> Reflection:
        """
        Evaluate own output against quality criteria
        
        Args:
            output: The output to evaluate
            criteria: Dict of criterion_name -> description
            context: Optional context for evaluation
            
        Returns:
            Reflection with critique and quality score
        """
        reflection_prompt = self._build_reflection_prompt(output, criteria, context)
        critique = await self.llm_provider.generate(reflection_prompt)
        quality_score = self._extract_quality_score(critique)
        improvements = self._extract_improvements(critique)
        
        reflection = Reflection(
            original_output=output,
            critique=critique,
            quality_score=quality_score,
            improvements_suggested=improvements
        )
        
        self.reflection_history.append(reflection)
        return reflection
    
    async def critique_and_improve(
        self,
        output: Any,
        criteria: Dict[str, str],
        max_iterations: int = 3
    ) -> Reflection:
        """
        Iteratively improve output through self-critique
        
        Args:
            output: Initial output to improve
            criteria: Quality criteria
            max_iterations: Maximum refinement iterations
            
        Returns:
            Final reflection with improved output
        """
        current_output = output
        
        for iteration in range(max_iterations):
            reflection = await self.reflect_on_output(
                current_output,
                criteria
            )
            
            if reflection.quality_score >= self.quality_threshold:
                reflection.improved_output = current_output
                return reflection
            
            # Generate improved version
            improved_output = await self._generate_improvement(
                current_output,
                reflection.critique,
                reflection.improvements_suggested
            )
            
            current_output = improved_output
        
        # Return final reflection
        reflection.improved_output = current_output
        return reflection
    
    async def adjust_strategy(
        self,
        feedback: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Modify approach based on performance feedback
        
        Args:
            feedback: Feedback from previous executions
            performance_metrics: Performance scores
            
        Returns:
            Adjusted strategy parameters
        """
        adjustment_prompt = self._build_strategy_adjustment_prompt(
            feedback,
            performance_metrics
        )
        
        adjustments = await self.llm_provider.generate(adjustment_prompt)
        return self._parse_strategy_adjustments(adjustments)
    
    def detect_errors(
        self,
        execution_trace: List[Dict],
        expected_outcomes: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Self-detect errors in reasoning or execution
        
        Args:
            execution_trace: List of execution steps
            expected_outcomes: Optional expected results
            
        Returns:
            List of detected errors with descriptions
        """
        errors = []
        
        # Check for logical inconsistencies
        for i, step in enumerate(execution_trace):
            if self._has_logical_error(step, execution_trace[:i]):
                errors.append({
                    "step": i,
                    "type": "logical_inconsistency",
                    "description": "Step contradicts previous reasoning"
                })
        
        # Check against expected outcomes
        if expected_outcomes:
            for i, (step, expected) in enumerate(zip(execution_trace, expected_outcomes)):
                if not self._matches_expected(step, expected):
                    errors.append({
                        "step": i,
                        "type": "outcome_mismatch",
                        "description": f"Expected {expected}, got {step}"
                    })
        
        return errors
    
    def _build_reflection_prompt(
        self,
        output: Any,
        criteria: Dict[str, str],
        context: Optional[Dict]
    ) -> str:
        """Build prompt for self-reflection"""
        prompt = f"""You are critically evaluating your own output.

Output to evaluate:
{output}

Evaluation criteria:
"""
        for criterion, description in criteria.items():
            prompt += f"- {criterion}: {description}\n"
        
        if context:
            prompt += f"\nContext:\n{context}\n"
        
        prompt += """
Please provide:
1. A critical analysis of the output
2. A quality score from 0.0 to 1.0
3. Specific improvements that could be made

Format your response as:
CRITIQUE: [your analysis]
SCORE: [0.0-1.0]
IMPROVEMENTS:
- [improvement 1]
- [improvement 2]
...
"""
        return prompt
    
    async def _generate_improvement(
        self,
        output: Any,
        critique: str,
        improvements: List[str]
    ) -> Any:
        """Generate improved version based on critique"""
        improvement_prompt = f"""Based on this critique, please improve the output.

Original output:
{output}

Critique:
{critique}

Suggested improvements:
{chr(10).join(f'- {imp}' for imp in improvements)}

Please provide an improved version that addresses the critique:
"""
        improved = await self.llm_provider.generate(improvement_prompt)
        return improved
    
    def _extract_quality_score(self, critique: str) -> float:
        """Extract quality score from critique text"""
        import re
        match = re.search(r'SCORE:\s*([0-9.]+)', critique)
        if match:
            return float(match.group(1))
        return 0.5  # Default score
    
    def _extract_improvements(self, critique: str) -> List[str]:
        """Extract list of improvements from critique"""
        import re
        improvements_section = re.search(
            r'IMPROVEMENTS:(.*?)(?:$|\n\n)',
            critique,
            re.DOTALL
        )
        if improvements_section:
            text = improvements_section.group(1)
            return [
                line.strip('- ').strip()
                for line in text.split('\n')
                if line.strip().startswith('-')
            ]
        return []
    
    def _has_logical_error(self, step: Dict, previous_steps: List[Dict]) -> bool:
        """Check if step has logical inconsistency with previous steps"""
        # Simplified - would need more sophisticated logic
        return False
    
    def _matches_expected(self, step: Dict, expected: Dict) -> bool:
        """Check if step matches expected outcome"""
        # Simplified comparison
        return True
    
    def _build_strategy_adjustment_prompt(
        self,
        feedback: Dict,
        metrics: Dict[str, float]
    ) -> str:
        """Build prompt for strategy adjustment"""
        return f"""Analyze the following performance feedback and suggest strategy adjustments:

Feedback: {feedback}
Performance Metrics: {metrics}

What adjustments would improve performance?
"""
    
    def _parse_strategy_adjustments(self, adjustments: str) -> Dict[str, Any]:
        """Parse strategy adjustments from LLM response"""
        # Would need proper parsing logic
        return {"adjustments": adjustments}


# Integration with existing pipeline system
class MetacognitivePipelineStep:
    """Pipeline step that uses metacognition"""
    
    def __init__(
        self,
        agent: MetacognitiveAgent,
        criteria: Dict[str, str],
        max_iterations: int = 3
    ):
        self.agent = agent
        self.criteria = criteria
        self.max_iterations = max_iterations
    
    async def execute(self, input_data: Dict) -> Dict:
        """Execute step with metacognitive refinement"""
        # Initial execution
        output = await self._initial_execution(input_data)
        
        # Self-improve through reflection
        reflection = await self.agent.critique_and_improve(
            output,
            self.criteria,
            self.max_iterations
        )
        
        return {
            "output": reflection.improved_output,
            "reflection_history": [
                {
                    "iteration": i,
                    "quality_score": r.quality_score,
                    "critique": r.critique
                }
                for i, r in enumerate(self.agent.reflection_history)
            ],
            "final_quality_score": reflection.quality_score
        }
    
    async def _initial_execution(self, input_data: Dict) -> Any:
        """Initial execution before refinement"""
        # Placeholder - would call actual execution logic
        return input_data.get("prompt", "")
```

**Tests**: `tests/unit/test_metacognition.py`

**Showcase Examples**:
1. **Writing Assistant** - Critiques and improves its own drafts
2. **Code Reviewer** - Re-evaluates its own code suggestions
3. **Research Agent** - Refines search strategy based on results

---

### 2. **Advanced Agentic RAG Module** ⭐⭐⭐ (HIGH PRIORITY)

**Location**: `ia_modules/rag/agentic_rag.py`

**Why**: Current RAG is basic - needs query refinement, relevance evaluation, and corrective retrieval

```python
"""
Agentic RAG Module

Advanced RAG with:
- Query refinement loops
- LLM-based relevance evaluation
- Re-ranking with reasoning
- Corrective retrieval
- Intent-aware search
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class RetrievalStrategy(Enum):
    """RAG retrieval strategies"""
    BASIC = "basic"
    QUERY_REFINEMENT = "query_refinement"
    CORRECTIVE = "corrective"
    AGENTIC = "agentic"

@dataclass
class Document:
    """Retrieved document"""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float = 0.0
    source: str = ""

@dataclass
class RAGIteration:
    """Single RAG iteration"""
    query: str
    retrieved_docs: List[Document]
    relevance_scores: List[float]
    refined_query: Optional[str] = None

class AgenticRAG:
    """
    Advanced RAG with query refinement and relevance evaluation
    
    Features:
    - Iterative query refinement
    - LLM-based relevance scoring
    - Re-ranking with reasoning
    - Corrective retrieval loops
    - Intent understanding
    """
    
    def __init__(
        self,
        retriever,
        llm_provider,
        relevance_threshold: float = 0.7,
        max_iterations: int = 3
    ):
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.relevance_threshold = relevance_threshold
        self.max_iterations = max_iterations
        self.iteration_history: List[RAGIteration] = []
    
    async def retrieve_with_refinement(
        self,
        query: str,
        context: Optional[Dict] = None,
        top_k: int = 5
    ) -> List[Document]:
        """
        Retrieve with iterative query refinement
        
        Args:
            query: Initial search query
            context: Optional context for query understanding
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        current_query = query
        best_docs = []
        best_score = 0.0
        
        for iteration in range(self.max_iterations):
            # Retrieve documents
            docs = await self.retriever.retrieve(current_query, top_k)
            
            # Evaluate relevance
            relevance_scores = await self.evaluate_relevance(
                docs,
                query,
                context
            )
            
            # Re-rank
            ranked_docs = self.re_rank_results(docs, relevance_scores)
            
            # Record iteration
            avg_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            iteration_data = RAGIteration(
                query=current_query,
                retrieved_docs=ranked_docs,
                relevance_scores=relevance_scores
            )
            self.iteration_history.append(iteration_data)
            
            # Check if we have good enough results
            if avg_score >= self.relevance_threshold:
                return ranked_docs
            
            # Refine query for next iteration
            refined_query = await self.refine_query(
                current_query,
                ranked_docs,
                query,
                context
            )
            
            iteration_data.refined_query = refined_query
            current_query = refined_query
            
            if avg_score > best_score:
                best_score = avg_score
                best_docs = ranked_docs
        
        return best_docs or ranked_docs
    
    async def refine_query(
        self,
        current_query: str,
        retrieved_docs: List[Document],
        original_query: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Iteratively refine search query for better results
        
        Args:
            current_query: Current query
            retrieved_docs: Documents retrieved with current query
            original_query: Original user query
            context: Optional context
            
        Returns:
            Refined query
        """
        refinement_prompt = f"""You are refining a search query to get better results.

Original query: {original_query}
Current query: {current_query}

Retrieved documents were not sufficiently relevant. Sample:
{self._format_docs_sample(retrieved_docs[:2])}

Please provide a refined query that would retrieve more relevant information.
Consider:
- Different keywords or synonyms
- More specific terminology
- Additional context

Refined query:"""
        
        refined = await self.llm_provider.generate(refinement_prompt)
        return refined.strip()
    
    async def evaluate_relevance(
        self,
        documents: List[Document],
        query: str,
        context: Optional[Dict] = None
    ) -> List[float]:
        """
        Use LLM to score document relevance
        
        Args:
            documents: Documents to evaluate
            query: Search query
            context: Optional context
            
        Returns:
            Relevance scores (0.0 to 1.0) for each document
        """
        scores = []
        
        for doc in documents:
            relevance_prompt = f"""Rate the relevance of this document to the query on a scale of 0.0 to 1.0.

Query: {query}

Document:
{doc.content[:500]}...

Consider:
- Direct relevance to query
- Accuracy and credibility
- Recency and timeliness
- Coverage of key topics

Provide only the numerical score (0.0 to 1.0):"""
            
            score_text = await self.llm_provider.generate(relevance_prompt)
            try:
                score = float(score_text.strip())
                scores.append(max(0.0, min(1.0, score)))
            except ValueError:
                scores.append(0.5)  # Default score on parse error
        
        return scores
    
    def re_rank_results(
        self,
        documents: List[Document],
        relevance_scores: List[float]
    ) -> List[Document]:
        """
        Re-rank retrieved documents using LLM-based scores
        
        Args:
            documents: Original documents
            relevance_scores: LLM-generated relevance scores
            
        Returns:
            Re-ranked documents
        """
        # Combine original scores with LLM scores
        for doc, llm_score in zip(documents, relevance_scores):
            doc.relevance_score = (doc.relevance_score + llm_score) / 2
        
        # Sort by combined score
        ranked = sorted(
            documents,
            key=lambda d: d.relevance_score,
            reverse=True
        )
        
        return ranked
    
    async def corrective_retrieval(
        self,
        query: str,
        failed_docs: List[Document],
        context: Optional[Dict] = None
    ) -> List[Document]:
        """
        Retrieve alternative documents if initial results are poor
        
        Args:
            query: Original query
            failed_docs: Previously retrieved docs that weren't relevant
            context: Optional context
            
        Returns:
            Alternative documents
        """
        # Analyze why previous docs failed
        failure_analysis = await self._analyze_failure(query, failed_docs)
        
        # Generate alternative search strategy
        alt_query = await self._generate_alternative_query(
            query,
            failure_analysis
        )
        
        # Retrieve with alternative query
        alt_docs = await self.retriever.retrieve(alt_query, top_k=5)
        
        return alt_docs
    
    async def _analyze_failure(
        self,
        query: str,
        docs: List[Document]
    ) -> str:
        """Analyze why retrieved documents weren't relevant"""
        analysis_prompt = f"""Analyze why these documents weren't relevant to the query:

Query: {query}

Documents:
{self._format_docs_sample(docs)}

Why weren't these relevant? What's missing?"""
        
        return await self.llm_provider.generate(analysis_prompt)
    
    async def _generate_alternative_query(
        self,
        original_query: str,
        failure_analysis: str
    ) -> str:
        """Generate alternative query based on failure analysis"""
        alt_prompt = f"""Generate an alternative search query:

Original query: {original_query}
Failure analysis: {failure_analysis}

Alternative query:"""
        
        return await self.llm_provider.generate(alt_prompt)
    
    def _format_docs_sample(self, docs: List[Document]) -> str:
        """Format document sample for prompts"""
        return "\n\n".join(
            f"Doc {i+1}: {doc.content[:200]}..."
            for i, doc in enumerate(docs)
        )


# Integration with pipeline
class AgenticRAGStep:
    """Pipeline step for agentic RAG"""
    
    def __init__(
        self,
        rag: AgenticRAG,
        output_format: str = "documents"
    ):
        self.rag = rag
        self.output_format = output_format
    
    async def execute(self, input_data: Dict) -> Dict:
        """Execute RAG retrieval"""
        query = input_data.get("query", "")
        context = input_data.get("context")
        
        documents = await self.rag.retrieve_with_refinement(
            query,
            context
        )
        
        return {
            "documents": [
                {
                    "content": doc.content,
                    "relevance": doc.relevance_score,
                    "metadata": doc.metadata
                }
                for doc in documents
            ],
            "iteration_history": [
                {
                    "query": iter.query,
                    "num_docs": len(iter.retrieved_docs),
                    "avg_relevance": sum(iter.relevance_scores) / len(iter.relevance_scores) if iter.relevance_scores else 0,
                    "refined_query": iter.refined_query
                }
                for iter in self.rag.iteration_history
            ]
        }
```

**Tests**: `tests/unit/test_agentic_rag.py`

---

### 3. **Context Engineering Module** ⭐⭐ (MEDIUM PRIORITY)

**Location**: `ia_modules/agents/context_engineering.py`

**Why**: Critical for effective agent prompting and context management within token limits

```python
"""
Context Engineering Module

Manages and optimizes context for LLM agents:
- Context window management
- Information prioritization
- Token budget optimization
- Model-specific formatting
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class PrioritizationStrategy(Enum):
    """Strategies for prioritizing information"""
    RECENCY = "recency"
    RELEVANCE = "relevance"
    IMPORTANCE = "importance"
    HYBRID = "hybrid"

@dataclass
class ContextItem:
    """Single item in context"""
    content: str
    type: str  # 'message', 'document', 'system', etc.
    priority: float = 0.5
    token_count: int = 0
    metadata: Dict[str, Any] = None

class ContextEngineer:
    """
    Manage and optimize context for LLM agents
    
    Features:
    - Token budget management
    - Information prioritization
    - Context compression
    - Model-specific formatting
    """
    
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 4096,
        prioritization_strategy: PrioritizationStrategy = PrioritizationStrategy.HYBRID
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.prioritization_strategy = prioritization_strategy
    
    def build_context(
        self,
        memory: List[Dict],
        tools: List[Dict],
        task: Dict,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Construct optimal context for agent
        
        Args:
            memory: Conversation history/memory
            tools: Available tools
            task: Current task
            system_prompt: Optional system prompt
            
        Returns:
            Formatted context string
        """
        # Convert inputs to ContextItems
        items = []
        
        if system_prompt:
            items.append(ContextItem(
                content=system_prompt,
                type="system",
                priority=1.0,
                token_count=self._count_tokens(system_prompt)
            ))
        
        # Add memory
        for msg in memory:
            items.append(ContextItem(
                content=self._format_message(msg),
                type="message",
                priority=self._calculate_priority(msg),
                token_count=self._count_tokens(self._format_message(msg))
            ))
        
        # Add tools
        tools_text = self._format_tools(tools)
        items.append(ContextItem(
            content=tools_text,
            type="tools",
            priority=0.8,
            token_count=self._count_tokens(tools_text)
        ))
        
        # Add task
        task_text = self._format_task(task)
        items.append(ContextItem(
            content=task_text,
            type="task",
            priority=1.0,
            token_count=self._count_tokens(task_text)
        ))
        
        # Prioritize and fit within budget
        selected_items = self.prioritize_information(items, self.max_tokens)
        
        # Format for specific model
        return self.format_for_model(selected_items, self.model_name)
    
    def prioritize_information(
        self,
        items: List[ContextItem],
        max_tokens: int
    ) -> List[ContextItem]:
        """
        Select most relevant information within token limit
        
        Args:
            items: All available context items
            max_tokens: Maximum tokens allowed
            
        Returns:
            Prioritized subset of items
        """
        # Sort by priority
        sorted_items = sorted(items, key=lambda x: x.priority, reverse=True)
        
        selected = []
        total_tokens = 0
        
        for item in sorted_items:
            if total_tokens + item.token_count <= max_tokens:
                selected.append(item)
                total_tokens += item.token_count
            elif item.priority >= 1.0:  # Must-include items
                # Compress or truncate
                compressed = self._compress_item(item, max_tokens - total_tokens)
                if compressed:
                    selected.append(compressed)
                    total_tokens += compressed.token_count
        
        return selected
    
    def format_for_model(
        self,
        items: List[ContextItem],
        model_type: str
    ) -> str:
        """
        Format context appropriately for specific LLM
        
        Args:
            items: Context items to format
            model_type: Target model (gpt-4, claude, gemini, etc.)
            
        Returns:
            Formatted context string
        """
        if "gpt" in model_type.lower():
            return self._format_for_openai(items)
        elif "claude" in model_type.lower():
            return self._format_for_anthropic(items)
        elif "gemini" in model_type.lower():
            return self._format_for_gemini(items)
        else:
            return self._format_generic(items)
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count"""
        # Simplified - would use actual tokenizer
        return len(text.split()) * 1.3  # Rough estimate
    
    def _calculate_priority(self, item: Dict) -> float:
        """Calculate priority score for item"""
        if self.prioritization_strategy == PrioritizationStrategy.RECENCY:
            return self._recency_score(item)
        elif self.prioritization_strategy == PrioritizationStrategy.RELEVANCE:
            return self._relevance_score(item)
        elif self.prioritization_strategy == PrioritizationStrategy.IMPORTANCE:
            return self._importance_score(item)
        else:  # HYBRID
            return (
                self._recency_score(item) * 0.3 +
                self._relevance_score(item) * 0.4 +
                self._importance_score(item) * 0.3
            )
    
    def _recency_score(self, item: Dict) -> float:
        """Score based on recency"""
        # Would use timestamps
        return 0.5
    
    def _relevance_score(self, item: Dict) -> float:
        """Score based on relevance to current task"""
        # Would use semantic similarity
        return 0.5
    
    def _importance_score(self, item: Dict) -> float:
        """Score based on inherent importance"""
        # Would use metadata or learned importance
        return 0.5
    
    def _compress_item(
        self,
        item: ContextItem,
        max_tokens: int
    ) -> Optional[ContextItem]:
        """Compress item to fit token budget"""
        if item.token_count <= max_tokens:
            return item
        
        # Truncate to fit
        ratio = max_tokens / item.token_count
        words = item.content.split()
        truncated_words = words[:int(len(words) * ratio)]
        
        return ContextItem(
            content=" ".join(truncated_words) + "...",
            type=item.type,
            priority=item.priority,
            token_count=max_tokens,
            metadata=item.metadata
        )
    
    def _format_message(self, msg: Dict) -> str:
        """Format message for context"""
        role = msg.get("role", "user")
        content = msg.get("content", "")
        return f"{role}: {content}"
    
    def _format_tools(self, tools: List[Dict]) -> str:
        """Format tools list for context"""
        if not tools:
            return ""
        
        formatted = "Available tools:\n"
        for tool in tools:
            formatted += f"- {tool.get('name')}: {tool.get('description')}\n"
        return formatted
    
    def _format_task(self, task: Dict) -> str:
        """Format task for context"""
        return f"Current task: {task.get('description', task)}"
    
    def _format_for_openai(self, items: List[ContextItem]) -> str:
        """Format for OpenAI models"""
        # Would use messages format
        return "\n\n".join(item.content for item in items)
    
    def _format_for_anthropic(self, items: List[ContextItem]) -> str:
        """Format for Anthropic Claude"""
        # Would use Claude-specific format
        return "\n\n".join(item.content for item in items)
    
    def _format_for_gemini(self, items: List[ContextItem]) -> str:
        """Format for Google Gemini"""
        # Would use Gemini-specific format
        return "\n\n".join(item.content for item in items)
    
    def _format_generic(self, items: List[ContextItem]) -> str:
        """Generic formatting"""
        return "\n\n".join(item.content for item in items)
```

**Tests**: `tests/unit/test_context_engineering.py`

---

## 📚 Example Pipelines (Microsoft-Inspired)

### 1. Research Team Multi-Agent Pipeline
```json
{
  "name": "Research Team Workflow",
  "description": "Multi-agent research pipeline with specialized roles",
  "pattern": "multi_agent",
  "agents": [
    {
      "name": "researcher",
      "role": "information_gatherer",
      "task": "Gather comprehensive information on topic",
      "tools": ["web_search", "academic_search", "news_api"],
      "output": "research_data"
    },
    {
      "name": "analyst",
      "role": "data_analyst", 
      "task": "Analyze findings and identify key patterns/insights",
      "depends_on": ["researcher"],
      "tools": ["data_analysis", "visualization"],
      "pattern": "reflection",
      "output": "analysis_results"
    },
    {
      "name": "writer",
      "role": "content_creator",
      "task": "Draft comprehensive research report",
      "depends_on": ["analyst"],
      "tools": ["document_generator"],
      "output": "initial_draft"
    },
    {
      "name": "reviewer",
      "role": "quality_assurance",
      "task": "Critique and improve report quality",
      "depends_on": ["writer"],
      "pattern": "metacognition",
      "config": {
        "quality_criteria": {
          "accuracy": "Information must be factually correct",
          "completeness": "All key points must be covered",
          "clarity": "Writing must be clear and accessible",
          "structure": "Report must be well-organized"
        },
        "max_iterations": 3,
        "quality_threshold": 0.85
      },
      "output": "final_report"
    }
  ]
}
```

### 2. Self-Improving Travel Planner
```json
{
  "name": "Travel Planning with Metacognition",
  "description": "Agent that iteratively improves travel plans through self-reflection",
  "pattern": "metacognition",
  "steps": [
    {
      "name": "bootstrap_plan",
      "type": "planning",
      "description": "Create initial travel plan based on preferences",
      "config": {
        "preferences": "{{user_preferences}}",
        "budget": "{{budget}}",
        "duration": "{{days}}"
      },
      "output": "initial_plan"
    },
    {
      "name": "evaluate_plan",
      "type": "metacognition_reflection",
      "description": "Evaluate plan quality against criteria",
      "input": "{{initial_plan}}",
      "config": {
        "criteria": {
          "budget_fit": "Plan must stay within budget",
          "preference_match": "Activities must match user preferences",
          "feasibility": "Schedule must be realistic and achievable",
          "variety": "Good mix of activities and experiences",
          "local_tips": "Include authentic local experiences"
        }
      },
      "output": "evaluation"
    },
    {
      "name": "refine_plan",
      "type": "metacognition_improve",
      "description": "Iteratively refine plan based on self-critique",
      "input": {
        "plan": "{{initial_plan}}",
        "evaluation": "{{evaluation}}"
      },
      "loop": {
        "condition": "evaluation.quality_score < 0.9",
        "max_iterations": 3,
        "improvement_strategy": "address_lowest_scoring_criteria"
      },
      "output": "refined_plan"
    },
    {
      "name": "generate_alternatives",
      "type": "planning",
      "description": "Generate 2-3 alternative plans for comparison",
      "input": "{{refined_plan}}",
      "parallel": true,
      "config": {
        "num_alternatives": 3,
        "variation_factors": ["budget", "pace", "activities"]
      },
      "output": "alternative_plans"
    },
    {
      "name": "human_approval",
      "type": "hitl",
      "description": "Present plans to user for selection",
      "input": {
        "primary": "{{refined_plan}}",
        "alternatives": "{{alternative_plans}}"
      },
      "ui": {
        "display_mode": "comparison",
        "show_ratings": true,
        "allow_modifications": true
      },
      "output": "selected_plan"
    }
  ]
}
```

### 3. Code Review Multi-Agent Pipeline
```json
{
  "name": "Comprehensive Code Review Workflow",
  "description": "Multi-agent code review with parallel checks and metacognition",
  "pattern": "multi_agent_parallel",
  "steps": [
    {
      "name": "parallel_checks",
      "type": "parallel_group",
      "description": "Run multiple code checks in parallel",
      "parallel_steps": [
        {
          "name": "linter",
          "type": "tool_use",
          "tool": "code_linter",
          "config": {
            "rules": "strict",
            "auto_fix": false
          },
          "output": "lint_results"
        },
        {
          "name": "security_scanner",
          "type": "tool_use",
          "tool": "security_scanner",
          "config": {
            "scan_dependencies": true,
            "check_vulnerabilities": true
          },
          "output": "security_results"
        },
        {
          "name": "test_coverage",
          "type": "tool_use",
          "tool": "coverage_analyzer",
          "output": "coverage_report"
        }
      ]
    },
    {
      "name": "aggregate_results",
      "type": "data_processing",
      "depends_on": ["parallel_checks"],
      "description": "Combine results from parallel checks",
      "input": {
        "lint": "{{lint_results}}",
        "security": "{{security_results}}",
        "coverage": "{{coverage_report}}"
      },
      "output": "combined_results"
    },
    {
      "name": "code_reviewer",
      "type": "agent",
      "role": "senior_developer",
      "depends_on": ["aggregate_results"],
      "description": "Detailed code review considering all automated checks",
      "pattern": "reflection",
      "input": {
        "code": "{{source_code}}",
        "automated_checks": "{{combined_results}}"
      },
      "config": {
        "review_aspects": [
          "architecture",
          "performance",
          "maintainability",
          "best_practices"
        ]
      },
      "output": "initial_review"
    },
    {
      "name": "review_refinement",
      "type": "metacognition_improve",
      "depends_on": ["code_reviewer"],
      "description": "Refine code review through self-critique",
      "input": "{{initial_review}}",
      "config": {
        "criteria": {
          "actionable": "Feedback must be specific and actionable",
          "constructive": "Feedback must be constructive and helpful",
          "prioritized": "Issues must be prioritized by importance",
          "examples": "Include code examples where helpful"
        },
        "max_iterations": 2
      },
      "output": "refined_review"
    },
    {
      "name": "approver",
      "type": "hitl",
      "depends_on": ["review_refinement"],
      "description": "Human final approval decision",
      "input": {
        "code": "{{source_code}}",
        "review": "{{refined_review}}",
        "automated_checks": "{{combined_results}}"
      },
      "ui": {
        "show_diff": true,
        "show_comments": true,
        "allow_inline_edits": true
      },
      "decision_options": ["approve", "request_changes", "reject"],
      "output": "approval_decision"
    }
  ]
}
```

### 4. Customer Service Multi-Agent Pipeline
```json
{
  "name": "Intelligent Customer Service Workflow",
  "description": "Multi-agent customer service with triage, specialist routing, and QA",
  "pattern": "multi_agent_conditional",
  "steps": [
    {
      "name": "triage",
      "type": "agent",
      "role": "triage_specialist",
      "description": "Classify customer issue and determine routing",
      "pattern": "tool_use",
      "input": "{{customer_message}}",
      "tools": ["issue_classifier", "sentiment_analyzer", "kb_search"],
      "output": {
        "category": "string",
        "priority": "high|medium|low",
        "sentiment": "positive|neutral|negative",
        "suggested_specialist": "string"
      }
    },
    {
      "name": "specialist_routing",
      "type": "conditional_routing",
      "depends_on": ["triage"],
      "routes": [
        {
          "condition": "triage.category == 'technical'",
          "next_step": "technical_specialist"
        },
        {
          "condition": "triage.category == 'billing'",
          "next_step": "billing_specialist"
        },
        {
          "condition": "triage.category == 'general'",
          "next_step": "general_specialist"
        }
      ]
    },
    {
      "name": "technical_specialist",
      "type": "agent",
      "role": "technical_support",
      "pattern": "agentic_rag",
      "tools": ["kb_search", "troubleshooting_guide", "diagnostic_tools"],
      "config": {
        "max_retrieval_iterations": 3,
        "relevance_threshold": 0.75
      },
      "output": "proposed_solution"
    },
    {
      "name": "billing_specialist",
      "type": "agent",
      "role": "billing_support",
      "tools": ["billing_system", "transaction_lookup", "refund_processor"],
      "output": "proposed_solution"
    },
    {
      "name": "general_specialist",
      "type": "agent",
      "role": "general_support",
      "pattern": "agentic_rag",
      "tools": ["kb_search", "faq_lookup"],
      "output": "proposed_solution"
    },
    {
      "name": "qa_reviewer",
      "type": "agent",
      "role": "quality_assurance",
      "depends_on": ["technical_specialist", "billing_specialist", "general_specialist"],
      "description": "Review proposed solution for quality",
      "pattern": "metacognition",
      "input": {
        "issue": "{{customer_message}}",
        "triage": "{{triage}}",
        "solution": "{{proposed_solution}}"
      },
      "config": {
        "criteria": {
          "addresses_issue": "Solution must directly address the customer's issue",
          "clear": "Solution must be easy for customer to understand",
          "complete": "Solution must include all necessary steps",
          "tone": "Response must be empathetic and professional"
        },
        "quality_threshold": 0.85
      },
      "output": "reviewed_solution"
    },
    {
      "name": "follow_up_planning",
      "type": "planning",
      "depends_on": ["qa_reviewer"],
      "description": "Plan follow-up actions",
      "config": {
        "schedule_followup": true,
        "followup_delay_hours": 24,
        "create_ticket": true
      },
      "output": "followup_plan"
    },
    {
      "name": "send_response",
      "type": "action",
      "depends_on": ["follow_up_planning"],
      "action": "send_email",
      "input": {
        "to": "{{customer_email}}",
        "subject": "Re: {{customer_issue_subject}}",
        "body": "{{reviewed_solution}}",
        "followup": "{{followup_plan}}"
      }
    }
  ]
}
```

---

## 🎨 Showcase App UI Enhancements

### Pattern Visualizations

**1. Reflection Pattern Viewer**
```javascript
// src/components/patterns/ReflectionPatternViz.jsx
export function ReflectionPatternViz({ execution }) {
  const reflections = execution.pattern_data?.reflections || []
  
  return (
    <div className="reflection-pattern-viz">
      <PatternHeader 
        icon={<ReflectIcon />}
        title="Reflection Pattern"
        description="Agent iteratively improves output through self-critique"
      />
      
      <ReflectionTimeline>
        {reflections.map((reflection, idx) => (
          <ReflectionIteration key={idx} iteration={idx + 1}>
            <IterationHeader>
              <span>Iteration {idx + 1}</span>
              <QualityBadge score={reflection.quality_score} />
            </IterationHeader>
            
            <OutputComparison>
              <OutputPane title="Current Output">
                <CodeBlock language="markdown">
                  {reflection.original_output}
                </CodeBlock>
              </OutputPane>
              
              <CritiquePane>
                <CritiqueHeader>Self-Critique</CritiqueHeader>
                <CritiqueText>{reflection.critique}</CritiqueText>
                <ImprovementsList>
                  {reflection.improvements.map((imp, i) => (
                    <ImprovementItem key={i}>{imp}</ImprovementItem>
                  ))}
                </ImprovementsList>
              </CritiquePane>
              
              {reflection.improved_output && (
                <OutputPane title="Improved Output">
                  <CodeBlock language="markdown">
                    {reflection.improved_output}
                  </CodeBlock>
                  <DiffIndicator 
                    before={reflection.original_output}
                    after={reflection.improved_output}
                  />
                </OutputPane>
              )}
            </OutputComparison>
            
            <QualityProgress>
              <ProgressBar 
                value={reflection.quality_score}
                max={1.0}
                label={`Quality: ${(reflection.quality_score * 100).toFixed(0)}%`}
              />
            </QualityProgress>
          </ReflectionIteration>
        ))}
      </ReflectionTimeline>
      
      <OverallImprovement>
        <ImprovementChart data={reflections} />
      </OverallImprovement>
    </div>
  )
}
```

**2. Agentic RAG Viewer**
```javascript
// src/components/patterns/AgenticRAGViz.jsx
export function AgenticRAGViz({ execution }) {
  const iterations = execution.pattern_data?.rag_iterations || []
  
  return (
    <div className="agentic-rag-viz">
      <PatternHeader
        icon={<SearchIcon />}
        title="Agentic RAG Pattern"
        description="Iterative query refinement and relevance evaluation"
      />
      
      <RAGIterationFlow>
        {iterations.map((iter, idx) => (
          <RAGIteration key={idx}>
            <IterationNumber>{idx + 1}</IterationNumber>
            
            <QueryDisplay>
              <QueryLabel>Query</QueryLabel>
              <QueryText>{iter.query}</QueryText>
              {iter.refined_query && (
                <RefinedQuery>
                  <Arrow />
                  <QueryText refined>{iter.refined_query}</QueryText>
                </RefinedQuery>
              )}
            </QueryDisplay>
            
            <RetrievedDocuments>
              <DocsHeader>
                Retrieved Documents ({iter.documents.length})
              </DocsHeader>
              {iter.documents.map((doc, i) => (
                <DocumentCard key={i}>
                  <DocHeader>
                    <DocTitle>{doc.title}</DocTitle>
                    <RelevanceScore score={doc.relevance_score} />
                  </DocHeader>
                  <DocPreview>{doc.content.substring(0, 200)}...</DocPreview>
                  <DocMetadata>
                    <MetaItem>Source: {doc.source}</MetaItem>
                    <MetaItem>Tokens: {doc.token_count}</MetaItem>
                  </DocMetadata>
                </DocumentCard>
              ))}
            </RetrievedDocuments>
            
            <IterationMetrics>
              <MetricCard>
                <MetricLabel>Avg Relevance</MetricLabel>
                <MetricValue>
                  {(iter.avg_relevance * 100).toFixed(0)}%
                </MetricValue>
              </MetricCard>
              <MetricCard>
                <MetricLabel>Retrieved</MetricLabel>
                <MetricValue>{iter.documents.length}</MetricValue>
              </MetricCard>
            </IterationMetrics>
          </RAGIteration>
        ))}
      </RAGIterationFlow>
      
      <RelevanceChart iterations={iterations} />
    </div>
  )
}
```

---

## 📝 Implementation Priority

### Phase 1: Core Modules (Weeks 1-4)
1. **Metacognition Module** - Weeks 1-2
   - Core metacognitive agent implementation
   - Pipeline integration
   - Basic showcase examples
   
2. **Advanced Agentic RAG** - Weeks 2-3
   - Query refinement logic
   - Relevance evaluation
   - Corrective retrieval
   
3. **Context Engineering** - Week 4
   - Token management
   - Prioritization strategies
   - Model-specific formatting

### Phase 2: Example Pipelines (Weeks 5-6)
1. Research Team pipeline
2. Self-improving travel planner
3. Code review pipeline
4. Customer service pipeline

### Phase 3: Showcase Visualizations (Weeks 7-8)
1. Reflection pattern viewer
2. RAG iteration viewer
3. Planning visualization
4. Multi-agent coordination display

---

## 🧪 Testing Strategy

### Unit Tests
```python
# tests/unit/test_metacognition.py
# tests/unit/test_agentic_rag.py
# tests/unit/test_context_engineering.py
```

### Integration Tests
```python
# tests/integration/test_metacognitive_pipeline.py
# tests/integration/test_agentic_rag_pipeline.py
# tests/integration/test_multi_agent_patterns.py
```

### Showcase Tests
```python
# showcase_app/tests/test_pattern_visualizations.py
# showcase_app/tests/test_example_pipelines.py
```

---

## 📚 Documentation

### User Guides
- `docs/METACOGNITION_GUIDE.md` - Using metacognitive agents
- `docs/AGENTIC_RAG_GUIDE.md` - Advanced RAG patterns
- `docs/CONTEXT_ENGINEERING_GUIDE.md` - Context optimization
- `docs/PATTERN_CATALOG.md` - All agentic design patterns

### API Documentation
- Auto-generated from docstrings
- Interactive examples in showcase app
- Jupyter notebooks for each pattern

---

## 🎯 Success Criteria

✅ **Metacognition module fully implemented and tested**
✅ **Agentic RAG with query refinement working**
✅ **Context engineering operational**
✅ **At least 4 Microsoft-inspired example pipelines**
✅ **Pattern visualizations in showcase app**
✅ **Comprehensive documentation**
✅ **90%+ test coverage for new modules**

---

**Let's make ia_modules the most comprehensive and educational agent framework!** 🚀
