"""
Decision Agent Step Implementation with LLM Integration
"""

from typing import Dict, Any
from datetime import datetime
import json

from ia_modules.pipeline.core import Step


class DecisionAgentStep(Step):
    """A step that makes decisions based on input data using LLM reasoning"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an intelligent decision based on the input data using LLM"""
        # Get LLM service from service registry
        llm_service = self.services.get('llm_provider')

        # Get decision logic type from config
        decision_logic = self.config.get('decision_logic', 'validation')

        # Extract relevant data for decision making
        task = data.get('task', 'make decision')
        content = data.get('text', str(data))
        previous_agent_data = data.get('llm_response', {})

        if llm_service:
            try:
                # Create decision prompt based on logic type
                if decision_logic == 'validation':
                    prompt = f"""You are an intelligent decision agent. Your role is to analyze data and make validation decisions.

Task: {task}
Content to evaluate: {content}
Previous processing: {json.dumps(previous_agent_data, indent=2) if previous_agent_data else "None"}

Please evaluate this data and make a decision. Consider:
1. Data quality and completeness
2. Relevance to the task
3. Potential risks or concerns
4. Confidence in your assessment

Respond in JSON format with:
- decision: "accept" or "reject"
- confidence: score from 0-1
- reasoning: detailed explanation of your decision
- recommendations: any suggested actions
- risk_assessment: potential risks identified"""

                elif decision_logic == 'classification':
                    prompt = f"""You are a classification agent. Analyze the content and classify it appropriately.

Task: {task}
Content: {content}
Previous processing: {json.dumps(previous_agent_data, indent=2) if previous_agent_data else "None"}

Please classify this content and provide your decision. Respond in JSON format with:
- decision: your classification category
- confidence: score from 0-1
- reasoning: why you chose this classification
- alternatives: other possible classifications
- certainty: how certain you are of this classification"""

                else:
                    prompt = f"""You are a decision agent. Analyze the following data and make an informed decision.

Task: {task}
Content: {content}
Previous processing: {json.dumps(previous_agent_data, indent=2) if previous_agent_data else "None"}

Make a decision and respond in JSON format with:
- decision: your decision
- confidence: score from 0-1
- reasoning: explanation of your decision
- factors: key factors that influenced your decision"""

                # Get LLM decision
                response = await llm_service.generate_structured_output(
                    prompt=prompt,
                    schema={
                        "type": "object",
                        "properties": {
                            "decision": {"type": "string"},
                            "confidence": {"type": "number"},
                            "reasoning": {"type": "string"},
                            "recommendations": {"type": "array", "items": {"type": "string"}},
                            "risk_assessment": {"type": "string"}
                        }
                    }
                )

                # Create structured decision response
                processed_data = {
                    "agent_name": self.name,
                    "decision_logic": decision_logic,
                    "processed_at": datetime.now().isoformat(),
                    "original_task": task,
                    "original_content": content,
                    "llm_decision": response,
                    "final_decision": response.get('decision', 'unknown'),
                    "confidence": response.get('confidence', 0.5),
                    "metadata": {
                        "step_id": self.name,
                        "processing_type": "llm_decision_agent",
                        "llm_used": True
                    }
                }

            except Exception as e:
                # Fallback to simple decision logic if LLM fails
                fallback_decision = self._simple_decision_logic(data)
                processed_data = {
                    "agent_name": self.name,
                    "decision_logic": decision_logic,
                    "processed_at": datetime.now().isoformat(),
                    "original_task": task,
                    "original_content": content,
                    "fallback_decision": fallback_decision,
                    "final_decision": fallback_decision['decision'],
                    "confidence": fallback_decision['confidence'],
                    "error": str(e),
                    "metadata": {
                        "step_id": self.name,
                        "processing_type": "fallback_decision_agent",
                        "llm_used": False
                    }
                }
        else:
            # No LLM service available, use simple decision logic
            simple_decision = self._simple_decision_logic(data)
            processed_data = {
                "agent_name": self.name,
                "decision_logic": decision_logic,
                "processed_at": datetime.now().isoformat(),
                "original_task": task,
                "original_content": content,
                "simple_decision": simple_decision,
                "final_decision": simple_decision['decision'],
                "confidence": simple_decision['confidence'],
                "metadata": {
                    "step_id": self.name,
                    "processing_type": "simple_decision_agent",
                    "llm_used": False
                }
            }

        return processed_data

    def _simple_decision_logic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback simple decision logic"""
        task = data.get('task', '')
        text = data.get('text', '')

        # Simple heuristics
        has_content = bool(text and len(text.strip()) > 0)
        has_task = bool(task and len(task.strip()) > 0)

        if has_content and has_task:
            decision = "accept"
            confidence = 0.8
            reasoning = "Has both task and content"
        elif has_content or has_task:
            decision = "accept"
            confidence = 0.6
            reasoning = "Has either task or content"
        else:
            decision = "reject"
            confidence = 0.9
            reasoning = "Missing both task and content"

        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning
        }
