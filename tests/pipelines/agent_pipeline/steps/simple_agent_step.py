"""
Simple Agent Step Implementation with LLM Integration
"""

from typing import Dict, Any
from datetime import datetime
import json

from ia_modules.pipeline.core import Step


class SimpleAgentStep(Step):
    """A simple step that acts as an agent for processing data using LLM"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using LLM-powered agent logic"""
        # Get LLM service from service registry
        llm_service = self.services.get('llm_provider')

        # Get processing type from config
        processing_type = self.config.get('processing_type', 'general')

        # Extract relevant data for processing
        task = data.get('task', 'process data')
        text_content = data.get('text', str(data))

        if llm_service:
            try:
                # Create appropriate prompt based on processing type
                if processing_type == 'ingestion':
                    prompt = f"""You are a data ingestion agent. Your task is to analyze and prepare the following data for processing:

Task: {task}
Content: {text_content}

Please provide:
1. A summary of the data
2. Key entities or concepts identified
3. Data quality assessment
4. Recommended processing steps

Respond in JSON format with keys: summary, entities, quality_score, recommendations"""

                elif processing_type == 'final_processing':
                    prompt = f"""You are a final processing agent. Your task is to create a final output based on the processed data:

Task: {task}
Content: {text_content}

Please provide:
1. Final processed result
2. Confidence score (0-1)
3. Processing summary
4. Any warnings or notes

Respond in JSON format with keys: result, confidence, summary, notes"""

                else:
                    prompt = f"""You are an AI agent processing the following data:

Task: {task}
Content: {text_content}

Please analyze and process this data appropriately. Respond in JSON format with keys: analysis, result, confidence"""

                # Get LLM response
                response = await llm_service.generate_structured_output(
                    prompt=prompt,
                    schema={
                        "type": "object",
                        "properties": {
                            "analysis": {"type": "string"},
                            "result": {"type": "string"},
                            "confidence": {"type": "number"}
                        }
                    }
                )

                # Create structured response
                processed_data = {
                    "agent_name": self.name,
                    "processing_type": processing_type,
                    "processed_at": datetime.now().isoformat(),
                    "original_task": task,
                    "original_content": text_content,
                    "llm_response": response,
                    "metadata": {
                        "step_id": self.name,
                        "processing_type": "llm_agent",
                        "llm_used": True
                    }
                }

            except Exception as e:
                # Fallback to simple processing if LLM fails
                processed_data = {
                    "agent_name": self.name,
                    "processing_type": processing_type,
                    "processed_at": datetime.now().isoformat(),
                    "original_task": task,
                    "original_content": text_content,
                    "fallback_result": f"Processed {task} with simple transformation",
                    "error": str(e),
                    "metadata": {
                        "step_id": self.name,
                        "processing_type": "fallback_agent",
                        "llm_used": False
                    }
                }
        else:
            # No LLM service available, use simple processing
            processed_data = {
                "agent_name": self.name,
                "processing_type": processing_type,
                "processed_at": datetime.now().isoformat(),
                "original_task": task,
                "original_content": text_content,
                "simple_result": f"Processed {task} with basic transformation",
                "metadata": {
                    "step_id": self.name,
                    "processing_type": "simple_agent",
                    "llm_used": False
                }
            }

        return processed_data
