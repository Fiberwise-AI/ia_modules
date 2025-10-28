"""
LLM Step using the existing LLMProviderService.

This step uses the actual LLM service infrastructure already in ia_modules.
"""
from typing import Dict, Any
from ia_modules.pipeline.core import Step
from ia_modules.pipeline.llm_provider_service import LLMProviderService


class LLMStep(Step):
    """
    Pipeline step that calls LLM using the LLMProviderService.

    Uses the existing ia_modules LLM infrastructure - no mocks.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize LLM step.

        Config options:
            provider_name: Name of registered provider (optional, uses default)
            temperature: LLM temperature (default: 0.7)
            max_tokens: Maximum tokens (default: 1000)
            system_prompt: System prompt to prepend (optional)
            input_field: Field containing user input (default: "user_input")
            output_field: Field to store LLM response (default: "llm_response")
        """
        super().__init__(name, config)

        self.provider_name = config.get("provider_name")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        self.system_prompt = config.get("system_prompt")
        self.input_field = config.get("input_field", "user_input")
        self.output_field = config.get("output_field", "llm_response")

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call LLM using the LLMProviderService.

        Args:
            data: Pipeline data containing user input

        Returns:
            Data with LLM response added
        """
        # Get LLM service from services registry
        if not self.services:
            raise RuntimeError("Services not injected - LLMProviderService not available")

        llm_service = self.services.get('llm_provider')
        if not llm_service:
            raise RuntimeError("LLMProviderService not registered in services")

        # Get user input
        user_input = data.get(self.input_field, "")
        if not user_input:
            self.logger.warning(f"No input found in field '{self.input_field}'")
            data[self.output_field] = ""
            return data

        # Build prompt
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\nUser: {user_input}\n\nAssistant:"
        else:
            prompt = user_input

        # Call LLM
        self.logger.info(f"Calling LLM provider: {self.provider_name or 'default'}")

        response = await llm_service.generate_completion(
            prompt=prompt,
            provider_name=self.provider_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # Store response
        data[self.output_field] = response.content

        # Store metadata
        data["llm_metadata"] = {
            "provider": response.provider.value,
            "model": response.model,
            "usage": response.usage,
            "timestamp": response.timestamp.isoformat()
        }

        self.logger.info(f"LLM response generated ({len(response.content)} chars)")

        return data
