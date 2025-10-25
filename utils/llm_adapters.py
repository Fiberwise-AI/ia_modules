"""
LLM Service Adapters

Adapter classes to bridge different LLM service interfaces with pattern requirements.
"""

from typing import Optional, Any
from ..pipeline.llm_provider_service import LLMProviderService


class LLMProviderAdapter:
    """
    Adapts LLMProviderService to the simple interface expected by patterns.
    
    Patterns expect: async generate(prompt: str) -> str
    LLMProviderService provides: async generate_completion(...) -> Response
    
    Usage:
        service = LLMProviderService()
        service.register_provider("openai", LLMProvider.OPENAI, api_key="...")
        
        adapter = LLMProviderAdapter(service)
        context = {'services': {'llm': adapter}}
        
        result = await pattern.execute(context)
    """
    
    def __init__(self, provider_service: LLMProviderService):
        """
        Initialize adapter with an LLMProviderService instance.
        
        Args:
            provider_service: Configured LLMProviderService instance
        """
        self.service = provider_service
    
    async def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: The prompt to generate from
            model: Optional model override
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text as string
        """
        response = await self.service.generate_completion(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens or 500,
            **kwargs
        )
        return response.content
