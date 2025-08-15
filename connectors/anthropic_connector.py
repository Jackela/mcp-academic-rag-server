"""
Anthropic Claude LLM Connector - Claude model integration
"""

import logging
from typing import Dict, Any, Optional, List
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .base_llm_connector import BaseLLMConnector

logger = logging.getLogger(__name__)

class AnthropicConnector(BaseLLMConnector):
    """Anthropic Claude LLM connector"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        timeout: int = 60,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Anthropic connector
        
        Args:
            api_key: Anthropic API key
            model: Claude model name
            timeout: Request timeout
            parameters: Generation parameters (max_tokens, temperature, etc.)
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )
        
        super().__init__(api_key, model, timeout, parameters)
        self._init_generator()
    
    def _get_provider_name(self) -> str:
        return "anthropic"
    
    def _init_generator(self):
        """Initialize Anthropic client"""
        try:
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=self.timeout
            )
            logger.info(f"Initialized Anthropic client: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise
    
    def generate(
        self,
        messages: List[Dict[str, str]], 
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response using Claude"""
        try:
            # Normalize messages
            normalized_messages = self.normalize_messages(messages)
            
            # Convert to Anthropic format
            anthropic_messages = []
            system_message = None
            
            for msg in normalized_messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": "user" if msg["role"] == "user" else "assistant",
                        "content": msg["content"]
                    })
            
            # Merge generation parameters
            params = self.parameters.copy()
            if generation_kwargs:
                params.update(generation_kwargs)
            
            # Set default parameters for Claude
            claude_params = {
                "model": self.model,
                "max_tokens": params.get("max_tokens", 1000),
                "messages": anthropic_messages
            }
            
            # Add optional parameters
            if "temperature" in params:
                claude_params["temperature"] = params["temperature"]
            if system_message:
                claude_params["system"] = system_message
            if "stop_sequences" in params:
                claude_params["stop_sequences"] = params["stop_sequences"]
            if "top_p" in params:
                claude_params["top_p"] = params["top_p"]
            if "top_k" in params:
                claude_params["top_k"] = params["top_k"]
            
            # Generate response
            logger.debug(f"Generating Claude response with {len(anthropic_messages)} messages")
            response = self.client.messages.create(**claude_params)
            
            return {
                "content": response.content[0].text,
                "role": "assistant", 
                "model": self.model,
                "provider": "anthropic"
            }
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            return {
                "content": f"Claude generation failed: {str(e)}",
                "role": "assistant",
                "model": self.model,
                "provider": "anthropic", 
                "error": str(e)
            }

class AnthropicConnectorFactory:
    """Factory for creating Anthropic connectors"""
    
    SUPPORTED_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229", 
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022"
    ]
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> AnthropicConnector:
        """Create Anthropic connector from config"""
        import os
        
        api_key = config.get("api_key", os.environ.get("ANTHROPIC_API_KEY", ""))
        if not api_key:
            raise ValueError("Anthropic API key is required")
        
        model = config.get("model", "claude-3-sonnet-20240229")
        if model not in cls.SUPPORTED_MODELS:
            logger.warning(f"Model {model} not in supported list: {cls.SUPPORTED_MODELS}")
        
        return AnthropicConnector(
            api_key=api_key,
            model=model,
            timeout=config.get("timeout", 60),
            parameters=config.get("parameters", {})
        )
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported Claude models"""
        return cls.SUPPORTED_MODELS.copy()