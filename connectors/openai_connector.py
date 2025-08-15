"""
OpenAI LLM Connector - OpenAI/GPT model integration
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from .base_llm_connector import BaseLLMConnector

logger = logging.getLogger(__name__)

class OpenAIConnector(BaseLLMConnector):
    """OpenAI LLM connector using Haystack"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        api_base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        streaming_callback: Optional[Callable] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize OpenAI connector
        
        Args:
            api_key: OpenAI API key
            model: Model name (e.g., gpt-3.5-turbo, gpt-4, gpt-4-turbo)
            api_base_url: API base URL
            timeout: Request timeout
            streaming_callback: Optional streaming callback
            parameters: Generation parameters (temperature, max_tokens, etc.)
        """
        super().__init__(api_key, model, timeout, parameters)
        self.api_base_url = api_base_url
        self.streaming_callback = streaming_callback
        self._init_generator()
    
    def _get_provider_name(self) -> str:
        return "openai"
    
    def _init_generator(self):
        """Initialize OpenAI generator"""
        try:
            self.generator = OpenAIChatGenerator(
                api_key=Secret.from_token(self.api_key),
                model=self.model,
                api_base_url=self.api_base_url,
                timeout=self.timeout,
                streaming_callback=self.streaming_callback,
                generation_kwargs=self.parameters
            )
            logger.info(f"Initialized OpenAI generator: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI generator: {e}")
            raise
    
    def _create_chat_message(self, message: Dict[str, str]) -> ChatMessage:
        """Create ChatMessage using new Haystack API based on role"""
        role = message["role"]
        content = message["content"]
        
        # Map roles to appropriate ChatMessage factory methods
        role_mapping = {
            "user": ChatMessage.from_user,
            "assistant": ChatMessage.from_assistant,
            "system": ChatMessage.from_system
        }
        
        # Use role-specific method or default to user
        factory_method = role_mapping.get(role, ChatMessage.from_user)
        return factory_method(content)
    
    def generate(
        self,
        messages: List[Dict[str, str]], 
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response using OpenAI"""
        try:
            # Normalize and convert messages using new Haystack API
            normalized_messages = self.normalize_messages(messages)
            chat_messages = [self._create_chat_message(msg) for msg in normalized_messages]
            
            # Merge generation parameters
            params = self.parameters.copy()
            if generation_kwargs:
                params.update(generation_kwargs)
            
            # Update generator parameters if needed
            if generation_kwargs:
                temp_generator = OpenAIChatGenerator(
                    api_key=Secret.from_token(self.api_key),
                    model=self.model,
                    api_base_url=self.api_base_url,
                    timeout=self.timeout,
                    streaming_callback=self.streaming_callback,
                    generation_kwargs=params
                )
            else:
                temp_generator = self.generator
            
            # Generate response
            logger.debug(f"Generating OpenAI response with {len(chat_messages)} messages")
            response = temp_generator.run(messages=chat_messages)
            
            return {
                "content": response["replies"][0].content,
                "role": "assistant",
                "model": self.model,
                "provider": "openai"
            }
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return {
                "content": f"OpenAI generation failed: {str(e)}",
                "role": "assistant",
                "model": self.model,
                "provider": "openai",
                "error": str(e)
            }

class OpenAIConnectorFactory:
    """Factory for creating OpenAI connectors"""
    
    SUPPORTED_MODELS = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k", 
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4o",
        "gpt-4o-mini"
    ]
    
    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        streaming_callback: Optional[Callable] = None
    ) -> OpenAIConnector:
        """Create OpenAI connector from config"""
        import os
        
        api_key = config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        model = config.get("model", "gpt-3.5-turbo")
        if model not in cls.SUPPORTED_MODELS:
            logger.warning(f"Model {model} not in supported list: {cls.SUPPORTED_MODELS}")
        
        return OpenAIConnector(
            api_key=api_key,
            model=model,
            api_base_url=config.get("api_base_url", "https://api.openai.com/v1"),
            timeout=config.get("timeout", 60),
            streaming_callback=streaming_callback,
            parameters=config.get("parameters", {})
        )
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported OpenAI models"""
        return cls.SUPPORTED_MODELS.copy()