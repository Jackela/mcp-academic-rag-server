"""
Base LLM Connector - Abstract base class for all LLM providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class BaseLLMConnector(ABC):
    """Abstract base class for LLM connectors"""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: int = 60,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base LLM connector
        
        Args:
            api_key (str): API key for the LLM provider
            model (str): Model name/identifier
            timeout (int): Request timeout in seconds
            parameters (dict, optional): LLM generation parameters
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.parameters = parameters or {}
        self.provider_name = self._get_provider_name()
    
    @abstractmethod
    def _get_provider_name(self) -> str:
        """Get the provider name (e.g., 'openai', 'anthropic', 'google')"""
        pass
    
    @abstractmethod
    def _init_generator(self):
        """Initialize the underlying generator/client"""
        pass
    
    @abstractmethod
    def generate(
        self, 
        messages: List[Dict[str, str]],
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate response from messages
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            generation_kwargs: Optional generation parameters
            
        Returns:
            Dict with 'content', 'role', 'model' and optionally 'error' keys
        """
        pass
    
    def update_parameters(self, parameters: Dict[str, Any]):
        """Update generation parameters"""
        self.parameters.update(parameters)
        logger.info(f"Updated {self.provider_name} parameters: {parameters}")
    
    def set_model(self, model: str):
        """Update the model"""
        if model != self.model:
            old_model = self.model
            self.model = model
            self._init_generator()
            logger.info(f"Updated {self.provider_name} model: {old_model} -> {model}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get connector information"""
        return {
            "provider": self.provider_name,
            "model": self.model,
            "timeout": self.timeout,
            "parameters": self.parameters.copy()
        }
    
    @staticmethod
    def normalize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Normalize message format across providers"""
        return [
            {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            }
            for msg in messages
        ]