"""
Google Gemini LLM Connector - Gemini model integration
"""

import logging
from typing import Dict, Any, Optional, List
try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from .base_llm_connector import BaseLLMConnector

logger = logging.getLogger(__name__)

class GoogleConnector(BaseLLMConnector):
    """Google Gemini LLM connector"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-pro",
        timeout: int = 60,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Google Gemini connector
        
        Args:
            api_key: Google AI API key
            model: Gemini model name
            timeout: Request timeout
            parameters: Generation parameters (temperature, max_output_tokens, etc.)
        """
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "Google generative AI package not installed. Install with: pip install google-generativeai"
            )
        
        super().__init__(api_key, model, timeout, parameters)
        self._init_generator()
    
    def _get_provider_name(self) -> str:
        return "google"
    
    def _init_generator(self):
        """Initialize Google Gemini client"""
        try:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
            logger.info(f"Initialized Google Gemini client: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini client: {e}")
            raise
    
    def generate(
        self,
        messages: List[Dict[str, str]], 
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response using Gemini"""
        try:
            # Normalize messages
            normalized_messages = self.normalize_messages(messages)
            
            # Convert to Gemini chat format
            history = []
            current_message = ""
            
            for msg in normalized_messages:
                if msg["role"] == "system":
                    # Add system message as context to the first user message
                    if current_message:
                        current_message = msg["content"] + "\n\n" + current_message
                    else:
                        current_message = msg["content"]
                elif msg["role"] == "user":
                    if current_message:
                        current_message += "\n\n" + msg["content"]
                    else:
                        current_message = msg["content"]
                elif msg["role"] == "assistant":
                    if current_message:
                        history.append({
                            "role": "user",
                            "parts": [current_message]
                        })
                        current_message = ""
                    history.append({
                        "role": "model", 
                        "parts": [msg["content"]]
                    })
            
            # Merge generation parameters
            params = self.parameters.copy()
            if generation_kwargs:
                params.update(generation_kwargs)
            
            # Configure generation parameters
            generation_config = {}
            if "temperature" in params:
                generation_config["temperature"] = params["temperature"]
            if "max_tokens" in params:
                generation_config["max_output_tokens"] = params["max_tokens"]
            elif "max_output_tokens" in params:
                generation_config["max_output_tokens"] = params["max_output_tokens"]
            if "top_p" in params:
                generation_config["top_p"] = params["top_p"]
            if "top_k" in params:
                generation_config["top_k"] = params["top_k"]
            if "stop_sequences" in params:
                generation_config["stop_sequences"] = params["stop_sequences"]
            
            # Start chat or generate single response
            if history:
                chat = self.client.start_chat(history=history)
                logger.debug(f"Generating Gemini response with {len(history)} history messages")
                response = chat.send_message(
                    current_message,
                    generation_config=genai.types.GenerationConfig(**generation_config) if generation_config else None
                )
            else:
                logger.debug("Generating Gemini response for single message")
                response = self.client.generate_content(
                    current_message,
                    generation_config=genai.types.GenerationConfig(**generation_config) if generation_config else None
                )
            
            return {
                "content": response.text,
                "role": "assistant",
                "model": self.model,
                "provider": "google"
            }
            
        except Exception as e:
            logger.error(f"Google Gemini generation failed: {e}")
            return {
                "content": f"Gemini generation failed: {str(e)}",
                "role": "assistant",
                "model": self.model,
                "provider": "google",
                "error": str(e)
            }

class GoogleConnectorFactory:
    """Factory for creating Google connectors"""
    
    SUPPORTED_MODELS = [
        "gemini-pro",
        "gemini-pro-vision", 
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b"
    ]
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> GoogleConnector:
        """Create Google connector from config"""
        import os
        
        api_key = config.get("api_key", os.environ.get("GOOGLE_API_KEY", ""))
        if not api_key:
            raise ValueError("Google AI API key is required")
        
        model = config.get("model", "gemini-pro")
        if model not in cls.SUPPORTED_MODELS:
            logger.warning(f"Model {model} not in supported list: {cls.SUPPORTED_MODELS}")
        
        return GoogleConnector(
            api_key=api_key,
            model=model,
            timeout=config.get("timeout", 60),
            parameters=config.get("parameters", {})
        )
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported Gemini models"""
        return cls.SUPPORTED_MODELS.copy()