"""
LLM Factory - Central factory for creating LLM connectors
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Union
from .base_llm_connector import BaseLLMConnector

logger = logging.getLogger(__name__)

class LLMFactory:
    """Central factory for creating LLM connectors"""
    
    SUPPORTED_PROVIDERS = {
        "openai": {
            "factory_class": "OpenAIConnectorFactory",
            "module": "connectors.openai_connector",
            "description": "OpenAI GPT models (GPT-3.5, GPT-4, etc.)"
        },
        "anthropic": {
            "factory_class": "AnthropicConnectorFactory", 
            "module": "connectors.anthropic_connector",
            "description": "Anthropic Claude models"
        },
        "google": {
            "factory_class": "GoogleConnectorFactory",
            "module": "connectors.google_connector", 
            "description": "Google Gemini models"
        }
    }
    
    @classmethod
    def create_connector(
        cls,
        provider: str,
        config: Dict[str, Any],
        streaming_callback: Optional[Callable] = None
    ) -> BaseLLMConnector:
        """
        Create LLM connector for specified provider
        
        Args:
            provider: Provider name ('openai', 'anthropic', 'google')
            config: Provider-specific configuration
            streaming_callback: Optional streaming callback (OpenAI only)
            
        Returns:
            BaseLLMConnector: Initialized connector instance
            
        Raises:
            ValueError: If provider is not supported
            ImportError: If provider dependencies are not available
        """
        provider = provider.lower()
        
        if provider not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: {list(cls.SUPPORTED_PROVIDERS.keys())}"
            )
        
        try:
            provider_info = cls.SUPPORTED_PROVIDERS[provider]
            module_name = provider_info["module"]
            factory_class_name = provider_info["factory_class"]
            
            # Import the module
            module = __import__(module_name, fromlist=[factory_class_name])
            factory_class = getattr(module, factory_class_name)
            
            # Create connector
            if provider == "openai":
                return factory_class.create(config, streaming_callback)
            else:
                return factory_class.create(config)
            
        except ImportError as e:
            logger.error(f"Failed to import {provider} connector: {e}")
            raise ImportError(
                f"Dependencies for {provider} provider not available. "
                f"Please install required packages: {cls._get_install_command(provider)}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to create {provider} connector: {e}")
            raise
    
    @classmethod
    def get_supported_providers(cls) -> Dict[str, str]:
        """Get supported providers with descriptions"""
        return {
            provider: info["description"] 
            for provider, info in cls.SUPPORTED_PROVIDERS.items()
        }
    
    @classmethod
    def get_supported_models(cls, provider: str) -> List[str]:
        """Get supported models for a provider"""
        provider = provider.lower()
        
        if provider not in cls.SUPPORTED_PROVIDERS:
            return []
        
        try:
            provider_info = cls.SUPPORTED_PROVIDERS[provider]
            module_name = provider_info["module"]
            factory_class_name = provider_info["factory_class"]
            
            # Import and get models
            module = __import__(module_name, fromlist=[factory_class_name])
            factory_class = getattr(module, factory_class_name)
            
            return factory_class.get_supported_models()
            
        except Exception as e:
            logger.warning(f"Could not get models for {provider}: {e}")
            return []
    
    @classmethod
    def validate_config(cls, provider: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration for a provider
        
        Returns:
            Dict with 'valid' (bool) and 'errors' (list) keys
        """
        provider = provider.lower()
        errors = []
        
        if provider not in cls.SUPPORTED_PROVIDERS:
            errors.append(f"Unsupported provider: {provider}")
            return {"valid": False, "errors": errors}
        
        # Common validation
        if not config.get("api_key"):
            env_var = cls._get_env_var_name(provider)
            errors.append(f"API key required. Set 'api_key' in config or {env_var} environment variable")
        
        model = config.get("model")
        if model:
            supported_models = cls.get_supported_models(provider)
            if supported_models and model not in supported_models:
                errors.append(f"Model {model} not supported. Supported models: {supported_models}")
        
        # Provider-specific validation
        if provider == "anthropic":
            max_tokens = config.get("parameters", {}).get("max_tokens")
            if max_tokens and max_tokens > 4096:
                errors.append("Anthropic models have a max_tokens limit of 4096")
        
        elif provider == "google":
            max_output_tokens = config.get("parameters", {}).get("max_output_tokens")
            if max_output_tokens and max_output_tokens > 8192:
                errors.append("Google Gemini models have a max_output_tokens limit of 8192")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    @classmethod
    def create_from_unified_config(
        cls,
        config: Dict[str, Any],
        streaming_callback: Optional[Callable] = None
    ) -> BaseLLMConnector:
        """
        Create connector from unified configuration format
        
        Config format:
        {
            "provider": "openai|anthropic|google",
            "model": "model-name",
            "api_key": "key-or-env-var",
            "parameters": {...}
        }
        """
        provider = config.get("provider")
        if not provider:
            raise ValueError("Provider must be specified in config")
        
        # Validate configuration
        validation = cls.validate_config(provider, config)
        if not validation["valid"]:
            raise ValueError(f"Invalid configuration: {', '.join(validation['errors'])}")
        
        return cls.create_connector(provider, config, streaming_callback)
    
    @staticmethod
    def _get_env_var_name(provider: str) -> str:
        """Get environment variable name for provider API key"""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", 
            "google": "GOOGLE_API_KEY"
        }
        return env_vars.get(provider.lower(), f"{provider.upper()}_API_KEY")
    
    @staticmethod
    def _get_install_command(provider: str) -> str:
        """Get pip install command for provider dependencies"""
        install_commands = {
            "openai": "pip install haystack-ai openai",
            "anthropic": "pip install anthropic",
            "google": "pip install google-generativeai"
        }
        return install_commands.get(provider.lower(), f"pip install {provider}")

# Convenience functions for backward compatibility
def create_openai_connector(config: Dict[str, Any], streaming_callback: Optional[Callable] = None) -> BaseLLMConnector:
    """Create OpenAI connector (backward compatibility)"""
    return LLMFactory.create_connector("openai", config, streaming_callback)

def create_anthropic_connector(config: Dict[str, Any]) -> BaseLLMConnector:
    """Create Anthropic connector (backward compatibility)"""
    return LLMFactory.create_connector("anthropic", config)

def create_google_connector(config: Dict[str, Any]) -> BaseLLMConnector:
    """Create Google connector (backward compatibility)"""
    return LLMFactory.create_connector("google", config)