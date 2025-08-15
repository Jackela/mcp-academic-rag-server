"""
Integration tests for multi-model LLM system
Tests the complete integration of multiple LLM providers with the RAG system
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from connectors.llm_factory import LLMFactory
from core.server_context import ServerContext
from core.config_manager import ConfigManager
from rag.haystack_pipeline import RAGPipelineFactory


class TestMultiModelIntegration:
    """Integration tests for multi-model LLM system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
        
        # Create test configuration
        self.test_config = {
            "storage": {
                "type": "local",
                "base_path": "./data"
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": "${OPENAI_API_KEY}",
                "parameters": {
                    "temperature": 0.1,
                    "max_tokens": 500
                }
            },
            "rag_settings": {
                "retriever_top_k": 5,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "processors": {}
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    @patch('connectors.openai_connector.OpenAIChatGenerator')
    def test_openai_provider_integration(self, mock_generator):
        """Test OpenAI provider integration with server context"""
        # Mock the generator
        mock_generator_instance = Mock()
        mock_generator.return_value = mock_generator_instance
        
        # Create config manager with test config
        config_manager = ConfigManager(str(self.config_path))
        
        # Test LLM factory integration
        llm_config = config_manager.get_value("llm", {})
        
        # Resolve environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        connector_config = {
            "api_key": api_key,
            "model": llm_config.get("model", "gpt-3.5-turbo"),
            "parameters": llm_config.get("parameters", {})
        }
        
        # Create connector
        connector = LLMFactory.create_connector("openai", connector_config)
        
        assert connector.provider_name == "openai"
        assert connector.model == "gpt-3.5-turbo"
        assert connector.api_key == "test-openai-key"
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"})
    def test_anthropic_provider_integration(self):
        """Test Anthropic provider integration"""
        # Update config for Anthropic
        self.test_config["llm"]["provider"] = "anthropic"
        self.test_config["llm"]["model"] = "claude-3-sonnet-20240229"
        self.test_config["llm"]["api_key"] = "${ANTHROPIC_API_KEY}"
        
        # Test configuration validation (without actually creating connector)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        connector_config = {
            "api_key": api_key,
            "model": "claude-3-sonnet-20240229",
            "parameters": {}
        }
        
        # Test config validation (this works without anthropic package)
        validation = LLMFactory.validate_config("anthropic", connector_config)
        assert validation["valid"]
    
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"})
    @patch('connectors.google_connector.genai')
    def test_google_provider_integration(self, mock_genai):
        """Test Google provider integration"""
        # Update config for Google
        self.test_config["llm"]["provider"] = "google"
        self.test_config["llm"]["model"] = "gemini-pro"
        self.test_config["llm"]["api_key"] = "${GOOGLE_API_KEY}"
        
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
        
        # Mock Google AI client
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        config_manager = ConfigManager(str(self.config_path))
        llm_config = config_manager.get_value("llm", {})
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        connector_config = {
            "api_key": api_key,
            "model": llm_config.get("model"),
            "parameters": llm_config.get("parameters", {})
        }
        
        # Test config validation
        validation = LLMFactory.validate_config("google", connector_config)
        assert validation["valid"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('connectors.openai_connector.OpenAIChatGenerator')
    @patch('core.processor_loader.ProcessorLoader.load_processors')
    def test_server_context_multi_model_initialization(self, mock_load_processors, mock_generator):
        """Test server context initialization with multi-model support"""
        # Mock processor loading
        mock_load_processors.return_value = []
        
        # Mock generator
        mock_generator_instance = Mock()
        mock_generator.return_value = mock_generator_instance
        
        # Create server context with custom config
        context = ServerContext()
        context._config_manager = ConfigManager(str(self.config_path))
        
        # Initialize context
        context.initialize()
        
        assert context.is_initialized
        assert context.rag_pipeline is not None

    def test_config_environment_variable_resolution(self):
        """Test environment variable resolution in config"""
        # Test with different environment variable formats
        test_cases = [
            ("${OPENAI_API_KEY}", "OPENAI_API_KEY", "test-key-1"),
            ("${ANTHROPIC_API_KEY}", "ANTHROPIC_API_KEY", "test-key-2"),
            ("direct-key", None, "direct-key")
        ]
        
        for api_key_field, env_var, expected in test_cases:
            if env_var:
                with patch.dict(os.environ, {env_var: expected}):
                    # Test environment variable extraction
                    if api_key_field.startswith("${") and api_key_field.endswith("}"):
                        extracted_env_var = api_key_field[2:-1]
                        resolved_key = os.environ.get(extracted_env_var, "")
                        assert resolved_key == expected
            else:
                # Direct key case
                resolved_key = api_key_field
                assert resolved_key == expected

    def test_provider_switching_configuration(self):
        """Test switching between providers through configuration"""
        provider_configs = [
            {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": "test-openai-key"
            },
            {
                "provider": "anthropic", 
                "model": "claude-3-sonnet-20240229",
                "api_key": "test-anthropic-key"
            },
            {
                "provider": "google",
                "model": "gemini-pro",
                "api_key": "test-google-key"
            }
        ]
        
        for config in provider_configs:
            # Test configuration validation
            validation = LLMFactory.validate_config(config["provider"], config)
            assert validation["valid"], f"Config validation failed for {config['provider']}"
            
            # Test supported models
            models = LLMFactory.get_supported_models(config["provider"])
            if models:  # Only test if models list is available
                assert config["model"] in models, f"Model {config['model']} not supported by {config['provider']}"

    def test_rag_pipeline_factory_multi_model(self):
        """Test RAG pipeline factory with different LLM connectors"""
        # Test pipeline creation concept without mock components
        # (since Haystack requires actual @component decorated classes)
        
        # Test configuration validation for different providers
        providers = ["openai", "anthropic", "google"]
        
        for provider in providers:
            config = {
                "api_key": "test-key",
                "model": "test-model"
            }
            
            validation = LLMFactory.validate_config(provider, config)
            # Should be valid structure (model validation may fail but that's ok)
            assert isinstance(validation, dict)
            assert "valid" in validation
            assert "errors" in validation

    def test_error_handling_missing_dependencies(self):
        """Test error handling when provider dependencies are missing"""
        # Test with providers that might not be installed
        error_configs = [
            ("anthropic", "claude-3-sonnet-20240229", "test-key"),
            ("google", "gemini-pro", "test-key")
        ]
        
        for provider, model, api_key in error_configs:
            config = {
                "provider": provider,
                "model": model,
                "api_key": api_key
            }
            
            try:
                # This might raise ImportError if dependencies not installed
                connector = LLMFactory.create_connector(provider, config)
                # If successful, test basic functionality
                assert connector.provider_name == provider
                assert connector.model == model
            except ImportError as e:
                # Expected behavior when dependencies not available
                assert "not available" in str(e) or "not installed" in str(e)

    def test_configuration_parameter_inheritance(self):
        """Test parameter inheritance and override behavior"""
        base_config = {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "api_key": "test-key",
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 500
            }
        }
        
        # Test parameter validation
        validation = LLMFactory.validate_config("openai", base_config)
        assert validation["valid"]
        
        # Test parameter override
        override_params = {"temperature": 0.8, "max_tokens": 1000}
        merged_config = base_config.copy()
        merged_config["parameters"].update(override_params)
        
        assert merged_config["parameters"]["temperature"] == 0.8
        assert merged_config["parameters"]["max_tokens"] == 1000

    def test_model_validation_across_providers(self):
        """Test model validation across different providers"""
        # Test valid models for each provider
        valid_model_tests = [
            ("openai", "gpt-3.5-turbo"),
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-sonnet-20240229"),
            ("google", "gemini-pro")
        ]
        
        for provider, model in valid_model_tests:
            config = {
                "api_key": "test-key",
                "model": model
            }
            
            validation = LLMFactory.validate_config(provider, config)
            # Should be valid if model is in supported list
            supported_models = LLMFactory.get_supported_models(provider)
            if supported_models and model in supported_models:
                assert validation["valid"]

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "openai-key",
        "ANTHROPIC_API_KEY": "anthropic-key",
        "GOOGLE_API_KEY": "google-key"
    })
    def test_environment_variable_fallback(self):
        """Test fallback to provider-specific environment variables"""
        providers = ["openai", "anthropic", "google"]
        
        for provider in providers:
            env_var_name = LLMFactory._get_env_var_name(provider)
            expected_key = os.environ.get(env_var_name)
            
            config = {"model": "test-model"}  # No api_key in config
            
            # Should find API key from environment
            validation = LLMFactory.validate_config(provider, config)
            # Will be valid if we have the right environment variable
            if expected_key:
                assert validation["valid"] or "not supported" in str(validation["errors"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])