"""
Unit tests for LLM connectors and factory system
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the connectors
from connectors.base_llm_connector import BaseLLMConnector
from connectors.llm_factory import LLMFactory
from connectors.openai_connector import OpenAIConnector, OpenAIConnectorFactory

class TestBaseLLMConnector:
    """Test the base LLM connector abstract class"""
    
    def test_base_connector_cannot_be_instantiated(self):
        """BaseLLMConnector is abstract and cannot be instantiated"""
        with pytest.raises(TypeError):
            BaseLLMConnector("api_key", "model")
    
    def test_normalize_messages(self):
        """Test message normalization"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"content": "No role specified"}  # Missing role
        ]
        
        normalized = BaseLLMConnector.normalize_messages(messages)
        
        assert len(normalized) == 3
        assert normalized[0] == {"role": "user", "content": "Hello"}
        assert normalized[1] == {"role": "assistant", "content": "Hi there"}
        assert normalized[2] == {"role": "user", "content": "No role specified"}  # Default role


class TestOpenAIConnector:
    """Test OpenAI connector implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.api_key = "test-api-key"
        self.model = "gpt-3.5-turbo"
    
    @patch('connectors.openai_connector.OpenAIChatGenerator')
    def test_openai_connector_initialization(self, mock_generator):
        """Test OpenAI connector initialization"""
        connector = OpenAIConnector(
            api_key=self.api_key,
            model=self.model
        )
        
        assert connector.api_key == self.api_key
        assert connector.model == self.model
        assert connector.provider_name == "openai"
        mock_generator.assert_called_once()
    
    @patch('connectors.openai_connector.OpenAIChatGenerator')
    def test_openai_connector_generate_success(self, mock_generator):
        """Test successful response generation"""
        # Mock the generator response
        mock_reply = Mock()
        mock_reply.content = "Test response"
        mock_generator_instance = Mock()
        mock_generator_instance.run.return_value = {"replies": [mock_reply]}
        mock_generator.return_value = mock_generator_instance
        
        connector = OpenAIConnector(
            api_key=self.api_key,
            model=self.model
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        result = connector.generate(messages)
        
        assert result["content"] == "Test response"
        assert result["role"] == "assistant"
        assert result["model"] == self.model
        assert result["provider"] == "openai"
        assert "error" not in result
    
    @patch('connectors.openai_connector.OpenAIChatGenerator')
    def test_openai_connector_generate_error(self, mock_generator):
        """Test error handling in generation"""
        mock_generator_instance = Mock()
        mock_generator_instance.run.side_effect = Exception("API Error")
        mock_generator.return_value = mock_generator_instance
        
        connector = OpenAIConnector(
            api_key=self.api_key,
            model=self.model
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        result = connector.generate(messages)
        
        assert "error" in result
        assert "API Error" in result["content"]
        assert result["provider"] == "openai"


class TestOpenAIConnectorFactory:
    """Test OpenAI connector factory"""
    
    def test_supported_models(self):
        """Test supported models list"""
        models = OpenAIConnectorFactory.get_supported_models()
        
        assert isinstance(models, list)
        assert "gpt-3.5-turbo" in models
        assert "gpt-4" in models
        assert "gpt-4o" in models
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('connectors.openai_connector.OpenAIChatGenerator')
    def test_factory_create_success(self, mock_generator):
        """Test successful connector creation via factory"""
        config = {
            "model": "gpt-4",
            "parameters": {"temperature": 0.5}
        }
        
        connector = OpenAIConnectorFactory.create(config)
        
        assert isinstance(connector, OpenAIConnector)
        assert connector.model == "gpt-4"
        assert connector.parameters["temperature"] == 0.5
    
    def test_factory_create_missing_api_key(self):
        """Test factory creation fails without API key"""
        config = {"model": "gpt-4"}
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAIConnectorFactory.create(config)


class TestLLMFactory:
    """Test the central LLM factory"""
    
    def test_supported_providers(self):
        """Test supported providers list"""
        providers = LLMFactory.get_supported_providers()
        
        assert isinstance(providers, dict)
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
    
    def test_get_supported_models_openai(self):
        """Test getting supported models for OpenAI"""
        models = LLMFactory.get_supported_models("openai")
        
        assert isinstance(models, list)
        assert "gpt-3.5-turbo" in models
        assert "gpt-4" in models
    
    def test_get_supported_models_invalid_provider(self):
        """Test getting models for invalid provider"""
        models = LLMFactory.get_supported_models("invalid_provider")
        assert models == []
    
    def test_validate_config_missing_provider(self):
        """Test configuration validation with missing provider"""
        config = {"model": "gpt-4"}
        
        validation = LLMFactory.validate_config("invalid_provider", config)
        
        assert not validation["valid"]
        assert "Unsupported provider" in validation["errors"][0]
    
    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key"""
        config = {"model": "gpt-4"}
        
        with patch.dict(os.environ, {}, clear=True):
            validation = LLMFactory.validate_config("openai", config)
            
            assert not validation["valid"]
            assert "API key required" in validation["errors"][0]
    
    def test_validate_config_invalid_model(self):
        """Test configuration validation with invalid model"""
        config = {
            "api_key": "test-key",
            "model": "invalid-model"
        }
        
        with patch('connectors.llm_factory.LLMFactory.get_supported_models') as mock_models:
            mock_models.return_value = ["gpt-3.5-turbo", "gpt-4"]
            
            validation = LLMFactory.validate_config("openai", config)
            
            assert not validation["valid"]
            assert "not supported" in validation["errors"][0]
    
    def test_validate_config_valid(self):
        """Test successful configuration validation"""
        config = {
            "api_key": "test-key",
            "model": "gpt-3.5-turbo"
        }
        
        validation = LLMFactory.validate_config("openai", config)
        
        assert validation["valid"]
        assert len(validation["errors"]) == 0
    
    @patch('connectors.llm_factory.LLMFactory.create_connector')
    def test_create_from_unified_config(self, mock_create):
        """Test creating connector from unified config"""
        mock_connector = Mock()
        mock_create.return_value = mock_connector
        
        config = {
            "provider": "openai",
            "api_key": "test-key",
            "model": "gpt-4"
        }
        
        result = LLMFactory.create_from_unified_config(config)
        
        assert result == mock_connector
        mock_create.assert_called_once_with("openai", config, None)
    
    def test_create_from_unified_config_missing_provider(self):
        """Test unified config creation fails without provider"""
        config = {"model": "gpt-4"}
        
        with pytest.raises(ValueError, match="Provider must be specified"):
            LLMFactory.create_from_unified_config(config)
    
    def test_env_var_names(self):
        """Test environment variable name generation"""
        assert LLMFactory._get_env_var_name("openai") == "OPENAI_API_KEY"
        assert LLMFactory._get_env_var_name("anthropic") == "ANTHROPIC_API_KEY"
        assert LLMFactory._get_env_var_name("google") == "GOOGLE_API_KEY"
        assert LLMFactory._get_env_var_name("custom") == "CUSTOM_API_KEY"
    
    def test_install_commands(self):
        """Test install command generation"""
        assert "openai" in LLMFactory._get_install_command("openai")
        assert "anthropic" in LLMFactory._get_install_command("anthropic")
        assert "google-generativeai" in LLMFactory._get_install_command("google")


# Mock connector for testing
class MockConnector(BaseLLMConnector):
    """Mock connector for testing base functionality"""
    
    def _get_provider_name(self) -> str:
        return "mock"
    
    def _init_generator(self):
        self.generator = Mock()
    
    def generate(self, messages, generation_kwargs=None):
        return {
            "content": "Mock response",
            "role": "assistant",
            "model": self.model,
            "provider": "mock"
        }


class TestMockConnector:
    """Test mock connector implementation"""
    
    def test_mock_connector_basic_functionality(self):
        """Test basic connector functionality with mock"""
        connector = MockConnector("test-key", "test-model")
        
        assert connector.provider_name == "mock"
        assert connector.model == "test-model"
        assert connector.api_key == "test-key"
    
    def test_mock_connector_generate(self):
        """Test mock connector generation"""
        connector = MockConnector("test-key", "test-model")
        
        messages = [{"role": "user", "content": "Hello"}]
        result = connector.generate(messages)
        
        assert result["content"] == "Mock response"
        assert result["provider"] == "mock"
        assert result["model"] == "test-model"
    
    def test_mock_connector_update_parameters(self):
        """Test parameter updates"""
        connector = MockConnector("test-key", "test-model")
        
        new_params = {"temperature": 0.8}
        connector.update_parameters(new_params)
        
        assert connector.parameters["temperature"] == 0.8
    
    def test_mock_connector_set_model(self):
        """Test model updates"""
        connector = MockConnector("test-key", "test-model")
        
        connector.set_model("new-model")
        
        assert connector.model == "new-model"
    
    def test_mock_connector_get_info(self):
        """Test connector info retrieval"""
        connector = MockConnector("test-key", "test-model", 
                                 parameters={"temperature": 0.5})
        
        info = connector.get_info()
        
        assert info["provider"] == "mock"
        assert info["model"] == "test-model"
        assert info["parameters"]["temperature"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])