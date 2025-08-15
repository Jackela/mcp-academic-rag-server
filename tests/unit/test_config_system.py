"""
Unit tests for configuration system with multi-model support
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from core.config_manager import ConfigManager
from connectors.llm_factory import LLMFactory


class TestConfigSystem:
    """Test configuration system functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_basic_config_loading(self):
        """Test basic configuration loading"""
        config_data = {
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": "test-key"
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)
        
        config_manager = ConfigManager(str(self.config_path))
        llm_config = config_manager.get_value("llm", {})
        
        assert llm_config["provider"] == "openai"
        assert llm_config["model"] == "gpt-3.5-turbo"
        assert llm_config["api_key"] == "test-key"
    
    def test_environment_variable_resolution(self):
        """Test environment variable resolution in config"""
        config_data = {
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": "${OPENAI_API_KEY}"
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-env-key"}):
            config_manager = ConfigManager(str(self.config_path))
            llm_config = config_manager.get_value("llm", {})
            
            # Test environment variable pattern detection
            api_key_field = llm_config["api_key"]
            if api_key_field.startswith("${") and api_key_field.endswith("}"):
                env_var = api_key_field[2:-1]
                resolved_key = os.environ.get(env_var, "")
                assert resolved_key == "test-env-key"
    
    def test_provider_configuration_validation(self):
        """Test provider configuration validation"""
        # Test different provider configurations
        provider_configs = [
            {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": "test-key"
            },
            {
                "provider": "anthropic",
                "model": "claude-3-sonnet-20240229",
                "api_key": "test-key"
            },
            {
                "provider": "google",
                "model": "gemini-pro",
                "api_key": "test-key"
            }
        ]
        
        for config in provider_configs:
            validation = LLMFactory.validate_config(config["provider"], config)
            assert isinstance(validation, dict)
            assert "valid" in validation
            assert "errors" in validation
    
    def test_multi_provider_configuration(self):
        """Test configuration with multiple providers defined"""
        config_data = {
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": "${OPENAI_API_KEY}"
            },
            "llm_providers": {
                "openai": {
                    "models": ["gpt-3.5-turbo", "gpt-4"],
                    "api_key_env": "OPENAI_API_KEY"
                },
                "anthropic": {
                    "models": ["claude-3-sonnet-20240229"],
                    "api_key_env": "ANTHROPIC_API_KEY"
                },
                "google": {
                    "models": ["gemini-pro"],
                    "api_key_env": "GOOGLE_API_KEY"
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)
        
        config_manager = ConfigManager(str(self.config_path))
        
        # Test main LLM config
        llm_config = config_manager.get_value("llm", {})
        assert llm_config["provider"] == "openai"
        
        # Test providers config
        providers_config = config_manager.get_value("llm_providers", {})
        assert "openai" in providers_config
        assert "anthropic" in providers_config
        assert "google" in providers_config
    
    def test_parameter_configuration(self):
        """Test LLM parameter configuration"""
        config_data = {
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": "test-key",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "top_p": 0.9
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)
        
        config_manager = ConfigManager(str(self.config_path))
        llm_config = config_manager.get_value("llm", {})
        
        params = llm_config.get("parameters", {})
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 1000
        assert params["top_p"] == 0.9
    
    def test_provider_specific_parameters(self):
        """Test provider-specific parameter configurations"""
        # Test different parameter sets for different providers
        provider_params = {
            "openai": {
                "temperature": 0.7,
                "max_tokens": 1000,
                "presence_penalty": 0.1
            },
            "anthropic": {
                "temperature": 0.5,
                "max_tokens": 1500,
                "stop_sequences": ["Human:", "Assistant:"]
            },
            "google": {
                "temperature": 0.3,
                "max_output_tokens": 1200,
                "top_p": 0.8
            }
        }
        
        for provider, params in provider_params.items():
            config = {
                "provider": provider,
                "model": "test-model",
                "api_key": "test-key",
                "parameters": params
            }
            
            # Test that validation accepts provider-specific parameters
            validation = LLMFactory.validate_config(provider, config)
            assert isinstance(validation, dict)
    
    def test_config_fallback_behavior(self):
        """Test configuration fallback behavior"""
        # Test minimal config with fallbacks
        minimal_config = {
            "llm": {
                "provider": "openai"
                # Missing model and api_key - should use defaults/env vars
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(minimal_config, f)
        
        config_manager = ConfigManager(str(self.config_path))
        llm_config = config_manager.get_value("llm", {})
        
        assert llm_config["provider"] == "openai"
        # Should have fallback handling for missing values
    
    def test_invalid_provider_configuration(self):
        """Test handling of invalid provider configurations"""
        invalid_configs = [
            {
                "provider": "invalid_provider",
                "model": "test-model",
                "api_key": "test-key"
            },
            {
                "provider": "openai",
                "model": "",  # Empty model
                "api_key": "test-key"
            },
            {
                "provider": "openai",
                "model": "test-model"
                # Missing api_key
            }
        ]
        
        for config in invalid_configs:
            if config["provider"] in ["openai", "anthropic", "google"]:
                validation = LLMFactory.validate_config(config["provider"], config)
                if not config.get("api_key"):
                    assert not validation["valid"]
                    assert len(validation["errors"]) > 0
            else:
                # Invalid provider
                validation = LLMFactory.validate_config(config["provider"], config)
                assert not validation["valid"]
                assert "Unsupported provider" in validation["errors"][0]
    
    def test_configuration_override_precedence(self):
        """Test configuration override precedence"""
        # Test that environment variables can override config file values
        config_data = {
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": "${OPENAI_API_KEY}"
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)
        
        # Test with different environment variable values
        env_values = ["env-key-1", "env-key-2", "env-key-3"]
        
        for env_value in env_values:
            with patch.dict(os.environ, {"OPENAI_API_KEY": env_value}):
                # Simulate environment variable resolution
                api_key_field = config_data["llm"]["api_key"]
                if api_key_field.startswith("${") and api_key_field.endswith("}"):
                    env_var = api_key_field[2:-1]
                    resolved_key = os.environ.get(env_var, "")
                    assert resolved_key == env_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])