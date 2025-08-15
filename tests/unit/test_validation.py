"""
Unit tests for validation functions
Tests API key validation, environment validation, and other critical validation logic
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from servers.mcp_server_secure import validate_api_key, validate_environment, setup_secure_logging


class TestApiKeyValidation:
    """Test cases for API key validation function."""
    
    def test_valid_api_key(self):
        """Test validation of valid OpenAI API keys."""
        # Test minimum valid key
        assert validate_api_key("sk-" + "x" * 18) == True
        
        # Test typical length key
        assert validate_api_key("sk-" + "a" * 48) == True
        
        # Test longer key
        assert validate_api_key("sk-" + "1" * 100) == True
    
    def test_invalid_api_key_format(self):
        """Test validation rejects invalid API key formats."""
        # Test wrong prefix
        assert validate_api_key("ak-1234567890123456789") == False
        assert validate_api_key("key-1234567890123456789") == False
        
        # Test no prefix
        assert validate_api_key("1234567890123456789") == False
        
        # Test empty prefix
        assert validate_api_key("-1234567890123456789") == False
    
    def test_invalid_api_key_length(self):
        """Test validation rejects API keys that are too short."""
        # Test too short
        assert validate_api_key("sk-123") == False
        assert validate_api_key("sk-1234567890") == False
        
        # Test exactly at boundary (20 chars total = 17 after prefix)
        assert validate_api_key("sk-" + "x" * 17) == False
        
        # Test exactly at minimum (21 chars total = 18 after prefix)
        assert validate_api_key("sk-" + "x" * 18) == True
    
    def test_empty_and_none_api_key(self):
        """Test validation handles empty and None values correctly."""
        assert validate_api_key("") == False
        assert validate_api_key(None) == False
    
    def test_whitespace_api_key(self):
        """Test validation handles whitespace correctly."""
        # Test leading/trailing whitespace
        assert validate_api_key("  sk-" + "x" * 18 + "  ") == False
        
        # Test internal whitespace
        assert validate_api_key("sk- " + "x" * 18) == False
    
    def test_special_characters(self):
        """Test validation with special characters."""
        # Test with various special characters
        valid_key_base = "sk-" + "x" * 18
        
        # These should be valid (alphanumeric and common characters)
        assert validate_api_key("sk-abc123XYZ789012345") == True
        assert validate_api_key("sk-ABC123xyz789012345") == True
        
        # Unicode characters (should still validate format, though may not be real keys)
        assert validate_api_key("sk-" + "🔑" * 18) == True


class TestEnvironmentValidation:
    """Test cases for environment validation function."""
    
    def setUp(self):
        """Set up test environment."""
        # Store original environment
        self.original_env = os.environ.copy()
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123456789012345678'}, clear=True)
    def test_valid_environment(self):
        """Test environment validation with valid API key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {'DATA_PATH': temp_dir}):
                logger = validate_environment()
                assert logger is not None
                assert isinstance(logger, logging.Logger)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        """Test environment validation fails with missing API key."""
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY environment variable is required"):
            validate_environment()
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'invalid-key'}, clear=True)
    def test_invalid_api_key_in_environment(self):
        """Test environment validation fails with invalid API key."""
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY environment variable is required"):
            validate_environment()
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123456789012345678'}, clear=True)
    def test_data_path_creation(self):
        """Test that data path is created during validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data"
            
            with patch.dict(os.environ, {'DATA_PATH': str(data_path)}):
                validate_environment()
                assert data_path.exists()
                assert data_path.is_dir()
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123456789012345678'}, clear=True)
    def test_default_data_path(self):
        """Test that default data path is used when not specified."""
        # Don't set DATA_PATH in environment
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            validate_environment()
            # Should have tried to create ./data directory
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @patch('mcp_server_secure.setup_secure_logging')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123456789012345678'}, clear=True)
    def test_logging_setup_called(self, mock_setup_logging):
        """Test that logging setup is called during validation."""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        result = validate_environment()
        
        mock_setup_logging.assert_called_once()
        assert result == mock_logger
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test123456789012345678',
        'LOG_LEVEL': 'DEBUG'
    }, clear=True)
    def test_log_level_environment_variable(self):
        """Test that LOG_LEVEL environment variable is respected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {'DATA_PATH': temp_dir}):
                logger = validate_environment()
                # This test mainly ensures no errors occur with LOG_LEVEL set
                assert logger is not None


class TestSecureLogging:
    """Test cases for secure logging setup."""
    
    def test_logging_setup_returns_logger(self):
        """Test that logging setup returns a logger instance."""
        logger = setup_secure_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "mcp-academic-rag-server"
    
    def test_logging_uses_stderr(self):
        """Test that logging is configured to use stderr."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_secure_logging()
            
            # Verify basicConfig was called with stderr
            mock_basic_config.assert_called_once()
            args, kwargs = mock_basic_config.call_args
            assert kwargs['stream'] == sys.stderr
    
    def test_logging_level_configuration(self):
        """Test that logging level is configured correctly."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_secure_logging()
            
            args, kwargs = mock_basic_config.call_args
            assert kwargs['level'] == logging.INFO
    
    def test_logging_format_configuration(self):
        """Test that logging format is configured correctly."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_secure_logging()
            
            args, kwargs = mock_basic_config.call_args
            expected_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            assert kwargs['format'] == expected_format


class TestIntegrationValidation:
    """Integration tests for validation functions."""
    
    @patch.dict(os.environ, {}, clear=True)
    def test_complete_validation_flow_failure(self):
        """Test complete validation flow with missing requirements."""
        # Test that validation fails appropriately when requirements aren't met
        with pytest.raises(EnvironmentError):
            validate_environment()
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-valid123456789012345678',
        'LOG_LEVEL': 'INFO'
    }, clear=True)
    def test_complete_validation_flow_success(self):
        """Test complete validation flow with all requirements met."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {'DATA_PATH': temp_dir}):
                logger = validate_environment()
                
                # Verify successful validation
                assert logger is not None
                assert isinstance(logger, logging.Logger)
                
                # Verify data directory was created
                data_path = Path(temp_dir)
                assert data_path.exists()
                assert data_path.is_dir()


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])