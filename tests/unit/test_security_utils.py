"""
Unit tests for security utilities
"""

import pytest
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from werkzeug.test import EnvironBuilder
from flask import Flask, request

from utils.security_utils import (
    SecurityConfig,
    InputValidator,
    APIKeyManager,
    RateLimiter,
    require_api_key,
    rate_limit,
    validate_content_type,
    SecurityHeaders,
    sanitize_path
)


class TestInputValidator:
    """Test input validation functionality"""
    
    def test_validate_filename_valid(self):
        """Test valid filename validation"""
        validator = InputValidator()
        
        valid_filenames = [
            "document.pdf",
            "image.png",
            "scan.jpg",
            "file.txt",
            "report.docx"
        ]
        
        for filename in valid_filenames:
            is_valid, error = validator.validate_filename(filename)
            assert is_valid is True
            assert error is None
    
    def test_validate_filename_invalid(self):
        """Test invalid filename validation"""
        validator = InputValidator()
        
        invalid_cases = [
            ("", "Filename cannot be empty"),
            ("a" * 256 + ".pdf", "Filename too long"),
            ("../../../etc/passwd", "Invalid filename: potential path traversal"),
            ("file\x00.pdf", "Invalid filename: null byte detected"),
            ("script.sh", "File type not allowed: .sh"),
            ("<script>alert('xss')</script>.pdf", "Invalid filename: contains illegal characters"),
        ]
        
        for filename, expected_error in invalid_cases:
            is_valid, error = validator.validate_filename(filename)
            assert is_valid is False
            assert expected_error in error
    
    def test_validate_file_content(self):
        """Test file content validation"""
        validator = InputValidator()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_file = f.name
        
        try:
            # Valid file
            is_valid, error = validator.validate_file_content(temp_file)
            assert is_valid is True
            assert error is None
        finally:
            os.unlink(temp_file)
        
        # Test non-existent file
        is_valid, error = validator.validate_file_content("/non/existent/file")
        assert is_valid is False
        assert "File validation failed" in error
    
    def test_sanitize_input(self):
        """Test input sanitization"""
        validator = InputValidator()
        
        # Test HTML escaping
        dangerous_input = "<script>alert('xss')</script>"
        sanitized = validator.sanitize_input(dangerous_input)
        assert "&lt;script&gt;" in sanitized
        assert "<script>" not in sanitized
        
        # Test null byte removal
        null_input = "test\x00data"
        sanitized = validator.sanitize_input(null_input)
        assert "\x00" not in sanitized
        assert sanitized == "testdata"
        
        # Test with allow_html=True
        html_input = "<p>Hello</p><script>alert('xss')</script>"
        sanitized = validator.sanitize_input(html_input, allow_html=True)
        assert "<p>Hello</p>" in sanitized
        assert "<script>" not in sanitized
    
    def test_validate_json_input(self):
        """Test JSON input validation"""
        validator = InputValidator()
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }
        
        # Valid input
        valid_data = {"name": "John", "age": 30}
        is_valid, error = validator.validate_json_input(valid_data, schema)
        assert is_valid is True
        assert error is None
        
        # Invalid input - missing required field
        invalid_data = {"age": 30}
        is_valid, error = validator.validate_json_input(invalid_data, schema)
        assert is_valid is False
        assert "'name' is a required property" in error


class TestAPIKeyManager:
    """Test API key management"""
    
    def test_generate_api_key(self):
        """Test API key generation"""
        # Test with default prefix
        key1 = APIKeyManager.generate_api_key()
        assert key1.startswith("sk_")
        assert len(key1) == 67  # sk_ + 64 hex chars
        
        # Test with custom prefix
        key2 = APIKeyManager.generate_api_key(prefix="pk")
        assert key2.startswith("pk_")
        
        # Keys should be unique
        assert key1 != key2
    
    def test_hash_and_verify_api_key(self):
        """Test API key hashing and verification"""
        api_key = "sk_test_key_12345"
        
        # Hash the key
        hashed_key, salt = APIKeyManager.hash_api_key(api_key)
        assert isinstance(hashed_key, str)
        assert isinstance(salt, bytes)
        
        # Verify correct key
        is_valid = APIKeyManager.verify_api_key(api_key, hashed_key, salt)
        assert is_valid is True
        
        # Verify incorrect key
        is_valid = APIKeyManager.verify_api_key("wrong_key", hashed_key, salt)
        assert is_valid is False
    
    def test_mask_api_key(self):
        """Test API key masking"""
        # Normal key
        masked = APIKeyManager.mask_api_key("sk_1234567890abcdef")
        assert masked == "sk_1...cdef"
        
        # Short key
        masked = APIKeyManager.mask_api_key("short")
        assert masked == "*****"


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limit"""
        limiter = RateLimiter()
        
        # Make requests within limit
        for i in range(3):
            allowed, retry_after = limiter.is_allowed("user1", 5, 60)
            assert allowed is True
            assert retry_after is None
    
    def test_rate_limiter_blocks_excess_requests(self):
        """Test rate limiter blocks excess requests"""
        limiter = RateLimiter()
        
        # Make requests up to limit
        for i in range(3):
            limiter.is_allowed("user2", 3, 60)
        
        # Next request should be blocked
        allowed, retry_after = limiter.is_allowed("user2", 3, 60)
        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0
    
    def test_rate_limiter_window_expiry(self):
        """Test rate limiter window expiry"""
        limiter = RateLimiter()
        
        # Fill up the limit
        for i in range(2):
            limiter.is_allowed("user3", 2, 0.1)  # 100ms window
        
        # Should be blocked
        allowed, _ = limiter.is_allowed("user3", 2, 0.1)
        assert allowed is False
        
        # Wait for window to expire
        time.sleep(0.15)
        
        # Should be allowed again
        allowed, _ = limiter.is_allowed("user3", 2, 0.1)
        assert allowed is True
    
    def test_rate_limiter_reset(self):
        """Test rate limiter reset"""
        limiter = RateLimiter()
        
        # Make some requests
        limiter.is_allowed("user4", 3, 60)
        limiter.is_allowed("user4", 3, 60)
        
        # Reset the limiter
        limiter.reset("user4")
        
        # Should be able to make full quota again
        for i in range(3):
            allowed, _ = limiter.is_allowed("user4", 3, 60)
            assert allowed is True


class TestFlaskDecorators:
    """Test Flask decorators"""
    
    def setup_method(self):
        """Set up Flask app for testing"""
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
    
    def test_require_api_key_decorator(self):
        """Test API key requirement decorator"""
        # Set valid API key in environment
        os.environ['VALID_API_KEYS'] = 'sk_test_key_123,sk_test_key_456'
        
        @self.app.route('/test')
        @require_api_key()
        def test_endpoint():
            return {"status": "success"}
        
        with self.app.test_client() as client:
            # Without API key
            response = client.get('/test')
            assert response.status_code == 401
            assert response.json['code'] == 'MISSING_API_KEY'
            
            # With invalid API key
            response = client.get('/test', headers={'X-API-Key': 'invalid_key'})
            assert response.status_code == 401
            assert response.json['code'] == 'INVALID_API_KEY'
            
            # With valid API key
            response = client.get('/test', headers={'X-API-Key': 'sk_test_key_123'})
            assert response.status_code == 200
            assert response.json['status'] == 'success'
    
    def test_rate_limit_decorator(self):
        """Test rate limiting decorator"""
        @self.app.route('/test')
        @rate_limit(max_requests=2, window_seconds=60)
        def test_endpoint():
            return {"status": "success"}
        
        with self.app.test_client() as client:
            # Make requests up to limit
            for i in range(2):
                response = client.get('/test')
                assert response.status_code == 200
            
            # Next request should be rate limited
            response = client.get('/test')
            assert response.status_code == 429
            assert response.json['code'] == 'RATE_LIMIT_EXCEEDED'
            assert 'retry_after' in response.json
    
    def test_validate_content_type_decorator(self):
        """Test content type validation decorator"""
        @self.app.route('/test', methods=['POST'])
        @validate_content_type('application/json', 'text/plain')
        def test_endpoint():
            return {"status": "success"}
        
        with self.app.test_client() as client:
            # Without Content-Type
            response = client.post('/test')
            assert response.status_code == 400
            assert response.json['code'] == 'MISSING_CONTENT_TYPE'
            
            # With invalid Content-Type
            response = client.post('/test', content_type='application/xml')
            assert response.status_code == 415
            assert response.json['code'] == 'INVALID_CONTENT_TYPE'
            
            # With valid Content-Type
            response = client.post('/test', content_type='application/json')
            assert response.status_code == 200


class TestSecurityHeaders:
    """Test security headers"""
    
    def test_add_security_headers(self):
        """Test adding security headers to response"""
        app = Flask(__name__)
        
        @app.route('/test')
        def test_endpoint():
            return {"status": "success"}
        
        @app.after_request
        def add_headers(response):
            return SecurityHeaders.add_security_headers(response)
        
        with app.test_client() as client:
            response = client.get('/test')
            
            # Check security headers
            assert response.headers['X-Content-Type-Options'] == 'nosniff'
            assert response.headers['X-Frame-Options'] == 'DENY'
            assert response.headers['X-XSS-Protection'] == '1; mode=block'
            assert 'Strict-Transport-Security' in response.headers
            assert 'Content-Security-Policy' in response.headers


class TestPathSanitization:
    """Test path sanitization"""
    
    def test_sanitize_path_valid(self):
        """Test valid path sanitization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid path within base directory
            valid_path = os.path.join(temp_dir, "subdir", "file.txt")
            sanitized = sanitize_path(valid_path, temp_dir)
            assert sanitized is not None
            assert sanitized.startswith(temp_dir)
    
    def test_sanitize_path_traversal(self):
        """Test path traversal prevention"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Path traversal attempt
            evil_path = os.path.join(temp_dir, "..", "..", "etc", "passwd")
            sanitized = sanitize_path(evil_path, temp_dir)
            assert sanitized is None
    
    @pytest.mark.skipif(os.name == 'nt', reason="Symlinks require admin on Windows")
    def test_sanitize_path_symlink(self):
        """Test symbolic link detection"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a symlink
            target = os.path.join(temp_dir, "target.txt")
            link = os.path.join(temp_dir, "link.txt")
            
            with open(target, 'w') as f:
                f.write("test")
            
            os.symlink(target, link)
            
            # Should reject symlink
            sanitized = sanitize_path(link, temp_dir)
            assert sanitized is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])