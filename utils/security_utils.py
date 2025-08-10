"""
Security utilities for the Academic RAG Server

This module provides security features including:
- Input validation and sanitization
- File upload security
- API key management
- Rate limiting
- CORS handling
- Security headers
"""

import os
import re
import hashlib
import secrets
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Set
from functools import wraps
from datetime import datetime, timedelta
import magic
import hmac
from werkzeug.utils import secure_filename
from flask import request, jsonify, current_app
import time
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class SecurityConfig:
    """Security configuration constants"""
    
    # File upload settings
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'txt', 'docx'}
    ALLOWED_MIME_TYPES = {
        'application/pdf',
        'image/png',
        'image/jpeg',
        'image/tiff',
        'text/plain',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\.',
        r'\.\./',
        r'%2e%2e',
        r'%252e%252e',
        r'\x00',
        r'[\\/]etc[\\/]',
        r'[\\/]proc[\\/]',
        r'[\\/]dev[\\/]'
    ]
    
    # SQL injection patterns (basic)
    SQL_INJECTION_PATTERNS = [
        r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
        r"(-{2}|/\*|\*/|;|\||&&)",
        r"(\'|\"|`|\\)",
        r"(\bor\b\s*\d+\s*=\s*\d+)",
        r"(\band\b\s*\d+\s*=\s*\d+)"
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe",
        r"<object",
        r"<embed"
    ]


class InputValidator:
    """Input validation utilities"""
    
    @staticmethod
    def validate_filename(filename: str) -> Tuple[bool, Optional[str]]:
        """
        Validate filename for security issues
        
        Args:
            filename: Filename to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not filename:
            return False, "Filename cannot be empty"
        
        # Check length
        if len(filename) > 255:
            return False, "Filename too long"
        
        # Check for path traversal
        for pattern in SecurityConfig.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                return False, "Invalid filename: potential path traversal"
        
        # Check for null bytes
        if '\x00' in filename:
            return False, "Invalid filename: null byte detected"
        
        # Secure the filename
        secured = secure_filename(filename)
        if not secured or secured != filename:
            return False, "Invalid filename: contains illegal characters"
        
        # Check extension
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if ext not in SecurityConfig.ALLOWED_EXTENSIONS:
            return False, f"File type not allowed: .{ext}"
        
        return True, None
    
    @staticmethod
    def validate_file_content(file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate file content using magic numbers
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > SecurityConfig.MAX_FILE_SIZE:
                return False, "File too large"
            
            if file_size == 0:
                return False, "File is empty"
            
            # Check MIME type
            try:
                mime = magic.from_file(file_path, mime=True)
                if mime not in SecurityConfig.ALLOWED_MIME_TYPES:
                    return False, f"File type not allowed: {mime}"
            except Exception:
                # Fallback to extension-based check
                logger.warning("python-magic not available, using extension check only")
            
            return True, None
            
        except Exception as e:
            logger.error(f"File validation error: {str(e)}")
            return False, f"File validation failed: {str(e)}"
    
    @staticmethod
    def sanitize_input(text: str, allow_html: bool = False) -> str:
        """
        Sanitize text input
        
        Args:
            text: Input text
            allow_html: Whether to allow HTML
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        if not allow_html:
            # Escape HTML entities
            text = (
                text.replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#x27;')
            )
        else:
            # Remove dangerous tags
            for pattern in SecurityConfig.XSS_PATTERNS:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def validate_json_input(data: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate JSON input against schema
        
        Args:
            data: Input data
            schema: Expected schema
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            from jsonschema import validate, ValidationError
            validate(instance=data, schema=schema)
            return True, None
        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Validation error: {str(e)}"


class APIKeyManager:
    """API key management utilities"""
    
    @staticmethod
    def generate_api_key(prefix: str = "sk") -> str:
        """Generate a secure API key"""
        random_bytes = secrets.token_bytes(32)
        key = hashlib.sha256(random_bytes).hexdigest()
        return f"{prefix}_{key}"
    
    @staticmethod
    def hash_api_key(api_key: str, salt: Optional[bytes] = None) -> Tuple[str, bytes]:
        """
        Hash an API key for storage
        
        Args:
            api_key: The API key to hash
            salt: Optional salt (will be generated if not provided)
            
        Returns:
            Tuple of (hashed_key, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        
        hashed = hashlib.pbkdf2_hmac('sha256', api_key.encode(), salt, 100000)
        return hashed.hex(), salt
    
    @staticmethod
    def verify_api_key(api_key: str, hashed_key: str, salt: bytes) -> bool:
        """Verify an API key against its hash"""
        calculated_hash, _ = APIKeyManager.hash_api_key(api_key, salt)
        return hmac.compare_digest(calculated_hash, hashed_key)
    
    @staticmethod
    def mask_api_key(api_key: str) -> str:
        """Mask API key for display (show only first and last 4 characters)"""
        if len(api_key) <= 8:
            return "*" * len(api_key)
        return f"{api_key[:4]}...{api_key[-4:]}"


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if request is allowed under rate limit
        
        Args:
            identifier: Unique identifier (IP, user ID, etc.)
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        now = time.time()
        
        with self.lock:
            # Clean old requests
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < window_seconds
            ]
            
            # Check rate limit
            if len(self.requests[identifier]) >= max_requests:
                oldest_request = min(self.requests[identifier])
                retry_after = int(window_seconds - (now - oldest_request))
                return False, retry_after
            
            # Add current request
            self.requests[identifier].append(now)
            return True, None
    
    def reset(self, identifier: str):
        """Reset rate limit for identifier"""
        with self.lock:
            self.requests.pop(identifier, None)


# Global rate limiter instance
rate_limiter = RateLimiter()


def require_api_key(check_permission: Optional[Callable] = None):
    """
    Decorator to require API key authentication
    
    Args:
        check_permission: Optional function to check specific permissions
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get API key from header
            api_key = request.headers.get('X-API-Key')
            if not api_key:
                return jsonify({
                    'error': 'API key required',
                    'code': 'MISSING_API_KEY'
                }), 401
            
            # Validate API key format
            if not api_key.startswith(('sk_', 'pk_')):
                return jsonify({
                    'error': 'Invalid API key format',
                    'code': 'INVALID_API_KEY'
                }), 401
            
            # Here you would typically validate against database
            # For now, we'll use environment variable
            valid_keys = os.environ.get('VALID_API_KEYS', '').split(',')
            if api_key not in valid_keys:
                return jsonify({
                    'error': 'Invalid API key',
                    'code': 'INVALID_API_KEY'
                }), 401
            
            # Check additional permissions if provided
            if check_permission and not check_permission(api_key):
                return jsonify({
                    'error': 'Insufficient permissions',
                    'code': 'FORBIDDEN'
                }), 403
            
            # Add API key to request context
            request.api_key = api_key
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit(max_requests: int = 60, window_seconds: int = 60):
    """
    Decorator for rate limiting
    
    Args:
        max_requests: Maximum requests allowed
        window_seconds: Time window in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get identifier (IP address or API key)
            identifier = request.headers.get('X-API-Key', request.remote_addr)
            
            # Check rate limit
            allowed, retry_after = rate_limiter.is_allowed(
                identifier,
                max_requests,
                window_seconds
            )
            
            if not allowed:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'code': 'RATE_LIMIT_EXCEEDED',
                    'retry_after': retry_after
                }), 429
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_content_type(*allowed_types: str):
    """
    Decorator to validate request content type
    
    Args:
        allowed_types: Allowed content types
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            content_type = request.content_type
            
            if not content_type:
                return jsonify({
                    'error': 'Content-Type header required',
                    'code': 'MISSING_CONTENT_TYPE'
                }), 400
            
            # Check base content type (ignore charset)
            base_type = content_type.split(';')[0].strip()
            
            if base_type not in allowed_types:
                return jsonify({
                    'error': f'Invalid content type: {base_type}',
                    'code': 'INVALID_CONTENT_TYPE',
                    'allowed': list(allowed_types)
                }), 415
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class SecurityHeaders:
    """Security headers middleware"""
    
    HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'",
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
    }
    
    @staticmethod
    def add_security_headers(response):
        """Add security headers to response"""
        for header, value in SecurityHeaders.HEADERS.items():
            response.headers[header] = value
        return response


def sanitize_path(path: str, base_path: str) -> Optional[str]:
    """
    Sanitize and validate file path
    
    Args:
        path: Input path
        base_path: Base directory path
        
    Returns:
        Sanitized absolute path or None if invalid
    """
    try:
        # Resolve to absolute path
        abs_path = Path(path).resolve()
        abs_base = Path(base_path).resolve()
        
        # Check if path is within base directory
        if not str(abs_path).startswith(str(abs_base)):
            logger.warning(f"Path traversal attempt: {path}")
            return None
        
        # Check for symbolic links
        if abs_path.is_symlink():
            logger.warning(f"Symbolic link detected: {path}")
            return None
        
        return str(abs_path)
        
    except Exception as e:
        logger.error(f"Path sanitization error: {str(e)}")
        return None


# Example usage
if __name__ == "__main__":
    # Test input validation
    validator = InputValidator()
    
    # Test filename validation
    filenames = [
        "document.pdf",
        "../../../etc/passwd",
        "file\x00.pdf",
        "<script>alert('xss')</script>.pdf"
    ]
    
    for filename in filenames:
        valid, error = validator.validate_filename(filename)
        print(f"Filename '{filename}': Valid={valid}, Error={error}")
    
    # Test API key generation
    api_key = APIKeyManager.generate_api_key()
    print(f"Generated API key: {APIKeyManager.mask_api_key(api_key)}")
    
    # Test rate limiting
    for i in range(5):
        allowed, retry = rate_limiter.is_allowed("test_user", 3, 60)
        print(f"Request {i+1}: Allowed={allowed}, Retry after={retry}")
        time.sleep(0.1)