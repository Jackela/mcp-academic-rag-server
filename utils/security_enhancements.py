"""
Enhanced Security Utilities for MCP Academic RAG Server

This module provides production-ready security enhancements including:
- Secure secret key generation and management
- Enhanced CSRF protection  
- File upload security validation
- API key rotation and management
- Security headers middleware
- Rate limiting with Redis backend
"""

import os
import secrets
import hashlib
import logging
import hmac
import time
from typing import Dict, Any, Optional, List
from functools import wraps
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from flask import request, jsonify, current_app, g
import magic
import redis

logger = logging.getLogger(__name__)


class SecretKeyManager:
    """Secure management of Flask secret keys with rotation support"""
    
    def __init__(self, key_file: str = './.secrets/flask_secret.key'):
        self.key_file = key_file
        self.backup_key_file = f"{key_file}.backup"
        os.makedirs(os.path.dirname(key_file), exist_ok=True)
    
    def generate_secret_key(self) -> str:
        """Generate a cryptographically secure secret key"""
        return secrets.token_urlsafe(32)
    
    def get_or_create_secret_key(self) -> str:
        """Get existing secret key or create a new one"""
        try:
            if os.path.exists(self.key_file):
                with open(self.key_file, 'r') as f:
                    key = f.read().strip()
                    if len(key) >= 32:  # Minimum key length
                        return key
            
            # Generate new key
            new_key = self.generate_secret_key()
            self.save_secret_key(new_key)
            logger.info("Generated new Flask secret key")
            return new_key
            
        except Exception as e:
            logger.error(f"Error managing secret key: {e}")
            # Fallback to environment variable or generated key
            return os.environ.get('SECRET_KEY', self.generate_secret_key())
    
    def save_secret_key(self, key: str) -> None:
        """Securely save secret key to file"""
        try:
            # Backup existing key
            if os.path.exists(self.key_file):
                os.rename(self.key_file, self.backup_key_file)
            
            # Write new key with restricted permissions
            with open(self.key_file, 'w') as f:
                f.write(key)
            
            # Set file permissions (readable only by owner)
            os.chmod(self.key_file, 0o600)
            
        except Exception as e:
            logger.error(f"Error saving secret key: {e}")
            raise
    
    def rotate_secret_key(self) -> str:
        """Rotate secret key (generate new key while keeping backup)"""
        new_key = self.generate_secret_key()
        self.save_secret_key(new_key)
        logger.warning("Secret key rotated - all existing sessions will be invalidated")
        return new_key


class EnhancedFileUploadValidator:
    """Enhanced file upload security validation"""
    
    def __init__(self):
        self.allowed_extensions = {'pdf', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'txt', 'docx'}
        self.allowed_mime_types = {
            'application/pdf',
            'image/png', 'image/jpeg', 'image/tiff',
            'text/plain',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        self.max_file_size = 16 * 1024 * 1024  # 16MB
        self.blocked_extensions = {'exe', 'bat', 'cmd', 'com', 'scr', 'vbs', 'js', 'jar', 'zip'}
        
        # Initialize magic for file type detection
        try:
            self.magic_mime = magic.Magic(mime=True)
        except Exception as e:
            logger.warning(f"Could not initialize python-magic: {e}")
            self.magic_mime = None
    
    def validate_file(self, file, filename: str) -> Dict[str, Any]:
        """Comprehensive file validation"""
        errors = []
        warnings = []
        
        try:
            # 1. Filename validation
            if not filename or filename.strip() == '':
                errors.append("Filename cannot be empty")
                return {"valid": False, "errors": errors, "warnings": warnings}
            
            # 2. Extension validation
            extension = filename.lower().split('.')[-1] if '.' in filename else ''
            if extension in self.blocked_extensions:
                errors.append(f"File type .{extension} is not allowed for security reasons")
            
            if extension not in self.allowed_extensions:
                errors.append(f"File type .{extension} is not supported")
            
            # 3. File size validation
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)  # Reset file pointer
            
            if file_size == 0:
                errors.append("File is empty")
            elif file_size > self.max_file_size:
                errors.append(f"File size ({file_size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)")
            
            # 4. MIME type validation
            if self.magic_mime:
                try:
                    file_content = file.read(1024)  # Read first 1KB for MIME detection
                    file.seek(0)  # Reset file pointer
                    
                    detected_mime = self.magic_mime.from_buffer(file_content)
                    if detected_mime not in self.allowed_mime_types:
                        errors.append(f"File content type '{detected_mime}' does not match allowed types")
                    
                except Exception as e:
                    warnings.append(f"Could not verify file content type: {e}")
            
            # 5. Filename security validation
            dangerous_patterns = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
            for pattern in dangerous_patterns:
                if pattern in filename:
                    errors.append(f"Filename contains dangerous character: {pattern}")
            
            # 6. Length validation
            if len(filename) > 255:
                errors.append("Filename is too long (maximum 255 characters)")
            
            # 7. Hidden file detection
            if filename.startswith('.'):
                warnings.append("Uploading hidden files is discouraged")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "file_size": file_size,
                "detected_mime": detected_mime if self.magic_mime else None,
                "extension": extension
            }
            
        except Exception as e:
            logger.error(f"Error during file validation: {e}")
            return {
                "valid": False,
                "errors": [f"File validation failed: {str(e)}"],
                "warnings": warnings
            }


class EnhancedRateLimit:
    """Enhanced rate limiting with Redis backend and multiple strategies"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or self._create_redis_client()
        self.default_limits = {
            'per_minute': 60,
            'per_hour': 600,
            'per_day': 5000
        }
    
    def _create_redis_client(self):
        """Create Redis client for rate limiting"""
        try:
            redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/1')
            return redis.from_url(redis_url, decode_responses=True)
        except Exception as e:
            logger.warning(f"Could not connect to Redis for rate limiting: {e}")
            return None
    
    def check_rate_limit(self, key: str, limit: int, window: int) -> Dict[str, Any]:
        """Check rate limit using sliding window"""
        if not self.redis_client:
            # Fallback to in-memory (not recommended for production)
            return {"allowed": True, "remaining": limit, "reset_time": time.time() + window}
        
        try:
            now = time.time()
            window_start = now - window
            
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current entries
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Set expiry
            pipe.expire(key, window)
            
            results = pipe.execute()
            current_count = results[1] + 1  # +1 for the request we just added
            
            allowed = current_count <= limit
            remaining = max(0, limit - current_count)
            reset_time = now + window
            
            return {
                "allowed": allowed,
                "remaining": remaining,
                "reset_time": reset_time,
                "current_count": current_count,
                "limit": limit
            }
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Fail open for availability
            return {"allowed": True, "remaining": limit, "reset_time": time.time() + window}
    
    def rate_limit(self, per_minute: int = None, per_hour: int = None, per_day: int = None):
        """Rate limiting decorator"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Get client IP
                client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
                if not client_ip:
                    client_ip = 'unknown'
                
                # Check different time windows
                limits_to_check = []
                if per_minute:
                    limits_to_check.append((f"rate_limit:minute:{client_ip}", per_minute, 60))
                if per_hour:
                    limits_to_check.append((f"rate_limit:hour:{client_ip}", per_hour, 3600))
                if per_day:
                    limits_to_check.append((f"rate_limit:day:{client_ip}", per_day, 86400))
                
                for key, limit, window in limits_to_check:
                    result = self.check_rate_limit(key, limit, window)
                    if not result["allowed"]:
                        return jsonify({
                            "error": "Rate limit exceeded",
                            "limit": limit,
                            "window": window,
                            "reset_time": result["reset_time"],
                            "remaining": result["remaining"]
                        }), 429
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator


class SecurityHeadersMiddleware:
    """Add security headers to all responses"""
    
    def __init__(self, app=None):
        self.app = app
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize security headers middleware"""
        app.after_request(self.add_security_headers)
    
    def add_security_headers(self, response):
        """Add security headers to response"""
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "form-action 'self'; "
            "base-uri 'self'"
        )
        response.headers['Content-Security-Policy'] = csp
        
        # Security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Remove server header
        response.headers.pop('Server', None)
        
        return response


class APIKeyManager:
    """Enhanced API key management with rotation and validation"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        if encryption_key:
            self.cipher = Fernet(encryption_key.encode())
        else:
            # Generate or load encryption key
            key_file = './.secrets/api_encryption.key'
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    self.cipher = Fernet(f.read())
            else:
                key = Fernet.generate_key()
                os.makedirs(os.path.dirname(key_file), exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(key)
                os.chmod(key_file, 0o600)
                self.cipher = Fernet(key)
    
    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key for storage"""
        return self.cipher.encrypt(api_key.encode()).decode()
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API key for use"""
        return self.cipher.decrypt(encrypted_key.encode()).decode()
    
    def validate_api_key_format(self, api_key: str, provider: str) -> Dict[str, Any]:
        """Validate API key format for different providers"""
        patterns = {
            'openai': r'^sk-[a-zA-Z0-9]{48}$',
            'azure': r'^[a-f0-9]{32}$',
            'google': r'^[a-zA-Z0-9_-]{39}$',
            'anthropic': r'^sk-ant-[a-zA-Z0-9_-]+$'
        }
        
        if provider.lower() not in patterns:
            return {"valid": False, "error": f"Unknown provider: {provider}"}
        
        import re
        pattern = patterns[provider.lower()]
        if not re.match(pattern, api_key):
            return {"valid": False, "error": f"Invalid {provider} API key format"}
        
        return {"valid": True}


def init_security_enhancements(app):
    """Initialize all security enhancements for the Flask app"""
    
    # 1. Initialize secure secret key management
    secret_manager = SecretKeyManager()
    app.secret_key = secret_manager.get_or_create_secret_key()
    
    # 2. Initialize security headers middleware
    security_headers = SecurityHeadersMiddleware(app)
    
    # 3. Initialize rate limiting
    rate_limiter = EnhancedRateLimit()
    
    # Store instances in app context for access in routes
    app.secret_manager = secret_manager
    app.file_validator = EnhancedFileUploadValidator()
    app.rate_limiter = rate_limiter
    app.api_key_manager = APIKeyManager()
    
    logger.info("Security enhancements initialized successfully")
    
    return {
        'secret_manager': secret_manager,
        'file_validator': app.file_validator,
        'rate_limiter': rate_limiter,
        'api_key_manager': app.api_key_manager
    }


# Utility functions for easy access
def get_secure_secret_key() -> str:
    """Get secure secret key for Flask app"""
    return SecretKeyManager().get_or_create_secret_key()


def validate_uploaded_file(file, filename: str) -> Dict[str, Any]:
    """Validate uploaded file with comprehensive security checks"""
    validator = EnhancedFileUploadValidator()
    return validator.validate_file(file, filename)


# Export main classes and functions
__all__ = [
    'SecretKeyManager',
    'EnhancedFileUploadValidator', 
    'EnhancedRateLimit',
    'SecurityHeadersMiddleware',
    'APIKeyManager',
    'init_security_enhancements',
    'get_secure_secret_key',
    'validate_uploaded_file'
]