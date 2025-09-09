"""
Security middleware and utilities for the Contract Analysis API.
"""

import os
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import hmac
import secrets

class RateLimiter:
    """Simple rate limiter using sliding window."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        client_requests = self.requests[client_ip]

        # Remove old requests outside window
        while client_requests and client_requests[0] <= now - self.window_seconds:
            client_requests.popleft()

        # Check if under limit
        if len(client_requests) >= self.max_requests:
            return False

        # Add current request
        client_requests.append(now)
        return True

    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for client."""
        now = time.time()
        client_requests = self.requests[client_ip]

        # Remove old requests
        while client_requests and client_requests[0] <= now - self.window_seconds:
            client_requests.popleft()

        return max(0, self.max_requests - len(client_requests))

class FileValidator:
    """Validate uploaded files."""

    def __init__(self, max_size_mb: int = 10, allowed_mime_types: List[str] = None):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.allowed_mime_types = allowed_mime_types or [
            "application/pdf",
            "text/plain",
            "text/csv"
        ]

    def validate_size(self, file_size: int) -> bool:
        """Validate file size."""
        return file_size <= self.max_size_bytes

    def validate_mime_type(self, mime_type: str) -> bool:
        """Validate MIME type."""
        return mime_type in self.allowed_mime_types

    def validate_file(self, file_size: int, mime_type: str) -> Dict[str, Any]:
        """Validate file completely."""
        issues = []

        if not self.validate_size(file_size):
            issues.append(f"File too large: {file_size / 1024 / 1024:.1f}MB > {self.max_size_bytes / 1024 / 1024}MB")

        if not self.validate_mime_type(mime_type):
            issues.append(f"Unsupported file type: {mime_type}")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "file_size_mb": file_size / 1024 / 1024,
            "mime_type": mime_type
        }

class TokenValidator:
    """Validate API tokens."""

    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.getenv("SECRET_KEY", "dev-secret-key")
        self.valid_tokens = {
            "devtoken": "development",
            "prodtoken": "production"
        }

    def validate_token(self, token: str) -> bool:
        """Validate API token."""
        return token in self.valid_tokens

    def get_token_info(self, token: str) -> Optional[Dict[str, str]]:
        """Get token information."""
        if token in self.valid_tokens:
            return {
                "token": token,
                "environment": self.valid_tokens[token],
                "valid": True
            }
        return None

    def generate_token(self, environment: str = "development") -> str:
        """Generate a new token."""
        token = secrets.token_urlsafe(32)
        self.valid_tokens[token] = environment
        return token

class SecurityManager:
    """Main security manager."""

    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.file_validator = FileValidator()
        self.token_validator = TokenValidator()

    def check_security(self, client_ip: str, token: str, 
                      file_size: int = None, mime_type: str = None) -> Dict[str, Any]:
        """Perform comprehensive security check."""
        issues = []

        # Rate limiting
        if not self.rate_limiter.is_allowed(client_ip):
            issues.append("Rate limit exceeded")

        # Token validation
        if not self.token_validator.validate_token(token):
            issues.append("Invalid API token")

        # File validation (if provided)
        if file_size is not None and mime_type is not None:
            file_validation = self.file_validator.validate_file(file_size, mime_type)
            if not file_validation["is_valid"]:
                issues.extend(file_validation["issues"])

        return {
            "is_secure": len(issues) == 0,
            "issues": issues,
            "rate_limit_remaining": self.rate_limiter.get_remaining(client_ip),
            "token_info": self.token_validator.get_token_info(token)
        }
