"""
Security configuration for Contract Analysis System.
"""

import os
from typing import Dict, List, Any

class SecurityConfig:
    """Security configuration management."""

    def __init__(self):
        # Mock settings for demo - in production, these would come from environment
        self.settings = {
            'cors_origins': ["http://localhost:8501", "http://127.0.0.1:8501"],
            'rate_limit_requests': 100,
            'rate_limit_window': 3600,
            'max_file_size_mb': 10,
            'allowed_mime_types': ["application/pdf", "text/plain"],
            'max_pages_per_request': 50,
            'jwt_secret_key': 'your-secret-key-change-in-production',
            'jwt_algorithm': 'HS256',
            'jwt_expiration_hours': 24,
            'api_token': 'devtoken',
            'log_level': 'INFO',
            'log_format': 'json',
            'log_file': None,
            'prometheus_enabled': True,
            'prometheus_port': 9090,
            'opentelemetry_enabled': False,
            'jaeger_endpoint': None,
            'debug': False,
            'environment': 'development'
        }

    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses."""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }

    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration."""
        return {
            'allow_origins': self.settings['cors_origins'],
            'allow_credentials': True,
            'allow_methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            'allow_headers': ['*'],
            'expose_headers': ['X-Total-Count', 'X-Request-ID']
        }

    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return {
            'requests_per_minute': self.settings['rate_limit_requests'],
            'window_minutes': self.settings['rate_limit_window'] // 60,
            'burst_size': 10,
            'enabled': True
        }

    def get_file_upload_config(self) -> Dict[str, Any]:
        """Get file upload security configuration."""
        return {
            'max_file_size_mb': self.settings['max_file_size_mb'],
            'allowed_mime_types': self.settings['allowed_mime_types'],
            'max_pages_per_request': self.settings['max_pages_per_request'],
            'scan_uploads': True,
            'quarantine_suspicious': True
        }

    def get_authentication_config(self) -> Dict[str, Any]:
        """Get authentication configuration."""
        return {
            'jwt_secret_key': self.settings['jwt_secret_key'],
            'jwt_algorithm': self.settings['jwt_algorithm'],
            'jwt_expiration_hours': self.settings['jwt_expiration_hours'],
            'api_token': self.settings['api_token'],
            'require_authentication': True,
            'session_timeout_minutes': 30
        }

    def get_logging_config(self) -> Dict[str, Any]:
        """Get security logging configuration."""
        return {
            'log_level': self.settings['log_level'],
            'log_format': self.settings['log_format'],
            'log_file': self.settings['log_file'],
            'log_sensitive_data': False,
            'audit_logging': True,
            'log_retention_days': 90
        }

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get security monitoring configuration."""
        return {
            'prometheus_enabled': self.settings['prometheus_enabled'],
            'prometheus_port': self.settings['prometheus_port'],
            'opentelemetry_enabled': self.settings['opentelemetry_enabled'],
            'jaeger_endpoint': self.settings['jaeger_endpoint'],
            'alert_on_failures': True,
            'alert_threshold': 5  # failures per minute
        }

    def validate_security_config(self) -> Dict[str, Any]:
        """Validate security configuration."""
        issues = []

        # Check for weak secrets
        if len(self.settings['jwt_secret_key']) < 32:
            issues.append('JWT secret key is too short (minimum 32 characters)')

        if self.settings['api_token'] == 'devtoken':
            issues.append('Using default API token - change in production')

        # Check for insecure settings
        if self.settings['debug'] and self.settings['environment'] == 'production':
            issues.append('Debug mode enabled in production')

        # Check for missing security features
        if not self.settings['prometheus_enabled']:
            issues.append('Prometheus monitoring disabled')

        if not self.settings['cors_origins']:
            issues.append('CORS origins not configured')

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'recommendations': self._get_security_recommendations()
        }

    def _get_security_recommendations(self) -> List[str]:
        """Get security recommendations."""
        recommendations = [
            'Enable HTTPS/TLS encryption',
            'Implement regular security scans',
            'Use strong, unique passwords',
            'Enable two-factor authentication',
            'Regular security updates',
            'Implement data backup and recovery',
            'Monitor for suspicious activity',
            'Regular security training for team'
        ]

        return recommendations
