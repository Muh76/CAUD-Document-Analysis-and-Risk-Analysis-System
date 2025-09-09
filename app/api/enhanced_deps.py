"""
Enhanced API dependencies with JWT authentication, Prometheus metrics, and structured logging.
"""

import time
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from functools import wraps

import jwt
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import Counter, Histogram, Gauge
import structlog

from app.config.settings import get_settings

# Initialize structured logging
logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('http_active_connections', 'Active HTTP connections')
AUTH_FAILURES = Counter('auth_failures_total', 'Authentication failures', ['reason'])

# JWT Security
security = HTTPBearer(auto_error=False)

# Settings
settings = get_settings()


class JWTAuth:
    """JWT Authentication handler."""

    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=settings.jwt_expiration_hours)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
        return encoded_jwt

    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            AUTH_FAILURES.labels(reason='token_expired').inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            AUTH_FAILURES.labels(reason='invalid_token').inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    """Get current authenticated user."""
    if not credentials:
        AUTH_FAILURES.labels(reason='no_credentials').inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    # Check API token first (simple auth)
    if credentials.credentials == settings.api_token:
        return {"user_id": "api_user", "role": "api", "auth_type": "token"}

    # Check JWT token
    try:
        payload = JWTAuth.verify_token(credentials.credentials)
        return payload
    except HTTPException:
        raise


def require_auth(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Require authentication for endpoint."""
    return user


def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Require admin role for endpoint."""
    if user.get("role") != "admin":
        AUTH_FAILURES.labels(reason='insufficient_privileges').inc()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user


def metrics_middleware(func):
    """Decorator to add metrics to endpoints."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        request = None

        # Find request object in args/kwargs
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break

        if not request:
            for value in kwargs.values():
                if isinstance(value, Request):
                    request = value
                    break

        try:
            result = await func(*args, **kwargs)

            # Record metrics
            if request:
                duration = time.time() - start_time
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=200
                ).inc()
                REQUEST_DURATION.observe(duration)

                logger.info(
                    "Request completed",
                    method=request.method,
                    endpoint=request.url.path,
                    duration_ms=int(duration * 1000),
                    status=200
                )

            return result

        except Exception as e:
            if request:
                duration = time.time() - start_time
                status_code = getattr(e, 'status_code', 500)

                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=status_code
                ).inc()
                REQUEST_DURATION.observe(duration)

                logger.error(
                    "Request failed",
                    method=request.method,
                    endpoint=request.url.path,
                    duration_ms=int(duration * 1000),
                    status=status_code,
                    error=str(e)
                )

            raise

    return wrapper
