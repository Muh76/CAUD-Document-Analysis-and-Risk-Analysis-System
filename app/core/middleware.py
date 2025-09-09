"""
FastAPI middleware for security, logging, and metrics.
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .logging import request_logger, metrics
from .security import SecurityManager

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for rate limiting and validation."""

    def __init__(self, app, security_manager: SecurityManager = None):
        super().__init__(app)
        self.security_manager = security_manager or SecurityManager()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security middleware."""
        client_ip = request.client.host
        token = request.headers.get("Authorization", "").replace("Bearer ", "")

        # Check security
        security_check = self.security_manager.check_security(client_ip, token)

        if not security_check["is_secure"]:
            return JSONResponse(
                status_code=401 if "Invalid API token" in security_check["issues"] else 429,
                content={
                    "error": "Security check failed",
                    "issues": security_check["issues"],
                    "rate_limit_remaining": security_check["rate_limit_remaining"]
                }
            )

        # Add security info to request state
        request.state.security_info = security_check

        # Continue to next middleware
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Remaining"] = str(security_check["rate_limit_remaining"])

        return response

class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging middleware for request/response logging."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through logging middleware."""
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Log request
        request_logger.log_request(
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            client_ip=request.client.host,
            user_agent=request.headers.get("User-Agent", ""),
            query_params=dict(request.query_params)
        )

        # Add request ID to request state
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Log response
        request_logger.log_response(
            request_id=request_id,
            status_code=response.status_code,
            latency_ms=latency_ms
        )

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{latency_ms}ms"

        return response

class MetricsMiddleware(BaseHTTPMiddleware):
    """Metrics middleware for collecting API metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through metrics middleware."""
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Record metrics
        metrics.increment_counter("api_requests_total", labels={
            "method": request.method,
            "path": str(request.url.path),
            "status_code": str(response.status_code)
        })

        metrics.record_histogram("api_request_duration_ms", latency_ms, labels={
            "method": request.method,
            "path": str(request.url.path)
        })

        return response

class CORSMiddleware(BaseHTTPMiddleware):
    """CORS middleware for UI access."""

    def __init__(self, app, allowed_origins: list = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or [
            "http://localhost:8501",
            "http://127.0.0.1:8501"
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through CORS middleware."""
        origin = request.headers.get("Origin")

        if origin in self.allowed_origins:
            response = await call_next(request)
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return response

        return await call_next(request)
