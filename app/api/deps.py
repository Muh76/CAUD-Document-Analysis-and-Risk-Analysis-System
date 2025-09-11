"""
FastAPI dependencies for authentication, settings, and analyzer singleton.
"""

import os
import time
from typing import Optional
from fastapi import HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.settings import Settings
from app.core.pipeline import ContractAnalyzer

# Global instances
settings = None
analyzer = None
start_time = time.time()

def get_settings() -> Settings:
    """Get application settings."""
    global settings
    if settings is None:
        artifacts_dir = os.getenv("ARTIFACTS_DIR", "phase3_mvp/artifacts/snapshot_20250909")
        settings = Settings(artifacts_dir)
    return settings

def get_analyzer() -> ContractAnalyzer:
    """Get contract analyzer singleton."""
    global analyzer
    if analyzer is None:
        artifacts_dir = os.getenv("ARTIFACTS_DIR", "phase3_mvp/artifacts/snapshot_20250909")
        analyzer = ContractAnalyzer(artifacts_dir)
    return analyzer

def get_uptime() -> float:
    """Get application uptime in seconds."""
    return time.time() - start_time

# Authentication
security = HTTPBearer(auto_error=False)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    """Verify API token."""
    expected_token = os.getenv("API_TOKEN", "devtoken")

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header"
        )

    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid API token"
        )

    return credentials.credentials

def verify_token_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """Verify API token (optional)."""
    expected_token = os.getenv("API_TOKEN", "devtoken")

    if not credentials:
        return None

    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid API token"
        )

    return credentials.credentials

# Rate limiting (simple in-memory implementation)
request_counts = {}
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

def check_rate_limit(client_ip: str = Header(None)):
    """Check rate limit for client."""
    current_time = time.time()
    minute_window = int(current_time // 60)

    # Clean old entries
    for key in list(request_counts.keys()):
        if key[1] < minute_window - 1:
            del request_counts[key]

    # Check current rate
    key = (client_ip or "unknown", minute_window)
    current_count = request_counts.get(key, 0)

    if current_count >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_PER_MINUTE} requests per minute."
        )

    # Increment counter
    request_counts[key] = current_count + 1
