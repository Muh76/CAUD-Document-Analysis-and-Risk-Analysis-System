

# Add these endpoints to your existing app/api/main.py file:

# Authentication endpoints
@app.post("/auth/login")
async def login(username: str, password: str):
    """Login endpoint for JWT authentication."""
    # In production, verify against database
    if username == "admin" and password == "admin123":
        token = JWTAuth.create_access_token({"user_id": username, "role": "admin"})
        logger.info("User logged in", user_id=username, role="admin")
        return {"access_token": token, "token_type": "bearer"}

    AUTH_FAILURES.labels(reason='invalid_credentials').inc()
    logger.warning("Login failed", username=username)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )


# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Admin endpoints
@app.get("/admin/status")
async def admin_status(user: Dict[str, Any] = Depends(require_admin)):
    """Get system status (admin only)."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {
            "total_requests": REQUEST_COUNT._value.sum(),
            "active_connections": ACTIVE_CONNECTIONS._value,
            "auth_failures": AUTH_FAILURES._value.sum()
        }
    }


@app.post("/admin/rebuild-index")
async def rebuild_rag_index(user: Dict[str, Any] = Depends(require_admin)):
    """Rebuild RAG index (admin only)."""
    logger.info("RAG index rebuild initiated", admin_user=user.get("user_id"))
    # This would trigger the RAG index rebuild
    return {"message": "RAG index rebuild initiated", "status": "success"}


@app.get("/admin/logs")
async def get_logs(user: Dict[str, Any] = Depends(require_admin)):
    """Get recent logs (admin only)."""
    # In production, this would read from log files
    return {
        "message": "Log retrieval endpoint",
        "note": "In production, this would return actual log entries"
    }

