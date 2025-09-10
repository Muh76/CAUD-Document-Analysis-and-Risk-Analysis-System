"""
Application configuration using Pydantic Settings.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = Field(default="Contract Analysis System", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    api_reload: bool = Field(default=False, env="API_RELOAD")

    # Security
    api_token: str = Field(default="devtoken", env="API_TOKEN")
    jwt_secret_key: str = Field(default="your-secret-key-change-in-production", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")

    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # seconds

    # File Upload Limits
    max_file_size_mb: int = Field(default=10, env="MAX_FILE_SIZE_MB")
    max_pages_per_request: int = Field(default=50, env="MAX_PAGES_PER_REQUEST")
    allowed_mime_types: List[str] = Field(
        default=["application/pdf", "text/plain", "application/msword", 
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
        env="ALLOWED_MIME_TYPES"
    )

    # Model Configuration
    model_snapshot: str = Field(default="snapshot_20250909", env="MODEL_SNAPSHOT")
    artifacts_dir: str = Field(default="app/artifacts", env="ARTIFACTS_DIR")
    model_cache_size: int = Field(default=100, env="MODEL_CACHE_SIZE")

    # RAG Configuration
    rag_collection: str = Field(default="cuad-safe-clauses", env="RAG_COLLECTION")
    rag_similarity_threshold: float = Field(default=0.7, env="RAG_SIMILARITY_THRESHOLD")
    rag_top_k: int = Field(default=5, env="RAG_TOP_K")
    rag_index_dir: str = Field(default="app/var/indices", env="RAG_INDEX_DIR")

    # Database Configuration (for production)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")

    # Monitoring
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    opentelemetry_enabled: bool = Field(default=False, env="OPENTELEMETRY_ENABLED")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:8501", "http://127.0.0.1:8501"],
        env="CORS_ORIGINS"
    )

    # Cloud Deployment (Azure Container Apps)
    azure_container_app_name: Optional[str] = Field(default=None, env="AZURE_CONTAINER_APP_NAME")
    azure_resource_group: Optional[str] = Field(default=None, env="AZURE_RESOURCE_GROUP")

    # Cloud Deployment (Google Cloud Run)
    gcp_project_id: Optional[str] = Field(default=None, env="GCP_PROJECT_ID")
    gcp_service_name: Optional[str] = Field(default=None, env="GCP_SERVICE_NAME")
    gcp_region: Optional[str] = Field(default="us-central1", env="GCP_REGION")

    # Streamlit Share
    streamlit_share_url: Optional[str] = Field(default=None, env="STREAMLIT_SHARE_URL")

    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

    @field_validator('allowed_mime_types', mode='before')
    @classmethod
    def parse_mime_types(cls, v):
        if isinstance(v, str):
            return [mime.strip() for mime in v.split(',')]
        return v

    @property
    def artifacts_path(self) -> Path:
        """Get artifacts directory path."""
        return Path(self.artifacts_dir)

    @property
    def rag_index_path(self) -> Path:
        """Get RAG index directory path."""
        return Path(self.rag_index_dir)

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment.lower() == "development"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
