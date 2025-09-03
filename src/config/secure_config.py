"""
Secure Configuration Management
All sensitive data is loaded from environment variables
"""

import os
from typing import Dict, List, Any
import hashlib
import base64

# Project Configuration
PROJECT_NAME = os.getenv("PROJECT_NAME", "Contract Review & Risk Analysis System")
VERSION = os.getenv("VERSION", "1.0.0")
AUTHOR = os.getenv("AUTHOR", "Mohammad Babaie")
EMAIL = os.getenv("EMAIL", "mj.babaie@gmail.com")
GITHUB_URL = os.getenv("GITHUB_URL", "https://github.com/Muh76")
LINKEDIN_URL = os.getenv("LINKEDIN_URL", "https://www.linkedin.com/in/mohammadbabaie/")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "500"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_STORAGE_BUCKET = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./legalai.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "legal-contract-analysis")

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# File Processing Configuration
ALLOWED_EXTENSIONS = ["pdf", "txt", "docx"]
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB

# Risk Scoring Configuration
RISK_WEIGHTS = {
    "uncapped_liability": 0.25,
    "non_compete": 0.20,
    "ip_assignment": 0.15,
    "termination_convenience": 0.10,
    "audit_rights": 0.05,
    "confidentiality": 0.10,
    "governing_law": 0.05,
    "dispute_resolution": 0.10,
}

# CUAD Dataset Categories (simplified for readability)
CUAD_CATEGORIES = [
    "Document Type",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Contract Value",
    "Payment Terms",
    "Notices",
    "Termination for Convenience",
    "Termination for Cause",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Non-Solicitation",
    "Change of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Audit Rights",
    "Insurance",
    "Subcontracting",
    "Minimum Commitment",
]

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv(
    "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOG_FILE = os.getenv("LOG_FILE", "logs/legalai.log")

# Monitoring Configuration
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))
GRAFANA_PORT = int(os.getenv("GRAFANA_PORT", "3000"))

# Development Configuration
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
TESTING = os.getenv("TESTING", "false").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data for logging purposes"""
    if not data:
        return "NOT_SET"
    return hashlib.sha256(data.encode()).hexdigest()[:8] + "..."


def get_config_summary() -> Dict[str, Any]:
    """Get a safe configuration summary for logging"""
    return {
        "project_name": PROJECT_NAME,
        "version": VERSION,
        "author": AUTHOR,
        "email": EMAIL,
        "api_host": API_HOST,
        "api_port": API_PORT,
        "openai_model": OPENAI_MODEL,
        "openai_key_set": bool(OPENAI_API_KEY),
        "openai_key_hash": hash_sensitive_data(OPENAI_API_KEY),
        "azure_endpoint_set": bool(AZURE_OPENAI_ENDPOINT),
        "azure_key_set": bool(AZURE_OPENAI_API_KEY),
        "azure_key_hash": hash_sensitive_data(AZURE_OPENAI_API_KEY),
        "gcp_project_set": bool(GOOGLE_CLOUD_PROJECT),
        "gcp_bucket_set": bool(GOOGLE_CLOUD_STORAGE_BUCKET),
        "database_url_set": bool(DATABASE_URL),
        "redis_url_set": bool(REDIS_URL),
        "environment": ENVIRONMENT,
        "debug": DEBUG,
        "testing": TESTING,
    }


def validate_configuration() -> List[str]:
    """Validate that required configuration is present"""
    errors = []

    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required")

    if not SECRET_KEY or SECRET_KEY == "your-secret-key-change-this-in-production":
        errors.append("SECRET_KEY should be set to a secure value")

    return errors


# Security utilities
def mask_api_key(api_key: str) -> str:
    """Mask API key for display purposes"""
    if not api_key:
        return "NOT_SET"
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]


def is_production() -> bool:
    """Check if running in production environment"""
    return ENVIRONMENT.lower() == "production"


def is_development() -> bool:
    """Check if running in development environment"""
    return ENVIRONMENT.lower() == "development"
