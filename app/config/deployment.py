"""
Deployment configuration for Contract Analysis System.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from app.config.settings import get_settings

class DeploymentConfig:
    """Deployment configuration management."""

    def __init__(self):
        self.settings = get_settings()

    def get_gcp_config(self) -> Dict[str, Any]:
        """Get Google Cloud Platform deployment configuration."""
        return {
            "project_id": self.settings.gcp_project_id or "your-project-id",
            "region": self.settings.gcp_region,
            "service_name": self.settings.gcp_service_name or "contract-analysis-api",
            "image": f"gcr.io/{self.settings.gcp_project_id}/contract-analysis-api",
            "memory": "2Gi",
            "cpu": "2",
            "min_instances": 0,
            "max_instances": 10,
            "port": 8000,
            "environment_vars": {
                "ENVIRONMENT": "production",
                "LOG_LEVEL": "INFO",
                "PROMETHEUS_ENABLED": "true",
                "API_HOST": "0.0.0.0",
                "API_PORT": "8000"
            }
        }

    def get_azure_config(self) -> Dict[str, Any]:
        """Get Azure Container Apps deployment configuration."""
        return {
            "resource_group": self.settings.azure_resource_group or "contract-analysis-rg",
            "location": self.settings.azure_location or "East US",
            "container_app_name": self.settings.azure_container_app_name or "contract-analysis",
            "registry_name": f"{self.settings.azure_container_app_name}registry",
            "image": f"{self.settings.azure_container_app_name}registry.azurecr.io/contract-analysis-api:latest",
            "memory": "4.0Gi",
            "cpu": "2.0",
            "min_replicas": 0,
            "max_replicas": 10,
            "port": 8000,
            "environment_vars": {
                "ENVIRONMENT": "production",
                "LOG_LEVEL": "INFO",
                "PROMETHEUS_ENABLED": "true"
            }
        }

    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get Streamlit Share deployment configuration."""
        return {
            "app_file": "app/ui/app.py",
            "requirements_file": "requirements.txt",
            "secrets": {
                "API_URL": self.settings.api_url or "https://your-api-url.com",
                "API_TOKEN": self.settings.api_token
            },
            "environment_vars": {
                "ENVIRONMENT": "production",
                "LOG_LEVEL": "INFO"
            }
        }

    def validate_deployment_config(self, platform: str) -> bool:
        """Validate deployment configuration for a platform."""
        if platform == "gcp":
            config = self.get_gcp_config()
            return bool(config["project_id"] and config["project_id"] != "your-project-id")
        elif platform == "azure":
            config = self.get_azure_config()
            return bool(config["resource_group"] and config["container_app_name"])
        elif platform == "streamlit":
            config = self.get_streamlit_config()
            return bool(config["secrets"]["API_URL"] and config["secrets"]["API_TOKEN"])
        return False
