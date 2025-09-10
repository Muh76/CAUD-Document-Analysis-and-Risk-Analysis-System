"""
Basic tests for the Contract Analysis System app package.
"""

import pytest
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

def test_app_import():
    """Test that the app package can be imported."""
    try:
        import app
        assert hasattr(app, '__version__')
        assert app.__version__ == "1.0.0"
    except ImportError as e:
        pytest.fail(f"Failed to import app package: {e}")

def test_config_import():
    """Test that config module can be imported."""
    try:
        from app.config import Settings
        assert Settings is not None
    except ImportError as e:
        pytest.fail(f"Failed to import Settings: {e}")

def test_settings_creation():
    """Test that Settings can be instantiated."""
    try:
        from app.config import Settings
        settings = Settings()
        assert settings.app_name == "Contract Analysis System"
        assert settings.app_version == "1.0.0"
    except Exception as e:
        pytest.fail(f"Failed to create Settings instance: {e}")

def test_api_schemas():
    """Test that API schemas can be imported."""
    try:
        from app.api import schemas
        assert schemas is not None
    except ImportError as e:
        pytest.fail(f"Failed to import API schemas: {e}")

def test_core_pipeline():
    """Test that core pipeline can be imported."""
    try:
        from app.core import pipeline
        assert pipeline is not None
    except ImportError as e:
        pytest.fail(f"Failed to import core pipeline: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
