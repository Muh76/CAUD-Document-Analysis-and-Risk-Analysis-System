"""
Contract Review & Risk Analysis System - Production Package
"""

__version__ = "1.0.0"
__author__ = "Contract Analysis Team"
__description__ = "AI-powered contract review and risk analysis system"

from .config.settings import Settings
from .core.pipeline import ContractAnalyzer

__all__ = ["Settings", "ContractAnalyzer"]
