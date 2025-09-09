"""
FastAPI application package for Contract Review & Risk Analysis System.
"""

from .main import app
from .schemas import *
from .deps import *

__version__ = "1.0.0"
__all__ = ["app"]
