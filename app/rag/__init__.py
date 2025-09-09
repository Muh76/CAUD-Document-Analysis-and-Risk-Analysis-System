"""
RAG (Retrieval-Augmented Generation) package for Contract Review & Risk Analysis System.
"""

from .build_index import RAGIndexBuilder
from .retrieval import RAGRetrieval

__version__ = "1.0.0"
__all__ = ["RAGIndexBuilder", "RAGRetrieval"]
