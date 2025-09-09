"""
Core inference package for Contract Review & Risk Analysis System.
"""

from .settings import Settings
from .pipeline import ContractAnalyzer
from .schemas import ContractAnalysis, ClauseResult, TextChunk, ModelPrediction
from .text_ingest import TextIngestion
from .io import IOUtils
from .pdf_processor import PDFProcessor
from .clause_chunker import ClauseChunker

__version__ = "1.0.0"
__all__ = [
    "Settings",
    "ContractAnalyzer", 
    "ContractAnalysis",
    "ClauseResult",
    "TextChunk",
    "ModelPrediction",
    "TextIngestion",
    "IOUtils",
    "PDFProcessor",
    "ClauseChunker"
]
