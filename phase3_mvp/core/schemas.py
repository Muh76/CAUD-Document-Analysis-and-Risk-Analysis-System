"""
Internal dataclasses and schemas for the Contract Analysis Pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class ClauseResult:
    """Result of analyzing a single clause."""
    clause_id: int
    text: str
    probs: List[float]  # 49 probabilities aligned to label_map
    risk_score: float
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    page_number: Optional[int] = None
    rationale: Optional[List[str]] = None
    detected_labels: Optional[List[str]] = None

@dataclass
class ContractAnalysis:
    """Complete analysis result for a contract."""
    contract_id: str
    results: List[ClauseResult]
    total_clauses: int
    high_risk_clauses: int
    medium_risk_clauses: int
    low_risk_clauses: int
    overall_risk_score: float
    thresholds_used: Dict[str, Any]
    model_snapshot: str
    calibration_version: str
    latency_ms: int
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    text: str
    start_offset: int
    end_offset: int
    page_number: Optional[int] = None
    chunk_id: Optional[int] = None

@dataclass
class ModelPrediction:
    """Raw model prediction result."""
    probabilities: List[float]
    predicted_labels: List[str]
    confidence_scores: List[float]
    model_name: str
    inference_time_ms: float
