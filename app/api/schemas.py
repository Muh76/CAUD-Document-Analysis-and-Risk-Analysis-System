"""
Pydantic schemas for FastAPI request/response models.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ClauseResult(BaseModel):
    """Result of analyzing a single clause."""
    clause_id: int
    probs: List[float] = Field(..., description="49-length probs aligned to label_map")
    risk: float
    start: Optional[int] = None
    end: Optional[int] = None
    snippet: Optional[str] = None
    rationale: Optional[List[str]] = None
    detected_labels: Optional[List[str]] = None

class AnalyzeResponse(BaseModel):
    """Response for contract analysis."""
    contract_id: str
    label_map: List[str]
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

class AnalyzeRequest(BaseModel):
    """Request for contract analysis."""
    contract_id: str
    text: Optional[str] = None
    file_b64: Optional[str] = None  # PDF/text; server extracts
    mime: Optional[str] = "application/pdf"
    token: Optional[str] = None

class BatchAnalyzeRequest(BaseModel):
    """Request for batch analysis."""
    contracts: List[AnalyzeRequest]
    token: Optional[str] = None

class BatchAnalyzeResponse(BaseModel):
    """Response for batch analysis."""
    job_id: str
    total_contracts: int
    status: str
    created_at: datetime

class RiskReportRequest(BaseModel):
    """Request for risk report generation."""
    contract_ids: List[str]
    include_suggestions: bool = False
    token: Optional[str] = None

class RiskReportResponse(BaseModel):
    """Response for risk report."""
    report_id: str
    total_contracts: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    top_red_flags: List[Dict[str, Any]]
    missing_clauses: List[str]
    recommendations: Optional[List[Dict[str, Any]]] = None
    generated_at: datetime

class ExportRequest(BaseModel):
    """Request for data export."""
    contract_id: str
    format: str = Field(..., pattern="^(csv|json)$")
    token: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_snapshot: str
    calibration_version: str
    uptime_seconds: float
    timestamp: datetime

class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
