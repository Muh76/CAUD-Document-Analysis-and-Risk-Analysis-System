"""
FastAPI main application with all endpoints.
"""

import os
import base64
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import json

from .schemas import (
    AnalyzeRequest, AnalyzeResponse, BatchAnalyzeRequest, BatchAnalyzeResponse,
    RiskReportRequest, RiskReportResponse, ExportRequest, HealthResponse, ErrorResponse
)
from .deps import (
    get_settings, get_analyzer, get_uptime, verify_token, verify_token_optional, check_rate_limit
)
from core.text_ingest import TextIngestion
from core.logging import logger, request_logger, metrics
from core.security import SecurityManager
from core.middleware import SecurityMiddleware, LoggingMiddleware, MetricsMiddleware, CORSMiddleware

from core.export import ExportManager
from core.io import IOUtils

# Create FastAPI app
app = FastAPI(
    title="Contract Review & Risk Analysis API",
    description="API for analyzing contracts and assessing risk",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware after app creation
security_manager = SecurityManager()
app.add_middleware(SecurityMiddleware, security_manager=security_manager)
app.add_middleware(LoggingMiddleware)
app.add_middleware(MetricsMiddleware)

# CORS middleware
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for batch jobs (in production, use Redis or database)
batch_jobs = {}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    analyzer = get_analyzer()

    return HealthResponse(
        status="ok",
        model_snapshot=settings.model_snapshot,
        calibration_version="v1",
        uptime_seconds=get_uptime(),
        timestamp=datetime.now()
    )

@app.post("/analyze_contract", response_model=AnalyzeResponse)
async def analyze_contract(
    request: AnalyzeRequest,
    token: str = Depends(verify_token),
    client_ip: str = Depends(check_rate_limit)
):
    """Analyze a single contract."""
    try:
        analyzer = get_analyzer()
        settings = get_settings()

        # Extract text from request
        if request.text:
            clauses = [request.text]
        elif request.file_b64:
            # Decode base64 file
            try:
                file_content = base64.b64decode(request.file_b64).decode('utf-8')
                # For now, treat as single clause - in production, use TextIngestion
                clauses = [file_content]
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid file content: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Either text or file_b64 must be provided")

        # Analyze contract
        analysis = analyzer.analyze(request.contract_id, clauses)

        # Convert to response format
        clause_results = []
        for result in analysis.results:
            clause_results.append({
                "clause_id": result.clause_id,
                "probs": result.probs,
                "risk": result.risk_score,
                "start": result.start_offset,
                "end": result.end_offset,
                "snippet": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                "rationale": result.rationale,
                "detected_labels": result.detected_labels
            })

        return AnalyzeResponse(
            contract_id=analysis.contract_id,
            label_map=list(settings.label_map.values()),
            results=clause_results,
            total_clauses=analysis.total_clauses,
            high_risk_clauses=analysis.high_risk_clauses,
            medium_risk_clauses=analysis.medium_risk_clauses,
            low_risk_clauses=analysis.low_risk_clauses,
            overall_risk_score=analysis.overall_risk_score,
            thresholds_used=analysis.thresholds_used,
            model_snapshot=analysis.model_snapshot,
            calibration_version=analysis.calibration_version,
            latency_ms=analysis.latency_ms,
            timestamp=analysis.timestamp
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/batch_analyze", response_model=BatchAnalyzeResponse)
async def batch_analyze(
    request: BatchAnalyzeRequest,
    token: str = Depends(verify_token),
    client_ip: str = Depends(check_rate_limit)
):
    """Start batch analysis of multiple contracts."""
    try:
        job_id = str(uuid.uuid4())

        # Store job info
        batch_jobs[job_id] = {
            "status": "queued",
            "total_contracts": len(request.contracts),
            "completed": 0,
            "results": [],
            "created_at": datetime.now(),
            "contracts": request.contracts
        }

        return BatchAnalyzeResponse(
            job_id=job_id,
            total_contracts=len(request.contracts),
            status="queued",
            created_at=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.get("/batch_analyze/{job_id}")
async def get_batch_status(job_id: str, token: str = Depends(verify_token)):
    """Get batch analysis status."""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = batch_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "total_contracts": job["total_contracts"],
        "completed": job["completed"],
        "created_at": job["created_at"]
    }

@app.post("/risk_report", response_model=RiskReportResponse)
async def generate_risk_report(
    request: RiskReportRequest,
    token: str = Depends(verify_token),
    client_ip: str = Depends(check_rate_limit)
):
    """Generate risk report for multiple contracts."""
    try:
        analyzer = get_analyzer()
        settings = get_settings()

        # Placeholder implementation - in production, analyze all contracts
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0

        # Mock analysis for demo
        for contract_id in request.contract_ids:
            # Simulate analysis
            mock_clauses = [f"Sample clause from {contract_id}"]
            analysis = analyzer.analyze(contract_id, mock_clauses)

            if analysis.overall_risk_score >= settings.get_global_threshold("HIGH_RISK_THRESHOLD"):
                high_risk_count += 1
            elif analysis.overall_risk_score >= settings.get_global_threshold("MEDIUM_RISK_THRESHOLD"):
                medium_risk_count += 1
            else:
                low_risk_count += 1

        # Generate mock red flags and missing clauses
        top_red_flags = [
            {"label": "Cap on Liability", "count": 15, "risk_level": "high"},
            {"label": "Termination for Convenience", "count": 12, "risk_level": "medium"},
            {"label": "Anti-Assignment", "count": 8, "risk_level": "medium"}
        ]

        missing_clauses = [
            "Governing Law",
            "Dispute Resolution",
            "Confidentiality"
        ]

        recommendations = []
        if request.include_suggestions:
            recommendations = [
                {
                    "clause": "Cap on Liability",
                    "suggestion": "Consider adding carve-outs for gross negligence and IP infringement",
                    "risk_level": "high"
                }
            ]

        return RiskReportResponse(
            report_id=str(uuid.uuid4()),
            total_contracts=len(request.contract_ids),
            high_risk_count=high_risk_count,
            medium_risk_count=medium_risk_count,
            low_risk_count=low_risk_count,
            top_red_flags=top_red_flags,
            missing_clauses=missing_clauses,
            recommendations=recommendations,
            generated_at=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk report generation failed: {str(e)}")

@app.get("/export")
async def export_contract_data(
    contract_id: str = Query(...),
    fmt: str = Query("csv", regex="^(csv|json)$"),
    token: str = Depends(verify_token)
):
    """Export contract analysis data."""
    try:
        # Mock export data
        export_data = {
            "contract_id": contract_id,
            "analysis_date": datetime.now().isoformat(),
            "total_clauses": 5,
            "high_risk_clauses": 2,
            "medium_risk_clauses": 2,
            "low_risk_clauses": 1,
            "overall_risk_score": 0.65
        }

        if fmt == "json":
            return JSONResponse(content=export_data)
        else:
            # CSV export would be implemented here
            return JSONResponse(content={"message": "CSV export not yet implemented", "data": export_data})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


    # Export endpoints
    @app.post("/export/contract")
    async def export_contract_data(
        request: ExportRequest,
        token: str = Depends(verify_token),
        client_ip: str = Depends(check_rate_limit)
    ):
        """Export contract analysis data."""
        try:
            analyzer = get_analyzer()
            export_manager = ExportManager()

            # Analyze contract first
            if request.contract_id:
                # Mock analysis for demo - in production, get from database
                mock_clauses = [f"Sample clause from {request.contract_id}"]
                analysis = analyzer.analyze(request.contract_id, mock_clauses)

                # Export based on format
                if request.format == "csv":
                    filepath = export_manager.export_to_csv(analysis)
                elif request.format == "json":
                    filepath = export_manager.export_to_json(analysis)
                else:
                    raise HTTPException(status_code=400, detail="Unsupported format")

                # Return file
                return FileResponse(
                    path=filepath,
                    filename=Path(filepath).name,
                    media_type="application/octet-stream"
                )
            else:
                raise HTTPException(status_code=400, detail="Contract ID required")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

    @app.get("/export/formats")
    async def get_export_formats(token: str = Depends(verify_token)):
        """Get available export formats."""
        export_manager = ExportManager()
        return {
            "formats": export_manager.get_export_formats(),
            "descriptions": {
                "csv": "Comma-separated values format",
                "json": "JSON format with full analysis data",
                "portfolio_csv": "CSV format for multiple contracts",
                "risk_report": "JSON risk report summary"
            }
        }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now()
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
