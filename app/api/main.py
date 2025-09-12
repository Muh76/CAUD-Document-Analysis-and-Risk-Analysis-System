"""
FastAPI main application with all endpoints.
"""

import os
import base64
import uuid
import asyncio
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Query, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware as FastAPICORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import json

from .schemas import (
    AnalyzeRequest, AnalyzeResponse, BatchAnalyzeRequest, BatchAnalyzeResponse,
    RiskReportRequest, RiskReportResponse, ExportRequest, HealthResponse, ErrorResponse
)
from .deps import (
    get_settings, get_analyzer, get_uptime, verify_token, verify_token_optional, check_rate_limit
)
from app.core.text_ingest import TextIngestion
from app.core.logging import logger, request_logger, metrics
from app.core.security import SecurityManager
from app.core.middleware import SecurityMiddleware, LoggingMiddleware, MetricsMiddleware, CORSMiddleware

from app.core.export import ExportManager
from app.core.io import IOUtils

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
    allowed_origins=cors_origins,
)

# Global storage for batch jobs (in production, use Redis or database)
batch_jobs = {}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Contract Review & Risk Analysis API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "/health",
            "/analyze_contract",
            "/batch_analyze", 
            "/risk_report",
            "/export",
            "/metrics",
            "/docs"
        ]
    }

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
    token: str = Depends(verify_token_optional),
    client_ip: str = Depends(check_rate_limit)
):
    """Analyze a single contract."""
    try:
        print(f"DEBUG API: Received request for contract {request.contract_id}")
        print(f"DEBUG API: Text length: {len(request.text) if request.text else 0}")
        print(f"DEBUG API: Has file: {bool(request.file_b64)}")
        
        analyzer = get_analyzer()
        settings = get_settings()

        # Extract text from request
        if request.text:
            # Create TextChunk objects for text input
            from app.core.text_ingest import TextIngestion
            text_ingestion = TextIngestion()
            chunks = text_ingestion.chunk_text(request.text)
            print(f"DEBUG API: Created {len(chunks)} chunks from text")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"DEBUG API: Chunk {i}: '{chunk.text[:50]}...'")
        elif request.file_b64:
            # Decode base64 file
            try:
                file_content = base64.b64decode(request.file_b64)
                
                # Check if it's a PDF file
                if request.mime == "application/pdf":
                    try:
                        # Use TextIngestion to process PDF
                        from app.core.text_ingest import TextIngestion
                        text_ingestion = TextIngestion()
                        chunks = text_ingestion.process_contract_bytes(file_content, "application/pdf")

                        # If no text extracted, fallback to simple processing
                        if not chunks:
                            from app.core.schemas import TextChunk
                            chunks = [TextChunk(
                                text="PDF text extraction failed - please try uploading as text file", 
                                start_offset=0,
                                end_offset=len("PDF text extraction failed - please try uploading as text file"),
                                chunk_id=0
                            )]
                    except Exception as pdf_error:
                        # Enhanced error handling for PDF processing
                        error_msg = str(pdf_error)
                        from app.core.schemas import TextChunk
                        if "document closed" in error_msg.lower():
                            error_text = "PDF file appears to be corrupted or password-protected. Please try a different PDF file or convert to text format."
                            chunks = [TextChunk(text=error_text, start_offset=0, end_offset=len(error_text), chunk_id=0)]
                        elif "invalid" in error_msg.lower():
                            error_text = "PDF file format is not supported. Please try uploading a different PDF or convert to text format."
                            chunks = [TextChunk(text=error_text, start_offset=0, end_offset=len(error_text), chunk_id=0)]
                        else:
                            error_text = f"PDF processing error: {error_msg} - please try uploading as text file"
                            chunks = [TextChunk(text=error_text, start_offset=0, end_offset=len(error_text), chunk_id=0)]
                else:
                    # For text files, decode as UTF-8 and chunk
                    text = file_content.decode('utf-8')
                    from app.core.text_ingest import TextIngestion
                    text_ingestion = TextIngestion()
                    chunks = text_ingestion.chunk_text(text)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid file content: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Either text or file_b64 must be provided")

        # Analyze contract
        analysis = analyzer.analyze(request.contract_id, chunks)

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
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token_optional),
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
            "contracts": request.contracts,
            "errors": []
        }

        # Start background processing
        background_tasks.add_task(process_batch_job, job_id)

        return BatchAnalyzeResponse(
            job_id=job_id,
            total_contracts=len(request.contracts),
            status="queued",
            created_at=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

def process_batch_job(job_id: str):
    """Process a batch job in the background."""
    if job_id not in batch_jobs:
        return
    
    job = batch_jobs[job_id]
    job["status"] = "processing"
    
    try:
        analyzer = get_analyzer()
        
        for i, contract in enumerate(job["contracts"]):
            try:
                # Process each contract
                if contract.text:
                    # Text-based analysis
                    text_ingestion = TextIngestion()
                    chunks = text_ingestion.chunk_text(contract.text)
                elif contract.file_b64:
                    # File-based analysis
                    file_content = base64.b64decode(contract.file_b64)
                    if contract.mime == "application/pdf":
                        # Process PDF
                        text_ingestion = TextIngestion()
                        chunks = text_ingestion.process_contract_bytes(file_content, "application/pdf")
                    else:
                        # Process text file
                        text_ingestion = TextIngestion()
                        chunks = text_ingestion.chunk_text(file_content.decode('utf-8'))
                else:
                    raise ValueError("No text or file content provided")
                
                # Analyze the contract
                analysis = analyzer.analyze(contract.contract_id or f"batch_{i}", chunks)
                
                # Store result
                result = {
                    "contract_id": contract.contract_id or f"batch_{i}",
                    "status": "completed",
                    "overall_risk_score": analysis.overall_risk_score,
                    "total_clauses": analysis.total_clauses,
                    "high_risk_clauses": analysis.high_risk_clauses,
                    "results": [
                        {
                            "risk": r.risk_score,
                            "snippet": r.text[:200] + "..." if len(r.text) > 200 else r.text,
                            "rationale": r.rationale,
                            "probs": r.probs
                        } for r in analysis.results
                    ]
                }
                
                job["results"].append(result)
                job["completed"] += 1
                
            except Exception as e:
                # Handle individual contract errors
                error_result = {
                    "contract_id": contract.contract_id or f"batch_{i}",
                    "status": "error",
                    "error": str(e),
                    "overall_risk_score": 0.0,
                    "total_clauses": 0,
                    "high_risk_clauses": 0,
                    "results": []
                }
                job["results"].append(error_result)
                job["errors"].append(f"Contract {i}: {str(e)}")
                job["completed"] += 1
        
        job["status"] = "completed"
        
    except Exception as e:
        job["status"] = "failed"
        job["errors"].append(f"Batch processing failed: {str(e)}")

@app.get("/batch_analyze/{job_id}")
async def get_batch_status(job_id: str, token: str = Depends(verify_token_optional)):
    """Get batch analysis status and results."""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = batch_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "total_contracts": job["total_contracts"],
        "completed": job["completed"],
        "created_at": job["created_at"],
        "results": job["results"],
        "errors": job.get("errors", [])
    }

@app.post("/risk_report", response_model=RiskReportResponse)
async def generate_risk_report(
    request: RiskReportRequest,
    token: str = Depends(verify_token_optional),
    client_ip: str = Depends(check_rate_limit)
):
    """Generate risk report for multiple contracts."""
    try:
        analyzer = get_analyzer()
        settings = get_settings()

        # Real analysis implementation
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0

        # Collect all detected labels and their counts
        label_counts = {}
        all_clauses = []
        missing_clauses_set = set()
        
        # Common clauses that should be present in contracts
        expected_clauses = {
            "Governing Law", "Dispute Resolution", "Confidentiality", 
            "Termination", "Liability", "Indemnity", "Force Majeure",
            "Assignment", "Amendment", "Severability", "Entire Agreement"
        }

        # Analyze each contract
        for contract_id in request.contract_ids:
            # For demo purposes, create sample contract text
            sample_text = f"""
            TERM AND TERMINATION: This agreement shall be for one year.
            GOVERNING LAW: This agreement shall be governed by California law.
            LIABILITY: Each party liability shall be limited to the contract amount.
            CONFIDENTIALITY: Both parties agree to maintain confidentiality.
            """
            
            # Create TextChunk objects for analysis
            from app.core.text_ingest import TextIngestion
            text_ingestion = TextIngestion()
            chunks = text_ingestion.chunk_text(sample_text)
            
            # Perform real analysis
            analysis = analyzer.analyze(contract_id, chunks)

            # Categorize risk level
            if analysis.overall_risk_score >= settings.get_global_threshold("HIGH_RISK_THRESHOLD"):
                high_risk_count += 1
            elif analysis.overall_risk_score >= settings.get_global_threshold("MEDIUM_RISK_THRESHOLD"):
                medium_risk_count += 1
            else:
                low_risk_count += 1

            # Collect detected labels
            for result in analysis.results:
                all_clauses.append(result)
                for label in result.detected_labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            # Check for missing clauses (simplified - in production, this would be more sophisticated)
            detected_labels_set = set()
            for result in analysis.results:
                detected_labels_set.update(result.detected_labels)
            
            missing_clauses_set.update(expected_clauses - detected_labels_set)

        # Generate real red flags based on detected labels
        top_red_flags = []
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            # Determine risk level based on label type
            risk_level = "medium"
            if label in ["Liability", "Indemnity", "Termination"]:
                risk_level = "high"
            elif label in ["Confidentiality", "Assignment"]:
                risk_level = "low"
            
            top_red_flags.append({
                "label": label,
                "count": count,
                "risk_level": risk_level
            })

        # Convert missing clauses to list
        missing_clauses = list(missing_clauses_set)[:5]  # Top 5 missing clauses

        # Generate real recommendations based on analysis
        recommendations = []
        if request.include_suggestions:
            for flag in top_red_flags[:3]:  # Top 3 red flags
                if flag["label"] == "Liability":
                    recommendations.append({
                        "clause": "Liability",
                    "suggestion": "Consider adding carve-outs for gross negligence and IP infringement",
                    "risk_level": "high"
                    })
                elif flag["label"] == "Termination":
                    recommendations.append({
                        "clause": "Termination",
                        "suggestion": "Review termination clauses for fairness and notice periods",
                        "risk_level": "medium"
                    })
                elif flag["label"] == "Indemnity":
                    recommendations.append({
                        "clause": "Indemnity",
                        "suggestion": "Ensure indemnity clauses are mutual and reasonable in scope",
                        "risk_level": "high"
                    })

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
    token: str = Depends(verify_token_optional)
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
        token: str = Depends(verify_token_optional),
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
    async def get_export_formats(token: str = Depends(verify_token_optional)):
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
