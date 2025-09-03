"""
FastAPI Backend for Contract Review & Risk Analysis System
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
import uvicorn
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.config import *
from src.models.contract_analyzer import ContractAnalyzer
from src.models.risk_scorer import RiskScorer
from src.utils.file_processor import FileProcessor
from src.utils.auth import AuthManager

# Initialize FastAPI app
app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    description="Production-ready legal AI system for contract review and risk analysis",
    contact={"name": AUTHOR, "email": EMAIL, "url": "https://github.com/Muh76"},
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
contract_analyzer = ContractAnalyzer()
risk_scorer = RiskScorer()
file_processor = FileProcessor()
auth_manager = AuthManager()

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@app.get("/")
async def root():
    """Root endpoint with project information"""
    return {
        "project": PROJECT_NAME,
        "version": VERSION,
        "author": AUTHOR,
        "email": EMAIL,
        "status": "running",
        "endpoints": {"analyze": "/analyze", "health": "/health", "docs": "/docs"},
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "contract_analyzer": "running",
            "risk_scorer": "running",
            "file_processor": "running",
        },
    }


@app.post("/analyze")
async def analyze_contract(
    file: UploadFile = File(...), token: str = Depends(oauth2_scheme)
):
    """
    Analyze uploaded contract for risk assessment

    Args:
        file: Contract file (PDF, TXT, DOCX)
        token: Authentication token

    Returns:
        Analysis results with risk score and extracted clauses
    """
    try:
        # Validate file
        if not file_processor.is_valid_file(file):
            raise HTTPException(
                status_code=400, detail="Invalid file type. Supported: PDF, TXT, DOCX"
            )

        # Extract text from file
        contract_text = await file_processor.extract_text(file)

        # Analyze contract
        extracted_clauses = contract_analyzer.extract_clauses(contract_text)

        # Calculate risk score
        risk_score = risk_scorer.calculate_risk_score(extracted_clauses)

        # Get similar clauses (RAG)
        similar_clauses = contract_analyzer.find_similar_clauses(extracted_clauses)

        # Generate recommendations
        recommendations = contract_analyzer.generate_recommendations(extracted_clauses)

        return {
            "file_name": file.filename,
            "risk_score": risk_score,
            "risk_level": risk_scorer.get_risk_level(risk_score),
            "extracted_clauses": extracted_clauses,
            "similar_clauses": similar_clauses,
            "recommendations": recommendations,
            "business_impact": risk_scorer.calculate_business_impact(risk_score),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch")
async def analyze_contracts_batch(
    files: list[UploadFile] = File(...), token: str = Depends(oauth2_scheme)
):
    """
    Analyze multiple contracts in batch

    Args:
        files: List of contract files
        token: Authentication token

    Returns:
        Batch analysis results
    """
    results = []

    for file in files:
        try:
            # Process each file
            contract_text = await file_processor.extract_text(file)
            extracted_clauses = contract_analyzer.extract_clauses(contract_text)
            risk_score = risk_scorer.calculate_risk_score(extracted_clauses)

            results.append(
                {
                    "file_name": file.filename,
                    "risk_score": risk_score,
                    "risk_level": risk_scorer.get_risk_level(risk_score),
                    "extracted_clauses": extracted_clauses,
                }
            )

        except Exception as e:
            results.append({"file_name": file.filename, "error": str(e)})

    return {
        "batch_results": results,
        "summary": {
            "total_files": len(files),
            "successful_analyses": len([r for r in results if "error" not in r]),
            "average_risk_score": (
                sum([r.get("risk_score", 0) for r in results if "error" not in r])
                / len([r for r in results if "error" not in r])
                if results
                else 0
            ),
        },
    }


@app.get("/clauses/similar")
async def get_similar_clauses(
    clause_text: str, clause_type: str, token: str = Depends(oauth2_scheme)
):
    """
    Find similar clauses using RAG

    Args:
        clause_text: Query clause text
        clause_type: Type of clause
        token: Authentication token

    Returns:
        Similar clauses from precedent database
    """
    try:
        similar_clauses = contract_analyzer.find_similar_clauses_by_text(
            clause_text, clause_type
        )
        return {
            "query_clause": clause_text,
            "clause_type": clause_type,
            "similar_clauses": similar_clauses,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clauses/suggest")
async def suggest_alternative_wording(
    clause_text: str, clause_type: str, token: str = Depends(oauth2_scheme)
):
    """
    Suggest alternative wording for risky clauses

    Args:
        clause_text: Original clause text
        clause_type: Type of clause
        token: Authentication token

    Returns:
        Alternative wording suggestions
    """
    try:
        suggestions = contract_analyzer.suggest_alternative_wording(
            clause_text, clause_type
        )
        return {
            "original_clause": clause_text,
            "clause_type": clause_type,
            "suggestions": suggestions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics(token: str = Depends(oauth2_scheme)):
    """
    Get system metrics and performance data

    Args:
        token: Authentication token

    Returns:
        System metrics
    """
    try:
        metrics = {
            "total_contracts_analyzed": contract_analyzer.get_total_analyzed(),
            "average_processing_time": contract_analyzer.get_avg_processing_time(),
            "model_accuracy": contract_analyzer.get_model_accuracy(),
            "risk_distribution": risk_scorer.get_risk_distribution(),
        }
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)
