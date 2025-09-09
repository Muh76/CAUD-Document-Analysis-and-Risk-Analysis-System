"""
Enhanced pipeline with integrated monitoring and quality checks.
"""

import json
import pickle
import numpy as np
from time import perf_counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from app.config.settings import get_settings
from app.core.model_monitoring import model_monitor

# Settings
settings = get_settings()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedContractAnalyzer:
    """Enhanced contract analyzer with monitoring and quality checks."""

    def __init__(self):
        self.settings = settings
        self.model_version = settings.model_snapshot
        self.artifacts_path = settings.artifacts_path

        # Load models and configurations
        self._load_models()

        logger.info(f"Enhanced ContractAnalyzer initialized with version {self.model_version}")

    def _load_models(self):
        """Load models and configurations."""
        try:
            # Load label map
            label_map_path = self.artifacts_path / self.model_version / "label_map.json"
            if label_map_path.exists():
                with open(label_map_path, "r") as f:
                    self.label_map = json.load(f)
            else:
                self.label_map = {"unknown": 0}

            # Load thresholds
            thresholds_path = self.artifacts_path / self.model_version / "thresholds.json"
            if thresholds_path.exists():
                with open(thresholds_path, "r") as f:
                    self.thresholds = json.load(f)
            else:
                self.thresholds = {"high": 0.8, "medium": 0.6, "low": 0.4}

            # Load models
            models_dir = self.artifacts_path / self.model_version / "models"
            if models_dir.exists():
                self.models = {}
                for model_file in models_dir.glob("*.pkl"):
                    with open(model_file, "rb") as f:
                        self.models[model_file.stem] = pickle.load(f)
            else:
                self.models = {}

            logger.info("Models and configurations loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            # Fallback to default values
            self.label_map = {"unknown": 0}
            self.thresholds = {"high": 0.8, "medium": 0.6, "low": 0.4}
            self.models = {}

    def analyze_contract(self, text: str, **kwargs) -> Dict[str, Any]:
        """Analyze contract with enhanced monitoring and quality checks."""
        start_time = perf_counter()
        prediction_id = None
        error = None

        try:
            # Data quality validation
            quality_check = model_monitor.validate_data_quality(text)
            if not quality_check["passed"]:
                logger.warning(f"Data quality issues: {quality_check['issues']}")

            # Perform analysis
            result = self._perform_analysis(text, **kwargs)

            # Calculate processing time
            processing_time_ms = (perf_counter() - start_time) * 1000

            # Calculate overall confidence
            confidence = self._calculate_confidence(result)

            # Log prediction for monitoring
            prediction_id = model_monitor.log_prediction(
                input_text=text,
                prediction=result,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                model_version=self.model_version
            )

            # Add monitoring metadata
            result["monitoring"] = {
                "prediction_id": prediction_id,
                "model_version": self.model_version,
                "confidence": confidence,
                "processing_time_ms": processing_time_ms,
                "data_quality": quality_check,
                "timestamp": datetime.utcnow().isoformat()
            }

            logger.info(f"Analysis completed successfully - ID: {prediction_id}")
            return result

        except Exception as e:
            error = str(e)
            processing_time_ms = (perf_counter() - start_time) * 1000

            # Log error prediction
            prediction_id = model_monitor.log_prediction(
                input_text=text,
                prediction={"error": error},
                confidence=0.0,
                processing_time_ms=processing_time_ms,
                model_version=self.model_version,
                error=error
            )

            logger.error(f"Analysis failed: {e}")

            return {
                "error": error,
                "monitoring": {
                    "prediction_id": prediction_id,
                    "model_version": self.model_version,
                    "confidence": 0.0,
                    "processing_time_ms": processing_time_ms,
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

    def _perform_analysis(self, text: str, **kwargs) -> Dict[str, Any]:
        """Perform the actual contract analysis."""
        # This is where your existing analysis logic would go
        # For now, we'll create a realistic simulation

        # Simulate clause extraction
        clauses = self._extract_clauses(text)

        # Simulate risk assessment
        risk_assessment = self._assess_risks(clauses)

        # Simulate RAG retrieval
        similar_clauses = self._retrieve_similar_clauses(text, **kwargs)

        return {
            "contract_summary": {
                "total_clauses": len(clauses),
                "high_risk_count": risk_assessment["high_risk_count"],
                "medium_risk_count": risk_assessment["medium_risk_count"],
                "low_risk_count": risk_assessment["low_risk_count"]
            },
            "clauses": clauses,
            "risk_assessment": risk_assessment,
            "similar_clauses": similar_clauses,
            "analysis_metadata": {
                "model_version": self.model_version,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "input_length": len(text)
            }
        }

    def _extract_clauses(self, text: str) -> List[Dict[str, Any]]:
        """Extract contract clauses."""
        # Simulate clause extraction
        clauses = []

        # Simple keyword-based clause detection
        clause_keywords = {
            "liability": "Liability Clause",
            "indemnification": "Indemnification Clause", 
            "termination": "Termination Clause",
            "payment": "Payment Clause",
            "confidentiality": "Confidentiality Clause",
            "warranty": "Warranty Clause",
            "force majeure": "Force Majeure Clause"
        }

        text_lower = text.lower()
        for keyword, clause_type in clause_keywords.items():
            if keyword in text_lower:
                # Find the sentence containing the keyword
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        clauses.append({
                            "clause_type": clause_type,
                            "text": sentence.strip(),
                            "risk_level": self._assess_clause_risk(sentence),
                            "confidence": np.random.uniform(0.7, 0.95)
                        })
                        break

        return clauses

    def _assess_clause_risk(self, clause_text: str) -> str:
        """Assess risk level of a clause."""
        high_risk_keywords = ["liability", "indemnification", "penalty", "breach"]
        medium_risk_keywords = ["termination", "payment", "warranty"]

        clause_lower = clause_text.lower()

        if any(keyword in clause_lower for keyword in high_risk_keywords):
            return "high"
        elif any(keyword in clause_lower for keyword in medium_risk_keywords):
            return "medium"
        else:
            return "low"

    def _assess_risks(self, clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall contract risks."""
        risk_counts = {"high": 0, "medium": 0, "low": 0}

        for clause in clauses:
            risk_level = clause.get("risk_level", "low")
            risk_counts[risk_level] += 1

        # Calculate risk score
        total_clauses = len(clauses)
        if total_clauses > 0:
            risk_score = (risk_counts["high"] * 3 + risk_counts["medium"] * 2 + risk_counts["low"] * 1) / (total_clauses * 3)
        else:
            risk_score = 0.0

        return {
            "high_risk_count": risk_counts["high"],
            "medium_risk_count": risk_counts["medium"],
            "low_risk_count": risk_counts["low"],
            "risk_score": risk_score,
            "overall_risk": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low"
        }

    def _retrieve_similar_clauses(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve similar clauses using RAG."""
        # Simulate RAG retrieval
        similarity_threshold = kwargs.get("similarity_threshold", 0.7)
        top_k = kwargs.get("top_k", 5)

        # Mock similar clauses
        similar_clauses = [
            {
                "text": "The Company shall indemnify and hold harmless the Client from any claims...",
                "similarity": 0.85,
                "source": "Sample Contract A",
                "clause_type": "Indemnification Clause"
            },
            {
                "text": "In case of breach of contract, the defaulting party shall pay damages...",
                "similarity": 0.78,
                "source": "Sample Contract B", 
                "clause_type": "Breach Clause"
            }
        ]

        # Filter by similarity threshold
        filtered_clauses = [c for c in similar_clauses if c["similarity"] >= similarity_threshold]

        return filtered_clauses[:top_k]

    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate overall confidence score."""
        if "error" in result:
            return 0.0

        # Calculate confidence based on clause confidences
        clauses = result.get("clauses", [])
        if not clauses:
            return 0.5

        confidences = [clause.get("confidence", 0.5) for clause in clauses]
        return np.mean(confidences)

    def get_model_metrics(self) -> Dict[str, Any]:
        """Get current model performance metrics."""
        try:
            metrics = model_monitor.calculate_model_metrics()
            return {
                "status": "success",
                "metrics": {
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "avg_confidence": metrics.avg_confidence,
                    "avg_processing_time_ms": metrics.avg_processing_time_ms,
                    "total_predictions": metrics.total_predictions,
                    "error_rate": metrics.error_rate
                },
                "model_version": metrics.model_version,
                "timestamp": metrics.timestamp
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def detect_drift(self) -> Dict[str, Any]:
        """Detect model drift."""
        return model_monitor.detect_drift()

    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get monitoring dashboard data."""
        return model_monitor.get_monitoring_dashboard_data()

# Global instance
enhanced_analyzer = EnhancedContractAnalyzer()
