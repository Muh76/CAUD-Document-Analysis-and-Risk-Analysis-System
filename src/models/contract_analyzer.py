"""
Contract Analyzer - Core ML model for clause extraction and analysis
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, List, Optional
import mlflow
import logging
from pathlib import Path

from src.config.config import CUAD_CATEGORIES, RISK_WEIGHTS
from src.models.legal_rag import LegalRAGSystem


class LegalContractModel(nn.Module):
    """Multi-task learning model for legal contract analysis"""

    def __init__(self, num_categories: int = 41, model_name: str = "roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifiers = nn.ModuleDict(
            {
                "binary": nn.Linear(768, 2),  # Yes/No categories
                "extractive": nn.Linear(768, 2),  # Span extraction
                "regression": nn.Linear(768, 1),  # Dates, amounts
            }
        )
        self.num_categories = num_categories

    def forward(self, input_ids, attention_mask, task_type="binary"):
        outputs = self.encoder(input_ids, attention_mask)
        return self.classifiers[task_type](outputs.pooler_output)


class ContractAnalyzer:
    """Main contract analysis engine"""

    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self._load_model(model_path)

        # Initialize RAG system
        self.rag_system = LegalRAGSystem()

        # Metrics tracking
        self.total_analyzed = 0
        self.processing_times = []
        self.model_accuracy = 0.85  # Mock value

    def _load_model(self, model_path: Optional[str] = None):
        """Load the trained model"""
        try:
            if model_path and Path(model_path).exists():
                # Load custom trained model
                self.model = torch.load(model_path, map_location=self.device)
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            else:
                # Load pre-trained model for demo
                self.model = LegalContractModel()
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

            self.model.to(self.device)
            self.model.eval()
            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            # Fallback to rule-based system
            self.model = None

    def extract_clauses(self, contract_text: str) -> Dict[str, str]:
        """
        Extract legal clauses from contract text

        Args:
            contract_text: Raw contract text

        Returns:
            Dictionary of extracted clauses by type
        """
        import time

        start_time = time.time()

        try:
            if self.model is not None:
                # ML-based extraction
                extracted_clauses = self._extract_with_ml(contract_text)
            else:
                # Rule-based extraction
                extracted_clauses = self._extract_with_rules(contract_text)

            # Update metrics
            self.total_analyzed += 1
            self.processing_times.append(time.time() - start_time)

            return extracted_clauses

        except Exception as e:
            self.logger.error(f"Error extracting clauses: {e}")
            return {}

    def _extract_with_ml(self, contract_text: str) -> Dict[str, str]:
        """Extract clauses using ML model"""
        # Tokenize input
        inputs = self.tokenizer(
            contract_text, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs, task_type="binary")
            predictions = torch.softmax(outputs, dim=-1)

        # Extract clauses based on predictions
        extracted_clauses = {}

        # Mock extraction for demo
        extracted_clauses = {
            "governing_law": "This agreement shall be governed by the laws of California",
            "liquidated_damages": "Party A shall pay $50,000 in liquidated damages",
            "confidentiality": "All information shall be kept confidential",
            "termination": "Either party may terminate this agreement with 30 days notice",
            "ip_assignment": "All intellectual property shall be assigned to Company",
        }

        return extracted_clauses

    def _extract_with_rules(self, contract_text: str) -> Dict[str, str]:
        """Extract clauses using rule-based system"""
        extracted_clauses = {}

        # Simple keyword-based extraction
        keywords = {
            "governing_law": ["governed by", "laws of", "jurisdiction"],
            "liquidated_damages": ["liquidated damages", "penalty", "fine"],
            "confidentiality": ["confidential", "non-disclosure", "secret"],
            "termination": ["terminate", "termination", "end"],
            "ip_assignment": ["intellectual property", "IP", "assign"],
        }

        for clause_type, search_terms in keywords.items():
            for term in search_terms:
                if term.lower() in contract_text.lower():
                    # Extract sentence containing the term
                    sentences = contract_text.split(".")
                    for sentence in sentences:
                        if term.lower() in sentence.lower():
                            extracted_clauses[clause_type] = sentence.strip()
                            break
                    break

        return extracted_clauses

    def find_similar_clauses(
        self, extracted_clauses: Dict[str, str]
    ) -> Dict[str, List[Dict]]:
        """
        Find similar clauses using RAG

        Args:
            extracted_clauses: Dictionary of extracted clauses

        Returns:
            Dictionary of similar clauses for each extracted clause
        """
        similar_clauses = {}

        for clause_type, clause_text in extracted_clauses.items():
            similar = self.find_similar_clauses_by_text(clause_text, clause_type)
            similar_clauses[clause_type] = similar

        return similar_clauses

    def find_similar_clauses_by_text(
        self, clause_text: str, clause_type: str
    ) -> List[Dict]:
        """
        Find similar clauses by text using vector similarity

        Args:
            clause_text: Query clause text
            clause_type: Type of clause

        Returns:
            List of similar clauses with metadata
        """
        return self.rag_system.find_similar_clauses(clause_text, clause_type)

    def suggest_alternative_wording(
        self, clause_text: str, clause_type: str
    ) -> List[str]:
        """
        Suggest alternative wording for risky clauses

        Args:
            clause_text: Original clause text
            clause_type: Type of clause

        Returns:
            List of alternative suggestions
        """
        return self.rag_system.suggest_alternative_wording(clause_text, clause_type)

    def analyze_clause_risk(self, clause_text: str, clause_type: str) -> Dict:
        """
        Analyze risk of a specific clause

        Args:
            clause_text: Text of the clause
            clause_type: Type of clause

        Returns:
            Risk analysis with score and explanation
        """
        return self.rag_system.analyze_clause_risk(clause_text, clause_type)

    def generate_recommendations(self, extracted_clauses: Dict[str, str]) -> List[str]:
        """
        Generate recommendations based on extracted clauses

        Args:
            extracted_clauses: Dictionary of extracted clauses

        Returns:
            List of recommendations
        """
        recommendations = []

        # Risk-based recommendations
        if "liquidated_damages" in extracted_clauses:
            recommendations.append("Consider negotiating liquidated damages amount")

        if "confidentiality" in extracted_clauses:
            recommendations.append("Review confidentiality duration and scope")

        if "termination" in extracted_clauses:
            recommendations.append("Ensure termination notice period is reasonable")

        # General recommendations
        recommendations.extend(
            [
                "Review governing law jurisdiction",
                "Consider adding force majeure clause",
                "Verify IP assignment terms",
            ]
        )

        return recommendations

    def get_total_analyzed(self) -> int:
        """Get total number of contracts analyzed"""
        return self.total_analyzed

    def get_avg_processing_time(self) -> float:
        """Get average processing time"""
        return np.mean(self.processing_times) if self.processing_times else 0.0

    def get_model_accuracy(self) -> float:
        """Get current model accuracy"""
        return self.model_accuracy
