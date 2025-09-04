"""
Risk Scoring Engine - Calculate risk scores for legal clauses
"""

import numpy as np
from typing import Dict, List, Optional
import logging
from collections import defaultdict
from collections import defaultdict

from src.config.config import RISK_WEIGHTS


class RiskScorer:
    """Risk scoring engine for legal contracts"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.risk_weights = RISK_WEIGHTS
        self.risk_categories = {
            "critical": ["uncapped_liability", "ip_assignment", "non_compete"],
            "high": ["termination_convenience", "audit_rights", "liquidated_damages"],
            "medium": ["governing_law", "warranty_duration", "confidentiality"],
            "low": ["document_name", "parties", "effective_date"],
        }

        # Risk distribution tracking
        self.risk_distribution: Dict[str, int] = defaultdict(int)

    def calculate_risk_score(self, extracted_clauses: Dict[str, str]) -> float:
        """
        Calculate overall risk score for extracted clauses

        Args:
            extracted_clauses: Dictionary of extracted clauses

        Returns:
            Risk score between 0 and 1
        """
        try:
            risk_score = 0.0

            for clause_type, clause_text in extracted_clauses.items():
                if clause_type in self.risk_weights:
                    weight = self.risk_weights[clause_type]
                    clause_risk = self._calculate_clause_risk(clause_text, clause_type)
                    risk_score += weight * clause_risk

            # Normalize to [0, 1]
            risk_score = min(risk_score, 1.0)

            # Track distribution
            risk_level = self.get_risk_level(risk_score)
            self.risk_distribution[risk_level] += 1

            return risk_score

        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.5  # Default medium risk

    def _calculate_clause_risk(self, clause_text: str, clause_type: str) -> float:
        """
        Calculate risk for individual clause

        Args:
            clause_text: Text of the clause
            clause_type: Type of clause

        Returns:
            Risk value between 0 and 1
        """
        # Risk indicators for different clause types
        risk_indicators = {
            "uncapped_liability": {
                "high_risk": ["unlimited", "uncapped", "all damages", "any damages"],
                "medium_risk": ["reasonable", "actual", "direct"],
                "low_risk": ["limited", "capped", "maximum"],
            },
            "liquidated_damages": {
                "high_risk": ["penalty", "fine", "damages"],
                "medium_risk": ["reasonable", "actual"],
                "low_risk": ["limited", "capped"],
            },
            "confidentiality": {
                "high_risk": ["permanent", "indefinite", "forever"],
                "medium_risk": ["reasonable", "necessary"],
                "low_risk": ["limited", "specific", "defined"],
            },
            "termination": {
                "high_risk": ["immediate", "at will", "without cause"],
                "medium_risk": ["reasonable", "notice"],
                "low_risk": ["cause", "breach", "default"],
            },
            "ip_assignment": {
                "high_risk": ["all", "any", "everything"],
                "medium_risk": ["related", "arising"],
                "low_risk": ["specific", "defined", "limited"],
            },
        }

        if clause_type not in risk_indicators:
            return 0.5  # Default medium risk

        indicators = risk_indicators[clause_type]
        clause_lower = clause_text.lower()

        # Calculate risk based on indicators
        high_risk_count = sum(
            1 for term in indicators["high_risk"] if term in clause_lower
        )
        medium_risk_count = sum(
            1 for term in indicators["medium_risk"] if term in clause_lower
        )
        low_risk_count = sum(
            1 for term in indicators["low_risk"] if term in clause_lower
        )

        # Weighted risk calculation
        if high_risk_count > 0:
            return 0.8 + (high_risk_count * 0.1)
        elif medium_risk_count > 0:
            return 0.5 + (medium_risk_count * 0.1)
        elif low_risk_count > 0:
            return 0.2 + (low_risk_count * 0.1)
        else:
            return 0.5  # Default medium risk

    def get_risk_level(self, risk_score: float) -> str:
        """
        Get risk level category based on score

        Args:
            risk_score: Risk score between 0 and 1

        Returns:
            Risk level: 'low', 'medium', 'high', 'critical'
        """
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"

    def calculate_business_impact(self, risk_score: float) -> Dict[str, float]:
        """
        Calculate business impact of risk score

        Args:
            risk_score: Risk score between 0 and 1

        Returns:
            Dictionary of business impact metrics
        """
        # Mock business impact calculation
        base_contract_value = 100000  # $100k base contract value

        # Time impact (hours saved/lost)
        time_impact = risk_score * 40  # 40 hours for high risk contracts

        # Cost impact
        cost_impact = risk_score * base_contract_value * 0.1  # 10% of contract value

        # Legal fees impact
        legal_fees = risk_score * 5000  # $5k base legal fees

        return {
            "time_impact_hours": time_impact,
            "cost_impact_usd": cost_impact,
            "legal_fees_usd": legal_fees,
            "total_impact_usd": cost_impact + legal_fees,
        }

    def prioritize_clauses(self, extracted_clauses: Dict[str, str]) -> List[Dict]:
        """
        Prioritize clauses by risk and business impact

        Args:
            extracted_clauses: Dictionary of extracted clauses

        Returns:
            List of prioritized clauses with metadata
        """
        prioritized = []

        for clause_type, clause_text in extracted_clauses.items():
            risk_score = self._calculate_clause_risk(clause_text, clause_type)
            risk_level = self.get_risk_level(risk_score)
            business_impact = self.calculate_business_impact(risk_score)

            prioritized.append(
                {
                    "clause_type": clause_type,
                    "text": clause_text,
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "business_impact": business_impact,
                    "action_required": self._get_action_required(risk_level),
                }
            )

        # Sort by business impact
        prioritized.sort(
            key=lambda x: float(x["business_impact"]["total_impact_usd"]), reverse=True
        )

        return prioritized

    def _get_action_required(self, risk_level: str) -> str:
        """
        Get action required based on risk level

        Args:
            risk_level: Risk level category

        Returns:
            Action required description
        """
        actions = {
            "critical": "Immediate review and negotiation required",
            "high": "Review and consider modifications",
            "medium": "Monitor and review periodically",
            "low": "Standard review process",
        }

        return actions.get(risk_level, "Review as needed")

    def get_risk_distribution(self) -> Dict[str, int]:
        """
        Get current risk distribution

        Returns:
            Dictionary of risk level counts
        """
        return dict(self.risk_distribution)

    def generate_risk_report(self, extracted_clauses: Dict[str, str]) -> Dict:
        """
        Generate comprehensive risk report

        Args:
            extracted_clauses: Dictionary of extracted clauses

        Returns:
            Comprehensive risk report
        """
        risk_score = self.calculate_risk_score(extracted_clauses)
        risk_level = self.get_risk_level(risk_score)
        business_impact = self.calculate_business_impact(risk_score)
        prioritized_clauses = self.prioritize_clauses(extracted_clauses)

        return {
            "overall_risk_score": risk_score,
            "risk_level": risk_level,
            "business_impact": business_impact,
            "prioritized_clauses": prioritized_clauses,
            "recommendations": self._generate_risk_recommendations(
                risk_level, prioritized_clauses
            ),
            "summary": {
                "total_clauses": len(extracted_clauses),
                "high_risk_clauses": len(
                    [
                        c
                        for c in prioritized_clauses
                        if c["risk_level"] in ["high", "critical"]
                    ]
                ),
                "estimated_review_time": business_impact["time_impact_hours"],
                "estimated_cost": business_impact["total_impact_usd"],
            },
        }

    def _generate_risk_recommendations(
        self, risk_level: str, prioritized_clauses: List[Dict]
    ) -> List[str]:
        """
        Generate risk-based recommendations

        Args:
            risk_level: Overall risk level
            prioritized_clauses: Prioritized clauses

        Returns:
            List of recommendations
        """
        recommendations = []

        # Overall risk recommendations
        if risk_level == "critical":
            recommendations.extend(
                [
                    "Immediate legal review required",
                    "Consider contract renegotiation",
                    "Implement additional safeguards",
                ]
            )
        elif risk_level == "high":
            recommendations.extend(
                [
                    "Schedule legal review within 1 week",
                    "Consider risk mitigation strategies",
                    "Review similar contracts for best practices",
                ]
            )

        # Clause-specific recommendations
        for clause in prioritized_clauses[:3]:  # Top 3 highest risk clauses
            if clause["risk_level"] in ["high", "critical"]:
                recommendations.append(
                    f"Review {clause['clause_type']} clause for modifications"
                )

        return recommendations
