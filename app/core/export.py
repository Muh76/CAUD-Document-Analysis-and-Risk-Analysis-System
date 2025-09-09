"""
Export functionality for contract analysis results.
"""

import csv
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from .schemas import ContractAnalysis

class ExportManager:
    """Manage export of contract analysis results."""

    def __init__(self, output_dir: str = "exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def export_to_csv(self, analysis: ContractAnalysis, filename: Optional[str] = None) -> str:
        """Export analysis results to CSV format."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"contract_analysis_{analysis.contract_id}_{timestamp}.csv"

        filepath = self.output_dir / filename

        # Prepare CSV data
        csv_data = []
        for result in analysis.results:
            csv_data.append({
                "contract_id": analysis.contract_id,
                "clause_id": result.clause_id,
                "risk_score": result.risk_score,
                "detected_labels": ", ".join(result.detected_labels or []),
                "top_probability": max(result.probs) if result.probs else 0.0,
                "start_offset": result.start_offset or 0,
                "end_offset": result.end_offset or 0,
                "page_number": result.page_number or 0,
                "text": result.text,
                "rationale": "; ".join(result.rationale or [])
            })

        # Write CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False)

        return str(filepath)

    def export_to_json(self, analysis: ContractAnalysis, filename: Optional[str] = None) -> str:
        """Export analysis results to JSON format."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"contract_analysis_{analysis.contract_id}_{timestamp}.json"

        filepath = self.output_dir / filename

        # Convert to dict
        analysis_dict = {
            "contract_id": analysis.contract_id,
            "total_clauses": analysis.total_clauses,
            "high_risk_clauses": analysis.high_risk_clauses,
            "medium_risk_clauses": analysis.medium_risk_clauses,
            "low_risk_clauses": analysis.low_risk_clauses,
            "overall_risk_score": analysis.overall_risk_score,
            "thresholds_used": analysis.thresholds_used,
            "model_snapshot": analysis.model_snapshot,
            "calibration_version": analysis.calibration_version,
            "latency_ms": analysis.latency_ms,
            "timestamp": analysis.timestamp.isoformat(),
            "results": [
                {
                    "clause_id": result.clause_id,
                    "text": result.text,
                    "probs": result.probs,
                    "risk_score": result.risk_score,
                    "start_offset": result.start_offset,
                    "end_offset": result.end_offset,
                    "page_number": result.page_number,
                    "rationale": result.rationale,
                    "detected_labels": result.detected_labels
                }
                for result in analysis.results
            ]
        }

        # Write JSON
        with open(filepath, 'w') as f:
            json.dump(analysis_dict, f, indent=2, default=str)

        return str(filepath)

    def export_portfolio_csv(self, analyses: List[ContractAnalysis], filename: Optional[str] = None) -> str:
        """Export multiple analyses to a single CSV file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_analysis_{timestamp}.csv"

        filepath = self.output_dir / filename

        # Prepare CSV data
        csv_data = []
        for analysis in analyses:
            for result in analysis.results:
                csv_data.append({
                    "contract_id": analysis.contract_id,
                    "clause_id": result.clause_id,
                    "risk_score": result.risk_score,
                    "detected_labels": ", ".join(result.detected_labels or []),
                    "top_probability": max(result.probs) if result.probs else 0.0,
                    "start_offset": result.start_offset or 0,
                    "end_offset": result.end_offset or 0,
                    "page_number": result.page_number or 0,
                    "text": result.text,
                    "rationale": "; ".join(result.rationale or []),
                    "overall_risk_score": analysis.overall_risk_score,
                    "analysis_timestamp": analysis.timestamp.isoformat()
                })

        # Write CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False)

        return str(filepath)

    def export_risk_report(self, analyses: List[ContractAnalysis], filename: Optional[str] = None) -> str:
        """Export risk report summary."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"risk_report_{timestamp}.json"

        filepath = self.output_dir / filename

        # Calculate portfolio metrics
        total_contracts = len(analyses)
        total_clauses = sum(a.total_clauses for a in analyses)
        total_high_risk = sum(a.high_risk_clauses for a in analyses)
        total_medium_risk = sum(a.medium_risk_clauses for a in analyses)
        total_low_risk = sum(a.low_risk_clauses for a in analyses)
        avg_risk_score = sum(a.overall_risk_score for a in analyses) / total_contracts if total_contracts > 0 else 0

        # Count red flags
        red_flags = {}
        for analysis in analyses:
            for result in analysis.results:
                for label in result.detected_labels or []:
                    red_flags[label] = red_flags.get(label, 0) + 1

        # Sort red flags by frequency
        top_red_flags = sorted(red_flags.items(), key=lambda x: x[1], reverse=True)[:10]

        # Generate report
        report = {
            "report_id": f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "portfolio_summary": {
                "total_contracts": total_contracts,
                "total_clauses": total_clauses,
                "high_risk_clauses": total_high_risk,
                "medium_risk_clauses": total_medium_risk,
                "low_risk_clauses": total_low_risk,
                "average_risk_score": avg_risk_score
            },
            "top_red_flags": [
                {"label": label, "count": count, "percentage": count / total_clauses * 100}
                for label, count in top_red_flags
            ],
            "contract_details": [
                {
                    "contract_id": analysis.contract_id,
                    "total_clauses": analysis.total_clauses,
                    "high_risk_clauses": analysis.high_risk_clauses,
                    "medium_risk_clauses": analysis.medium_risk_clauses,
                    "low_risk_clauses": analysis.low_risk_clauses,
                    "overall_risk_score": analysis.overall_risk_score,
                    "analysis_timestamp": analysis.timestamp.isoformat()
                }
                for analysis in analyses
            ]
        }

        # Write report
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return str(filepath)

    def get_export_formats(self) -> List[str]:
        """Get available export formats."""
        return ["csv", "json", "portfolio_csv", "risk_report"]

    def validate_export_data(self, analysis: ContractAnalysis) -> Dict[str, Any]:
        """Validate data before export."""
        issues = []

        if not analysis.results:
            issues.append("No analysis results to export")

        if not analysis.contract_id:
            issues.append("Missing contract ID")

        if analysis.total_clauses == 0:
            issues.append("No clauses found in analysis")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "total_clauses": analysis.total_clauses,
            "has_results": len(analysis.results) > 0
        }
