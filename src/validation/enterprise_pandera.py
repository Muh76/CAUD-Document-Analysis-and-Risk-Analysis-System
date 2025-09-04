"""
Enterprise-Ready Pandera Validation
Enhanced data quality checks for production use
"""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class EnterpriseDataValidator:
    """Enterprise-ready data validation with Pandera"""
    
    def __init__(self):
        self.schemas = self._create_enterprise_schemas()
    
    def _create_enterprise_schemas(self) -> Dict[str, DataFrameSchema]:
        """Create enterprise-grade validation schemas"""
        
        # Contracts Schema - Strict validation (updated to match actual data)
        contracts_schema = DataFrameSchema({
            # Column presence & dtypes (strict schema)
            "contract_id": Column(str, nullable=False, unique=True),
            "file_name": Column(str, nullable=False),
            "contract_type": Column(str, nullable=False),
            "parties": Column(str, nullable=False),
            "effective_date": Column(str, nullable=True),
            "jurisdiction": Column(str, nullable=True),
            "total_clauses": Column(int, nullable=False),
            "file_size": Column(int, nullable=False),
            "parsed_ok": Column(int, nullable=False),
            
            # Value ranges
            "total_clauses": Column(int, Check.in_range(5, 1000), nullable=False),
            "file_size": Column(int, Check.greater_than(0), nullable=False),
            "parsed_ok": Column(int, Check.isin([0, 1]), nullable=False),
            
            # Regex checks
            "file_name": Column(str, Check.str_matches(r"^[a-zA-Z0-9_\-\.]+$"), nullable=False),
        })
        
        # Clauses Schema - Strict validation (updated to match actual data)
        clauses_schema = DataFrameSchema({
            # Column presence & dtypes
            "clause_id": Column(str, nullable=False, unique=True),
            "contract_id": Column(str, nullable=False),
            "text": Column(str, nullable=False),
            "clause_type": Column(str, nullable=False),
            "confidence": Column(float, nullable=False),
            "text_length": Column(int, nullable=False),
            
            # Value ranges
            "text": Column(str, Check.str_length(min_value=10, max_value=5000), nullable=False),
            "confidence": Column(float, Check.in_range(0.0, 1.0), nullable=False),
            "text_length": Column(int, Check.greater_than(0), nullable=False),
            
            # Allowed sets
            "clause_type": Column(str, Check.isin([
                "governing_law", "liability", "confidentiality", "termination",
                "ip_assignment", "non_compete", "audit_rights", "force_majeure",
                "warranty", "indemnification", "dispute_resolution", "other"
            ]), nullable=False),
        })
        
        return {
            "contracts": contracts_schema,
            "clauses": clauses_schema
        }
    
    def validate_contracts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate contracts data with enterprise checks"""
        try:
            self.schemas["contracts"].validate(df)
            return {"passed": True, "errors": []}
        except Exception as e:
            return {"passed": False, "errors": [str(e)]}
    
    def validate_clauses(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate clauses data with enterprise checks"""
        try:
            self.schemas["clauses"].validate(df)
            return {"passed": True, "errors": []}
        except Exception as e:
            return {"passed": False, "errors": [str(e)]}
    
    def run_enterprise_validation(self, data_dir: str) -> Dict[str, Any]:
        """Run comprehensive enterprise validation"""
        results = {}
        
        # Load and validate contracts
        try:
            contracts_df = pd.read_csv(f"{data_dir}/contracts_index.csv")
            results["contracts"] = self.validate_contracts(contracts_df)
        except Exception as e:
            results["contracts"] = {"passed": False, "errors": [f"Failed to load contracts: {e}"]}
        
        # Load and validate clauses
        try:
            clauses_df = pd.read_csv(f"{data_dir}/clause_spans.csv")
            results["clauses"] = self.validate_clauses(clauses_df)
        except Exception as e:
            results["clauses"] = {"passed": False, "errors": [f"Failed to load clauses: {e}"]}
        
        # Overall result
        all_passed = all(result["passed"] for result in results.values())
        results["overall_passed"] = all_passed
        
        return results
