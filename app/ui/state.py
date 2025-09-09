"""
State management for Streamlit UI.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import json

class UIState:
    """Manage UI state across pages."""

    @staticmethod
    def initialize_session_state():
        """Initialize session state variables."""
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = {}

        if "portfolio_contracts" not in st.session_state:
            st.session_state.portfolio_contracts = []

        if "current_contract_id" not in st.session_state:
            st.session_state.current_contract_id = None

        if "selected_clause_id" not in st.session_state:
            st.session_state.selected_clause_id = None

    @staticmethod
    def set_analysis_result(contract_id: str, analysis_data: Dict[str, Any]):
        """Store analysis result in session state."""
        st.session_state.analysis_results[contract_id] = analysis_data
        st.session_state.current_contract_id = contract_id

    @staticmethod
    def get_analysis_result(contract_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis result from session state."""
        return st.session_state.analysis_results.get(contract_id)

    @staticmethod
    def get_current_analysis() -> Optional[Dict[str, Any]]:
        """Get current contract analysis."""
        if st.session_state.current_contract_id:
            return st.session_state.get_analysis_result(st.session_state.current_contract_id)
        return None

    @staticmethod
    def add_to_portfolio(contract_id: str, analysis_data: Dict[str, Any]):
        """Add contract to portfolio."""
        if contract_id not in [c.get("contract_id") for c in st.session_state.portfolio_contracts]:
            st.session_state.portfolio_contracts.append(analysis_data)

    @staticmethod
    def remove_from_portfolio(contract_id: str):
        """Remove contract from portfolio."""
        st.session_state.portfolio_contracts = [
            c for c in st.session_state.portfolio_contracts 
            if c.get("contract_id") != contract_id
        ]

    @staticmethod
    def get_portfolio() -> List[Dict[str, Any]]:
        """Get portfolio contracts."""
        return st.session_state.portfolio_contracts

    @staticmethod
    def clear_portfolio():
        """Clear portfolio."""
        st.session_state.portfolio_contracts = []

    @staticmethod
    def set_selected_clause(clause_id: int):
        """Set selected clause for detailed view."""
        st.session_state.selected_clause_id = clause_id

    @staticmethod
    def get_selected_clause() -> Optional[int]:
        """Get selected clause ID."""
        return st.session_state.selected_clause_id

    @staticmethod
    def clear_selected_clause():
        """Clear selected clause."""
        st.session_state.selected_clause_id = None

    @staticmethod
    def export_portfolio_csv() -> str:
        """Export portfolio as CSV."""
        import pandas as pd

        if not st.session_state.portfolio_contracts:
            return ""

        # Flatten portfolio data for CSV
        csv_data = []
        for contract in st.session_state.portfolio_contracts:
            for result in contract.get("results", []):
                csv_data.append({
                    "contract_id": contract.get("contract_id", ""),
                    "clause_id": result.get("clause_id", 0),
                    "risk_score": result.get("risk", 0),
                    "detected_labels": ", ".join(result.get("detected_labels", [])),
                    "top_probability": max(result.get("probs", [0])),
                    "text": result.get("text", "")
                })

        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)

    @staticmethod
    def export_portfolio_json() -> str:
        """Export portfolio as JSON."""
        return json.dumps(st.session_state.portfolio_contracts, indent=2, default=str)
