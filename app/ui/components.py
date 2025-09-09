"""
Reusable Streamlit components for the Contract Analysis UI.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import json

class ContractAnalysisComponents:
    """Reusable components for contract analysis UI."""

    @staticmethod
    def display_risk_summary(analysis_data: Dict[str, Any]):
        """Display risk summary metrics."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total Clauses",
                value=analysis_data.get("total_clauses", 0)
            )

        with col2:
            st.metric(
                label="High Risk",
                value=analysis_data.get("high_risk_clauses", 0),
                delta=f"{analysis_data.get('high_risk_clauses', 0) / max(analysis_data.get('total_clauses', 1), 1) * 100:.1f}%"
            )

        with col3:
            st.metric(
                label="Medium Risk",
                value=analysis_data.get("medium_risk_clauses", 0),
                delta=f"{analysis_data.get('medium_risk_clauses', 0) / max(analysis_data.get('total_clauses', 1), 1) * 100:.1f}%"
            )

        with col4:
            st.metric(
                label="Overall Risk",
                value=f"{analysis_data.get('overall_risk_score', 0):.2f}",
                delta="Score"
            )

    @staticmethod
    def display_risk_chart(analysis_data: Dict[str, Any]):
        """Display risk distribution chart."""
        risk_data = {
            "Risk Level": ["High Risk", "Medium Risk", "Low Risk"],
            "Count": [
                analysis_data.get("high_risk_clauses", 0),
                analysis_data.get("medium_risk_clauses", 0),
                analysis_data.get("low_risk_clauses", 0)
            ]
        }

        df = pd.DataFrame(risk_data)

        fig = px.bar(
            df, 
            x="Risk Level", 
            y="Count",
            color="Risk Level",
            color_discrete_map={
                "High Risk": "#ff4444",
                "Medium Risk": "#ffaa00",
                "Low Risk": "#44ff44"
            },
            title="Risk Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def display_clauses_table(results: List[Dict[str, Any]]):
        """Display clauses analysis table."""
        if not results:
            st.info("No clauses to display")
            return

        # Prepare data for table
        table_data = []
        for result in results:
            table_data.append({
                "Clause ID": result.get("clause_id", 0),
                "Risk Score": f"{result.get('risk', 0):.3f}",
                "Top Label": result.get("detected_labels", [""])[0] if result.get("detected_labels") else "",
                "Confidence": f"{max(result.get('probs', [0])):.3f}",
                "Snippet": result.get("snippet", "")[:100] + "..." if len(result.get("snippet", "")) > 100 else result.get("snippet", "")
            })

        df = pd.DataFrame(table_data)

        # Add risk-based coloring
        def highlight_risk(row):
            risk = float(row["Risk Score"])
            if risk >= 0.5:
                return ["background-color: #ffcccc"] * len(row)
            elif risk >= 0.3:
                return ["background-color: #fff2cc"] * len(row)
            else:
                return ["background-color: #ccffcc"] * len(row)

        styled_df = df.style.apply(highlight_risk, axis=1)
        st.dataframe(styled_df, use_container_width=True)

    @staticmethod
    def display_clause_details(result: Dict[str, Any]):
        """Display detailed clause analysis."""
        st.subheader(f"Clause {result.get('clause_id', 0)} Analysis")

        # Risk score with color coding
        risk_score = result.get("risk", 0)
        if risk_score >= 0.5:
            risk_color = "🔴"
            risk_level = "High Risk"
        elif risk_score >= 0.3:
            risk_color = "🟡"
            risk_level = "Medium Risk"
        else:
            risk_color = "🟢"
            risk_level = "Low Risk"

        st.write(f"**Risk Level:** {risk_color} {risk_level} ({risk_score:.3f})")

        # Detected labels
        detected_labels = result.get("detected_labels", [])
        if detected_labels:
            st.write("**Detected Labels:**")
            for label in detected_labels:
                st.write(f"- {label}")

        # Probabilities
        probs = result.get("probs", [])
        if probs:
            st.write("**Top Probabilities:**")
            # Get top 5 probabilities
            top_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:5]
            for i, prob in top_probs:
                st.write(f"- Label {i}: {prob:.3f}")

        # Rationale
        rationale = result.get("rationale", [])
        if rationale:
            st.write("**Rationale:**")
            for reason in rationale:
                st.write(f"- {reason}")

        # Full text
        st.write("**Full Text:**")
        st.text_area("", result.get("text", ""), height=200, disabled=True)

    @staticmethod
    def display_export_options(analysis_data: Dict[str, Any]):
        """Display export options."""
        st.subheader("Export Options")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Export as CSV"):
                # Convert to CSV format
                csv_data = []
                for result in analysis_data.get("results", []):
                    csv_data.append({
                        "clause_id": result.get("clause_id", 0),
                        "risk_score": result.get("risk", 0),
                        "detected_labels": ", ".join(result.get("detected_labels", [])),
                        "top_probability": max(result.get("probs", [0])),
                        "text": result.get("text", "")
                    })

                df = pd.DataFrame(csv_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"contract_analysis_{analysis_data.get('contract_id', 'unknown')}.csv",
                    mime="text/csv"
                )

        with col2:
            if st.button("📄 Export as JSON"):
                json_data = json.dumps(analysis_data, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"contract_analysis_{analysis_data.get('contract_id', 'unknown')}.json",
                    mime="application/json"
                )

    @staticmethod
    def display_portfolio_summary(portfolio_data: List[Dict[str, Any]]):
        """Display portfolio analysis summary."""
        if not portfolio_data:
            st.info("No contracts in portfolio")
            return

        # Calculate portfolio metrics
        total_contracts = len(portfolio_data)
        total_clauses = sum(contract.get("total_clauses", 0) for contract in portfolio_data)
        avg_risk = sum(contract.get("overall_risk_score", 0) for contract in portfolio_data) / total_contracts

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Contracts", total_contracts)

        with col2:
            st.metric("Total Clauses", total_clauses)

        with col3:
            st.metric("Average Risk", f"{avg_risk:.3f}")

        # Portfolio chart
        portfolio_df = pd.DataFrame([
            {
                "Contract": contract.get("contract_id", f"Contract {i}"),
                "Risk Score": contract.get("overall_risk_score", 0),
                "Clauses": contract.get("total_clauses", 0)
            }
            for i, contract in enumerate(portfolio_data)
        ])

        fig = px.scatter(
            portfolio_df,
            x="Clauses",
            y="Risk Score",
            size="Clauses",
            hover_name="Contract",
            title="Portfolio Risk Analysis"
        )

        st.plotly_chart(fig, use_container_width=True)
