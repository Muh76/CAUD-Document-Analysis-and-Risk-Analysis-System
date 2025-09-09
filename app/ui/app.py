"""
Main Streamlit application for Contract Review & Risk Analysis.
"""

import streamlit as st
import requests
import json
from typing import Dict, Any, List
import time

from .components import ContractAnalysisComponents
from .state import UIState

# Page configuration
st.set_page_config(
    page_title="Contract Review & Risk Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "devtoken"

class ContractAnalysisApp:
    """Main application class."""

    def __init__(self):
        self.components = ContractAnalysisComponents()
        self.state = UIState()
        self.state.initialize_session_state()

    def run(self):
        """Run the application."""
        st.title("📄 Contract Review & Risk Analysis System")
        st.markdown("---")

        # Sidebar
        self.render_sidebar()

        # Main content
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📋 Clauses", "📁 Portfolio"])

        with tab1:
            self.render_overview_tab()

        with tab2:
            self.render_clauses_tab()

        with tab3:
            self.render_portfolio_tab()

    def render_sidebar(self):
        """Render sidebar with file upload and controls."""
        st.sidebar.header("📁 Contract Upload")

        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Choose a contract file",
            type=['txt', 'pdf'],
            help="Upload a contract file for analysis"
        )

        # Text input
        contract_text = st.sidebar.text_area(
            "Or paste contract text:",
            height=200,
            help="Paste contract text directly"
        )

        # Contract ID input
        contract_id = st.sidebar.text_input(
            "Contract ID:",
            value="contract_001",
            help="Unique identifier for this contract"
        )

        # Analyze button
        if st.sidebar.button("�� Analyze Contract", type="primary"):
            if uploaded_file or contract_text:
                self.analyze_contract(uploaded_file, contract_text, contract_id)
            else:
                st.sidebar.error("Please upload a file or paste text")

        # API status
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔗 API Status")
        if self.check_api_status():
            st.sidebar.success("✅ API Connected")
        else:
            st.sidebar.error("❌ API Disconnected")

    def render_overview_tab(self):
        """Render overview tab."""
        st.header("📊 Contract Analysis Overview")

        current_analysis = self.state.get_current_analysis()

        if current_analysis:
            # Risk summary
            self.components.display_risk_summary(current_analysis)

            st.markdown("---")

            # Risk chart
            self.components.display_risk_chart(current_analysis)

            st.markdown("---")

            # Export options
            self.components.display_export_options(current_analysis)

            # Add to portfolio button
            if st.button("📁 Add to Portfolio"):
                self.state.add_to_portfolio(
                    current_analysis.get("contract_id", ""),
                    current_analysis
                )
                st.success("Contract added to portfolio!")
        else:
            st.info("👆 Upload a contract to see analysis overview")

    def render_clauses_tab(self):
        """Render clauses tab."""
        st.header("📋 Clause Analysis")

        current_analysis = self.state.get_current_analysis()

        if current_analysis:
            # Clauses table
            self.components.display_clauses_table(current_analysis.get("results", []))

            st.markdown("---")

            # Clause selection
            clause_ids = [r.get("clause_id", 0) for r in current_analysis.get("results", [])]
            if clause_ids:
                selected_clause = st.selectbox(
                    "Select clause for detailed analysis:",
                    clause_ids,
                    format_func=lambda x: f"Clause {x}"
                )

                if selected_clause is not None:
                    # Find selected clause result
                    selected_result = next(
                        (r for r in current_analysis.get("results", []) if r.get("clause_id") == selected_clause),
                        None
                    )

                    if selected_result:
                        st.markdown("---")
                        self.components.display_clause_details(selected_result)
        else:
            st.info("👆 Upload a contract to see clause analysis")

    def render_portfolio_tab(self):
        """Render portfolio tab."""
        st.header("📁 Portfolio Analysis")

        portfolio = self.state.get_portfolio()

        if portfolio:
            # Portfolio summary
            self.components.display_portfolio_summary(portfolio)

            st.markdown("---")

            # Portfolio management
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📊 Export Portfolio CSV"):
                    csv_data = self.state.export_portfolio_csv()
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="portfolio_analysis.csv",
                        mime="text/csv"
                    )

            with col2:
                if st.button("📄 Export Portfolio JSON"):
                    json_data = self.state.export_portfolio_json()
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name="portfolio_analysis.json",
                        mime="application/json"
                    )

            with col3:
                if st.button("🗑️ Clear Portfolio"):
                    self.state.clear_portfolio()
                    st.rerun()

            # Individual contract management
            st.markdown("---")
            st.subheader("Individual Contracts")

            for i, contract in enumerate(portfolio):
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.write(f"**{contract.get('contract_id', f'Contract {i}')}** - Risk: {contract.get('overall_risk_score', 0):.3f}")

                with col2:
                    if st.button(f"View", key=f"view_{i}"):
                        self.state.set_analysis_result(
                            contract.get("contract_id", ""),
                            contract
                        )
                        st.rerun()

                with col3:
                    if st.button(f"Remove", key=f"remove_{i}"):
                        self.state.remove_from_portfolio(contract.get("contract_id", ""))
                        st.rerun()
        else:
            st.info("📁 No contracts in portfolio. Add contracts from the Overview tab.")

    def analyze_contract(self, uploaded_file, contract_text: str, contract_id: str):
        """Analyze contract using API."""
        with st.spinner("🔍 Analyzing contract..."):
            try:
                # Prepare request data
                if uploaded_file:
                    # Read file content
                    file_content = uploaded_file.read()
                    if uploaded_file.type == "application/pdf":
                        # For PDF, encode as base64
                        import base64
                        file_b64 = base64.b64encode(file_content).decode('utf-8')
                        request_data = {
                            "contract_id": contract_id,
                            "file_b64": file_b64,
                            "mime": "application/pdf"
                        }
                    else:
                        # For text files
                        text_content = file_content.decode('utf-8')
                        request_data = {
                            "contract_id": contract_id,
                            "text": text_content
                        }
                else:
                    # Use text input
                    request_data = {
                        "contract_id": contract_id,
                        "text": contract_text
                    }

                # Make API request
                response = requests.post(
                    f"{API_BASE_URL}/analyze_contract",
                    json=request_data,
                    headers={"Authorization": f"Bearer {API_TOKEN}"},
                    timeout=30
                )

                if response.status_code == 200:
                    analysis_data = response.json()
                    self.state.set_analysis_result(contract_id, analysis_data)
                    st.success("✅ Contract analyzed successfully!")
                else:
                    st.error(f"❌ Analysis failed: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the API server is running.")
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

    def check_api_status(self) -> bool:
        """Check if API is available."""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

# Run the app
if __name__ == "__main__":
    app = ContractAnalysisApp()
    app.run()
