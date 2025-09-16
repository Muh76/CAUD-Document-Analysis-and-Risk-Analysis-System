"""
Contract Review & Risk Analysis System - Streamlit App
Self-contained version for Streamlit Share deployment.
"""

import streamlit as st
import requests
import json
import base64
from typing import Dict, Any, List
import time
import pandas as pd

# Try to import plotly, fallback to basic charts if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available, using basic charts")

# Advanced Visualization Components (from Phase 3)
class AdvancedVisualizations:
    """Advanced visualization components inspired by Phase 3."""
    
    @staticmethod
    def display_portfolio_metrics(contracts_data: List[Dict[str, Any]]):
        """Display portfolio-level metrics."""
        if not contracts_data:
            st.info("No contract data available for portfolio metrics")
            return
            
        total_contracts = len(contracts_data)
        total_clauses = sum(contract.get("total_clauses", 0) for contract in contracts_data)
        avg_risk = sum(contract.get("overall_risk_score", 0) for contract in contracts_data) / total_contracts
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Contracts",
                value=total_contracts,
                delta=None
            )
        
        with col2:
            st.metric(
                label="Total Clauses", 
                value=total_clauses,
                delta=f"{total_clauses/total_contracts:.1f} avg/contract"
            )
        
        with col3:
            st.metric(
                label="Average Risk",
                value=f"{avg_risk:.3f}",
                delta="Portfolio Risk"
            )
            
        with col4:
            high_risk_contracts = len([c for c in contracts_data if c.get("overall_risk_score", 0) > 0.3])
            st.metric(
                label="High Risk Contracts",
                value=high_risk_contracts,
                delta=f"{high_risk_contracts/total_contracts*100:.1f}%"
            )
    
    @staticmethod
    def display_portfolio_scatter(contracts_data: List[Dict[str, Any]]):
        """Display portfolio risk scatter plot."""
        if not contracts_data:
            return
            
        portfolio_df = pd.DataFrame([
            {
                "Contract": contract.get("contract_id", f"Contract {i}"),
                "Risk Score": contract.get("overall_risk_score", 0),
                "Clauses": contract.get("total_clauses", 0),
                "High Risk": contract.get("high_risk_clauses", 0)
            }
            for i, contract in enumerate(contracts_data)
        ])
        
        if PLOTLY_AVAILABLE:
            fig = px.scatter(
                portfolio_df,
                x="Clauses",
                y="Risk Score",
                size="High Risk",
                hover_name="Contract",
                title="Portfolio Risk Analysis: Risk Score vs Clauses",
                labels={
                    "Risk Score": "Overall Risk Score",
                    "Clauses": "Number of Clauses",
                    "High Risk": "High Risk Clauses"
                },
                color="Risk Score",
                color_continuous_scale="RdYlGn_r"  # Red to Green (high risk = red)
            )
            fig.update_layout(
                xaxis_title="Number of Clauses",
                yaxis_title="Risk Score (0-1)",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.scatter_chart(portfolio_df[["Clauses", "Risk Score"]])
    
    @staticmethod
    def display_risk_trend_chart(historical_data: List[Dict[str, Any]]):
        """Display risk trend over time."""
        if not historical_data:
            return
            
        if PLOTLY_AVAILABLE:
            fig = px.line(
                historical_data,
                x="date",
                y="avg_risk_score",
                title="Portfolio Risk Trend Over Time",
                labels={"date": "Date", "avg_risk_score": "Average Risk Score"},
                markers=True
            )
            fig.update_layout(
                yaxis_range=[0, 1],
                xaxis_title="Date",
                yaxis_title="Average Risk Score"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def display_enhanced_risk_distribution(risk_data: Dict[str, int]):
        """Display enhanced risk distribution with Phase 3 styling."""
        if PLOTLY_AVAILABLE:
            fig = px.bar(
                x=list(risk_data.keys()),
                y=list(risk_data.values()),
                color=list(risk_data.keys()),
                color_discrete_map={
                    "High Risk": "#ff4444",
                    "Medium Risk": "#ffaa00", 
                    "Low Risk": "#44ff44"
                },
                title="Contract Risk Distribution",
                labels={'x': 'Risk Level', 'y': 'Number of Contracts'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# Page configuration
st.set_page_config(
    page_title="Contract Review & Risk Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "https://contract-analysis-api-77455288936.us-central1.run.app"

def get_api_health():
    """Check API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"API connection failed: {str(e)}")
        return None

def analyze_contract(text: str) -> Dict[str, Any]:
    """Analyze contract text using the API."""
    try:
        payload = {
            "contract_id": f"text_{hash(text) % 10000}",  # Generate a unique contract ID
            "text": text,
            "include_confidence": True,
            "include_explanations": True
        }
        
        # Try with authentication first, then without
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer demo-token"  # Try with token first
        }
        
        response = requests.post(
            f"{API_BASE_URL}/analyze_contract",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        # If 401, try without authentication
        if response.status_code == 401:
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                f"{API_BASE_URL}/analyze_contract",
                json=payload,
                headers=headers,
                timeout=30
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("📄 Contract Review & Risk Analysis System")
    st.markdown("AI-powered contract analysis and risk assessment")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # API Health Check
        health_data = get_api_health()
        if health_data:
            st.success("✅ API Connected")
            st.info(f"**Status:** {health_data.get('status', 'Unknown')}")
            st.info(f"**Model:** {health_data.get('model_snapshot', 'Unknown')}")
            st.info(f"**Uptime:** {health_data.get('uptime_seconds', 0):.1f}s")
        else:
            st.error("❌ API Disconnected")
            st.warning("Please check the API connection")
        
        st.divider()
        
        # Navigation
        st.header("📋 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["Contract Analysis", "Batch Analysis", "Portfolio Analysis", "Risk Reports", "System Info"]
        )
    
    # Main content based on selected page
    if page == "Contract Analysis":
        contract_analysis_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
    elif page == "Portfolio Analysis":
        portfolio_analysis_page()
    elif page == "Risk Reports":
        risk_reports_page()
    elif page == "System Info":
        system_info_page()

def contract_analysis_page():
    """Contract analysis page."""
    st.header("📝 Single Contract Analysis")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "File Upload"],
        horizontal=True
    )
    
    contract_text = ""
    
    if input_method == "Text Input":
        contract_text = st.text_area(
            "Enter contract text:",
            height=300,
            placeholder="Paste your contract text here..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload contract file",
            type=['txt', 'pdf'],
            help="Upload a text or PDF file containing the contract. PDF processing may take longer for large files."
        )
        
        if uploaded_file:
            if uploaded_file.type == "text/plain":
                contract_text = str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "application/pdf":
                # Handle PDF files
                st.info(f"📄 PDF file uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
                st.info("🔄 PDF processing may take a moment...")
                
                file_bytes = uploaded_file.read()
                file_b64 = base64.b64encode(file_bytes).decode('utf-8')
                
                # Send PDF to API for analysis
                payload = {
                    "contract_id": f"pdf_{uploaded_file.name}",
                    "file_b64": file_b64,
                    "mime": "application/pdf"
                }
                
                # Try with authentication first, then without
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer demo-token"
                }
                
                response = requests.post(
                    f"{API_BASE_URL}/analyze_contract",
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                
                # If 401, try without authentication
                if response.status_code == 401:
                    headers = {"Content-Type": "application/json"}
                    response = requests.post(
                        f"{API_BASE_URL}/analyze_contract",
                        json=payload,
                        headers=headers,
                        timeout=30
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    display_analysis_results(result)
                    return
                else:
                    st.error(f"PDF analysis failed: {response.status_code} - {response.text}")
                    return
            else:
                st.warning(f"Unsupported file type: {uploaded_file.type}")
                return
    
    # Analysis button
    if st.button("🔍 Analyze Contract", type="primary"):
        if not contract_text.strip():
            st.warning("Please enter some contract text to analyze")
            return
        
        with st.spinner("Analyzing contract..."):
            result = analyze_contract(contract_text)
        
        if "error" in result:
            st.error(f"Analysis failed: {result['error']}")
            return
        
        # Display results
        display_analysis_results(result)

def display_analysis_results(result: Dict[str, Any]):
    """Display contract analysis results."""
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Convert overall_risk_score (0-1) to 0-10 scale
        risk_score = result.get('overall_risk_score', 0) * 10
        st.metric("Risk Score", f"{risk_score:.1f}/10")
    
    with col2:
        # Calculate average confidence from results
        results = result.get('results', [])
        if results:
            avg_confidence = sum(r.get('risk', 0) for r in results) / len(results)
            st.metric("Confidence", f"{avg_confidence:.1%}")
        else:
            st.metric("Confidence", "N/A")
    
    with col3:
        total_clauses = result.get('total_clauses', 0)
        st.metric("Clauses Found", total_clauses)
    
    with col4:
        high_risk_clauses = result.get('high_risk_clauses', 0)
        st.metric("High Risk Issues", high_risk_clauses)
    
    st.divider()
    
    # Check if we have any results
    results = result.get('results', [])
    if not results:
        st.warning("⚠️ **No clauses detected with sufficient confidence**")
        st.info("The contract may not contain recognizable legal clauses, or the text format may not be suitable for analysis.")
        
        # Show raw text preview for debugging
        if 'text' in result:
            st.subheader("📄 Text Preview")
            preview_text = result['text'][:500] + "..." if len(result['text']) > 500 else result['text']
            st.text_area("First 500 characters:", preview_text, height=100, disabled=True)
        
        return
    
    # Detailed results
    if results:
        st.subheader("📋 Clause Analysis")
        
        # Create DataFrame for display
        clauses_data = []
        for i, clause_result in enumerate(results):
            # Determine risk level based on risk score
            risk_score = clause_result.get('risk', 0)
            if risk_score > 0.3:
                risk_level = 'high'
            elif risk_score > 0.1:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            clauses_data.append({
                'Clause': clause_result.get('snippet', '')[:100] + '...' if len(clause_result.get('snippet', '')) > 100 else clause_result.get('snippet', ''),
                'Type': 'Contract Clause',
                'Risk Level': risk_level,
                'Confidence': f"{risk_score:.1%}",
                'Explanation': ', '.join(clause_result.get('rationale', ['No explanation available']))
            })
        
        df = pd.DataFrame(clauses_data)
        st.dataframe(df, use_container_width=True)
        
        # Enhanced Risk distribution chart with Phase 3 styling
        if len(clauses_data) > 0:
            st.subheader("📊 Risk Distribution")
            risk_counts = df['Risk Level'].value_counts()
            
            if PLOTLY_AVAILABLE:
                # Enhanced pie chart with better styling
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title=f"Risk Level Distribution (N={len(clauses_data)} clauses)",
                    color_discrete_map={
                        'low': '#44ff44',
                        'medium': '#ffaa00', 
                        'high': '#ff4444'
                    },
                    hole=0.3  # Donut chart for modern look
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    font=dict(size=12),
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add bar chart for better comparison
                fig_bar = px.bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    color=risk_counts.index,
                    color_discrete_map={
                        'low': '#44ff44',
                        'medium': '#ffaa00', 
                        'high': '#ff4444'
                    },
                    title="Risk Level Count Comparison",
                    labels={'x': 'Risk Level', 'y': 'Number of Clauses'}
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                # Fallback to basic chart
                st.bar_chart(risk_counts)
    
    # Recommendations
    if 'recommendations' in result and result['recommendations']:
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(result['recommendations'], 1):
            st.info(f"**{i}.** {rec}")

def batch_analysis_page():
    """Batch analysis page."""
    st.header("📚 Batch Contract Analysis")
    st.info("Upload multiple contracts for batch analysis")
    
    uploaded_files = st.file_uploader(
        "Upload multiple contract files",
        type=['txt', 'pdf'],
        accept_multiple_files=True,
        help="Upload multiple text or PDF files for batch processing"
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} files uploaded")
        
        if st.button("🔄 Process Batch", type="primary"):
            with st.spinner("Processing batch analysis..."):
                try:
                    # Prepare contracts for batch processing
                    contracts = []
                    for file in uploaded_files:
                        if file.type == "text/plain":
                            text = str(file.read(), "utf-8")
                            contracts.append({
                                "contract_id": f"batch_{file.name}",
                                "text": text
                            })
                        elif file.type == "application/pdf":
                            file_bytes = file.read()
                            file_b64 = base64.b64encode(file_bytes).decode('utf-8')
                            contracts.append({
                                "contract_id": f"batch_{file.name}",
                                "file_b64": file_b64,
                                "mime": "application/pdf"
                            })
                    
                    # Send batch request
                    payload = {
                        "contracts": contracts
                    }
                    
                    headers = {"Content-Type": "application/json"}
                    response = requests.post(
                        f"{API_BASE_URL}/batch_analyze",
                        json=payload,
                        headers=headers,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        batch_response = response.json()
                        job_id = batch_response["job_id"]
                        
                        # Poll for results
                        st.info(f"Batch job started: {job_id}")
                        
                        # Wait for completion
                        max_attempts = 30
                        for attempt in range(max_attempts):
                            status_response = requests.get(
                                f"{API_BASE_URL}/batch_analyze/{job_id}",
                                headers=headers,
                                timeout=30
                            )
                            
                            if status_response.status_code == 200:
                                status_data = status_response.json()
                                
                                if status_data["status"] == "completed":
                                    # Display results
                                    display_batch_results(status_data["results"])
                                    break
                                elif status_data["status"] == "failed":
                                    st.error(f"Batch processing failed: {status_data.get('errors', ['Unknown error'])}")
                                    break
                                else:
                                    # Still processing
                                    progress = status_data["completed"] / status_data["total_contracts"]
                                    st.progress(progress)
                                    st.write(f"Processing... {status_data['completed']}/{status_data['total_contracts']} contracts")
                                    time.sleep(2)
                            else:
                                st.error(f"Failed to check batch status: {status_response.status_code}")
                                break
                        else:
                            st.warning("Batch processing is taking longer than expected. Please check back later.")
                    else:
                        st.error(f"Batch analysis failed: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    st.error(f"Batch processing error: {str(e)}")

def extract_confidence_from_rationale(rationale_list):
    """Extract average confidence from rationale list."""
    if not rationale_list:
        return 0.0
    
    confidences = []
    for rationale in rationale_list:
        if "confidence" in rationale.lower():
            # Extract confidence value from "High confidence (0.99)" format
            import re
            match = re.search(r'\(([0-9.]+)\)', rationale)
            if match:
                confidences.append(float(match.group(1)))
    
    return sum(confidences) / len(confidences) if confidences else 0.0

def display_batch_results(results):
    """Display batch analysis results."""
    st.subheader("📊 Batch Analysis Results")
    
    # Create results table
    results_data = []
    for result in results:
        # Extract confidence from rationale
        confidence = 0.0
        if result.get('results'):
            all_rationale = []
            for r in result['results']:
                all_rationale.extend(r.get('rationale', []))
            confidence = extract_confidence_from_rationale(all_rationale)
        
        results_data.append({
            "File": result["contract_id"],
            "Risk Score": f"{result['overall_risk_score'] * 10:.1f}/10",
            "Confidence": f"{confidence * 100:.1f}%",
            "Clauses": result["total_clauses"],
            "High Risk": result["high_risk_clauses"]
        })
    
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_risk = sum(r["overall_risk_score"] for r in results) / len(results) * 10
        st.metric("Average Risk Score", f"{avg_risk:.1f}/10")
    
    with col2:
        total_clauses = sum(r["total_clauses"] for r in results)
        st.metric("Total Clauses", total_clauses)
    
    with col3:
        total_high_risk = sum(r["high_risk_clauses"] for r in results)
        st.metric("Total High Risk", total_high_risk)
    
    with col4:
        completed_contracts = len([r for r in results if r.get("status") == "completed"])
        st.metric("Completed", f"{completed_contracts}/{len(results)}")
    
    # Risk distribution chart
    if results:
        st.subheader("📈 Risk Distribution")
        risk_scores = [r["overall_risk_score"] * 10 for r in results]
        
        if PLOTLY_AVAILABLE:
            fig = px.histogram(
                x=risk_scores,
                nbins=10,
                title="Risk Score Distribution",
                labels={"x": "Risk Score", "y": "Number of Contracts"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(pd.Series(risk_scores).value_counts().sort_index())

def generate_risk_report(contract_ids, include_suggestions=False):
    """Generate risk report from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/risk_report",
            json={
                "contract_ids": contract_ids,
                "include_suggestions": include_suggestions
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def display_risk_report(report_data):
    """Display risk report results."""
    st.success("✅ Risk report generated successfully!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Contracts", report_data['total_contracts'])
    
    with col2:
        st.metric("High Risk", report_data['high_risk_count'], delta=None)
    
    with col3:
        st.metric("Medium Risk", report_data['medium_risk_count'], delta=None)
    
    with col4:
        st.metric("Low Risk", report_data['low_risk_count'], delta=None)
    
    # Enhanced Risk distribution with Phase 3 styling
    st.subheader("📊 Risk Distribution")
    
    risk_data = {
        'High Risk': report_data['high_risk_count'],
        'Medium Risk': report_data['medium_risk_count'],
        'Low Risk': report_data['low_risk_count']
    }
    
    if PLOTLY_AVAILABLE:
        # Enhanced pie chart with donut style
        fig = px.pie(
            values=list(risk_data.values()),
            names=list(risk_data.keys()),
            title="Contract Risk Distribution",
            color_discrete_map={
                "High Risk": "#ff4444",
                "Medium Risk": "#ffaa00", 
                "Low Risk": "#44ff44"
            },
            hole=0.3  # Donut chart
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            font=dict(size=12),
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add bar chart for comparison
        fig_bar = px.bar(
            x=list(risk_data.keys()),
            y=list(risk_data.values()),
            color=list(risk_data.keys()),
            color_discrete_map={
                "High Risk": "#ff4444",
                "Medium Risk": "#ffaa00", 
                "Low Risk": "#44ff44"
            },
            title="Risk Level Count Comparison",
            labels={'x': 'Risk Level', 'y': 'Number of Contracts'}
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        # Fallback to basic chart
        st.bar_chart(pd.Series(risk_data))
    
    # Top red flags
    if report_data['top_red_flags']:
        st.subheader("🚨 Top Red Flags")
        
        flags_df = pd.DataFrame(report_data['top_red_flags'])
        st.dataframe(flags_df, use_container_width=True)
        
        # Red flags chart
        if PLOTLY_AVAILABLE:
            fig = px.bar(
                flags_df,
                x='label',
                y='count',
                color='risk_level',
                title="Top Risk Issues",
                labels={'x': 'Issue Type', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(flags_df.set_index('label')['count'])
    
    # Missing clauses
    if report_data['missing_clauses']:
        st.subheader("❌ Missing Clauses")
        missing_df = pd.DataFrame({
            'Missing Clause': report_data['missing_clauses']
        })
        st.dataframe(missing_df, use_container_width=True)
    
    # Recommendations
    if report_data.get('recommendations'):
        st.subheader("💡 Recommendations")
        
        for rec in report_data['recommendations']:
            with st.expander(f"{rec['clause']} - {rec['risk_level'].title()} Risk"):
                st.write(rec['suggestion'])
    
    # Report metadata
    st.subheader("📋 Report Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Report ID:** {report_data['report_id']}")
    
    with col2:
        st.write(f"**Generated:** {report_data['generated_at']}")

def risk_reports_page():
    """Risk reports page."""
    st.header("📈 Risk Reports & Analytics")
    
    # Input section for contract IDs
    st.subheader("📋 Generate Risk Report")
    
    # Explanation of Contract IDs
    st.info("""
    **What are Contract IDs?**
    
    Contract IDs are identifiers for contracts you've already analyzed. You can find them in:
    - **Single Contract Analysis**: Look at the contract ID in the results
    - **Batch Analysis**: Check the "File" column in the results table
    - **Portfolio Analysis**: Check the "Contract ID" column
    
    **Examples**: `contract_001`, `batch_2ThemartComInc_19990826_10-12G...`, etc.
    """)

    # Contract IDs input
    contract_ids_input = st.text_area(
        "Enter Contract IDs (one per line):",
        value="contract_001\ncontract_002\ncontract_003",
        help="Enter contract IDs from previous analyses to generate a risk report"
    )
    
    include_suggestions = st.checkbox("Include Recommendations", value=True)
    
    if st.button("Generate Risk Report", type="primary"):
        contract_ids = [cid.strip() for cid in contract_ids_input.split('\n') if cid.strip()]
        
        if not contract_ids:
            st.error("Please enter at least one contract ID")
            return
        
        # Generate risk report
        with st.spinner("Generating risk report..."):
            try:
                report_data = generate_risk_report(contract_ids, include_suggestions)
                
                if report_data:
                    display_risk_report(report_data)
                else:
                    st.error("Failed to generate risk report")
                    
            except Exception as e:
                st.error(f"Error generating risk report: {str(e)}")
    
    # Sample data section (fallback)
    st.subheader("📊 Sample Analytics (Demo Data)")
    st.info("📊 Below shows sample analytics for demonstration. Use the form above to generate real reports from your contract data.")
    
    # Sample risk trends
    st.subheader("📈 Risk Trends Over Time")
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    risk_scores = [6.2, 5.8, 7.1, 6.5, 5.9, 6.8, 7.3, 6.1, 5.7, 6.4, 6.9, 7.0]
    
    trend_df = pd.DataFrame({
        'Date': dates,
        'Average Risk Score': risk_scores
    })
    
    if PLOTLY_AVAILABLE:
        fig = px.line(
            trend_df,
            x='Date',
            y='Average Risk Score',
            title="Monthly Average Risk Scores (Sample Data)",
            markers=True
        )
        fig.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback to basic chart
        st.line_chart(trend_df.set_index('Date')['Average Risk Score'])
    
    # Risk categories
    st.subheader("🏷️ Risk Categories")
    
    categories = ['Payment Terms', 'Liability', 'Termination', 'Intellectual Property', 'Confidentiality']
    counts = [15, 12, 8, 6, 4]
    
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            x=categories,
            y=counts,
            title="Risk Issues by Category (Sample Data)",
            labels={'x': 'Category', 'y': 'Number of Issues'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback to basic chart
        category_df = pd.DataFrame({'Category': categories, 'Count': counts})
        st.bar_chart(category_df.set_index('Category')['Count'])

def portfolio_analysis_page():
    """Portfolio analysis page with advanced visualizations."""
    st.header("📊 Portfolio Analysis")
    st.markdown("Advanced portfolio-level contract analysis and visualization")
    
    # Portfolio data input section
    st.subheader("📋 Portfolio Data Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Option to analyze multiple contracts
        st.write("**Option 1: Analyze Multiple Contracts**")
        contract_texts = st.text_area(
            "Enter contract texts (separate each contract with '---' on its own line):",
            value="TERMINATION: Either party may terminate this agreement with 30 days notice.\nGOVERNING LAW: This agreement shall be governed by California law.\n\n---\n\nLIABILITY: Each party's liability shall be limited to the contract amount.\nCONFIDENTIALITY: Both parties agree to maintain confidentiality.\n\n---\n\nINDEMNITY: Each party shall indemnify the other against third-party claims.\nASSIGNMENT: This agreement may not be assigned without written consent.",
            height=200,
            help="Enter contract text for each contract, separated by '---' on its own line"
        )
    
    with col2:
        st.write("**Option 2: Use Sample Portfolio**")
        if st.button("Load Sample Portfolio", type="secondary"):
            # Generate sample portfolio data
            sample_contracts = [
                {
                    "contract_id": "contract_001",
                    "text": "TERMINATION: Either party may terminate this agreement with 30 days notice. GOVERNING LAW: This agreement shall be governed by California law.",
                    "overall_risk_score": 0.296,
                    "total_clauses": 2,
                    "high_risk_clauses": 1,
                    "medium_risk_clauses": 1,
                    "low_risk_clauses": 0
                },
                {
                    "contract_id": "contract_002", 
                    "text": "LIABILITY: Each party's liability shall be limited to the contract amount. CONFIDENTIALITY: Both parties agree to maintain confidentiality.",
                    "overall_risk_score": 0.156,
                    "total_clauses": 2,
                    "high_risk_clauses": 0,
                    "medium_risk_clauses": 1,
                    "low_risk_clauses": 1
                },
                {
                    "contract_id": "contract_003",
                    "text": "INDEMNITY: Each party shall indemnify the other. ASSIGNMENT: This agreement may not be assigned without consent.",
                    "overall_risk_score": 0.445,
                    "total_clauses": 2,
                    "high_risk_clauses": 2,
                    "medium_risk_clauses": 0,
                    "low_risk_clauses": 0
                }
            ]
            st.session_state.portfolio_data = sample_contracts
            st.success("Sample portfolio loaded!")
    
    # Analyze portfolio
    if st.button("Analyze Portfolio", type="primary"):
        if contract_texts:
            # Parse multiple contracts
            contracts = contract_texts.split("---")
            portfolio_data = []
            
            for i, contract_text in enumerate(contracts):
                if contract_text.strip():
                    # Analyze each contract
                    result = analyze_contract(contract_text.strip())
                    if result:
                        portfolio_data.append({
                            "contract_id": f"contract_{i+1:03d}",
                            "text": contract_text.strip(),
                            "overall_risk_score": result.get("overall_risk_score", 0),
                            "total_clauses": result.get("total_clauses", 0),
                            "high_risk_clauses": result.get("high_risk_clauses", 0),
                            "medium_risk_clauses": result.get("medium_risk_clauses", 0),
                            "low_risk_clauses": result.get("low_risk_clauses", 0)
                        })
            
            if portfolio_data:
                st.session_state.portfolio_data = portfolio_data
                st.success(f"Portfolio analyzed: {len(portfolio_data)} contracts")
            else:
                st.error("Failed to analyze portfolio contracts")
    
    # Display portfolio analysis
    if hasattr(st.session_state, 'portfolio_data') and st.session_state.portfolio_data:
        portfolio_data = st.session_state.portfolio_data
        
        st.subheader("📈 Portfolio Metrics")
        AdvancedVisualizations.display_portfolio_metrics(portfolio_data)
        
        st.subheader("📊 Portfolio Risk Analysis")
        AdvancedVisualizations.display_portfolio_scatter(portfolio_data)
        
        # Individual contract details
        st.subheader("📋 Individual Contract Details")
        contract_details = []
        for contract in portfolio_data:
            contract_details.append({
                "Contract ID": contract["contract_id"],
                "Risk Score": f"{contract['overall_risk_score']:.3f}",
                "Total Clauses": contract["total_clauses"],
                "High Risk": contract["high_risk_clauses"],
                "Medium Risk": contract["medium_risk_clauses"],
                "Low Risk": contract["low_risk_clauses"]
            })
        
        df_details = pd.DataFrame(contract_details)
        st.dataframe(df_details, use_container_width=True)
        
        # Risk distribution
        risk_distribution = {
            "High Risk": len([c for c in portfolio_data if c["overall_risk_score"] > 0.3]),
            "Medium Risk": len([c for c in portfolio_data if 0.1 < c["overall_risk_score"] <= 0.3]),
            "Low Risk": len([c for c in portfolio_data if c["overall_risk_score"] <= 0.1])
        }
        
        st.subheader("📊 Portfolio Risk Distribution")
        
        # Debug info
        st.write(f"Debug: Risk distribution = {risk_distribution}")
        st.write(f"Debug: PLOTLY_AVAILABLE = {PLOTLY_AVAILABLE}")
        
        # Always show chart, regardless of Plotly availability
        if PLOTLY_AVAILABLE:
            # Create a proper pie chart for portfolio risk distribution
            fig = px.pie(
                values=list(risk_distribution.values()),
                names=list(risk_distribution.keys()),
                title="Portfolio Risk Distribution",
                color_discrete_map={
                    "High Risk": "#ff4444",
                    "Medium Risk": "#ffaa00", 
                    "Low Risk": "#44ff44"
                },
                hole=0.3  # Donut chart
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to basic chart
            st.write("Using basic chart (Plotly not available)")
            # Create a simple bar chart using Streamlit
            chart_data = pd.DataFrame({
                'Risk Level': list(risk_distribution.keys()),
                'Count': list(risk_distribution.values())
            })
            st.bar_chart(chart_data.set_index('Risk Level'))
        
        # Historical trend (simulated)
        st.subheader("📈 Risk Trend Analysis")
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        trend_data = []
        for i, date in enumerate(dates):
            # Simulate risk trend
            base_risk = sum(c["overall_risk_score"] for c in portfolio_data) / len(portfolio_data)
            trend_risk = base_risk + (i * 0.01) + (0.05 if i % 3 == 0 else -0.02)
            trend_data.append({
                "date": date.strftime("%Y-%m"),
                "avg_risk_score": max(0, min(1, trend_risk))
            })
        
        # Debug info
        st.write(f"Debug: Trend data points = {len(trend_data)}")
        
        # Always show chart, regardless of Plotly availability
        if PLOTLY_AVAILABLE:
            # Create a proper line chart for risk trend
            trend_df = pd.DataFrame(trend_data)
            fig = px.line(
                trend_df,
                x="date",
                y="avg_risk_score",
                title="Portfolio Risk Trend Over Time",
                markers=True
            )
            fig.update_layout(
                yaxis_range=[0, 1],
                xaxis_title="Date",
                yaxis_title="Average Risk Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to basic chart
            st.write("Using basic chart (Plotly not available)")
            # Create a simple line chart using Streamlit
            trend_df = pd.DataFrame(trend_data)
            trend_df['date'] = pd.to_datetime(trend_df['date'])
            st.line_chart(trend_df.set_index('date'))

def system_info_page():
    """System information page."""
    st.header("ℹ️ System Information")
    
    # API Status
    st.subheader("🔌 API Status")
    health_data = get_api_health()
    
    if health_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("✅ API Connected")
            st.write(f"**Status:** {health_data.get('status', 'Unknown')}")
            st.write(f"**Model Version:** {health_data.get('model_snapshot', 'Unknown')}")
            st.write(f"**Calibration:** {health_data.get('calibration_version', 'Unknown')}")
        
        with col2:
            st.write(f"**Uptime:** {health_data.get('uptime_seconds', 0):.1f} seconds")
            st.write(f"**Last Check:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.error("❌ API Disconnected")
    
    st.divider()
    
    # System Features
    st.subheader("🚀 Features")
    
    features = [
        "📝 Single contract analysis",
        "📚 Batch contract processing", 
        "🎯 Risk assessment and scoring",
        "📊 Analytics and reporting",
        "💡 AI-powered recommendations",
        "🔍 Clause-by-clause analysis"
    ]
    
    for feature in features:
        st.write(feature)
    
    st.divider()
    
    # API Endpoints
    st.subheader("🔗 Available API Endpoints")
    
    endpoints = [
        ("/health", "GET", "System health check"),
        ("/analyze_contract", "POST", "Analyze single contract"),
        ("/batch_analyze", "POST", "Batch contract analysis"),
        ("/risk_report", "GET", "Generate risk reports"),
        ("/export", "POST", "Export analysis data"),
        ("/metrics", "GET", "System metrics"),
        ("/docs", "GET", "API documentation")
    ]
    
    endpoint_df = pd.DataFrame(endpoints, columns=['Endpoint', 'Method', 'Description'])
    st.dataframe(endpoint_df, use_container_width=True)

if __name__ == "__main__":
    main()
