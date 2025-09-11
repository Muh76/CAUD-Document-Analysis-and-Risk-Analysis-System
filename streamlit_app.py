"""
Contract Review & Risk Analysis System - Streamlit App
Self-contained version for Streamlit Share deployment.
"""

import streamlit as st
import requests
import json
from typing import Dict, Any, List
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Contract Review & Risk Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "https://contract-analysis-api-5wwrqt3oua-uc.a.run.app"

def get_api_health():
    """Check API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
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
            "text": text,
            "include_confidence": True,
            "include_explanations": True
        }
        
        response = requests.post(
            f"{API_BASE_URL}/analyze_contract",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
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
            ["Contract Analysis", "Batch Analysis", "Risk Reports", "System Info"]
        )
    
    # Main content based on selected page
    if page == "Contract Analysis":
        contract_analysis_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
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
            help="Upload a text or PDF file containing the contract"
        )
        
        if uploaded_file:
            if uploaded_file.type == "text/plain":
                contract_text = str(uploaded_file.read(), "utf-8")
            else:
                st.warning("PDF processing not available in this demo version")
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
        st.metric("Risk Score", f"{result.get('risk_score', 0):.1f}/10")
    
    with col2:
        st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
    
    with col3:
        clauses_count = len(result.get('clauses', []))
        st.metric("Clauses Found", clauses_count)
    
    with col4:
        issues_count = len([c for c in result.get('clauses', []) if c.get('risk_level', 'low') == 'high'])
        st.metric("High Risk Issues", issues_count)
    
    st.divider()
    
    # Detailed results
    if 'clauses' in result and result['clauses']:
        st.subheader("📋 Clause Analysis")
        
        # Create DataFrame for display
        clauses_data = []
        for clause in result['clauses']:
            clauses_data.append({
                'Clause': clause.get('text', '')[:100] + '...' if len(clause.get('text', '')) > 100 else clause.get('text', ''),
                'Type': clause.get('type', 'Unknown'),
                'Risk Level': clause.get('risk_level', 'Unknown'),
                'Confidence': f"{clause.get('confidence', 0):.1%}",
                'Explanation': clause.get('explanation', 'No explanation available')
            })
        
        df = pd.DataFrame(clauses_data)
        st.dataframe(df, use_container_width=True)
        
        # Risk distribution chart
        if len(clauses_data) > 0:
            st.subheader("📊 Risk Distribution")
            risk_counts = df['Risk Level'].value_counts()
            
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color_discrete_map={
                    'low': '#28a745',
                    'medium': '#ffc107', 
                    'high': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
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
        type=['txt'],
        accept_multiple_files=True,
        help="Upload multiple text files for batch processing"
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} files uploaded")
        
        if st.button("🔄 Process Batch", type="primary"):
            progress_bar = st.progress(0)
            results = []
            
            for i, file in enumerate(uploaded_files):
                try:
                    text = str(file.read(), "utf-8")
                    result = analyze_contract(text)
                    result['filename'] = file.name
                    results.append(result)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            # Display batch results
            if results:
                st.subheader("📊 Batch Analysis Results")
                
                batch_data = []
                for result in results:
                    batch_data.append({
                        'File': result.get('filename', 'Unknown'),
                        'Risk Score': result.get('risk_score', 0),
                        'Confidence': result.get('confidence', 0),
                        'Clauses': len(result.get('clauses', [])),
                        'High Risk': len([c for c in result.get('clauses', []) if c.get('risk_level') == 'high'])
                    })
                
                batch_df = pd.DataFrame(batch_data)
                st.dataframe(batch_df, use_container_width=True)
                
                # Batch summary chart
                fig = px.bar(
                    batch_df,
                    x='File',
                    y='Risk Score',
                    title="Risk Scores by File",
                    color='Risk Score',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)

def risk_reports_page():
    """Risk reports page."""
    st.header("📈 Risk Reports & Analytics")
    
    # Mock data for demonstration
    st.info("📊 This section shows sample analytics and reports")
    
    # Sample risk trends
    st.subheader("📈 Risk Trends Over Time")
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    risk_scores = [6.2, 5.8, 7.1, 6.5, 5.9, 6.8, 7.3, 6.1, 5.7, 6.4, 6.9, 7.0]
    
    trend_df = pd.DataFrame({
        'Date': dates,
        'Average Risk Score': risk_scores
    })
    
    fig = px.line(
        trend_df,
        x='Date',
        y='Average Risk Score',
        title="Monthly Average Risk Scores",
        markers=True
    )
    fig.update_layout(yaxis_range=[0, 10])
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk categories
    st.subheader("🏷️ Risk Categories")
    
    categories = ['Payment Terms', 'Liability', 'Termination', 'Intellectual Property', 'Confidentiality']
    counts = [15, 12, 8, 6, 4]
    
    fig = px.bar(
        x=categories,
        y=counts,
        title="Risk Issues by Category",
        labels={'x': 'Category', 'y': 'Number of Issues'}
    )
    st.plotly_chart(fig, use_container_width=True)

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
