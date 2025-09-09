"""
Enhanced Streamlit UI with production features.
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.config.settings import get_settings
from app.ui.enhanced_components import UIComponents

# Settings
settings = get_settings()

# Page configuration
st.set_page_config(
    page_title="Contract Analysis System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .risk-high { border-left-color: #e74c3c; }
    .risk-medium { border-left-color: #f39c12; }
    .risk-low { border-left-color: #27ae60; }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📋 Contract Review & Risk Analysis System</h1>
        <p>AI-powered contract analysis with production-grade features</p>
    </div>
    """, unsafe_allow_html=True)

    # Show system status
    UIComponents.show_system_status()

    # Show model version badges
    UIComponents.show_model_version_badge()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Analysis", "🔍 RAG Search", "📊 Analytics", "⚙️ Settings"])

    with tab1:
        st.header("Contract Analysis")

        # Enhanced file upload
        uploaded_file = UIComponents.enhanced_file_upload()

        if uploaded_file:
            # RAG controls
            rag_params = UIComponents.show_rag_controls()

            # Analysis button
            if st.button("🚀 Analyze Contract", type="primary"):
                with st.spinner("Analyzing contract..."):
                    try:
                        # Prepare request
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {
                            "similarity_threshold": rag_params["similarity_threshold"],
                            "top_k": rag_params["top_k"],
                            "show_sources": rag_params["show_sources"]
                        }

                        # Make API request
                        response = requests.post(
                            f"http://localhost:{settings.api_port}/analyze_contract",
                            files=files,
                            data=data,
                            timeout=60
                        )

                        if response.status_code == 200:
                            result = response.json()

                            # Display results
                            st.success("✅ Analysis completed!")

                            # Risk summary
                            st.subheader("🎯 Risk Summary")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("High Risk", result.get("high_risk_count", 0))
                            with col2:
                                st.metric("Medium Risk", result.get("medium_risk_count", 0))
                            with col3:
                                st.metric("Low Risk", result.get("low_risk_count", 0))

                            # Detailed results
                            st.subheader("📋 Detailed Analysis")
                            for clause in result.get("clauses", []):
                                risk_level = clause.get("risk_level", "unknown")
                                risk_color = {
                                    "high": "risk-high",
                                    "medium": "risk-medium", 
                                    "low": "risk-low"
                                }.get(risk_level, "")

                                with st.expander(f"🔍 {clause.get('clause_type', 'Unknown')} - {risk_level.title()} Risk"):
                                    st.markdown(f"""
                                    <div class="metric-card {risk_color}">
                                        <strong>Clause:</strong> {clause.get('text', 'N/A')[:200]}...<br>
                                        <strong>Risk Level:</strong> {risk_level.title()}<br>
                                        <strong>Confidence:</strong> {clause.get('confidence', 0):.2f}<br>
                                        <strong>Explanation:</strong> {clause.get('explanation', 'N/A')}
                                    </div>
                                    """, unsafe_allow_html=True)

                            # Show feedback system
                            UIComponents.show_feedback_system(result)

                        else:
                            st.error(f"❌ Analysis failed: {response.text}")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    with tab2:
        st.header("🔍 RAG Search")
        st.info("Search for similar clauses in our knowledge base")

        # Search interface
        search_query = st.text_input("Enter search query:")
        if search_query:
            rag_params = UIComponents.show_rag_controls()

            if st.button("🔍 Search"):
                with st.spinner("Searching..."):
                    try:
                        response = requests.post(
                            f"http://localhost:{settings.api_port}/rag/search",
                            json={
                                "query": search_query,
                                "similarity_threshold": rag_params["similarity_threshold"],
                                "top_k": rag_params["top_k"]
                            },
                            timeout=30
                        )

                        if response.status_code == 200:
                            results = response.json()
                            st.success(f"Found {len(results.get('results', []))} similar clauses")

                            for i, result in enumerate(results.get("results", [])):
                                with st.expander(f"Result {i+1} - Similarity: {result.get('similarity', 0):.3f}"):
                                    st.write("**Text:**", result.get("text", "N/A"))
                                    st.write("**Source:**", result.get("source", "N/A"))
                                    if rag_params["show_sources"]:
                                        st.write("**Metadata:**", result.get("metadata", {}))

                        else:
                            st.error(f"❌ Search failed: {response.text}")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    with tab3:
        st.header("📊 Analytics & Monitoring")

        # Confidence calibration
        UIComponents.show_confidence_calibration()

        # System metrics
        st.subheader("System Metrics")
        try:
            response = requests.get(f"http://localhost:{settings.api_port}/metrics", timeout=5)
            if response.status_code == 200:
                st.success("📊 Metrics available")
                st.code(response.text[:1000] + "..." if len(response.text) > 1000 else response.text)
            else:
                st.warning("⚠️ Metrics unavailable")
        except:
            st.warning("⚠️ Cannot connect to metrics endpoint")

    with tab4:
        st.header("⚙️ Settings")

        st.subheader("Application Settings")
        st.write(f"**Version:** {settings.app_version}")
        st.write(f"**Environment:** {settings.environment}")
        st.write(f"**API Host:** {settings.api_host}:{settings.api_port}")

        st.subheader("Model Settings")
        st.write(f"**Model Snapshot:** {settings.model_snapshot}")
        st.write(f"**RAG Collection:** {settings.rag_collection}")
        st.write(f"**Default Similarity Threshold:** {settings.rag_similarity_threshold}")
        st.write(f"**Default Top K:** {settings.rag_top_k}")

        st.subheader("File Upload Limits")
        st.write(f"**Max File Size:** {settings.max_file_size_mb} MB")
        st.write(f"**Max Pages:** {settings.max_pages_per_request}")
        st.write(f"**Allowed Types:** {', '.join(settings.allowed_mime_types)}")

if __name__ == "__main__":
    main()
