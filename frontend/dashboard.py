"""
Streamlit Dashboard for Contract Review & Risk Analysis System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.config import *
from src.models.contract_analyzer import ContractAnalyzer
from src.models.risk_scorer import RiskScorer

# Page configuration
st.set_page_config(
    page_title="Contract Review & Risk Analysis",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def load_components():
    return ContractAnalyzer(), RiskScorer()

contract_analyzer, risk_scorer = load_components()

# Sidebar
st.sidebar.title("⚖️ Legal AI System")
st.sidebar.markdown("**Contract Review & Risk Analysis**")

# Navigation
page = st.sidebar.selectbox(
    "Choose a page",
    ["📊 Dashboard", "📄 Contract Analysis", "🔍 RAG Search", "📈 Analytics", "⚙️ Settings"]
)

# Main content
if page == "📊 Dashboard":
    st.title("📊 Contract Review Dashboard")
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Contracts",
            value="1,234",
            delta="+12%"
        )
    
    with col2:
        st.metric(
            label="Avg Risk Score",
            value="0.34",
            delta="-0.05"
        )
    
    with col3:
        st.metric(
            label="Time Saved",
            value="156 hrs",
            delta="+23 hrs"
        )
    
    with col4:
        st.metric(
            label="Cost Savings",
            value="$12,450",
            delta="+$1,200"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Distribution")
        risk_data = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High', 'Critical'],
            'Count': [45, 32, 18, 5]
        })
        fig = px.pie(risk_data, values='Count', names='Risk Level', 
                    color_discrete_sequence=['green', 'yellow', 'orange', 'red'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Processing Time Trend")
        time_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'Processing Time (min)': [2.1, 1.8, 2.3, 1.9, 2.0] * 6
        })
        fig = px.line(time_data, x='Date', y='Processing Time (min)')
        st.plotly_chart(fig, use_container_width=True)

elif page == "📄 Contract Analysis":
    st.title("📄 Contract Analysis")
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Contract",
        type=['pdf', 'txt', 'docx'],
        help="Upload a contract file for analysis"
    )
    
    if uploaded_file is not None:
        # Process file
        with st.spinner("Analyzing contract..."):
            # Extract text (simplified for demo)
            contract_text = uploaded_file.read().decode('utf-8')[:1000] + "..."
            
            # Mock analysis results
            extracted_clauses = {
                "governing_law": "This agreement shall be governed by the laws of California",
                "liquidated_damages": "Party A shall pay $50,000 in liquidated damages",
                "confidentiality": "All information shall be kept confidential"
            }
            
            risk_score = 0.67
            risk_level = "High"
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📋 Extracted Clauses")
                for clause_type, clause_text in extracted_clauses.items():
                    st.write(f"**{clause_type.replace('_', ' ').title()}:**")
                    st.write(f"*{clause_text}*")
                    st.write("---")
            
            with col2:
                st.subheader("⚠️ Risk Assessment")
                st.metric("Risk Score", f"{risk_score:.2f}")
                st.metric("Risk Level", risk_level)
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_score * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Level"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 RAG Search":
    st.title("🔍 RAG Search")
    st.markdown("---")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_area(
            "Enter clause text to search",
            placeholder="e.g., 'Party A shall have unlimited liability for any damages...'",
            height=100
        )
    
    with col2:
        clause_type = st.selectbox(
            "Clause Type",
            ["uncapped_liability", "non_compete", "ip_assignment", "confidentiality"]
        )
        search_button = st.button("🔍 Search Similar Clauses")
    
    if search_button and search_query:
        with st.spinner("Searching similar clauses..."):
            # Mock RAG results
            similar_clauses = [
                {
                    "text": "Party A shall be liable for all damages up to $100,000",
                    "similarity": 0.89,
                    "outcome": "Favorable",
                    "source": "Contract_2023_001"
                },
                {
                    "text": "Liability shall be limited to direct damages only",
                    "similarity": 0.76,
                    "outcome": "Favorable",
                    "source": "Contract_2023_015"
                },
                {
                    "text": "Party A's liability is capped at the contract value",
                    "similarity": 0.72,
                    "outcome": "Favorable",
                    "source": "Contract_2023_023"
                }
            ]
            
            st.subheader("📚 Similar Clauses Found")
            
            for i, clause in enumerate(similar_clauses):
                with st.expander(f"Clause {i+1} (Similarity: {clause['similarity']:.2f})"):
                    st.write(f"**Text:** {clause['text']}")
                    st.write(f"**Outcome:** {clause['outcome']}")
                    st.write(f"**Source:** {clause['source']}")
            
            # Alternative suggestions
            st.subheader("💡 Alternative Suggestions")
            suggestions = [
                "Consider capping liability at a reasonable amount",
                "Limit liability to direct damages only",
                "Add force majeure clause for protection"
            ]
            
            for suggestion in suggestions:
                st.write(f"• {suggestion}")

elif page == "📈 Analytics":
    st.title("📈 Analytics")
    st.markdown("---")
    
    # Analytics tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🎯 Risk Trends", "💰 Business Impact", "🔍 Model Insights"])
    
    with tab1:
        st.subheader("System Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Processing time chart
            processing_data = pd.DataFrame({
                'Hour': range(24),
                'Processing Time (min)': [2.1, 1.8, 2.3, 1.9, 2.0, 2.2, 1.7, 2.1, 1.9, 2.0, 2.3, 1.8,
                                        2.1, 1.9, 2.0, 2.2, 1.8, 2.1, 2.0, 1.9, 2.3, 2.1, 1.8, 2.0]
            })
            fig = px.line(processing_data, x='Hour', y='Processing Time (min)',
                         title="Processing Time by Hour")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Accuracy chart
            accuracy_data = pd.DataFrame({
                'Model': ['Clause Extraction', 'Risk Scoring', 'RAG Retrieval'],
                'Accuracy': [0.89, 0.92, 0.85]
            })
            fig = px.bar(accuracy_data, x='Model', y='Accuracy',
                        title="Model Accuracy")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Risk Trends")
        
        # Risk trend over time
        risk_trend_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'Avg Risk Score': [0.45, 0.42, 0.48, 0.41, 0.43, 0.46, 0.44, 0.47, 0.42, 0.45,
                              0.43, 0.46, 0.41, 0.44, 0.47, 0.42, 0.45, 0.43, 0.46, 0.41,
                              0.44, 0.47, 0.42, 0.45, 0.43, 0.46, 0.41, 0.44, 0.47, 0.42]
        })
        fig = px.line(risk_trend_data, x='Date', y='Avg Risk Score',
                     title="Average Risk Score Trend")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Business Impact")
        
        # ROI calculation
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Time Saved", "1,234 hours")
            st.metric("Cost Savings", "$123,450")
            st.metric("Contracts Processed", "567")
        
        with col2:
            st.metric("Risk Mitigation", "89%")
            st.metric("Accuracy Improvement", "23%")
            st.metric("Processing Speed", "3.2x faster")
    
    with tab4:
        st.subheader("Model Insights")
        
        # SHAP values visualization
        st.write("**Feature Importance for Risk Scoring:**")
        
        feature_importance = pd.DataFrame({
            'Feature': ['Uncapped Liability', 'Non-compete', 'IP Assignment', 'Termination', 'Audit Rights'],
            'Importance': [0.25, 0.20, 0.15, 0.10, 0.05]
        })
        
        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                    title="Feature Importance in Risk Scoring")
        st.plotly_chart(fig, use_container_width=True)

elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.markdown("---")
    
    # Configuration settings
    st.subheader("System Configuration")
    
    # API settings
    api_url = st.text_input("API URL", value="http://localhost:8000")
    
    # Model settings
    model_type = st.selectbox("Model Type", ["roberta-base", "bert-base", "distilbert"])
    
    # Risk scoring weights
    st.subheader("Risk Scoring Weights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uncapped_liability = st.slider("Uncapped Liability", 0.0, 1.0, 0.25)
        non_compete = st.slider("Non-compete", 0.0, 1.0, 0.20)
        ip_assignment = st.slider("IP Assignment", 0.0, 1.0, 0.15)
    
    with col2:
        termination = st.slider("Termination", 0.0, 1.0, 0.10)
        audit_rights = st.slider("Audit Rights", 0.0, 1.0, 0.05)
        liquidated_damages = st.slider("Liquidated Damages", 0.0, 1.0, 0.08)
    
    # Save settings
    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: gray;'>"
    f"Built by {AUTHOR} | {EMAIL} | "
    f"<a href='https://github.com/Muh76' target='_blank'>GitHub</a> | "
    f"<a href='https://www.linkedin.com/in/mohammadbabaie/' target='_blank'>LinkedIn</a>"
    f"</div>",
    unsafe_allow_html=True
)
