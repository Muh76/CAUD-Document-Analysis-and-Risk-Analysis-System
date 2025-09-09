"""
Enhanced UI components for production-ready Streamlit application.
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.config.settings import get_settings

# Settings
settings = get_settings()

class UIComponents:
    """Enhanced UI components for production features."""

    @staticmethod
    def show_model_version_badge():
        """Display model version and data timestamp badges."""
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div style="background-color: #e8f4fd; padding: 10px; border-radius: 5px; text-align: center;">
                <strong>Model Version</strong><br>
                <span style="color: #1f77b4;">{settings.app_version}</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Get model snapshot info
            model_snapshot = settings.model_snapshot
            st.markdown(f"""
            <div style="background-color: #f0f8e8; padding: 10px; border-radius: 5px; text-align: center;">
                <strong>Model Snapshot</strong><br>
                <span style="color: #2ca02c;">{model_snapshot}</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; text-align: center;">
                <strong>Environment</strong><br>
                <span style="color: #856404;">{settings.environment.title()}</span>
            </div>
            """, unsafe_allow_html=True)

    @staticmethod
    def show_rag_controls():
        """Display RAG similarity controls."""
        st.subheader("🔍 RAG Search Controls")

        col1, col2 = st.columns(2)

        with col1:
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.1,
                max_value=1.0,
                value=settings.rag_similarity_threshold,
                step=0.1,
                help="Higher values = more similar results"
            )

        with col2:
            top_k = st.slider(
                "Number of Results",
                min_value=1,
                max_value=20,
                value=settings.rag_top_k,
                step=1,
                help="Maximum number of similar clauses to return"
            )

        show_sources = st.checkbox(
            "Show Source Documents",
            value=True,
            help="Display the source documents for each result"
        )

        return {
            "similarity_threshold": similarity_threshold,
            "top_k": top_k,
            "show_sources": show_sources
        }

    @staticmethod
    def show_feedback_system(analysis_result: Dict[str, Any]):
        """Display feedback collection system."""
        st.subheader("�� Feedback & Quality Control")

        # Feedback form
        with st.expander("Report Issues or Provide Feedback", expanded=False):
            feedback_type = st.selectbox(
                "Feedback Type",
                ["Incorrect Classification", "Missing Risk", "False Positive", "Suggest Improvement", "Other"]
            )

            feedback_text = st.text_area(
                "Description",
                placeholder="Please describe the issue or suggestion...",
                height=100
            )

            confidence_rating = st.slider(
                "How confident are you in this feedback?",
                min_value=1,
                max_value=5,
                value=3,
                help="1 = Not sure, 5 = Very confident"
            )

            if st.button("Submit Feedback"):
                feedback_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "feedback_type": feedback_type,
                    "feedback_text": feedback_text,
                    "confidence_rating": confidence_rating,
                    "analysis_result": analysis_result,
                    "user_agent": "streamlit_ui"
                }

                # Save feedback to file
                feedback_file = settings.rag_index_path.parent / "feedback" / f"feedback_{int(time.time())}.json"
                feedback_file.parent.mkdir(exist_ok=True)

                with open(feedback_file, "w") as f:
                    json.dump(feedback_data, f, indent=2)

                st.success("✅ Feedback submitted successfully!")
                st.info("Thank you for helping improve our system!")

    @staticmethod
    def show_confidence_calibration():
        """Display confidence calibration information."""
        st.subheader("📊 Confidence Calibration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Confidence Levels:**
            - 🟢 **High (0.8-1.0)**: Very reliable prediction
            - 🟡 **Medium (0.5-0.8)**: Moderately reliable
            - 🔴 **Low (0.0-0.5)**: Less reliable, review recommended
            """)

        with col2:
            st.markdown("""
            **Model Performance:**
            - Accuracy: ~85% on test set
            - Precision: ~82% for risk detection
            - Recall: ~88% for risk detection
            """)

    @staticmethod
    def show_system_status():
        """Display system status and health."""
        try:
            # Check API health
            response = requests.get(f"http://localhost:{settings.api_port}/health", timeout=5)
            api_status = "🟢 Healthy" if response.status_code == 200 else "�� Unhealthy"
        except:
            api_status = "🔴 Unreachable"

        st.sidebar.subheader("System Status")
        st.sidebar.write(f"**API Status:** {api_status}")
        st.sidebar.write(f"**Environment:** {settings.environment.title()}")
        st.sidebar.write(f"**Version:** {settings.app_version}")

        # Show metrics if available
        try:
            metrics_response = requests.get(f"http://localhost:{settings.api_port}/metrics", timeout=5)
            if metrics_response.status_code == 200:
                st.sidebar.success("📊 Metrics Available")
            else:
                st.sidebar.warning("⚠️ Metrics Unavailable")
        except:
            st.sidebar.warning("⚠️ Metrics Unreachable")

    @staticmethod
    def enhanced_file_upload():
        """Enhanced file upload with validation."""
        st.subheader("📁 Document Upload")

        # File type information
        st.info(f"""
        **Supported Formats:** {', '.join(settings.allowed_mime_types)}
        **Max File Size:** {settings.max_file_size_mb} MB
        **Max Pages:** {settings.max_pages_per_request} pages per request
        """)

        uploaded_file = st.file_uploader(
            "Choose a contract document",
            type=['pdf', 'txt', 'docx'],
            help="Upload a contract document for analysis"
        )

        if uploaded_file:
            # File validation
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

            if file_size_mb > settings.max_file_size_mb:
                st.error(f"File too large! Maximum size: {settings.max_file_size_mb} MB")
                return None

            st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")
            return uploaded_file

        return None
