#!/bin/bash
# Quick start script for Contract Review & Risk Analysis System

echo "🚀 Starting Contract Review & Risk Analysis System"
echo "=================================================="

# Check if environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "legal-ai" && "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Please activate the legal-ai environment first:"
    echo "   conda activate legal-ai"
    echo "   or"
    echo "   source legal-ai-env/bin/activate"
    exit 1
fi

# Start FastAPI backend
echo "🔧 Starting FastAPI backend..."
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start Streamlit frontend
echo "🎨 Starting Streamlit dashboard..."
streamlit run frontend/dashboard.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!

# Wait a moment for frontend to start
sleep 3

echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Access your application:"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Dashboard: http://localhost:8501"
echo "   Health Check: http://localhost:8000/health"
echo ""
echo "🛑 To stop the services, press Ctrl+C"
echo ""

# Wait for user to stop
wait
