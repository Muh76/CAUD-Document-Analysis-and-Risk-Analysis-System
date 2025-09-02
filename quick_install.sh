#!/bin/bash
# Quick Install Script for Contract Review System

echo "🚀 Quick Installation for Contract Review & Risk Analysis System"
echo "================================================================"

# Check if we're in a conda environment
if [[ "$CONDA_DEFAULT_ENV" == "legal-ai" ]]; then
    echo "✅ Conda environment 'legal-ai' is active"
elif [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment is active"
else
    echo "⚠️  No virtual environment detected"
    echo "   Run: conda activate legal-ai"
    echo "   Or: source legal-ai-env/bin/activate"
    exit 1
fi

echo ""
echo "📦 Installing core dependencies..."

# Install essential packages for testing
pip install torch transformers sentence-transformers chromadb openai python-docx PyPDF2

echo ""
echo "✅ Core dependencies installed!"
echo ""
echo "🧪 Running system tests..."
python test_system.py

echo ""
echo "🎉 Installation complete!"
echo ""
echo "🚀 Next steps:"
echo "   1. Run: ./start.sh to launch the application"
echo "   2. Open: http://localhost:8000/docs for API docs"
echo "   3. Open: http://localhost:8501 for dashboard"
