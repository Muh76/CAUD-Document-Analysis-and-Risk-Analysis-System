#!/bin/bash

# Contract Analysis System - Environment Setup Script
# This script sets up the Python environment for the project

echo "🚀 Setting up Contract Analysis Environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment..."
conda create -n contract-analysis python=3.11 -y

# Activate environment
echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate contract-analysis

# Install required packages
echo "📚 Installing required packages..."
pip install pandas numpy matplotlib seaborn scikit-learn joblib pandera great-expectations spacy transformers torch pytest pytest-cov black flake8 mypy dvc

# Download spaCy model
echo "🤖 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Register with Jupyter
echo "📓 Registering with Jupyter..."
python -m ipykernel install --user --name contract-analysis --display-name "Contract Analysis (Python 3.11)"

# Test installation
echo "🧪 Testing installation..."
python -c "
import pandas as pd
import numpy as np
import spacy
import transformers
import torch
import sklearn
import pandera
import great_expectations
import dvc
print('✅ All packages imported successfully!')
"

echo "🎉 Environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  conda activate contract-analysis"
echo ""
echo "To start Jupyter:"
echo "  jupyter notebook"
echo ""
echo "To run Phase 1:"
echo "  python run_phase1.py"
