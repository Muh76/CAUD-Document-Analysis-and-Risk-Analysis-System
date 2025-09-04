#!/bin/bash
# Setup DVC Remote Configuration

echo "🔧 Setting up DVC Remote..."

# Create local remote directory
mkdir -p data/dvc_remote

# Add local remote
dvc remote add -d local_remote data/dvc_remote

# Configure remote
dvc remote modify local_remote url data/dvc_remote

echo "✅ DVC remote configured!"
echo "📁 Remote location: data/dvc_remote"
echo ""
echo "🚀 Next steps:"
echo "   dvc add data/raw/CUAD_v1.json"
echo "   dvc push"
echo "   dvc repro data_processing"
