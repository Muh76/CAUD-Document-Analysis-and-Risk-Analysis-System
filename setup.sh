#!/bin/bash
# Setup script for Contract Review & Risk Analysis System

echo "🚀 Setting up Contract Review & Risk Analysis System"
echo "=================================================="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✅ Conda found, creating environment..."
    
    # Create conda environment
    conda env create -f environment.yml
    
    # Activate environment
    echo "🔧 Activating conda environment..."
    conda activate legal-ai
    
    echo "✅ Conda environment setup complete!"
else
    echo "⚠️  Conda not found, using pip..."
    
    # Create virtual environment
    python -m venv legal-ai-env
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source legal-ai-env/Scripts/activate
    else
        source legal-ai-env/bin/activate
    fi
    
    echo "✅ Virtual environment created and activated!"
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,features}
mkdir -p models/{checkpoints,artifacts}
mkdir -p logs
mkdir -p notebooks

# Set up environment variables
echo "⚙️  Setting up environment variables..."
if [ ! -f .env ]; then
    cp env_template.txt .env
    echo "✅ Environment file created. Please edit .env with your configuration."
else
    echo "✅ Environment file already exists."
fi

# Initialize git repository (if not already done)
if [ ! -d .git ]; then
    echo "🔧 Initializing git repository..."
    git init
    echo "✅ Git repository initialized."
else
    echo "✅ Git repository already exists."
fi

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pip install pre-commit
pre-commit install

echo ""
echo "🎉 Setup complete! Your environment is ready."
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run: python -m uvicorn src.api.main:app --reload"
echo "3. Run: streamlit run frontend/dashboard.py"
echo "4. Open http://localhost:8000 for API docs"
echo "5. Open http://localhost:8501 for dashboard"
echo ""
echo "Happy coding! 🚀"
