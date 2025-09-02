# Contract Analysis System - Jupyter Notebooks

## Overview
This directory contains Jupyter notebooks for all phases of the Contract Analysis System project.

## Environment Setup

### Option 1: Automated Setup
```bash
./setup_environment.sh
```

### Option 2: Manual Setup
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate contract-analysis

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Notebook Structure

### 📚 Available Notebooks

1. **`00_overview_demo.ipynb`** - Complete project overview
   - All phases summary
   - Technology stack overview
   - Business impact & ROI
   - Portfolio value for recruiters

2. **`01_phase1_foundations.ipynb`** - Phase 1: Foundations & Data Pipeline ✅
   - CUAD dataset ingestion and processing
   - Contract metadata extraction using NER
   - Clause segmentation and classification
   - Baseline models (TF-IDF + Logistic Regression)
   - Simple risk scoring with keyword patterns
   - Data validation with Pandera

3. **`02_phase2_modeling.ipynb`** - Phase 2: Advanced Modeling & Risk Scoring 🚧
   - Fine-tune DistilBERT/Legal-BERT for clause classification
   - Implement calibrated probabilities and confidence scoring
   - Build advanced risk engine with policy rules + learned anomalies
   - Add explainability with token-level highlights + SHAP
   - Contract type classification
   - Risk trend analysis

4. **`03_phase3_product_mvp.ipynb`** - Phase 3: Product MVP (API & UI) 📋
   - FastAPI endpoints: /analyze_contract, /risk_report, /health
   - Pydantic I/O schemas
   - Streamlit UI with upload → clause highlights → per-clause analysis
   - Contract summary: total risk, missing clauses, red-flag list
   - Portfolio tab: multi-contract view (vendor, type, jurisdiction)
   - Comparison tools and dashboards
   - Export formats (PDF, Excel, JSON)

5. **`04_phase4_mlops.ipynb`** - Phase 4: MLOps & Deployment 📋
   - MLflow tracking + Model Registry (Staging → Prod)
   - Experiment tags tied to data hash
   - DVC for data/version lineage
   - GitHub Actions: tests, lint, data checks, build, deploy
   - Hosting: API on Railway, UI on Streamlit Cloud
   - Monitoring: request/latency logs, drift detection
   - Security: remove PII, HTTPS, basic auth, audit log
   - RBAC and compliance reporting

6. **`05_phase5_scale_compliance.ipynb`** - Phase 5: Scale & Compliance 📋
   - Multi-tenant architecture
   - Advanced RBAC with legal team roles
   - Compliance reporting (GDPR, CCPA, SOX)
   - Advanced analytics dashboard
   - Contract lifecycle management
   - Negotiation suggestions
   - ROI calculator
   - Integration APIs (DocuSign, Salesforce)

## Running Notebooks

### Start Jupyter
```bash
# Activate environment
conda activate contract-analysis

# Start Jupyter
jupyter notebook
```

### Select Kernel
When opening notebooks, make sure to select the **"Contract Analysis (Python 3.11)"** kernel.

## Quick Start

1. **Setup Environment**: Run `./setup_environment.sh`
2. **Start Jupyter**: `jupyter notebook`
3. **Open Overview**: Start with `00_overview_demo.ipynb`
4. **Run Phase 1**: Execute `01_phase1_foundations.ipynb`
5. **Continue with Phase 2**: Work on `02_phase2_modeling.ipynb`

## Dependencies

### Core Libraries
- **Data Processing**: pandas, numpy, spacy, transformers
- **Machine Learning**: scikit-learn, torch, transformers
- **Data Validation**: pandera, great-expectations
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest, pytest-cov
- **Code Quality**: black, flake8, mypy
- **MLOps**: dvc

### Models
- **spaCy**: en_core_web_sm (English language model)
- **Transformers**: DistilBERT, Legal-BERT (for Phase 2)

## Project Status

- ✅ **Phase 1**: Foundations & Data Pipeline - COMPLETED
- 🚧 **Phase 2**: Advanced Modeling & Risk Scoring - IN PROGRESS
- 📋 **Phase 3**: Product MVP (API & UI) - PLANNED
- 📋 **Phase 4**: MLOps & Deployment - PLANNED
- 📋 **Phase 5**: Scale & Compliance - PLANNED

## Contact

**Author**: Mohammad Babaie  
**Email**: mj.babaie@gmail.com  
**LinkedIn**: https://www.linkedin.com/in/mohammadbabaie/  
**GitHub**: https://github.com/Muh76

## Repository

📁 **Repository**: https://github.com/Muh76/CAUD-Document-Analysis-and-Risk-Analysis-System
