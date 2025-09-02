# Contract Review & Risk Analysis System

A production-ready legal AI system for contract review and risk analysis, built with advanced ML techniques and multi-cloud deployment capabilities.

## 🎯 Project Overview

This system demonstrates enterprise-level legal AI capabilities including:
- **Clause Extraction & Classification** using CUAD dataset
- **Risk Scoring & Analysis** with weighted algorithms
- **RAG-powered Similar Clause Retrieval** with vector databases
- **Alternative Wording Suggestions** using OpenAI/Azure
- **Production-ready API** with FastAPI
- **Interactive Dashboard** with Streamlit
- **Multi-cloud Deployment** (Azure + Google Cloud)

## 🚀 Key Features

### 🔍 **Intelligent Contract Analysis**
- Extract and classify 41+ legal clause types
- Risk scoring with business impact assessment
- SHAP-based explainability for clause highlights

### 🤖 **RAG-Powered Legal Assistant**
- Find similar clauses from precedent database
- Suggest safer alternative wording
- Analyze clause risk levels

### ☁️ **Multi-Cloud Architecture**
- **OpenAI**: Primary LLM for intelligent analysis
- **Azure OpenAI**: Enterprise-grade backup
- **Google Cloud**: Document storage and model artifacts
- **ChromaDB**: Vector database for similar clause retrieval

### 📊 **Production-Ready Features**
- FastAPI backend with authentication
- Streamlit dashboard with real-time analytics
- Docker containerization
- MLOps integration (MLflow)
- Monitoring and logging

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   ML Pipeline   │
│   (Streamlit)   │◄──►│   (FastAPI)     │◄──►│   (PyTorch)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vector DB     │    │   OpenAI/Azure   │    │   Google Cloud   │
│   (ChromaDB)    │    │   (LLM)          │    │   (Storage)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

### **Core ML**
- **PyTorch Lightning**: Model training framework
- **Hugging Face Transformers**: RoBERTa/BERT models
- **Sentence Transformers**: Embedding generation
- **SHAP**: Model explainability

### **RAG System**
- **ChromaDB**: Vector database
- **OpenAI GPT-4**: LLM for suggestions
- **Azure OpenAI**: Enterprise LLM backup

### **Backend**
- **FastAPI**: Production API
- **PostgreSQL**: Metadata storage
- **Redis**: Caching layer
- **JWT**: Authentication

### **Frontend**
- **Streamlit**: Interactive dashboard
- **Plotly**: Data visualization
- **React/Vue.js**: Web application (optional)

### **DevOps**
- **Docker**: Containerization
- **MLflow**: Experiment tracking
- **Prometheus**: Monitoring
- **Grafana**: Dashboards

## 📁 Project Structure

```
Contract Review & Risk Analysis System/
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI application
│   ├── models/
│   │   ├── contract_analyzer.py    # Core ML model
│   │   ├── risk_scorer.py          # Risk scoring engine
│   │   ├── legal_rag.py            # RAG system
│   │   └── enhanced_legal_rag.py   # Enhanced RAG with Azure
│   ├── utils/
│   │   ├── file_processor.py       # Document processing
│   │   └── auth.py                 # Authentication
│   └── config/
│       ├── config.py               # Configuration
│       └── secure_config.py        # Secure config management
├── frontend/
│   └── dashboard.py               # Streamlit dashboard
├── tests/                         # Test suite
├── docs/                          # Documentation
├── docker-compose.yml             # Multi-service deployment
├── Dockerfile                     # Container definition
├── requirements.txt               # Python dependencies
├── environment.yml               # Conda environment
├── .env                          # Environment variables (not in repo)
├── env_template.txt              # Environment template
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. **Clone Repository**
```bash
git clone https://github.com/Muh76/CUAD.git
cd CUAD
```

### 2. **Set Up Environment**
```bash
# Create conda environment
conda env create -f environment.yml
conda activate legal-ai

# Or use pip
pip install -r requirements.txt
```

### 3. **Configure Credentials**
```bash
# Copy environment template
cp env_template.txt .env

# Edit .env with your credentials
# Required: OPENAI_API_KEY
# Optional: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
# Optional: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_STORAGE_BUCKET
```

### 4. **Install Dependencies**
```bash
./quick_install.sh
```

### 5. **Test System**
```bash
python test_system.py
```

### 6. **Launch Application**
```bash
./start.sh
```

### 7. **Access Applications**
- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501

## 🔐 Security Features

### **Credential Protection**
- ✅ Environment variables for all sensitive data
- ✅ `.env` file excluded from version control
- ✅ API key hashing for logging
- ✅ Secure configuration management

### **Authentication & Authorization**
- ✅ JWT-based authentication
- ✅ Role-based access control
- ✅ Password hashing with bcrypt
- ✅ Data encryption with Fernet

### **Input Validation**
- ✅ File type validation
- ✅ Size limits enforcement
- ✅ Rate limiting
- ✅ SQL injection prevention

## 📊 Employment Portfolio Value

### **Technical Skills Demonstrated**
- **Production ML Pipeline**: End-to-end ML system
- **Multi-Cloud Deployment**: Azure + Google Cloud integration
- **RAG Implementation**: Vector databases and LLM integration
- **Risk Scoring Algorithms**: Business-focused ML
- **Full-Stack Development**: FastAPI + Streamlit
- **Docker Containerization**: Production deployment
- **MLOps Integration**: Experiment tracking and monitoring

### **Business Value Demonstrated**
- **Legal Tech Innovation**: AI for legal industry
- **Cost Optimization**: Multi-cloud cost management
- **Enterprise Deployment**: Production-ready architecture
- **Risk Mitigation**: Business impact assessment

### **Target Companies**
- **Legal Tech Startups**: DocuSign, LegalZoom, Clio
- **Law Firms**: With tech initiatives
- **Enterprise Legal Departments**: Fortune 500 companies
- **Consulting Firms**: McKinsey, BCG, Deloitte
- **Tech Companies**: Google, Microsoft, Amazon

## 💰 Cost Analysis

### **Monthly Costs (MVP)**
- **OpenAI API**: $50-100 (depending on usage)
- **Vector Database**: $0 (ChromaDB local)
- **Storage**: $0 (local)
- **Total**: $50-100/month

### **Monthly Costs (Production)**
- **OpenAI API**: $100-200
- **Azure OpenAI**: $50-100 (included in existing subscription)
- **Google Cloud**: $20-50
- **PostgreSQL**: $20-50
- **Redis**: $10-20
- **Total**: $200-500/month

## 🔄 Development Phases

### **Phase 1: MVP (Complete)**
- ✅ Basic clause extraction
- ✅ Risk scoring
- ✅ RAG system
- ✅ API and dashboard

### **Phase 2: Enhanced (In Progress)**
- 🔄 Multi-cloud integration
- 🔄 Advanced monitoring
- 🔄 Performance optimization

### **Phase 3: Production (Planned)**
- 📋 Cloud deployment
- 📋 Advanced security
- 📋 Compliance features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Mohammad Babaie**
- **Email**: mj.babaie@gmail.com
- **LinkedIn**: [https://www.linkedin.com/in/mohammadbabaie/](https://www.linkedin.com/in/mohammadbabaie/)
- **GitHub**: [https://github.com/Muh76](https://github.com/Muh76)

## 🙏 Acknowledgments

- **CUAD Dataset**: Contract Understanding Atticus Dataset
- **OpenAI**: GPT-4 API for intelligent analysis
- **Azure**: Enterprise OpenAI services
- **Google Cloud**: Storage and deployment platform
- **Hugging Face**: Transformers library
- **FastAPI**: Modern web framework
- **Streamlit**: Interactive dashboard framework

---

**Built with ❤️ for the legal tech community**
