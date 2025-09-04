# Project Configuration
PROJECT_NAME = "Contract Review & Risk Analysis System"
VERSION = "1.0.0"
AUTHOR = "Mohammad Babaie"
EMAIL = "mj.babaie@gmail.com"

# Data Paths
DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
FEATURES_DIR = "data/features"

# Model Paths
MODELS_DIR = "models"
CHECKPOINTS_DIR = "models/checkpoints"
ARTIFACTS_DIR = "models/artifacts"

# Logging
LOG_DIR = "logs"
LOG_LEVEL = "INFO"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True

# Database Configuration
DATABASE_URL = "postgresql://user:password@localhost/legalai"
REDIS_URL = "redis://localhost:6379"

# MLflow Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "contract_review"

# OpenAI Configuration
OPENAI_API_KEY = ""  # Set via environment variable

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = ""  # Set via environment variable
AZURE_OPENAI_API_KEY = ""  # Set via environment variable
AZURE_OPENAI_DEPLOYMENT_NAME = ""  # Set via environment variable

# Vector Database
CHROMA_PERSIST_DIR = "data/chroma_db"

# Security
SECRET_KEY = ""  # Set via environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# File Upload
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = [".pdf", ".txt", ".docx"]

# Risk Scoring Weights
RISK_WEIGHTS = {
    "uncapped_liability": 0.25,
    "non_compete": 0.20,
    "ip_assignment": 0.15,
    "termination_convenience": 0.10,
    "audit_rights": 0.05,
    "liquidated_damages": 0.08,
    "governing_law": 0.03,
    "warranty_duration": 0.04,
    "confidentiality": 0.05,
    "force_majeure": 0.05,
}

# CUAD Dataset Configuration
CUAD_CATEGORIES = [
    "document_name",
    "parties",
    "effective_date",
    "contract_type",
    "contract_value",
    "governing_law",
    "jurisdiction",
    "choice_of_law",
    "venue",
    "arbitration",
    "liquidated_damages",
    "warranty_duration",
    "warranty_scope",
    "warranty_exclusions",
    "warranty_limitations",
    "warranty_assignability",
    "warranty_termination",
    "warranty_renewal",
    "warranty_extension",
    "warranty_modification",
    "warranty_waiver",
    "warranty_disclaimer",
    "warranty_representation",
    "warranty_covenant",
    "warranty_condition",
    "warranty_breach",
    "warranty_remedy",
    "warranty_indemnification",
    "warranty_insurance",
    "warranty_audit",
    "warranty_inspection",
    "warranty_testing",
    "warranty_acceptance",
    "warranty_rejection",
    "warranty_correction",
    "warranty_replacement",
    "warranty_refund",
    "warranty_credit",
    "warranty_discount",
    "warranty_penalty",
    "warranty_bonus",
    "warranty_incentive",
]
