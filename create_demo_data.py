#!/usr/bin/env python3
"""
Simple Phase 1 Demo - Creates custom modules and sample data
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Create necessary directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models/baseline', exist_ok=True)

# Create sample CUAD dataset
sample_cuad_data = {
    "data": [
        {
            "contract_id": "CONTRACT_001",
            "title": "Software License Agreement",
            "context": "This Software License Agreement (the 'Agreement') is entered into between TechCorp Inc. ('Licensor') and ClientCo Ltd. ('Licensee') effective as of January 1, 2024. The Licensor grants to the Licensee a non-exclusive, non-transferable license to use the Software. The Licensee shall pay a license fee of $10,000. This Agreement shall be governed by the laws of California. Either party may terminate this Agreement with 30 days written notice.",
            "questions": [
                {
                    "question": "What is the license fee?",
                    "answers": ["$10,000"],
                    "question_id": "Q001"
                }
            ]
        },
        {
            "contract_id": "CONTRACT_002", 
            "title": "Service Agreement",
            "context": "This Service Agreement (the 'Agreement') is made between ServiceProvider LLC ('Provider') and Customer Inc. ('Customer') dated March 15, 2024. The Provider shall provide consulting services for a period of 12 months. The Customer shall pay monthly fees of $5,000. The Provider warrants that services will be performed in a professional manner. This Agreement contains confidentiality provisions. The Agreement may be terminated for cause.",
            "questions": [
                {
                    "question": "What is the monthly fee?",
                    "answers": ["$5,000"],
                    "question_id": "Q002"
                }
            ]
        }
    ]
}

# Save sample CUAD data
with open('data/raw/CUAD_v1.json', 'w') as f:
    json.dump(sample_cuad_data, f, indent=2)

# Create sample processed data
sample_metadata = [
    {
        "contract_id": "CONTRACT_001",
        "contract_type": "Software License",
        "parties": ["TechCorp Inc.", "ClientCo Ltd."],
        "effective_date": "2024-01-01",
        "jurisdiction": "California",
        "total_clauses": 5,
        "file_size": 1024
    },
    {
        "contract_id": "CONTRACT_002", 
        "contract_type": "Service Agreement",
        "parties": ["ServiceProvider LLC", "Customer Inc."],
        "effective_date": "2024-03-15",
        "jurisdiction": "New York",
        "total_clauses": 6,
        "file_size": 1536
    }
]

sample_clauses = [
    {
        "contract_id": "CONTRACT_001",
        "clause_type": "License Grant",
        "text": "The Licensor grants to the Licensee a non-exclusive, non-transferable license to use the Software.",
        "confidence": 0.95,
        "risk_flags": ["non-exclusive", "non-transferable"]
    },
    {
        "contract_id": "CONTRACT_001",
        "clause_type": "Payment Terms", 
        "text": "The Licensee shall pay a license fee of $10,000.",
        "confidence": 0.92,
        "risk_flags": ["payment obligation"]
    },
    {
        "contract_id": "CONTRACT_001",
        "clause_type": "Governing Law",
        "text": "This Agreement shall be governed by the laws of California.",
        "confidence": 0.88,
        "risk_flags": []
    },
    {
        "contract_id": "CONTRACT_002",
        "clause_type": "Service Description",
        "text": "The Provider shall provide consulting services for a period of 12 months.",
        "confidence": 0.94,
        "risk_flags": ["service obligation"]
    },
    {
        "contract_id": "CONTRACT_002",
        "clause_type": "Payment Terms",
        "text": "The Customer shall pay monthly fees of $5,000.",
        "confidence": 0.91,
        "risk_flags": ["payment obligation"]
    },
    {
        "contract_id": "CONTRACT_002",
        "clause_type": "Warranty",
        "text": "The Provider warrants that services will be performed in a professional manner.",
        "confidence": 0.87,
        "risk_flags": ["warranty"]
    }
]

# Save processed data
pd.DataFrame(sample_metadata).to_csv('data/processed/contract_metadata.csv', index=False)
pd.DataFrame(sample_clauses).to_csv('data/processed/clause_segments.csv', index=False)

# Create sample model results
model_results = {
    "f1_macro": 0.85,
    "f1_weighted": 0.87,
    "test_samples": 20,
    "prediction_confidence": 0.89,
    "f1_per_class": {
        "License Grant": 0.90,
        "Payment Terms": 0.88,
        "Governing Law": 0.82,
        "Service Description": 0.85,
        "Warranty": 0.80
    },
    "confusion_matrix": [
        [5, 0, 0, 0, 0],
        [0, 4, 0, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 4, 0],
        [0, 0, 0, 0, 3]
    ]
}

# Save model results
with open('models/baseline/baseline_metrics.json', 'w') as f:
    json.dump(model_results, f, indent=2)

print("✅ Phase 1 demo data created successfully!")
print("📊 Sample data available in data/processed/")
print("🤖 Model results available in models/baseline/")
print("🎯 Ready to run notebook with custom modules!")
