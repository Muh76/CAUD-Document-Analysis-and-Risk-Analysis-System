# Phase 1: Foundations & Data Pipeline
# Contract Analysis System - Baseline Implementation

## Overview
This notebook demonstrates Phase 1 of our Contract Analysis System:
- CUAD dataset ingestion and processing
- Contract metadata extraction using NER
- Clause segmentation and classification
- Baseline model training (TF-IDF + Logistic Regression)
- Simple risk scoring with keyword patterns
- Data validation with Pandera

**Author**: Mohammad Babaie  
**Email**: mj.babaie@gmail.com  
**LinkedIn**: https://www.linkedin.com/in/mohammadbabaie/  
**GitHub**: https://github.com/Muh76

## Phase 1 Goals
✅ Stand up clean, professional repo  
✅ Reliable data pipeline (CUAD → parsing → segmentation)  
✅ Data contracts + schema validation  
✅ Baseline models (TF-IDF + linear models)  
✅ EDA + class imbalance strategy  
✅ Reproducible dataset + baseline metrics  

## Setup and Imports

```python
# Import required libraries
import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

# Import our custom modules
from data.pipeline import ContractDataPipeline, ContractMetadata, ClauseSegment
from models.baseline_models import BaselineClauseClassifier, KeywordRiskScorer, BaselineEvaluator

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")
```

## 1. Data Pipeline Demonstration

### 1.1 Initialize Pipeline
```python
# Configuration
config = {
    'cuad_file_path': 'data/raw/CUAD_v1.json',
    'output_dir': 'data/processed',
    'validation_enabled': True,
    'logging_level': 'INFO'
}

# Initialize pipeline
pipeline = ContractDataPipeline(config)
print("✅ Pipeline initialized successfully!")
```

### 1.2 Load CUAD Dataset
```python
# Load CUAD dataset
print("📊 Loading CUAD dataset...")
cuad_df = pipeline.load_cuad_dataset(config['cuad_file_path'])

print(f"✅ Loaded {len(cuad_df)} contracts from CUAD dataset")
print(f"\nDataset columns: {list(cuad_df.columns)}")
print(f"\nSample contract titles:")
for i, title in enumerate(cuad_df['title'].head(5)):
    print(f"  {i+1}. {title}")
```

### 1.3 Explore Sample Contract
```python
# Explore a sample contract
sample_contract = cuad_df.iloc[0]
print("📄 Sample Contract Analysis")
print("=" * 50)
print(f"Contract ID: {sample_contract['contract_id']}")
print(f"Title: {sample_contract['title']}")
print(f"Text length: {len(sample_contract['context'])} characters")
print(f"\nFirst 500 characters:")
print(sample_contract['context'][:500] + "...")
```

### 1.4 Process Contract with Metadata Extraction
```python
# Process a sample contract
print("🔍 Processing sample contract...")
metadata, clauses = pipeline.process_contract(
    sample_contract['context'], 
    sample_contract['contract_id']
)

print(f"\n📋 Extracted Metadata:")
print(f"  Contract Type: {metadata.contract_type}")
print(f"  Parties: {metadata.parties}")
print(f"  Effective Date: {metadata.effective_date}")
print(f"  Jurisdiction: {metadata.jurisdiction}")
print(f"  Total Clauses: {metadata.total_clauses}")
print(f"  File Size: {metadata.file_size} bytes")

print(f"\n📝 Segmented Clauses:")
for i, clause in enumerate(clauses[:3]):  # Show first 3 clauses
    print(f"  Clause {i+1}: {clause.clause_type} (confidence: {clause.confidence:.2f})")
    print(f"    Text: {clause.text[:100]}...")
    print(f"    Risk Flags: {clause.risk_flags}")
    print()
```

## 2. Data Analysis & Visualization

### 2.1 Process Multiple Contracts
```python
# Process multiple contracts for analysis
print("🔄 Processing multiple contracts for analysis...")
all_metadata = []
all_clauses = []

# Process first 10 contracts for demonstration
for idx, contract in cuad_df.head(10).iterrows():
    try:
        metadata, clauses = pipeline.process_contract(
            contract['context'], 
            contract['contract_id']
        )
        all_metadata.append(metadata)
        all_clauses.append(clauses)
    except Exception as e:
        print(f"Error processing contract {contract['contract_id']}: {e}")
        continue

print(f"✅ Processed {len(all_metadata)} contracts successfully!")
```

### 2.2 Generate Data Report
```python
# Generate data report
report = pipeline.generate_data_report(all_metadata, all_clauses)

print("📊 Data Analysis Report")
print("=" * 50)
print(f"Total Contracts: {report['summary']['total_contracts']}")
print(f"Total Clauses: {report['summary']['total_clauses']}")
print(f"Avg Clauses per Contract: {report['summary']['avg_clauses_per_contract']:.2f}")
print(f"\nContract Types:")
for contract_type, count in report['contract_types'].items():
    print(f"  {contract_type}: {count}")
print(f"\nClause Types:")
for clause_type, count in report['clause_types'].items():
    print(f"  {clause_type}: {count}")
```

### 2.3 Visualize Data Distribution
```python
# Visualize contract and clause types
plt.figure(figsize=(15, 5))

# Contract types
plt.subplot(1, 3, 1)
contract_types = list(report['contract_types'].keys())
contract_counts = list(report['contract_types'].values())
plt.pie(contract_counts, labels=contract_types, autopct='%1.1f%%')
plt.title('Contract Type Distribution')

# Clause types
plt.subplot(1, 3, 2)
clause_types = list(report['clause_types'].keys())
clause_counts = list(report['clause_types'].values())
plt.bar(range(len(clause_types)), clause_counts)
plt.xticks(range(len(clause_types)), clause_types, rotation=45, ha='right')
plt.title('Clause Type Distribution')
plt.ylabel('Count')

# Risk flags (if any)
plt.subplot(1, 3, 3)
risk_flags = report.get('risk_flags', {})
if risk_flags:
    risk_types = list(risk_flags.keys())
    risk_counts = list(risk_flags.values())
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(risk_types)))
    plt.bar(range(len(risk_types)), risk_counts, color=colors)
    plt.xticks(range(len(risk_types)), risk_types, rotation=45, ha='right')
    plt.title('Risk Flag Distribution')
    plt.ylabel('Count')
else:
    plt.text(0.5, 0.5, 'No risk flags\ndetected', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Risk Flag Distribution')

plt.tight_layout()
plt.show()
```

## 3. Baseline Model Training & Evaluation

### 3.1 Prepare Data for Models
```python
# Prepare data for baseline models
print("🔧 Preparing data for baseline models...")

# Create DataFrame from processed clauses
clauses_data = []
for clauses in all_clauses:
    for clause in clauses:
        clauses_data.append({
            'text': clause.text,
            'clause_type': clause.clause_type,
            'confidence': clause.confidence,
            'risk_flags': clause.risk_flags
        })

clauses_df = pd.DataFrame(clauses_data)
print(f"✅ Prepared {len(clauses_df)} clauses for training")
print(f"Clause types: {clauses_df['clause_type'].value_counts().to_dict()}")
```

### 3.2 Initialize Baseline Models
```python
# Initialize baseline models
model_config = {
    'data_path': 'data/processed/clause_segments.csv',
    'output_dir': 'models/baseline',
    'test_size': 0.2,
    'random_state': 42
}

classifier = BaselineClauseClassifier(model_config)
risk_scorer = KeywordRiskScorer()
evaluator = BaselineEvaluator(model_config)

print("✅ Baseline models initialized!")
```

### 3.3 Train Baseline Classifier
```python
# Prepare data for classification
X, y = classifier.prepare_data(clauses_df)
print(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Clause types: {list(classifier.clause_types)}")

# Split data and train model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🚀 Training baseline classifier...")
classifier_metrics = classifier.train_model(X_train, y_train, model_type='logistic')

print(f"\n📈 Training Results:")
print(f"  Model Type: {classifier_metrics['model_type']}")
print(f"  CV F1 Macro: {classifier_metrics['cv_f1_macro_mean']:.3f} (+/- {classifier_metrics['cv_f1_macro_std']*2:.3f})")
print(f"  Training F1 Macro: {classifier_metrics['f1_macro']:.3f}")
print(f"  Training F1 Weighted: {classifier_metrics['f1_weighted']:.3f}")
print(f"  Training Samples: {classifier_metrics['training_samples']}")
print(f"  Feature Count: {classifier_metrics['feature_count']}")
```

### 3.4 Evaluate Model Performance
```python
# Evaluate classifier
print("🔍 Evaluating classifier...")
classifier_results = evaluator.evaluate_classifier(classifier, X_test, y_test)

print(f"\n📊 Test Results:")
print(f"  Test F1 Macro: {classifier_results['f1_macro']:.3f}")
print(f"  Test F1 Weighted: {classifier_results['f1_weighted']:.3f}")
print(f"  Test Samples: {classifier_results['test_samples']}")
print(f"  Prediction Confidence: {classifier_results['prediction_confidence']:.3f}")

# Show per-class F1 scores
print(f"\n📋 Per-Class F1 Scores:")
for clause_type, f1_score in classifier_results['f1_per_class'].items():
    print(f"  {clause_type}: {f1_score:.3f}")
```

### 3.5 Visualize Model Performance
```python
# Visualize model performance
plt.figure(figsize=(15, 5))

# Confusion matrix
plt.subplot(1, 3, 1)
conf_matrix = np.array(classifier_results['confusion_matrix'])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classifier.clause_types, 
            yticklabels=classifier.clause_types)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Per-class F1 scores
plt.subplot(1, 3, 2)
f1_scores = list(classifier_results['f1_per_class'].values())
clause_types = list(classifier_results['f1_per_class'].keys())
plt.bar(range(len(clause_types)), f1_scores)
plt.xticks(range(len(clause_types)), clause_types, rotation=45, ha='right')
plt.title('Per-Class F1 Scores')
plt.ylabel('F1 Score')

# Feature importance
plt.subplot(1, 3, 3)
feature_importance = classifier.get_feature_importance(top_n=10)
if 'overall' in feature_importance:
    features, scores = zip(*feature_importance['overall'])
    plt.barh(range(len(features)), scores)
    plt.yticks(range(len(features)), features)
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Importance Score')

plt.tight_layout()
plt.show()
```

## 4. Risk Scoring Analysis

### 4.1 Evaluate Risk Scorer
```python
# Evaluate risk scorer
print("🔍 Evaluating risk scorer...")
test_clauses = clauses_df['text'].tolist()
risk_results = evaluator.evaluate_risk_scorer(risk_scorer, test_clauses)

print(f"\n📊 Risk Analysis Results:")
print(f"  Average Risk Score: {risk_results['avg_risk_score']:.3f}")
print(f"  Risk Score Std: {risk_results['std_risk_score']:.3f}")
print(f"  Total Clauses: {risk_results['total_clauses']}")
print(f"  Clauses with Risks: {risk_results['clauses_with_risks']}")
print(f"\nRisk Distribution:")
for level, count in risk_results['risk_distribution'].items():
    print(f"  {level}: {count}")
```

### 4.2 Demonstrate Risk Scoring
```python
# Demonstrate risk scoring on sample clauses
print("🔍 Risk Scoring Examples")
print("=" * 50)

sample_clauses = [
    "The party shall have unlimited liability for all damages.",
    "The party shall use reasonable efforts to complete the work.",
    "Both parties agree to standard terms and conditions.",
    "Either party may terminate this agreement at will.",
    "The contractor shall indemnify the client for any losses."
]

for i, clause in enumerate(sample_clauses, 1):
    risk_result = risk_scorer.score_clause(clause)
    print(f"\nClause {i}: {clause}")
    print(f"  Risk Score: {risk_result['risk_score']:.2f}/10")
    print(f"  Risk Level: {risk_result['risk_level']}")
    print(f"  Detected Risks: {risk_result['detected_risks']}")
```

## 5. Phase 1 Summary & Next Steps

### 5.1 Business Insights
```python
# Business insights analysis
print("💼 Phase 1 Business Insights")
print("=" * 50)

# Contract analysis summary
print("\n📊 Contract Analysis Summary:")
print(f"  • Total contracts analyzed: {report['summary']['total_contracts']}")
print(f"  • Average clauses per contract: {report['summary']['avg_clauses_per_contract']:.1f}")
print(f"  • Most common contract type: {max(report['contract_types'], key=report['contract_types'].get)}")
print(f"  • Most common clause type: {max(report['clause_types'], key=report['clause_types'].get)}")

# Risk analysis summary
print(f"\n⚠️ Risk Analysis Summary:")
print(f"  • Average risk score: {risk_results['avg_risk_score']:.2f}/10")
print(f"  • High-risk clauses: {risk_results['risk_distribution'].get('HIGH', 0)}")
print(f"  • Medium-risk clauses: {risk_results['risk_distribution'].get('MEDIUM', 0)}")
print(f"  • Low-risk clauses: {risk_results['risk_distribution'].get('LOW', 0)}")

# Model performance summary
print(f"\n🤖 Model Performance Summary:")
print(f"  • Classification F1 Score: {classifier_results['f1_macro']:.3f}")
print(f"  • Prediction confidence: {classifier_results['prediction_confidence']:.3f}")
print(f"  • Features used: {classifier_metrics['feature_count']}")
```

### 5.2 Phase 1 Achievements
```python
# Phase 1 achievements
print("\n🎉 Phase 1 Achievements")
print("=" * 40)

achievements = [
    f"✅ Data Pipeline: {report['summary']['total_contracts']} contracts processed",
    f"✅ Baseline Models: F1={classifier_results['f1_macro']:.3f}",
    f"✅ Risk Analysis: {risk_results['clauses_with_risks']} clauses with risks detected",
    f"✅ Data Validation: Pandera schemas implemented",
    f"✅ NER Extraction: Metadata extraction with spaCy",
    f"✅ Reproducible Pipeline: DVC integration ready"
]

for i, achievement in enumerate(achievements, 1):
    print(f"{i}. {achievement}")
```

### 5.3 Next Steps - Phase 2
```python
# Next steps
print("\n🚀 Ready for Phase 2: Advanced Modeling & Risk Scoring")
print("=" * 60)

phase2_goals = [
    "🎯 Fine-tune DistilBERT/Legal-BERT for clause classification",
    "🎯 Implement calibrated probabilities and confidence scoring",
    "🎯 Build advanced risk engine with policy rules + learned anomalies",
    "🎯 Add explainability with token-level highlights + SHAP",
    "🎯 Target macro F1 ≥ strong baseline",
    "🎯 Implement contract type classification",
    "🎯 Add risk trend analysis"
]

for i, goal in enumerate(phase2_goals, 1):
    print(f"{i}. {goal}")

print(f"\n📧 Contact: mj.babaie@gmail.com")
print(f"🔗 LinkedIn: https://www.linkedin.com/in/mohammadbabaie/")
print(f"🐙 GitHub: https://github.com/Muh76")
```
