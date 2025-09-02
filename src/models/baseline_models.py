"""
Baseline Models for Contract Analysis - Phase 1
Implements TF-IDF + linear models for clause classification and simple keyword risk heuristics
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineClauseClassifier:
    """Baseline clause classification using TF-IDF and linear models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.model = None
        self.clause_types = None
        self.feature_names = None
        
    def prepare_data(self, clauses_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        logger.info("Preparing data for baseline classifier")
        
        # Extract text and labels
        texts = clauses_df['text'].fillna('').astype(str)
        labels = clauses_df['clause_type'].fillna('unknown')
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.clause_types = self.label_encoder.classes_
        
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"Prepared {X.shape[0]} clauses with {X.shape[1]} features")
        logger.info(f"Clause types: {list(self.clause_types)}")
        
        return X, encoded_labels
    
    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str = 'logistic') -> Dict[str, Any]:
        """Train baseline model"""
        logger.info(f"Training {model_type} baseline model")
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(X, y)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='f1_macro')
        
        # Predictions
        y_pred = self.model.predict(X)
        
        # Metrics
        metrics = {
            'model_type': model_type,
            'cv_f1_macro_mean': cv_scores.mean(),
            'cv_f1_macro_std': cv_scores.std(),
            'f1_macro': f1_score(y, y_pred, average='macro'),
            'f1_weighted': f1_score(y, y_pred, average='weighted'),
            'training_samples': len(y),
            'feature_count': X.shape[1],
            'clause_types': list(self.clause_types),
            'training_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Model trained successfully!")
        logger.info(f"CV F1 Macro: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        logger.info(f"Training F1 Macro: {metrics['f1_macro']:.3f}")
        
        return metrics
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict clause types and confidence scores"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Vectorize texts
        X = self.vectorizer.transform(texts)
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Get confidence scores (max probability)
        confidence_scores = np.max(probabilities, axis=1)
        
        return predictions, confidence_scores
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """Get feature importance for each clause type"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        feature_importance = {}
        
        if hasattr(self.model, 'coef_'):
            # Logistic Regression
            for i, clause_type in enumerate(self.clause_types):
                coef = self.model.coef_[i]
                feature_scores = list(zip(self.feature_names, coef))
                feature_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                feature_importance[clause_type] = feature_scores[:top_n]
        elif hasattr(self.model, 'feature_importances_'):
            # Random Forest
            importances = self.model.feature_importances_
            feature_scores = list(zip(self.feature_names, importances))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            feature_importance['overall'] = feature_scores[:top_n]
        
        return feature_importance
    
    def save_model(self, model_path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'clause_types': self.clause_types,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_path: str) -> 'BaselineClauseClassifier':
        """Load trained model"""
        model_data = joblib.load(model_path)
        
        classifier = cls(model_data['config'])
        classifier.model = model_data['model']
        classifier.vectorizer = model_data['vectorizer']
        classifier.label_encoder = model_data['label_encoder']
        classifier.clause_types = model_data['clause_types']
        classifier.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {model_path}")
        return classifier

class KeywordRiskScorer:
    """Simple keyword-based risk scoring system"""
    
    def __init__(self):
        self.risk_patterns = {
            'high_risk': {
                'uncapped_liability': [
                    r'\b(?:unlimited|uncapped|no\s+limit)\s+liability',
                    r'\b(?:indemnify|hold\s+harmless)\s+(?:all|any|every)',
                    r'\b(?:penalty|liquidated\s+damages)\s+(?:of|in\s+amount)',
                ],
                'unilateral_termination': [
                    r'\b(?:terminate|cancel)\s+(?:at\s+will|without\s+cause)',
                    r'\b(?:immediate|instant)\s+(?:termination|cancellation)',
                ],
                'broad_scope': [
                    r'\b(?:all|any|every)\s+(?:damages|losses|claims)',
                    r'\b(?:unlimited|broad)\s+(?:scope|application)',
                ]
            },
            'medium_risk': {
                'vague_terms': [
                    r'\b(?:reasonable|appropriate|suitable)\s+(?:efforts|time)',
                    r'\b(?:best\s+efforts?|commercially\s+reasonable)',
                ],
                'flexible_pricing': [
                    r'\b(?:price\s+adjustment|rate\s+change)',
                    r'\b(?:market\s+rate|prevailing\s+rate)',
                ],
                'broad_termination': [
                    r'\b(?:terminate|cancel)\s+(?:with\s+notice)',
                    r'\b(?:breach|default)\s+(?:termination)',
                ]
            },
            'low_risk': {
                'standard_terms': [
                    r'\b(?:standard|usual|customary)\s+(?:terms|conditions)',
                    r'\b(?:mutual|both\s+parties)\s+(?:agree|consent)',
                ],
                'limited_scope': [
                    r'\b(?:limited|restricted)\s+(?:scope|application)',
                    r'\b(?:specific|defined)\s+(?:terms|conditions)',
                ]
            }
        }
        
        self.risk_weights = {
            'high_risk': 3.0,
            'medium_risk': 2.0,
            'low_risk': 1.0
        }
    
    def score_clause(self, text: str) -> Dict[str, Any]:
        """Score risk for a single clause"""
        text_lower = text.lower()
        risk_scores = {}
        total_score = 0
        detected_risks = []
        
        for risk_level, patterns in self.risk_patterns.items():
            level_score = 0
            for risk_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    if matches:
                        level_score += len(matches)
                        detected_risks.append({
                            'risk_type': risk_type,
                            'risk_level': risk_level,
                            'pattern': pattern,
                            'matches': len(matches)
                        })
            
            risk_scores[risk_level] = level_score
            total_score += level_score * self.risk_weights[risk_level]
        
        # Normalize score (0-10 scale)
        normalized_score = min(10.0, total_score / 5.0)
        
        return {
            'risk_score': normalized_score,
            'risk_level': self._get_risk_level(normalized_score),
            'risk_breakdown': risk_scores,
            'detected_risks': detected_risks,
            'text_length': len(text)
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Convert score to risk level"""
        if score >= 7.0:
            return 'HIGH'
        elif score >= 4.0:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def score_contract(self, clauses: List[str]) -> Dict[str, Any]:
        """Score risk for entire contract"""
        clause_scores = []
        total_risk_score = 0
        
        for i, clause in enumerate(clauses):
            score_result = self.score_clause(clause)
            clause_scores.append({
                'clause_index': i,
                'clause_text': clause[:100] + '...' if len(clause) > 100 else clause,
                'risk_score': score_result['risk_score'],
                'risk_level': score_result['risk_level'],
                'detected_risks': score_result['detected_risks']
            })
            total_risk_score += score_result['risk_score']
        
        # Calculate contract-level metrics
        avg_risk_score = total_risk_score / len(clauses) if clauses else 0
        high_risk_clauses = sum(1 for score in clause_scores if score['risk_level'] == 'HIGH')
        medium_risk_clauses = sum(1 for score in clause_scores if score['risk_level'] == 'MEDIUM')
        
        return {
            'contract_risk_score': avg_risk_score,
            'contract_risk_level': self._get_risk_level(avg_risk_score),
            'total_clauses': len(clauses),
            'high_risk_clauses': high_risk_clauses,
            'medium_risk_clauses': medium_risk_clauses,
            'low_risk_clauses': len(clauses) - high_risk_clauses - medium_risk_clauses,
            'clause_scores': clause_scores,
            'risk_distribution': {
                'high': high_risk_clauses,
                'medium': medium_risk_clauses,
                'low': len(clauses) - high_risk_clauses - medium_risk_clauses
            }
        }

class BaselineEvaluator:
    """Evaluate baseline models and generate reports"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
    
    def evaluate_classifier(self, classifier: BaselineClauseClassifier, 
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate clause classifier"""
        logger.info("Evaluating baseline classifier")
        
        # Predictions
        y_pred = classifier.model.predict(X_test)
        y_proba = classifier.model.predict_proba(X_test)
        
        # Metrics
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=classifier.clause_types,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        feature_importance = classifier.get_feature_importance(top_n=10)
        
        results = {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': dict(zip(classifier.clause_types, f1_per_class)),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance,
            'test_samples': len(y_test),
            'prediction_confidence': np.mean(np.max(y_proba, axis=1)),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        self.results['classifier'] = results
        
        logger.info(f"Classifier F1 Macro: {f1_macro:.3f}")
        logger.info(f"Classifier F1 Weighted: {f1_weighted:.3f}")
        
        return results
    
    def evaluate_risk_scorer(self, risk_scorer: KeywordRiskScorer, 
                           test_clauses: List[str]) -> Dict[str, Any]:
        """Evaluate risk scorer"""
        logger.info("Evaluating baseline risk scorer")
        
        # Score all clauses
        clause_scores = []
        for clause in test_clauses:
            score_result = risk_scorer.score_clause(clause)
            clause_scores.append(score_result)
        
        # Aggregate statistics
        risk_scores = [score['risk_score'] for score in clause_scores]
        risk_levels = [score['risk_level'] for score in clause_scores]
        
        # Risk level distribution
        risk_distribution = {}
        for level in ['HIGH', 'MEDIUM', 'LOW']:
            risk_distribution[level] = risk_levels.count(level)
        
        # Risk type distribution
        risk_type_distribution = {}
        for score in clause_scores:
            for risk in score['detected_risks']:
                risk_type = risk['risk_type']
                risk_type_distribution[risk_type] = risk_type_distribution.get(risk_type, 0) + 1
        
        results = {
            'avg_risk_score': np.mean(risk_scores),
            'std_risk_score': np.std(risk_scores),
            'risk_distribution': risk_distribution,
            'risk_type_distribution': risk_type_distribution,
            'total_clauses': len(test_clauses),
            'clauses_with_risks': sum(1 for score in clause_scores if score['detected_risks']),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        self.results['risk_scorer'] = results
        
        logger.info(f"Average risk score: {results['avg_risk_score']:.3f}")
        logger.info(f"Risk distribution: {risk_distribution}")
        
        return results
    
    def generate_report(self, output_path: str):
        """Generate comprehensive evaluation report"""
        logger.info(f"Generating evaluation report to {output_path}")
        
        report = {
            'evaluation_summary': {
                'total_evaluations': len(self.results),
                'evaluation_timestamp': datetime.now().isoformat()
            },
            'results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Evaluation report generated successfully")

def main():
    """Main execution function for baseline models"""
    logger.info("Starting Baseline Models Training and Evaluation")
    
    # Configuration
    config = {
        'data_path': 'data/processed/clause_segments.csv',
        'output_dir': 'models/baseline',
        'test_size': 0.2,
        'random_state': 42
    }
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {config['data_path']}")
    clauses_df = pd.read_csv(config['data_path'])
    
    # Filter out clauses without types
    clauses_df = clauses_df[clauses_df['clause_type'].notna()]
    clauses_df = clauses_df[clauses_df['clause_type'] != '']
    
    logger.info(f"Loaded {len(clauses_df)} clauses with types")
    
    # Initialize models
    classifier = BaselineClauseClassifier(config)
    risk_scorer = KeywordRiskScorer()
    evaluator = BaselineEvaluator(config)
    
    # Prepare data for classification
    X, y = classifier.prepare_data(clauses_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], 
        random_state=config['random_state'], stratify=y
    )
    
    # Train classifier
    logger.info("Training baseline classifier")
    classifier_metrics = classifier.train_model(X_train, y_train, model_type='logistic')
    
    # Evaluate classifier
    classifier_results = evaluator.evaluate_classifier(classifier, X_test, y_test)
    
    # Evaluate risk scorer
    test_clauses = clauses_df.iloc[y_test]['text'].tolist()
    risk_results = evaluator.evaluate_risk_scorer(risk_scorer, test_clauses)
    
    # Save models
    classifier.save_model(output_dir / 'baseline_classifier.joblib')
    
    # Save results
    evaluator.generate_report(output_dir / 'baseline_evaluation_report.json')
    
    # Save metrics
    with open(output_dir / 'baseline_metrics.json', 'w') as f:
        json.dump({
            'classifier_metrics': classifier_metrics,
            'classifier_results': classifier_results,
            'risk_results': risk_results
        }, f, indent=2)
    
    logger.info("Baseline models training and evaluation completed!")
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
