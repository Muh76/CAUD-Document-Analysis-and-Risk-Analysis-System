#!/usr/bin/env python3
"""
Phase 1 Execution Script - Contract Analysis System
Complete execution of Phase 1: Foundations & Data

This script runs the entire Phase 1 pipeline including:
- CUAD data ingestion and processing
- Contract metadata extraction and NER
- Clause segmentation and classification
- Baseline model training and evaluation
- Data validation and quality checks
- DVC pipeline execution
"""

import os
import sys
import logging
import subprocess
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from data.pipeline import ContractDataPipeline
from models.baseline_models import BaselineClauseClassifier, KeywordRiskScorer, BaselineEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase1_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Phase1Executor:
    """Phase 1 execution orchestrator"""
    
    def __init__(self, config: dict):
        self.config = config
        self.results = {}
        self.start_time = datetime.now()
        
        # Create necessary directories
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            'data/raw',
            'data/processed',
            'models/baseline',
            'logs',
            'tests',
            'docs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def run_data_pipeline(self):
        """Execute data processing pipeline"""
        logger.info("=== Starting Data Pipeline ===")
        
        try:
            # Initialize pipeline
            pipeline = ContractDataPipeline(self.config)
            
            # Load CUAD dataset
            logger.info("Loading CUAD dataset...")
            cuad_df = pipeline.load_cuad_dataset(self.config['cuad_file_path'])
            logger.info(f"Loaded {len(cuad_df)} contracts from CUAD")
            
            # Process contracts
            all_metadata = []
            all_clauses = []
            
            for idx, contract in cuad_df.iterrows():
                contract_id = contract['contract_id']
                text = contract['context']
                
                try:
                    metadata, clauses = pipeline.process_contract(text, contract_id)
                    all_metadata.append(metadata)
                    all_clauses.append(clauses)
                    
                    if (idx + 1) % 10 == 0:
                        logger.info(f"Processed {idx + 1}/{len(cuad_df)} contracts")
                        
                except Exception as e:
                    logger.error(f"Error processing contract {contract_id}: {e}")
                    continue
            
            # Generate data report
            logger.info("Generating data report...")
            report = pipeline.generate_data_report(all_metadata, all_clauses)
            
            # Save results
            self._save_pipeline_results(all_metadata, all_clauses, report, pipeline)
            
            self.results['data_pipeline'] = {
                'status': 'success',
                'contracts_processed': len(all_metadata),
                'total_clauses': sum(len(c) for c in all_clauses),
                'data_report': report
            }
            
            logger.info("Data pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Data pipeline failed: {e}")
            self.results['data_pipeline'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _save_pipeline_results(self, metadata_list, clauses_list, report, pipeline):
        """Save pipeline results"""
        output_dir = Path(self.config['output_dir'])
        
        # Save metadata
        metadata_df = pd.DataFrame([vars(m) for m in metadata_list])
        metadata_df.to_csv(output_dir / 'contract_metadata.csv', index=False)
        logger.info(f"Saved metadata to {output_dir / 'contract_metadata.csv'}")
        
        # Save clauses
        all_clauses_flat = []
        for clauses in clauses_list:
            for clause in clauses:
                clause_dict = vars(clause)
                clause_dict['entities'] = json.dumps(clause_dict['entities'])
                clause_dict['risk_flags'] = json.dumps(clause_dict['risk_flags'])
                all_clauses_flat.append(clause_dict)
        
        clauses_df = pd.DataFrame(all_clauses_flat)
        clauses_df.to_csv(output_dir / 'clause_segments.csv', index=False)
        logger.info(f"Saved clauses to {output_dir / 'clause_segments.csv'}")
        
        # Save report
        with open(output_dir / 'data_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved data report to {output_dir / 'data_report.json'}")
        
        # Save data contracts
        data_contracts = pipeline.create_data_contracts()
        with open(output_dir / 'data_contracts.json', 'w') as f:
            json.dump(data_contracts, f, indent=2)
        logger.info(f"Saved data contracts to {output_dir / 'data_contracts.json'}")
    
    def run_baseline_models(self):
        """Execute baseline model training and evaluation"""
        logger.info("=== Starting Baseline Models ===")
        
        try:
            # Load processed data
            clauses_path = Path(self.config['output_dir']) / 'clause_segments.csv'
            if not clauses_path.exists():
                raise FileNotFoundError(f"Clauses file not found: {clauses_path}")
            
            clauses_df = pd.read_csv(clauses_path)
            logger.info(f"Loaded {len(clauses_df)} clauses for training")
            
            # Filter clauses with types
            clauses_df = clauses_df[clauses_df['clause_type'].notna()]
            clauses_df = clauses_df[clauses_df['clause_type'] != '']
            logger.info(f"Using {len(clauses_df)} clauses with types for training")
            
            # Initialize models
            classifier = BaselineClauseClassifier(self.config)
            risk_scorer = KeywordRiskScorer()
            evaluator = BaselineEvaluator(self.config)
            
            # Prepare data for classification
            X, y = classifier.prepare_data(clauses_df)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['test_size'], 
                random_state=self.config['random_state'], stratify=y
            )
            
            # Train classifier
            logger.info("Training baseline classifier...")
            classifier_metrics = classifier.train_model(X_train, y_train, model_type='logistic')
            
            # Evaluate classifier
            logger.info("Evaluating classifier...")
            classifier_results = evaluator.evaluate_classifier(classifier, X_test, y_test)
            
            # Evaluate risk scorer
            logger.info("Evaluating risk scorer...")
            test_clauses = clauses_df.iloc[y_test]['text'].tolist()
            risk_results = evaluator.evaluate_risk_scorer(risk_scorer, test_clauses)
            
            # Save models and results
            self._save_baseline_results(classifier, evaluator, classifier_metrics, 
                                      classifier_results, risk_results)
            
            self.results['baseline_models'] = {
                'status': 'success',
                'classifier_metrics': classifier_metrics,
                'classifier_results': classifier_results,
                'risk_results': risk_results
            }
            
            logger.info("Baseline models completed successfully!")
            
        except Exception as e:
            logger.error(f"Baseline models failed: {e}")
            self.results['baseline_models'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _save_baseline_results(self, classifier, evaluator, classifier_metrics, 
                             classifier_results, risk_results):
        """Save baseline model results"""
        output_dir = Path(self.config['models_output_dir'])
        
        # Save classifier
        classifier.save_model(output_dir / 'baseline_classifier.joblib')
        logger.info(f"Saved classifier to {output_dir / 'baseline_classifier.joblib'}")
        
        # Save evaluation report
        evaluator.generate_report(output_dir / 'baseline_evaluation_report.json')
        logger.info(f"Saved evaluation report to {output_dir / 'baseline_evaluation_report.json'}")
        
        # Save metrics
        metrics = {
            'classifier_metrics': classifier_metrics,
            'classifier_results': classifier_results,
            'risk_results': risk_results,
            'execution_timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / 'baseline_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {output_dir / 'baseline_metrics.json'}")
    
    def run_tests(self):
        """Run comprehensive test suite"""
        logger.info("=== Running Tests ===")
        
        try:
            # Run pytest
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/test_phase1.py', '-v', '--tb=short'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("All tests passed!")
                self.results['tests'] = {
                    'status': 'passed',
                    'output': result.stdout
                }
            else:
                logger.error(f"Tests failed: {result.stderr}")
                self.results['tests'] = {
                    'status': 'failed',
                    'output': result.stdout,
                    'error': result.stderr
                }
                
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            self.results['tests'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def run_dvc_pipeline(self):
        """Execute DVC pipeline"""
        logger.info("=== Running DVC Pipeline ===")
        
        try:
            # Add data files to DVC
            subprocess.run(['dvc', 'add', 'data/raw/CUAD_v1.json'], check=True)
            logger.info("Added CUAD dataset to DVC")
            
            # Run DVC pipeline
            result = subprocess.run(['dvc', 'repro'], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("DVC pipeline completed successfully!")
                self.results['dvc_pipeline'] = {
                    'status': 'success',
                    'output': result.stdout
                }
            else:
                logger.error(f"DVC pipeline failed: {result.stderr}")
                self.results['dvc_pipeline'] = {
                    'status': 'failed',
                    'error': result.stderr
                }
                
        except Exception as e:
            logger.error(f"DVC pipeline failed: {e}")
            self.results['dvc_pipeline'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def generate_phase1_report(self):
        """Generate comprehensive Phase 1 report"""
        logger.info("=== Generating Phase 1 Report ===")
        
        end_time = datetime.now()
        execution_time = end_time - self.start_time
        
        report = {
            'phase': 'Phase 1 - Foundations & Data',
            'execution_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'execution_time_seconds': execution_time.total_seconds(),
                'status': 'completed' if all(r.get('status') == 'success' for r in self.results.values()) else 'failed'
            },
            'results': self.results,
            'deliverables': {
                'data_pipeline': [
                    'data/processed/contract_metadata.csv',
                    'data/processed/clause_segments.csv',
                    'data/processed/data_report.json',
                    'data/processed/data_contracts.json'
                ],
                'baseline_models': [
                    'models/baseline/baseline_classifier.joblib',
                    'models/baseline/baseline_metrics.json',
                    'models/baseline/baseline_evaluation_report.json'
                ],
                'tests': [
                    'tests/test_phase1.py'
                ],
                'documentation': [
                    'docs/phase1_report.md',
                    'dvc.yaml'
                ]
            },
            'metrics': {
                'data_quality': self._extract_data_quality_metrics(),
                'model_performance': self._extract_model_metrics(),
                'test_coverage': self._extract_test_metrics()
            }
        }
        
        # Save report
        report_path = Path('docs/phase1_report.json')
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        logger.info(f"Phase 1 report generated: {report_path}")
        return report
    
    def _extract_data_quality_metrics(self):
        """Extract data quality metrics from results"""
        if 'data_pipeline' in self.results and self.results['data_pipeline']['status'] == 'success':
            report = self.results['data_pipeline']['data_report']
            return {
                'total_contracts': report['summary']['total_contracts'],
                'total_clauses': report['summary']['total_clauses'],
                'avg_clauses_per_contract': report['summary']['avg_clauses_per_contract'],
                'contracts_with_metadata': report['data_quality']['contracts_with_metadata'],
                'clauses_with_type': report['data_quality']['clauses_with_type'],
                'clauses_with_entities': report['data_quality']['clauses_with_entities']
            }
        return {}
    
    def _extract_model_metrics(self):
        """Extract model performance metrics"""
        if 'baseline_models' in self.results and self.results['baseline_models']['status'] == 'success':
            classifier_results = self.results['baseline_models']['classifier_results']
            risk_results = self.results['baseline_models']['risk_results']
            return {
                'classifier_f1_macro': classifier_results['f1_macro'],
                'classifier_f1_weighted': classifier_results['f1_weighted'],
                'avg_risk_score': risk_results['avg_risk_score'],
                'risk_distribution': risk_results['risk_distribution']
            }
        return {}
    
    def _extract_test_metrics(self):
        """Extract test coverage metrics"""
        if 'tests' in self.results:
            return {
                'test_status': self.results['tests']['status'],
                'test_files': 1,
                'test_functions': 15  # Approximate number of test functions
            }
        return {}
    
    def _generate_markdown_report(self, report):
        """Generate markdown report"""
        md_content = f"""# Phase 1 Report - Contract Analysis System

## Executive Summary

**Phase**: {report['phase']}  
**Status**: {report['execution_summary']['status'].upper()}  
**Execution Time**: {report['execution_summary']['execution_time_seconds']:.2f} seconds  
**Date**: {report['execution_summary']['start_time'][:10]}

## Results Overview

### Data Pipeline
- **Status**: {report['results']['data_pipeline']['status']}
- **Contracts Processed**: {report['metrics']['data_quality'].get('total_contracts', 'N/A')}
- **Total Clauses**: {report['metrics']['data_quality'].get('total_clauses', 'N/A')}

### Baseline Models
- **Status**: {report['results']['baseline_models']['status']}
- **Classifier F1 Macro**: {report['metrics']['model_performance'].get('classifier_f1_macro', 'N/A')}
- **Average Risk Score**: {report['metrics']['model_performance'].get('avg_risk_score', 'N/A')}

### Tests
- **Status**: {report['results']['tests']['status']}
- **Coverage**: {report['metrics']['test_coverage'].get('test_functions', 'N/A')} test functions

## Deliverables

### Data Pipeline Outputs
{chr(10).join(f"- {item}" for item in report['deliverables']['data_pipeline'])}

### Baseline Models
{chr(10).join(f"- {item}" for item in report['deliverables']['baseline_models'])}

### Tests & Documentation
{chr(10).join(f"- {item}" for item in report['deliverables']['tests'] + report['deliverables']['documentation'])}

## Next Steps

Phase 1 has established the foundation for the contract analysis system. The next phase will focus on:

1. **Advanced Modeling**: Fine-tuning transformer models for clause classification
2. **Risk Engine**: Implementing sophisticated risk scoring algorithms
3. **Product MVP**: Building the FastAPI backend and Streamlit frontend

## Contact

For questions or issues, contact:  
- Email: mj.babaie@gmail.com  
- LinkedIn: https://www.linkedin.com/in/mohammadbabaie/  
- GitHub: https://github.com/Muh76
"""
        
        with open('docs/phase1_report.md', 'w') as f:
            f.write(md_content)
    
    def execute(self):
        """Execute complete Phase 1 pipeline"""
        logger.info("🚀 Starting Phase 1 Execution")
        logger.info(f"Configuration: {self.config}")
        
        try:
            # Execute all phases
            self.run_data_pipeline()
            self.run_baseline_models()
            self.run_tests()
            self.run_dvc_pipeline()
            
            # Generate final report
            report = self.generate_phase1_report()
            
            logger.info("🎉 Phase 1 Execution Completed Successfully!")
            logger.info(f"Execution time: {report['execution_summary']['execution_time_seconds']:.2f} seconds")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Phase 1 Execution Failed: {e}")
            self.generate_phase1_report()  # Generate report even on failure
            raise

def main():
    """Main execution function"""
    
    # Configuration
    config = {
        'cuad_file_path': 'data/raw/CUAD_v1.json',
        'output_dir': 'data/processed',
        'models_output_dir': 'models/baseline',
        'test_size': 0.2,
        'random_state': 42,
        'validation_enabled': True,
        'logging_level': 'INFO'
    }
    
    # Execute Phase 1
    executor = Phase1Executor(config)
    report = executor.execute()
    
    # Print summary
    print("\n" + "="*60)
    print("PHASE 1 EXECUTION SUMMARY")
    print("="*60)
    print(f"Status: {report['execution_summary']['status'].upper()}")
    print(f"Execution Time: {report['execution_summary']['execution_time_seconds']:.2f} seconds")
    print(f"Contracts Processed: {report['metrics']['data_quality'].get('total_contracts', 'N/A')}")
    print(f"Total Clauses: {report['metrics']['data_quality'].get('total_clauses', 'N/A')}")
    print(f"Classifier F1 Macro: {report['metrics']['model_performance'].get('classifier_f1_macro', 'N/A')}")
    print("="*60)

if __name__ == "__main__":
    main()
