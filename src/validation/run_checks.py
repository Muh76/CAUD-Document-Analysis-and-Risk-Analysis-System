#!/usr/bin/env python3
"""
Data Validation Module
Phase 1: Great Expectations and Pandera Integration
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandera as pa
from pandera import DataFrameSchema, Column, Check
import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import DataContextConfig
from great_expectations.data_context.types.resource_identifiers import GeCloudIdentifier

class DataValidator:
    """Comprehensive data validation using Pandera and Great Expectations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.setup_great_expectations()
        
    def setup_great_expectations(self):
        """Initialize Great Expectations context"""
        try:
            # Create GE context
            data_context_config = DataContextConfig(
                store_backend_default=ge.data_context.store_backend_defaults.FileSystemStoreBackendDefaults(
                    root_directory="great_expectations"
                ),
                checkpoint_store_name="checkpoint_store",
                evaluation_parameter_store_name="evaluation_parameter_store",
                expectations_store_name="expectations_store",
                validations_store_name="validations_store"
            )
            
            self.ge_context = BaseDataContext(project_config=data_context_config)
            self.logger.info("Great Expectations context initialized")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize Great Expectations: {e}")
            self.ge_context = None
    
    def create_pandera_schemas(self) -> Dict[str, DataFrameSchema]:
        """Create Pandera schemas for data validation"""
        
        # Contract metadata schema
        contract_metadata_schema = DataFrameSchema({
            "contract_id": Column(str, nullable=False, unique=True),
            "contract_type": Column(str, nullable=False),
            "parties": Column(object, nullable=False),  # List of parties
            "effective_date": Column(str, nullable=True),
            "jurisdiction": Column(str, nullable=True),
            "total_clauses": Column(int, Check.greater_than(0), nullable=False),
            "file_size": Column(int, Check.greater_than(0), nullable=False),
            "processing_date": Column(str, nullable=False)
        })
        
        # Clause segments schema
        clause_segments_schema = DataFrameSchema({
            "contract_id": Column(str, nullable=False),
            "clause_id": Column(str, nullable=False),
            "clause_type": Column(str, nullable=False),
            "text": Column(str, Check.str_length(min_value=50, max_value=5000), nullable=False),
            "confidence": Column(float, Check.in_range(0.0, 1.0), nullable=False),
            "start_pos": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
            "end_pos": Column(int, Check.greater_than(0), nullable=False),
            "risk_flags": Column(object, nullable=True),  # List of risk flags
            "entities": Column(object, nullable=True)  # List of entities
        })
        
        # Enriched metadata schema
        enriched_metadata_schema = DataFrameSchema({
            "contract_id": Column(str, nullable=False, unique=True),
            "contract_type": Column(str, nullable=False),
            "parties": Column(object, nullable=False),
            "effective_date": Column(str, nullable=True),
            "jurisdiction": Column(str, nullable=True),
            "governing_law": Column(str, nullable=True),
            "total_clauses": Column(int, Check.greater_than(0), nullable=False),
            "high_risk_clauses": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
            "medium_risk_clauses": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
            "low_risk_clauses": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
            "total_entities": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
            "processing_date": Column(str, nullable=False)
        })
        
        return {
            "contract_metadata": contract_metadata_schema,
            "clause_segments": clause_segments_schema,
            "enriched_metadata": enriched_metadata_schema
        }
    
    def validate_with_pandera(self, data_dir: str) -> Dict[str, Any]:
        """Validate data using Pandera schemas"""
        self.logger.info("Starting Pandera validation...")
        
        schemas = self.create_pandera_schemas()
        validation_results = {}
        
        try:
            # Validate contract metadata
            metadata_path = Path(data_dir) / "contract_metadata.csv"
            if metadata_path.exists():
                metadata_df = pd.read_csv(metadata_path)
                try:
                    schemas["contract_metadata"].validate(metadata_df)
                    validation_results["contract_metadata"] = {"passed": True, "error": None}
                    self.logger.info("Contract metadata validation passed")
                except Exception as e:
                    validation_results["contract_metadata"] = {"passed": False, "error": str(e)}
                    self.logger.error(f"Contract metadata validation failed: {e}")
            
            # Validate clause segments
            clauses_path = Path(data_dir) / "clause_segments.csv"
            if clauses_path.exists():
                clauses_df = pd.read_csv(clauses_path)
                try:
                    schemas["clause_segments"].validate(clauses_df)
                    validation_results["clause_segments"] = {"passed": True, "error": None}
                    self.logger.info("Clause segments validation passed")
                except Exception as e:
                    validation_results["clause_segments"] = {"passed": False, "error": str(e)}
                    self.logger.error(f"Clause segments validation failed: {e}")
            
            # Validate enriched metadata
            enriched_path = Path(data_dir) / "enriched_metadata.csv"
            if enriched_path.exists():
                enriched_df = pd.read_csv(enriched_path)
                try:
                    schemas["enriched_metadata"].validate(enriched_df)
                    validation_results["enriched_metadata"] = {"passed": True, "error": None}
                    self.logger.info("Enriched metadata validation passed")
                except Exception as e:
                    validation_results["enriched_metadata"] = {"passed": False, "error": str(e)}
                    self.logger.error(f"Enriched metadata validation failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error during Pandera validation: {e}")
            validation_results["error"] = {"passed": False, "error": str(e)}
        
        return validation_results
    
    def create_great_expectations_suite(self, suite_name: str = "contract_analysis_suite"):
        """Create Great Expectations validation suite"""
        if not self.ge_context:
            self.logger.warning("Great Expectations not available")
            return None
        
        try:
            # Create suite
            suite = self.ge_context.create_expectation_suite(
                expectation_suite_name=suite_name,
                overwrite_existing=True
            )
            
            # Add expectations for contract metadata
            batch_request = RuntimeBatchRequest(
                datasource_name="contract_data",
                data_connector_name="default_runtime_data_connector_name",
                data_asset_name="contract_metadata",
                runtime_parameters={"path": "data/processed/contract_metadata.csv"},
                batch_identifiers={"default_identifier_name": "default_identifier"}
            )
            
            # Add expectations
            validator = self.ge_context.get_validator(
                batch_request=batch_request,
                expectation_suite=suite
            )
            
            # Contract metadata expectations
            validator.expect_column_to_exist("contract_id")
            validator.expect_column_values_to_be_unique("contract_id")
            validator.expect_column_values_to_not_be_null("contract_type")
            validator.expect_column_values_to_be_between("total_clauses", 1, 1000)
            
            # Clause segments expectations
            validator.expect_column_to_exist("clause_id")
            validator.expect_column_values_to_not_be_null("text")
            validator.expect_column_value_lengths_to_be_between("text", 50, 5000)
            validator.expect_column_values_to_be_between("confidence", 0.0, 1.0)
            
            # Save suite
            validator.save_expectation_suite()
            self.logger.info(f"Great Expectations suite '{suite_name}' created")
            
            return suite
            
        except Exception as e:
            self.logger.error(f"Error creating Great Expectations suite: {e}")
            return None
    
    def validate_with_great_expectations(self, data_dir: str) -> Dict[str, Any]:
        """Validate data using Great Expectations"""
        if not self.ge_context:
            return {"ge_available": False, "error": "Great Expectations not initialized"}
        
        self.logger.info("Starting Great Expectations validation...")
        
        try:
            # Create or load suite
            suite = self.create_great_expectations_suite()
            if not suite:
                return {"ge_validation": {"passed": False, "error": "Could not create validation suite"}}
            
            # Run validation
            checkpoint_config = {
                "name": "contract_validation_checkpoint",
                "config_version": 1.0,
                "class_name": "SimpleCheckpoint",
                "validations": [
                    {
                        "batch_request": {
                            "datasource_name": "contract_data",
                            "data_connector_name": "default_runtime_data_connector_name",
                            "data_asset_name": "contract_metadata",
                            "runtime_parameters": {"path": f"{data_dir}/contract_metadata.csv"},
                            "batch_identifiers": {"default_identifier_name": "default_identifier"}
                        },
                        "expectation_suite_name": "contract_analysis_suite"
                    }
                ]
            }
            
            checkpoint = self.ge_context.add_checkpoint(**checkpoint_config)
            results = checkpoint.run()
            
            # Process results
            validation_results = {
                "ge_validation": {
                    "passed": results.success,
                    "statistics": results.statistics,
                    "run_id": str(results.run_id)
                }
            }
            
            self.logger.info("Great Expectations validation completed")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error during Great Expectations validation: {e}")
            return {"ge_validation": {"passed": False, "error": str(e)}}
    
    def run_comprehensive_validation(self, data_dir: str) -> Dict[str, Any]:
        """Run comprehensive data validation"""
        self.logger.info(f"Starting comprehensive validation for {data_dir}")
        
        # Pandera validation
        pandera_results = self.validate_with_pandera(data_dir)
        
        # Great Expectations validation
        ge_results = self.validate_with_great_expectations(data_dir)
        
        # Combine results
        all_results = {
            "pandera": pandera_results,
            "great_expectations": ge_results,
            "overall_passed": all(
                result.get("passed", False) 
                for result in pandera_results.values() 
                if isinstance(result, dict) and "passed" in result
            )
        }
        
        # Save validation report
        report_path = Path(data_dir) / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        self.logger.info(f"Validation complete. Report saved to {report_path}")
        return all_results

def main():
    """Main function for running validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data validation")
    parser.add_argument("--data_dir", default="data/processed", help="Data directory to validate")
    parser.add_argument("--config", default="configs/parse.yaml", help="Configuration file")
    parser.add_argument("--strict", action="store_true", help="Fail on validation errors")
    
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    validator = DataValidator(config)
    results = validator.run_comprehensive_validation(args.data_dir)
    
    # Display results
    print("\n📊 Validation Results:")
    print("=" * 40)
    
    for validation_type, result in results.items():
        if validation_type == "overall_passed":
            continue
            
        print(f"\n{validation_type.upper()}:")
        if isinstance(result, dict):
            for check_name, check_result in result.items():
                if isinstance(check_result, dict) and "passed" in check_result:
                    status = "✅ PASS" if check_result["passed"] else "❌ FAIL"
                    print(f"  {check_name}: {status}")
                    if not check_result["passed"] and check_result.get("error"):
                        print(f"    Error: {check_result['error']}")
    
    print(f"\nOverall Status: {'✅ PASS' if results['overall_passed'] else '❌ FAIL'}")
    
    # Exit with error code if strict mode and validation failed
    if args.strict and not results['overall_passed']:
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
