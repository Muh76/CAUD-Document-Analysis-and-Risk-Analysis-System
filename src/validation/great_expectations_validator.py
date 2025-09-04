"""
Great Expectations Configuration and Validation System
Complements existing Pandera validation with comprehensive data quality checks
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Great Expectations imports
try:
    import great_expectations as ge
    from great_expectations.core.batch import RuntimeBatchRequest
    from great_expectations.data_context import BaseDataContext
    from great_expectations.data_context.types.base import DataContextConfig
    from great_expectations.data_context.types.resource_identifiers import GeCloudIdentifier
    from great_expectations.core import ExpectationConfiguration
    from great_expectations.execution_engine import PandasExecutionEngine
    from great_expectations.validator.validator import Validator
    GE_AVAILABLE = True
except ImportError:
    GE_AVAILABLE = False
    print("⚠️ Great Expectations not available - install with: pip install great-expectations")

logger = logging.getLogger(__name__)


class GreatExpectationsValidator:
    """Great Expectations validation system for contract data"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.context = None  # Initialize context as None
        self.setup_great_expectations()
        self.create_expectation_suites()

    def setup_great_expectations(self):
        """Setup Great Expectations context"""
        if not GE_AVAILABLE:
            logger.warning("Great Expectations not available")
            return

        try:
            # Create Great Expectations directory structure
            ge_dir = Path("great_expectations")
            ge_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            (ge_dir / "expectations").mkdir(exist_ok=True)
            (ge_dir / "plugins").mkdir(exist_ok=True)
            (ge_dir / "uncommitted").mkdir(exist_ok=True)
            (ge_dir / "checkpoints").mkdir(exist_ok=True)
            
            # Create data context config
            data_context_config = DataContextConfig(
                config_version=3.0,
                plugins_directory="plugins",
                config_variables_file_path=None,
                stores={
                    "expectations_store": {
                        "class_name": "ExpectationsStore",
                        "store_backend": {
                            "class_name": "TupleFilesystemStoreBackend",
                            "base_directory": "expectations"
                        }
                    },
                    "validations_store": {
                        "class_name": "ValidationsStore",
                        "store_backend": {
                            "class_name": "TupleFilesystemStoreBackend",
                            "base_directory": "uncommitted/validations"
                        }
                    },
                    "evaluation_parameter_store": {
                        "class_name": "EvaluationParameterStore",
                        "store_backend": {
                            "class_name": "TupleFilesystemStoreBackend",
                            "base_directory": "uncommitted/evaluation_parameters"
                        }
                    },
                    "checkpoint_store": {
                        "class_name": "CheckpointStore",
                        "store_backend": {
                            "class_name": "TupleFilesystemStoreBackend",
                            "base_directory": "checkpoints"
                        }
                    }
                }
            )
            
            # Initialize data context
            self.context = BaseDataContext(project_config=data_context_config)
            logger.info("Great Expectations context initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup Great Expectations: {e}")
            self.context = None

    def create_expectation_suites(self):
        """Create expectation suites for contract data"""
        if not self.context:
            return

        try:
            # Create contracts expectation suite
            contracts_suite = self.context.create_expectation_suite(
                expectation_suite_name="contracts_suite",
                overwrite_existing=True
            )
            
            # Add expectations for contracts
            expectations = [
                # Required fields
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "contract_id"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "file_name"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "n_clauses"}
                ),
                
                # Data types
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": "contract_id", "type_": "str"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": "n_clauses", "type_": "int"}
                ),
                
                # Value ranges
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={"column": "n_clauses", "min_value": 1, "max_value": 1000}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_in_set",
                    kwargs={"column": "parsed_ok", "value_set": [0, 1]}
                ),
                
                # Uniqueness
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_unique",
                    kwargs={"column": "contract_id"}
                ),
                
                # Completeness
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": "contract_id"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": "file_name"}
                ),
            ]
            
            for expectation in expectations:
                contracts_suite.add_expectation(expectation)
            
            self.context.save_expectation_suite(contracts_suite)
            logger.info("Created contracts expectation suite")
            
            # Create clauses expectation suite
            clauses_suite = self.context.create_expectation_suite(
                expectation_suite_name="clauses_suite",
                overwrite_existing=True
            )
            
            # Add expectations for clauses
            clause_expectations = [
                # Required fields
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "clause_id"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "contract_id"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "text"}
                ),
                
                # Data types
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": "clause_id", "type_": "str"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": "text", "type_": "str"}
                ),
                
                # Text quality
                ExpectationConfiguration(
                    expectation_type="expect_column_value_lengths_to_be_between",
                    kwargs={"column": "text", "min_value": 10, "max_value": 10000}
                ),
                
                # Uniqueness
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_unique",
                    kwargs={"column": "clause_id"}
                ),
                
                # Completeness
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": "clause_id"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": "text"}
                ),
            ]
            
            for expectation in clause_expectations:
                clauses_suite.add_expectation(expectation)
            
            self.context.save_expectation_suite(clauses_suite)
            logger.info("Created clauses expectation suite")
            
        except Exception as e:
            logger.error(f"Failed to create expectation suites: {e}")

    def validate_contracts_data(self, contracts_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate contracts data using Great Expectations"""
        if not self.context or contracts_df.empty:
            return {"status": "skipped", "reason": "no_context_or_data"}
        
        try:
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="pandas_datasource",
                data_connector_name="default_runtime_data_connector_name",
                data_asset_name="contracts_data",
                runtime_parameters={"batch_data": contracts_df},
                batch_identifiers={"default_identifier_name": "default_identifier"}
            )
            
            # Run validation
            results = self.context.run_validation_operator(
                "action_list_operator",
                assets_to_validate=[batch_request],
                expectation_suite_name="contracts_suite"
            )
            
            # Process results
            validation_result = results.run_results[list(results.run_results.keys())[0]]
            success = validation_result.success
            
            return {
                "status": "completed",
                "success": success,
                "statistics": validation_result.statistics,
                "results": validation_result.run_results,
            }
            
        except Exception as e:
            logger.error(f"Contract validation failed: {e}")
            return {"status": "failed", "error": str(e)}

    def validate_clauses_data(self, clauses_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate clauses data using Great Expectations"""
        if not self.context or clauses_df.empty:
            return {"status": "skipped", "reason": "no_context_or_data"}
        
        try:
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="pandas_datasource",
                data_connector_name="default_runtime_data_connector_name",
                data_asset_name="clauses_data",
                runtime_parameters={"batch_data": clauses_df},
                batch_identifiers={"default_identifier_name": "default_identifier"}
            )
            
            # Run validation
            results = self.context.run_validation_operator(
                "action_list_operator",
                assets_to_validate=[batch_request],
                expectation_suite_name="clauses_suite"
            )
            
            # Process results
            validation_result = results.run_results[list(results.run_results.keys())[0]]
            success = validation_result.success
            
            return {
                "status": "completed",
                "success": success,
                "statistics": validation_result.statistics,
                "results": validation_result.run_results,
            }
            
        except Exception as e:
            logger.error(f"Clauses validation failed: {e}")
            return {"status": "failed", "error": str(e)}

    def generate_data_quality_report(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "validation_summary": {},
            "quality_metrics": {},
            "recommendations": [],
        }
        
        # Process validation results
        for data_type, result in validation_results.items():
            if result.get("status") == "completed":
                report["validation_summary"][data_type] = {
                    "success": result.get("success", False),
                    "statistics": result.get("statistics", {}),
                }
                
                # Add quality metrics
                if "statistics" in result:
                    stats = result["statistics"]
                    report["quality_metrics"][data_type] = {
                        "evaluated_expectations": stats.get("evaluated_expectations", 0),
                        "successful_expectations": stats.get("successful_expectations", 0),
                        "unsuccessful_expectations": stats.get("unsuccessful_expectations", 0),
                        "success_percent": stats.get("success_percent", 0),
                    }
                    
                    # Add recommendations
                    if stats.get("success_percent", 100) < 90:
                        report["recommendations"].append(
                            f"Improve {data_type} data quality - {stats.get('success_percent', 0)}% success rate"
                        )
            else:
                report["validation_summary"][data_type] = {
                    "status": result.get("status", "unknown"),
                    "reason": result.get("reason", "unknown"),
                }
        
        return report

    def run_comprehensive_validation(self, data_dir: str = "data/processed") -> Dict[str, Any]:
        """Run comprehensive validation on all data files"""
        logger.info("Starting comprehensive data validation")
        
        results = {}
        
        # Validate contracts data
        contracts_file = Path(data_dir) / "contracts_index.csv"
        if contracts_file.exists():
            contracts_df = pd.read_csv(contracts_file)
            results["contracts"] = self.validate_contracts_data(contracts_df)
            logger.info(f"Validated contracts data: {len(contracts_df)} records")
        else:
            results["contracts"] = {"status": "skipped", "reason": "file_not_found"}
        
        # Validate clauses data
        clauses_file = Path(data_dir) / "clause_spans.csv"
        if clauses_file.exists():
            clauses_df = pd.read_csv(clauses_file)
            results["clauses"] = self.validate_clauses_data(clauses_df)
            logger.info(f"Validated clauses data: {len(clauses_df)} records")
        else:
            results["clauses"] = {"status": "skipped", "reason": "file_not_found"}
        
        # Generate report
        report = self.generate_data_quality_report(results)
        
        # Save report
        report_file = Path("reports") / "great_expectations_report.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {report_file}")
        return report


def main():
    """Main function for Great Expectations validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Great Expectations Data Validation")
    parser.add_argument("--data-dir", default="data/processed", help="Data directory")
    parser.add_argument("--config", default="configs/validation.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize validator and run validation
    config = {}
    validator = GreatExpectationsValidator(config)
    report = validator.run_comprehensive_validation(args.data_dir)
    
    # Print summary
    print(f"\n📊 Great Expectations Validation Results:")
    print(f"Timestamp: {report['timestamp']}")
    
    for data_type, summary in report["validation_summary"].items():
        if summary.get("status") == "completed":
            success = summary.get("success", False)
            print(f"{data_type.title()}: {'✅ PASSED' if success else '❌ FAILED'}")
        else:
            print(f"{data_type.title()}: ⚠️ {summary.get('status', 'unknown')}")
    
    if report["recommendations"]:
        print(f"\n🔧 Recommendations:")
        for rec in report["recommendations"]:
            print(f"   • {rec}")


if __name__ == "__main__":
    main()
