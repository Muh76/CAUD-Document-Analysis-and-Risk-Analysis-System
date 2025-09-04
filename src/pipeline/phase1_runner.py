#!/usr/bin/env python3
"""
Phase 1 Pipeline Runner for Contract Analysis System
Runs the complete Phase 1 pipeline: parsing, metadata extraction, NER, and validation
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.parsing_pipeline import ContractParser
from src.data.metadata_extractor import MetadataExtractor
from src.data.legal_ner import LegalNER
from src.validation.run_checks import DataValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/phase1_pipeline.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class Phase1Pipeline:
    """Complete Phase 1 pipeline runner"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_directories()
        self.results = {}

    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            "data/raw",
            "data/processed",
            "data/clause_spans",
            "data/metadata",
            "logs",
            "reports",
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def run_parsing_pipeline(self) -> Dict[str, Any]:
        """Run contract parsing pipeline"""
        logger.info("🚀 Starting Contract Parsing Pipeline")
        
        try:
            # Initialize parser
            parser_config = {
                "input_dir": self.config.get("input_dir", "data/raw"),
                "output_dir": self.config.get("output_dir", "data/processed"),
                "min_clause_length": 50,
                "confidence_threshold": 0.5,
            }
            
            parser = ContractParser(parser_config)
            
            # Process contracts
            results = parser.process_contracts_batch(
                parser_config["input_dir"],
                parser_config["output_dir"]
            )
            
            self.results["parsing"] = results
            logger.info(f"✅ Parsing complete: {results['successful_parses']}/{results['total_files']} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Parsing pipeline failed: {e}")
            raise

    def run_metadata_extraction(self) -> Dict[str, Any]:
        """Run metadata extraction pipeline"""
        logger.info("🔍 Starting Metadata Extraction Pipeline")
        
        try:
            # Load parsed data
            clause_file = Path("data/processed/clause_spans.csv")
            if not clause_file.exists():
                logger.warning("No clause data found, skipping metadata extraction")
                return {"status": "skipped", "reason": "no_clause_data"}
            
            df = pd.read_csv(clause_file)
            contracts_data = []
            
            # Group by contract_id
            for contract_id, group in df.groupby("contract_id"):
                text = " ".join(group["text"].fillna(""))
                contracts_data.append({
                    "contract_id": contract_id,
                    "text": text,
                })
            
            # Initialize extractor
            extractor_config = {}
            extractor = MetadataExtractor(extractor_config)
            
            # Process metadata
            results = extractor.process_contracts_batch(contracts_data)
            
            self.results["metadata"] = results
            logger.info(f"✅ Metadata extraction complete: {results['processed']}/{results['total_contracts']} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Metadata extraction failed: {e}")
            raise

    def run_legal_ner(self) -> Dict[str, Any]:
        """Run legal NER pipeline"""
        logger.info("🏷️ Starting Legal NER Pipeline")
        
        try:
            # Load parsed data
            clause_file = Path("data/processed/clause_spans.csv")
            if not clause_file.exists():
                logger.warning("No clause data found, skipping NER")
                return {"status": "skipped", "reason": "no_clause_data"}
            
            df = pd.read_csv(clause_file)
            contracts_data = []
            
            # Group by contract_id
            for contract_id, group in df.groupby("contract_id"):
                text = " ".join(group["text"].fillna(""))
                contracts_data.append({
                    "contract_id": contract_id,
                    "text": text,
                })
            
            # Initialize NER
            ner_config = {}
            ner = LegalNER(ner_config)
            
            # Process entities
            results = ner.process_contracts_batch(contracts_data)
            
            # Save entities
            all_entities = []
            for contract_data in results["entities_by_contract"].values():
                all_entities.extend(contract_data["entities"])
            
            ner.save_entities(all_entities, "data/processed")
            
            self.results["ner"] = results
            logger.info(f"✅ NER complete: {results['processed']}/{results['total_contracts']} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ NER pipeline failed: {e}")
            raise

    def run_data_validation(self) -> Dict[str, Any]:
        """Run data validation pipeline"""
        logger.info("✅ Starting Data Validation Pipeline")
        
        try:
            # Initialize validator
            validator_config = {
                "data_dir": "data/processed",
                "validation_config": "configs/validation.yaml",
            }
            
            validator = DataValidator(validator_config)
            
            # Run validation
            results = validator.validate_all()
            
            self.results["validation"] = results
            logger.info("✅ Data validation complete")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Data validation failed: {e}")
            raise

    def generate_phase1_report(self) -> Dict[str, Any]:
        """Generate comprehensive Phase 1 report"""
        logger.info("📊 Generating Phase 1 Report")
        
        report = {
            "phase": "Phase 1 - Data Foundation",
            "timestamp": pd.Timestamp.now().isoformat(),
            "pipeline_status": "completed",
            "results_summary": {},
            "data_quality_metrics": {},
            "next_steps": [],
        }
        
        # Summarize results
        if "parsing" in self.results:
            parsing = self.results["parsing"]
            report["results_summary"]["parsing"] = {
                "total_files": parsing.get("total_files", 0),
                "successful_parses": parsing.get("successful_parses", 0),
                "failed_parses": parsing.get("failed_parses", 0),
                "total_clauses": parsing.get("total_clauses", 0),
                "avg_clauses_per_contract": parsing.get("total_clauses", 0) / max(parsing.get("successful_parses", 1), 1),
            }
        
        if "metadata" in self.results:
            metadata = self.results["metadata"]
            report["results_summary"]["metadata"] = {
                "total_contracts": metadata.get("total_contracts", 0),
                "processed": metadata.get("processed", 0),
                "failed": metadata.get("failed", 0),
            }
            if "summary_stats" in metadata:
                report["data_quality_metrics"]["metadata"] = metadata["summary_stats"]
        
        if "ner" in self.results:
            ner = self.results["ner"]
            report["results_summary"]["ner"] = {
                "total_contracts": ner.get("total_contracts", 0),
                "processed": ner.get("processed", 0),
                "failed": ner.get("failed", 0),
                "total_entities": ner.get("total_entities", 0),
            }
            if "summary_report" in ner:
                report["data_quality_metrics"]["ner"] = ner["summary_report"]
        
        if "validation" in self.results:
            validation = self.results["validation"]
            report["results_summary"]["validation"] = {
                "status": "completed",
                "checks_passed": validation.get("checks_passed", 0),
                "checks_failed": validation.get("checks_failed", 0),
            }
        
        # Determine next steps
        if self._check_phase1_completion():
            report["next_steps"] = [
                "Phase 2: Model Development",
                "Train baseline models",
                "Implement risk scoring",
                "Develop RAG system",
            ]
        else:
            report["next_steps"] = [
                "Fix data quality issues",
                "Re-run failed components",
                "Validate data pipeline",
            ]
        
        # Save report
        report_file = Path("reports/phase1_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Report saved to {report_file}")
        return report

    def _check_phase1_completion(self) -> bool:
        """Check if Phase 1 objectives are met"""
        required_files = [
            "data/processed/contracts_index.csv",
            "data/processed/clause_spans.csv",
            "data/metadata.db",
            "data/processed/extracted_entities.json",
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                logger.warning(f"Missing required file: {file_path}")
                return False
        
        return True

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete Phase 1 pipeline"""
        logger.info("🎯 Starting Complete Phase 1 Pipeline")
        start_time = time.time()
        
        try:
            # Step 1: Contract Parsing
            self.run_parsing_pipeline()
            
            # Step 2: Metadata Extraction
            self.run_metadata_extraction()
            
            # Step 3: Legal NER
            self.run_legal_ner()
            
            # Step 4: Data Validation
            self.run_data_validation()
            
            # Step 5: Generate Report
            report = self.generate_phase1_report()
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"🎉 Phase 1 Pipeline Complete! Duration: {duration:.2f} seconds")
            
            # Print summary
            self._print_summary()
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Phase 1 Pipeline failed: {e}")
            raise

    def _print_summary(self):
        """Print pipeline summary"""
        print("\n" + "="*60)
        print("🎯 PHASE 1 PIPELINE SUMMARY")
        print("="*60)
        
        if "parsing" in self.results:
            parsing = self.results["parsing"]
            print(f"📄 Contract Parsing: {parsing.get('successful_parses', 0)}/{parsing.get('total_files', 0)} successful")
            print(f"   Total clauses extracted: {parsing.get('total_clauses', 0)}")
        
        if "metadata" in self.results:
            metadata = self.results["metadata"]
            print(f"🔍 Metadata Extraction: {metadata.get('processed', 0)}/{metadata.get('total_contracts', 0)} successful")
        
        if "ner" in self.results:
            ner = self.results["ner"]
            print(f"🏷️ Legal NER: {ner.get('processed', 0)}/{ner.get('total_contracts', 0)} successful")
            print(f"   Total entities extracted: {ner.get('total_entities', 0)}")
        
        if "validation" in self.results:
            validation = self.results["validation"]
            print(f"✅ Data Validation: Completed")
        
        print("\n📁 Output Files:")
        output_files = [
            "data/processed/contracts_index.csv",
            "data/processed/clause_spans.csv",
            "data/metadata.db",
            "data/processed/extracted_entities.json",
            "reports/phase1_report.json",
        ]
        
        for file_path in output_files:
            if Path(file_path).exists():
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path} (missing)")
        
        print("\n🚀 Next Steps:")
        if self._check_phase1_completion():
            print("   • Proceed to Phase 2: Model Development")
            print("   • Train baseline models")
            print("   • Implement risk scoring")
            print("   • Develop RAG system")
        else:
            print("   • Fix data quality issues")
            print("   • Re-run failed components")
            print("   • Validate data pipeline")
        
        print("="*60)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1 Pipeline Runner")
    parser.add_argument("--input", default="data/raw", help="Input directory")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--config", default="configs/phase1.yaml", help="Configuration file")
    parser.add_argument("--skip-parsing", action="store_true", help="Skip parsing step")
    parser.add_argument("--skip-metadata", action="store_true", help="Skip metadata extraction")
    parser.add_argument("--skip-ner", action="store_true", help="Skip NER step")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation step")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        "input_dir": args.input,
        "output_dir": args.output,
        "skip_parsing": args.skip_parsing,
        "skip_metadata": args.skip_metadata,
        "skip_ner": args.skip_ner,
        "skip_validation": args.skip_validation,
    }
    
    # Initialize and run pipeline
    pipeline = Phase1Pipeline(config)
    
    try:
        report = pipeline.run_complete_pipeline()
        print(f"\n📊 Pipeline completed successfully!")
        print(f"📄 Report saved to: reports/phase1_report.json")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
