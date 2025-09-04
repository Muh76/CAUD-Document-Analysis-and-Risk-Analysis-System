#!/usr/bin/env python3
"""
Simple Phase 1 Pipeline Runner
Runs the complete Phase 1 pipeline with proper imports
"""

import os
import sys
import logging
import json
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

from src.data.parsing_pipeline import ContractParser
from src.data.metadata_extractor import MetadataExtractor
from src.data.legal_ner import LegalNER

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_phase1_pipeline():
    """Run the complete Phase 1 pipeline"""
    print("🎯 Starting Phase 1 Pipeline")
    print("=" * 50)
    
    # Create directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    try:
        # Step 1: Process CUAD data with parsing pipeline
        print("📄 Step 1: Processing CUAD data...")
        
        # Load CUAD data
        cuad_file = Path("data/raw/CUAD_v1.json")
        if cuad_file.exists():
            import json
            with open(cuad_file, 'r') as f:
                cuad_data = json.load(f)
            
            print(f"   Found {len(cuad_data['data'])} contracts in CUAD dataset")
            
            # Process contracts
            contracts = []
            clauses = []
            
            for i, contract in enumerate(cuad_data['data']):
                contract_id = f"cuad_contract_{i:04d}"
                text = contract.get('text', '')
                
                if text:
                    # Parse and segment
                    parser = ContractParser({})
                    contract_clauses = parser.segment_clauses(text, contract_id)
                    
                    # Add contract metadata
                    contracts.append({
                        "contract_id": contract_id,
                        "file_name": f"cuad_contract_{i:04d}.json",
                        "text_length": len(text),
                        "n_clauses": len(contract_clauses),
                        "parsed_ok": 1,
                    })
                    
                    # Add clauses
                    clauses.extend(contract_clauses)
            
            # Save results
            import pandas as pd
            contracts_df = pd.DataFrame(contracts)
            clauses_df = pd.DataFrame(clauses)
            
            contracts_df.to_csv("data/processed/contracts_index.csv", index=False)
            clauses_df.to_csv("data/processed/clause_spans.csv", index=False)
            
            results["parsing"] = {
                "total_contracts": len(contracts),
                "total_clauses": len(clauses),
                "successful_parses": len(contracts),
                "failed_parses": 0,
            }
            
            print(f"   ✅ Processed {len(contracts)} contracts, {len(clauses)} clauses")
        else:
            print("   ⚠️ CUAD data not found, creating sample data")
            # Create sample data for testing
            sample_contracts = [
                {
                    "contract_id": "sample_contract_001",
                    "file_name": "sample_contract_001.txt",
                    "text_length": 1000,
                    "n_clauses": 5,
                    "parsed_ok": 1,
                }
            ]
            
            import pandas as pd
            contracts_df = pd.DataFrame(sample_contracts)
            contracts_df.to_csv("data/processed/contracts_index.csv", index=False)
            
            results["parsing"] = {
                "total_contracts": 1,
                "total_clauses": 5,
                "successful_parses": 1,
                "failed_parses": 0,
            }
        
        # Step 2: Metadata extraction
        print("🔍 Step 2: Extracting metadata...")
        
        if clauses_df is not None and not clauses_df.empty:
            extractor = MetadataExtractor({})
            
            # Group by contract_id
            contracts_data = []
            for contract_id, group in clauses_df.groupby("contract_id"):
                text = " ".join(group["text"].fillna(""))
                contracts_data.append({
                    "contract_id": contract_id,
                    "text": text,
                })
            
            metadata_results = extractor.process_contracts_batch(contracts_data)
            results["metadata"] = metadata_results
            
            print(f"   ✅ Extracted metadata for {metadata_results['processed']} contracts")
        
        # Step 3: Legal NER
        print("🏷️ Step 3: Legal NER...")
        
        if clauses_df is not None and not clauses_df.empty:
            ner = LegalNER({})
            
            # Group by contract_id
            contracts_data = []
            for contract_id, group in clauses_df.groupby("contract_id"):
                text = " ".join(group["text"].fillna(""))
                contracts_data.append({
                    "contract_id": contract_id,
                    "text": text,
                })
            
            ner_results = ner.process_contracts_batch(contracts_data)
            results["ner"] = ner_results
            
            # Save entities
            all_entities = []
            for contract_data in ner_results["entities_by_contract"].values():
                all_entities.extend(contract_data["entities"])
            
            ner.save_entities(all_entities, "data/processed")
            
            print(f"   ✅ Extracted {ner_results['total_entities']} entities")
        
        # Step 4: Generate report
        print("📊 Step 4: Generating report...")
        
        report = {
            "phase": "Phase 1 - Data Foundation",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_status": "completed",
            "results": results,
            "output_files": [
                "data/processed/contracts_index.csv",
                "data/processed/clause_spans.csv",
                "data/metadata.db",
                "data/processed/extracted_entities.json",
            ]
        }
        
        with open("reports/phase1_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("   ✅ Report saved to reports/phase1_report.json")
        
        # Print summary
        print("\n" + "=" * 50)
        print("🎉 PHASE 1 PIPELINE COMPLETE!")
        print("=" * 50)
        
        if "parsing" in results:
            parsing = results["parsing"]
            print(f"📄 Contracts processed: {parsing['successful_parses']}/{parsing['total_contracts']}")
            print(f"📄 Total clauses extracted: {parsing['total_clauses']}")
        
        if "metadata" in results:
            metadata = results["metadata"]
            print(f"🔍 Metadata extracted: {metadata['processed']}/{metadata['total_contracts']}")
        
        if "ner" in results:
            ner = results["ner"]
            print(f"🏷️ Entities extracted: {ner['total_entities']}")
        
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
        print("   • Proceed to Phase 2: Model Development")
        print("   • Train baseline models")
        print("   • Implement risk scoring")
        print("   • Develop RAG system")
        
        print("=" * 50)
        
        return report
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_phase1_pipeline()
