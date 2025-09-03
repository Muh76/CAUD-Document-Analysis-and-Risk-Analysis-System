#!/usr/bin/env python3
"""
Contract Analysis System CLI
Phase 1 Command Line Interface
"""

import typer
import yaml
import json
import sys
import os
from pathlib import Path
from typing import Optional
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.pipeline import ContractDataPipeline
from models.baseline_models import BaselineClauseClassifier, KeywordRiskScorer, BaselineEvaluator

app = typer.Typer(help="Contract Analysis System CLI")

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@app.command()
def prepare(
    config_path: str = typer.Option("configs/parse.yaml", "--config", "-c", help="Path to config file"),
    output_dir: str = typer.Option("data/processed", "--output", "-o", help="Output directory"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Run data validation")
):
    """Parse contracts and prepare data for analysis"""
    typer.echo("🔄 Starting contract preparation...")
    
    try:
        # Load configuration
        config = load_config(config_path)
        config['output_dir'] = output_dir
        config['validation_enabled'] = validate
        
        # Initialize pipeline
        pipeline = ContractDataPipeline(config)
        
        # Process contracts
        typer.echo("📊 Processing CUAD dataset...")
        all_metadata, all_clauses = pipeline.process_all_contracts()
        
        # Generate reports
        typer.echo("📈 Generating data reports...")
        data_report = pipeline.generate_data_report(all_metadata, all_clauses)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_df = pipeline.metadata_to_dataframe(all_metadata)
        metadata_df.to_csv(output_path / "contract_metadata.csv", index=False)
        
        # Save clauses
        clauses_df = pipeline.clauses_to_dataframe(all_clauses)
        clauses_df.to_csv(output_path / "clause_segments.csv", index=False)
        
        # Save reports
        with open(output_path / "data_report.json", 'w') as f:
            json.dump(data_report, f, indent=2)
        
        typer.echo(f"✅ Data preparation complete! Output saved to {output_dir}")
        typer.echo(f"📊 Processed {len(all_metadata)} contracts, {len(all_clauses)} clauses")
        
    except Exception as e:
        typer.echo(f"❌ Error during preparation: {e}")
        raise typer.Exit(1)

@app.command()
def extract(
    config_path: str = typer.Option("configs/parse.yaml", "--config", "-c", help="Path to config file"),
    input_dir: str = typer.Option("data/processed", "--input", "-i", help="Input directory"),
    output_dir: str = typer.Option("data/enriched", "--output", "-o", help="Output directory")
):
    """Extract metadata and perform NER enrichment"""
    typer.echo("🔍 Starting metadata extraction and NER enrichment...")
    
    try:
        # Load configuration
        config = load_config(config_path)
        config['input_dir'] = input_dir
        config['output_dir'] = output_dir
        
        # Initialize pipeline
        pipeline = ContractDataPipeline(config)
        
        # Load processed data
        metadata_df = pipeline.load_metadata(input_dir)
        clauses_df = pipeline.load_clauses(input_dir)
        
        # Extract metadata
        typer.echo("🏷️ Extracting contract metadata...")
        enriched_metadata = pipeline.extract_metadata(metadata_df)
        
        # Perform NER
        typer.echo("🧾 Performing NER enrichment...")
        ner_results = pipeline.perform_ner(clauses_df)
        
        # Save enriched data
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        enriched_metadata.to_csv(output_path / "enriched_metadata.csv", index=False)
        
        with open(output_path / "ner_results.json", 'w') as f:
            json.dump(ner_results, f, indent=2)
        
        typer.echo(f"✅ Extraction complete! Enriched data saved to {output_dir}")
        
    except Exception as e:
        typer.echo(f"❌ Error during extraction: {e}")
        raise typer.Exit(1)

@app.command()
def validate(
    config_path: str = typer.Option("configs/parse.yaml", "--config", "-c", help="Path to config file"),
    data_dir: str = typer.Option("data/processed", "--data", "-d", help="Data directory to validate"),
    strict: bool = typer.Option(False, "--strict", help="Fail on validation errors")
):
    """Run data quality checks"""
    typer.echo("✅ Running data validation...")
    
    try:
        # Import validation module directly
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from validation.run_checks import DataValidator
        
        # Load configuration
        config = load_config(config_path)
        config['data_dir'] = data_dir
        
        # Run validation
        validator = DataValidator(config)
        validation_results = validator.run_comprehensive_validation(data_dir)
        
        # Display results
        typer.echo("\n📊 Validation Results:")
        for validation_type, result in validation_results.items():
            if validation_type == "overall_passed":
                continue
                
            typer.echo(f"\n{validation_type.upper()}:")
            if isinstance(result, dict):
                for check_name, check_result in result.items():
                    if isinstance(check_result, dict) and "passed" in check_result:
                        status = "✅ PASS" if check_result["passed"] else "❌ FAIL"
                        typer.echo(f"  {check_name}: {status}")
                        if not check_result["passed"] and check_result.get("error"):
                            typer.echo(f"    Error: {check_result['error']}")
        
        # Check if all validations passed
        all_passed = validation_results.get('overall_passed', False)
        
        if all_passed:
            typer.echo("\n🎉 All validations passed!")
        else:
            typer.echo("\n⚠️ Some validations failed!")
            if strict:
                raise typer.Exit(1)
        
    except Exception as e:
        typer.echo(f"❌ Error during validation: {e}")
        raise typer.Exit(1)

@app.command()
def train(
    config_path: str = typer.Option("configs/models.yaml", "--config", "-c", help="Path to config file"),
    data_dir: str = typer.Option("data/processed", "--data", "-d", help="Data directory"),
    model_dir: str = typer.Option("models/baseline", "--model", "-m", help="Model output directory"),
    model_type: str = typer.Option("logistic", "--type", "-t", help="Model type (logistic, random_forest)")
):
    """Train baseline models"""
    typer.echo("🤖 Training baseline models...")
    
    try:
        # Load configuration
        config = load_config(config_path)
        config['data_dir'] = data_dir
        config['model_dir'] = model_dir
        config['model_type'] = model_type
        
        # Initialize models
        classifier = BaselineClauseClassifier(config)
        risk_scorer = KeywordRiskScorer()
        evaluator = BaselineEvaluator(config)
        
        # Load data
        clauses_df = classifier.load_data(data_dir)
        
        # Prepare data
        typer.echo("📊 Preparing training data...")
        X, y = classifier.prepare_data(clauses_df)
        
        # Train model
        typer.echo(f"🚀 Training {model_type} classifier...")
        model_metrics = classifier.train_model(X, y, model_type=model_type)
        
        # Evaluate model
        typer.echo("📈 Evaluating model performance...")
        evaluation_results = evaluator.evaluate_classifier(classifier, X, y)
        
        # Save model and results
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        classifier.save_model(model_path / "baseline_classifier.joblib")
        
        with open(model_path / "baseline_metrics.json", 'w') as f:
            json.dump(model_metrics, f, indent=2)
        
        with open(model_path / "evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        typer.echo(f"✅ Training complete! Model saved to {model_dir}")
        typer.echo(f"📊 F1 Score: {model_metrics.get('f1_macro', 'N/A')}")
        
    except Exception as e:
        typer.echo(f"❌ Error during training: {e}")
        raise typer.Exit(1)

@app.command()
def analyze(
    contract_path: str = typer.Argument(..., help="Path to contract file"),
    model_dir: str = typer.Option("models/baseline", "--model", "-m", help="Model directory"),
    output_dir: str = typer.Option("analysis_results", "--output", "-o", help="Output directory")
):
    """Analyze a single contract"""
    typer.echo(f"📄 Analyzing contract: {contract_path}")
    
    try:
        # Load models
        model_path = Path(model_dir)
        classifier = BaselineClauseClassifier.load_model(model_path / "baseline_classifier.joblib")
        risk_scorer = KeywordRiskScorer()
        
        # Process contract
        pipeline = ContractDataPipeline({})
        metadata, clauses = pipeline.process_contract(contract_path, "single_contract")
        
        # Analyze clauses
        results = []
        for clause in clauses:
            # Classify clause
            clause_type = classifier.predict_clause_type(clause.text)
            
            # Score risk
            risk_score = risk_scorer.score_clause(clause.text)
            
            results.append({
                'clause_text': clause.text[:100] + "..." if len(clause.text) > 100 else clause.text,
                'clause_type': clause_type,
                'risk_score': risk_score['risk_score'],
                'risk_level': risk_score['risk_level'],
                'detected_risks': risk_score['detected_risks']
            })
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "analysis_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Display summary
        typer.echo(f"\n📊 Analysis Summary:")
        typer.echo(f"  Total clauses: {len(clauses)}")
        typer.echo(f"  High risk clauses: {sum(1 for r in results if r['risk_level'] == 'HIGH')}")
        typer.echo(f"  Medium risk clauses: {sum(1 for r in results if r['risk_level'] == 'MEDIUM')}")
        typer.echo(f"  Low risk clauses: {sum(1 for r in results if r['risk_level'] == 'LOW')}")
        
        typer.echo(f"✅ Analysis complete! Results saved to {output_dir}")
        
    except Exception as e:
        typer.echo(f"❌ Error during analysis: {e}")
        raise typer.Exit(1)

@app.command()
def status():
    """Show system status"""
    typer.echo("📊 Contract Analysis System Status")
    typer.echo("=" * 40)
    
    # Check data files
    data_path = Path("data")
    if data_path.exists():
        typer.echo("📁 Data Directory:")
        for subdir in ["raw", "processed", "enriched"]:
            subdir_path = data_path / subdir
            if subdir_path.exists():
                file_count = len(list(subdir_path.glob("*")))
                typer.echo(f"  {subdir}: {file_count} files")
            else:
                typer.echo(f"  {subdir}: Not found")
    
    # Check models
    models_path = Path("models/baseline")
    if models_path.exists():
        typer.echo("🤖 Models:")
        model_files = list(models_path.glob("*"))
        for model_file in model_files:
            typer.echo(f"  {model_file.name}")
    
    # Check tests
    tests_path = Path("tests")
    if tests_path.exists():
        test_files = list(tests_path.glob("test_*.py"))
        typer.echo(f"🧪 Tests: {len(test_files)} test files")
    
    typer.echo("✅ Status check complete!")

if __name__ == "__main__":
    app()
