"""
Enhanced Data Validation System
Combines Pandera and Great Expectations for comprehensive data quality validation
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Add src to path
sys.path.insert(0, os.getcwd())

from src.validation.run_checks import DataValidator
from src.validation.great_expectations_validator import GreatExpectationsValidator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDataValidator:
    """Enhanced data validation combining Pandera and Great Expectations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pandera_validator = DataValidator(config)
        self.ge_validator = GreatExpectationsValidator(config)

    def run_comprehensive_validation(self, data_dir: str = "data/processed") -> Dict[str, Any]:
        """Run comprehensive validation using both Pandera and Great Expectations"""
        logger.info("🚀 Starting Enhanced Data Validation")
        
        results = {
            "pandera": {},
            "great_expectations": {},
            "summary": {},
        }
        
        try:
            # Run Pandera validation
            logger.info("📊 Running Pandera validation...")
            pandera_results = self.pandera_validator.validate_all()
            results["pandera"] = pandera_results
            
            # Run Great Expectations validation
            logger.info("📊 Running Great Expectations validation...")
            ge_results = self.ge_validator.run_comprehensive_validation(data_dir)
            results["great_expectations"] = ge_results
            
            # Generate combined summary
            results["summary"] = self._generate_combined_summary(results)
            
            # Save comprehensive report
            self._save_comprehensive_report(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"error": str(e)}

    def _generate_combined_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate combined summary from both validation systems"""
        summary = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "overall_status": "unknown",
            "validation_systems": {},
            "data_quality_score": 0,
            "recommendations": [],
        }
        
        # Process Pandera results
        if "pandera" in results and results["pandera"]:
            pandera = results["pandera"]
            summary["validation_systems"]["pandera"] = {
                "status": "completed",
                "checks_passed": pandera.get("checks_passed", 0),
                "checks_failed": pandera.get("checks_failed", 0),
                "success_rate": pandera.get("success_rate", 0),
            }
        
        # Process Great Expectations results
        if "great_expectations" in results and results["great_expectations"]:
            ge = results["great_expectations"]
            summary["validation_systems"]["great_expectations"] = {
                "status": "completed",
                "validation_summary": ge.get("validation_summary", {}),
                "quality_metrics": ge.get("quality_metrics", {}),
            }
            
            # Add recommendations from GE
            if "recommendations" in ge:
                summary["recommendations"].extend(ge["recommendations"])
        
        # Calculate overall data quality score
        pandera_score = 0
        ge_score = 0
        
        if "pandera" in summary["validation_systems"]:
            pandera_score = summary["validation_systems"]["pandera"].get("success_rate", 0)
        
        if "great_expectations" in summary["validation_systems"]:
            ge_metrics = summary["validation_systems"]["great_expectations"].get("quality_metrics", {})
            if ge_metrics:
                # Calculate average success rate across all data types
                success_rates = []
                for data_type, metrics in ge_metrics.items():
                    success_rates.append(metrics.get("success_percent", 0))
                ge_score = sum(success_rates) / len(success_rates) if success_rates else 0
        
        # Overall score (average of both systems)
        if pandera_score > 0 and ge_score > 0:
            summary["data_quality_score"] = (pandera_score + ge_score) / 2
        elif pandera_score > 0:
            summary["data_quality_score"] = pandera_score
        elif ge_score > 0:
            summary["data_quality_score"] = ge_score
        
        # Determine overall status
        if summary["data_quality_score"] >= 90:
            summary["overall_status"] = "excellent"
        elif summary["data_quality_score"] >= 80:
            summary["overall_status"] = "good"
        elif summary["data_quality_score"] >= 70:
            summary["overall_status"] = "fair"
        else:
            summary["overall_status"] = "poor"
        
        return summary

    def _save_comprehensive_report(self, results: Dict[str, Any]):
        """Save comprehensive validation report"""
        report_file = Path("reports") / "enhanced_validation_report.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📄 Comprehensive report saved to {report_file}")

    def print_validation_summary(self, results: Dict[str, Any]):
        """Print comprehensive validation summary"""
        print("\n" + "="*60)
        print("🎯 ENHANCED DATA VALIDATION SUMMARY")
        print("="*60)
        
        summary = results.get("summary", {})
        
        # Overall status
        overall_status = summary.get("overall_status", "unknown")
        quality_score = summary.get("data_quality_score", 0)
        
        print(f"📊 Overall Data Quality Score: {quality_score:.1f}%")
        print(f"📊 Overall Status: {overall_status.upper()}")
        
        # Pandera results
        if "pandera" in summary.get("validation_systems", {}):
            pandera = summary["validation_systems"]["pandera"]
            print(f"\n📊 Pandera Validation:")
            print(f"   Checks Passed: {pandera.get('checks_passed', 0)}")
            print(f"   Checks Failed: {pandera.get('checks_failed', 0)}")
            print(f"   Success Rate: {pandera.get('success_rate', 0):.1f}%")
        
        # Great Expectations results
        if "great_expectations" in summary.get("validation_systems", {}):
            ge = summary["validation_systems"]["great_expectations"]
            print(f"\n📊 Great Expectations Validation:")
            
            validation_summary = ge.get("validation_summary", {})
            for data_type, result in validation_summary.items():
                if result.get("status") == "completed":
                    success = result.get("success", False)
                    print(f"   {data_type.title()}: {'✅ PASSED' if success else '❌ FAILED'}")
                else:
                    print(f"   {data_type.title()}: ⚠️ {result.get('status', 'unknown')}")
            
            # Quality metrics
            quality_metrics = ge.get("quality_metrics", {})
            if quality_metrics:
                print(f"\n📊 Quality Metrics:")
                for data_type, metrics in quality_metrics.items():
                    success_percent = metrics.get("success_percent", 0)
                    print(f"   {data_type.title()}: {success_percent:.1f}% success rate")
        
        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            print(f"\n🔧 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
        
        print("="*60)


def main():
    """Main function for enhanced data validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Data Validation")
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
    validator = EnhancedDataValidator(config)
    
    try:
        results = validator.run_comprehensive_validation(args.data_dir)
        validator.print_validation_summary(results)
        
        print(f"\n🎉 Enhanced validation completed!")
        print(f"📄 Report saved to: reports/enhanced_validation_report.json")
        
    except Exception as e:
        logger.error(f"Enhanced validation failed: {e}")
        print(f"\n❌ Enhanced validation failed: {e}")


if __name__ == "__main__":
    main()

