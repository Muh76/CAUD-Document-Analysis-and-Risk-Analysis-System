#!/usr/bin/env python3
"""
Test script for Contract Review & Risk Analysis System
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_openai_connection():
    """Test OpenAI API connection"""
    print("🔤 Testing OpenAI API connection...")
    
    try:
        from src.models.legal_rag import LegalRAGSystem
        rag_system = LegalRAGSystem()
        
        # Test simple query
        test_clause = "Party A shall be liable for all damages"
        suggestions = rag_system.suggest_alternative_wording(test_clause, "liability")
        
        print(f"✅ OpenAI API working! Generated {len(suggestions)} suggestions")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False

def test_rag_system():
    """Test RAG system functionality"""
    print("\n🔍 Testing RAG system...")
    
    try:
        from src.models.legal_rag import LegalRAGSystem
        rag_system = LegalRAGSystem()
        
        # Test similar clause search
        test_clause = "Liability shall be unlimited"
        similar_clauses = rag_system.find_similar_clauses(test_clause, "liability")
        
        print(f"✅ RAG system working! Found {len(similar_clauses)} similar clauses")
        
        # Test risk analysis
        risk_analysis = rag_system.analyze_clause_risk(test_clause, "liability")
        print(f"✅ Risk analysis working! Risk level: {risk_analysis.get('risk_level', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG system error: {e}")
        return False

def test_contract_analyzer():
    """Test contract analyzer"""
    print("\n📄 Testing contract analyzer...")
    
    try:
        from src.models.contract_analyzer import ContractAnalyzer
        analyzer = ContractAnalyzer()
        
        # Test clause extraction
        test_contract = """
        This agreement is between Party A and Party B.
        Party A shall be liable for all damages.
        This agreement shall be governed by the laws of California.
        """
        
        extracted_clauses = analyzer.extract_clauses(test_contract)
        print(f"✅ Contract analyzer working! Extracted {len(extracted_clauses)} clauses")
        
        return True
        
    except Exception as e:
        print(f"❌ Contract analyzer error: {e}")
        return False

def test_risk_scorer():
    """Test risk scorer"""
    print("\n⚠️  Testing risk scorer...")
    
    try:
        from src.models.risk_scorer import RiskScorer
        scorer = RiskScorer()
        
        # Test risk scoring
        test_clauses = {
            "liability": "Party A shall be liable for all damages",
            "confidentiality": "All information shall be kept confidential"
        }
        
        risk_score = scorer.calculate_risk_score(test_clauses)
        risk_level = scorer.get_risk_level(risk_score)
        
        print(f"✅ Risk scorer working! Risk score: {risk_score:.2f}, Level: {risk_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk scorer error: {e}")
        return False

def test_file_processor():
    """Test file processor"""
    print("\n📁 Testing file processor...")
    
    try:
        from src.utils.file_processor import FileProcessor
        processor = FileProcessor()
        
        # Test text cleaning
        test_text = "This is a test contract.\n\nIt has multiple lines."
        cleaned_text = processor._clean_text(test_text)
        
        print(f"✅ File processor working! Cleaned text length: {len(cleaned_text)}")
        
        return True
        
    except Exception as e:
        print(f"❌ File processor error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Contract Review & Risk Analysis System")
    print("=" * 55)
    
    tests = [
        test_openai_connection,
        test_rag_system,
        test_contract_analyzer,
        test_risk_scorer,
        test_file_processor
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 55)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("🚀 Run './start.sh' to launch the application")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
