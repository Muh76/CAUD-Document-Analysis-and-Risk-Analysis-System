"""
Legal NER (Named Entity Recognition) for Contract Analysis System
Uses spaCy transformer pipeline with custom legal entity patterns
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

# spaCy imports
try:
    import spacy
    from spacy.tokens import Doc, Span
    from spacy.language import Language
    from spacy.pipeline import EntityRuler
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️ spaCy not available - install spacy and download en_core_web_trf")

logger = logging.getLogger(__name__)


class LegalNER:
    """Legal Named Entity Recognition with custom patterns"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_spacy()
        self.setup_legal_patterns()

    def setup_spacy(self):
        """Setup spaCy pipeline with transformer model"""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available - NER will be limited")
            self.nlp = None
            return

        try:
            # Try to load transformer model
            self.nlp = spacy.load("en_core_web_trf")
            logger.info("Loaded spaCy transformer model")
        except OSError:
            try:
                # Fallback to medium model
                self.nlp = spacy.load("en_core_web_md")
                logger.info("Loaded spaCy medium model")
            except OSError:
                try:
                    # Fallback to small model
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy small model")
                except OSError:
                    logger.error("No spaCy model available - install with: python -m spacy download en_core_web_sm")
                    self.nlp = None

    def setup_legal_patterns(self):
        """Setup custom legal entity patterns"""
        if not self.nlp:
            return

        # Create EntityRuler for custom patterns
        ruler = EntityRuler(self.nlp, overwrite_ents=True)
        
        # Legal entity patterns
        legal_patterns = [
            # Legal terms
            {"label": "LEGAL_TERM", "pattern": "Indemnified Party"},
            {"label": "LEGAL_TERM", "pattern": "Indemnifying Party"},
            {"label": "LEGAL_TERM", "pattern": "Force Majeure"},
            {"label": "LEGAL_TERM", "pattern": "Change of Control"},
            {"label": "LEGAL_TERM", "pattern": "Material Breach"},
            {"label": "LEGAL_TERM", "pattern": "Cure Period"},
            {"label": "LEGAL_TERM", "pattern": "Notice Period"},
            {"label": "LEGAL_TERM", "pattern": "Termination for Cause"},
            {"label": "LEGAL_TERM", "pattern": "Termination for Convenience"},
            {"label": "LEGAL_TERM", "pattern": "Liquidated Damages"},
            {"label": "LEGAL_TERM", "pattern": "Consequential Damages"},
            {"label": "LEGAL_TERM", "pattern": "Direct Damages"},
            {"label": "LEGAL_TERM", "pattern": "Punitive Damages"},
            {"label": "LEGAL_TERM", "pattern": "Actual Damages"},
            {"label": "LEGAL_TERM", "pattern": "Compensatory Damages"},
            
            # Contract types
            {"label": "CONTRACT_TYPE", "pattern": "Non-Disclosure Agreement"},
            {"label": "CONTRACT_TYPE", "pattern": "Service Agreement"},
            {"label": "CONTRACT_TYPE", "pattern": "Employment Agreement"},
            {"label": "CONTRACT_TYPE", "pattern": "Purchase Agreement"},
            {"label": "CONTRACT_TYPE", "pattern": "License Agreement"},
            {"label": "CONTRACT_TYPE", "pattern": "Partnership Agreement"},
            {"label": "CONTRACT_TYPE", "pattern": "Joint Venture Agreement"},
            {"label": "CONTRACT_TYPE", "pattern": "Master Service Agreement"},
            {"label": "CONTRACT_TYPE", "pattern": "Statement of Work"},
            
            # Legal entities
            {"label": "LEGAL_ENTITY", "pattern": "Board of Directors"},
            {"label": "LEGAL_ENTITY", "pattern": "Chief Executive Officer"},
            {"label": "LEGAL_ENTITY", "pattern": "Chief Financial Officer"},
            {"label": "LEGAL_ENTITY", "pattern": "General Counsel"},
            {"label": "LEGAL_ENTITY", "pattern": "Legal Department"},
            {"label": "LEGAL_ENTITY", "pattern": "Compliance Officer"},
            
            # Financial terms
            {"label": "FINANCIAL_TERM", "pattern": "Annual Recurring Revenue"},
            {"label": "FINANCIAL_TERM", "pattern": "Monthly Recurring Revenue"},
            {"label": "FINANCIAL_TERM", "pattern": "Cost of Goods Sold"},
            {"label": "FINANCIAL_TERM", "pattern": "Gross Margin"},
            {"label": "FINANCIAL_TERM", "pattern": "Net Present Value"},
            {"label": "FINANCIAL_TERM", "pattern": "Return on Investment"},
            
            # IP terms
            {"label": "IP_TERM", "pattern": "Intellectual Property"},
            {"label": "IP_TERM", "pattern": "Trade Secret"},
            {"label": "IP_TERM", "pattern": "Patent"},
            {"label": "IP_TERM", "pattern": "Trademark"},
            {"label": "IP_TERM", "pattern": "Copyright"},
            {"label": "IP_TERM", "pattern": "Confidential Information"},
            {"label": "IP_TERM", "pattern": "Proprietary Information"},
            
            # Regulatory terms
            {"label": "REGULATORY_TERM", "pattern": "General Data Protection Regulation"},
            {"label": "REGULATORY_TERM", "pattern": "California Consumer Privacy Act"},
            {"label": "REGULATORY_TERM", "pattern": "Health Insurance Portability and Accountability Act"},
            {"label": "REGULATORY_TERM", "pattern": "Sarbanes-Oxley Act"},
            {"label": "REGULATORY_TERM", "pattern": "Dodd-Frank Act"},
        ]
        
        ruler.add_patterns(legal_patterns)
        
        # Add ruler to pipeline
        if "entity_ruler" not in self.nlp.pipe_names:
            self.nlp.add_pipe("entity_ruler", before="ner")
        else:
            self.nlp.replace_pipe("entity_ruler", ruler)
        
        logger.info(f"Added {len(legal_patterns)} legal entity patterns")

    def extract_entities(self, text: str, contract_id: str) -> List[Dict[str, Any]]:
        """
        Extract legal entities from contract text
        
        Args:
            text: Contract text
            contract_id: Contract identifier
            
        Returns:
            List of extracted entities with metadata
        """
        if not self.nlp:
            logger.warning("spaCy not available - returning empty entity list")
            return []

        logger.info(f"Extracting entities for contract {contract_id}")
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entity_data = {
                "contract_id": contract_id,
                "entity": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": self._calculate_entity_confidence(ent),
                "context": self._get_entity_context(text, ent.start_char, ent.end_char),
            }
            entities.append(entity_data)
        
        # Remove duplicates (same entity, same position)
        unique_entities = []
        seen = set()
        for entity in entities:
            key = (entity["entity"], entity["start"], entity["end"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        logger.info(f"Extracted {len(unique_entities)} unique entities from contract {contract_id}")
        return unique_entities

    def _calculate_entity_confidence(self, entity: Span) -> float:
        """Calculate confidence score for entity extraction"""
        # Base confidence on entity type and length
        base_confidence = 0.7
        
        # Legal terms get higher confidence
        if entity.label_ in ["LEGAL_TERM", "CONTRACT_TYPE", "LEGAL_ENTITY"]:
            base_confidence = 0.9
        
        # Longer entities get slightly higher confidence
        if len(entity.text) > 20:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)

    def _get_entity_context(self, text: str, start: int, end: int, context_size: int = 50) -> str:
        """Get context around entity"""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        
        context = text[context_start:context_end]
        
        # Clean up context
        context = context.replace('\n', ' ').strip()
        context = ' '.join(context.split())  # Remove extra whitespace
        
        return context

    def classify_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Classify entities by type
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Dictionary of entities grouped by type
        """
        classified = {
            "legal_terms": [],
            "contract_types": [],
            "legal_entities": [],
            "financial_terms": [],
            "ip_terms": [],
            "regulatory_terms": [],
            "organizations": [],
            "persons": [],
            "dates": [],
            "money": [],
            "other": [],
        }
        
        for entity in entities:
            label = entity["label"]
            
            if label == "LEGAL_TERM":
                classified["legal_terms"].append(entity)
            elif label == "CONTRACT_TYPE":
                classified["contract_types"].append(entity)
            elif label == "LEGAL_ENTITY":
                classified["legal_entities"].append(entity)
            elif label == "FINANCIAL_TERM":
                classified["financial_terms"].append(entity)
            elif label == "IP_TERM":
                classified["ip_terms"].append(entity)
            elif label == "REGULATORY_TERM":
                classified["regulatory_terms"].append(entity)
            elif label == "ORG":
                classified["organizations"].append(entity)
            elif label == "PERSON":
                classified["persons"].append(entity)
            elif label == "DATE":
                classified["dates"].append(entity)
            elif label == "MONEY":
                classified["money"].append(entity)
            else:
                classified["other"].append(entity)
        
        return classified

    def generate_entity_report(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive entity extraction report
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Entity extraction report
        """
        if not entities:
            return {
                "total_entities": 0,
                "entity_types": {},
                "most_common_entities": [],
                "entity_distribution": {},
            }
        
        # Count entities by type
        entity_types = {}
        entity_texts = []
        
        for entity in entities:
            label = entity["label"]
            entity_types[label] = entity_types.get(label, 0) + 1
            entity_texts.append(entity["entity"])
        
        # Find most common entities
        from collections import Counter
        entity_counter = Counter(entity_texts)
        most_common = entity_counter.most_common(10)
        
        # Calculate distribution
        total_entities = len(entities)
        entity_distribution = {
            label: count / total_entities 
            for label, count in entity_types.items()
        }
        
        return {
            "total_entities": total_entities,
            "entity_types": entity_types,
            "most_common_entities": most_common,
            "entity_distribution": entity_distribution,
            "avg_confidence": np.mean([e["confidence"] for e in entities]),
        }

    def process_contracts_batch(self, contracts_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process entity extraction for multiple contracts
        
        Args:
            contracts_data: List of contract data with text
            
        Returns:
            Processing summary
        """
        logger.info(f"Processing entity extraction for {len(contracts_data)} contracts")
        
        results = {
            "total_contracts": len(contracts_data),
            "processed": 0,
            "failed": 0,
            "total_entities": 0,
            "entities_by_contract": {},
            "summary_report": {},
        }
        
        all_entities = []
        
        for contract_data in contracts_data:
            try:
                contract_id = contract_data["contract_id"]
                text = contract_data["text"]
                
                # Extract entities
                entities = self.extract_entities(text, contract_id)
                
                # Classify entities
                classified_entities = self.classify_entities(entities)
                
                # Store results
                results["entities_by_contract"][contract_id] = {
                    "entities": entities,
                    "classified": classified_entities,
                    "count": len(entities),
                }
                
                all_entities.extend(entities)
                results["total_entities"] += len(entities)
                results["processed"] += 1
                
                logger.info(f"Extracted {len(entities)} entities from {contract_id}")
                
            except Exception as e:
                results["failed"] += 1
                logger.error(f"Failed to extract entities from {contract_data.get('contract_id', 'unknown')}: {e}")
        
        # Generate summary report
        if all_entities:
            results["summary_report"] = self.generate_entity_report(all_entities)
        
        logger.info(f"Entity extraction complete: {results['processed']}/{results['total_contracts']} successful")
        return results

    def save_entities(self, entities: List[Dict[str, Any]], output_path: str):
        """Save entities to JSON file"""
        output_file = Path(output_path) / "extracted_entities.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(entities, f, indent=2)
        
        logger.info(f"Saved {len(entities)} entities to {output_file}")


def main():
    """Main function for legal NER"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Legal NER Pipeline")
    parser.add_argument("--input", default="data/processed/clause_spans.csv", help="Input CSV file")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    
    args = parser.parse_args()
    
    # Load contract data
    if Path(args.input).exists():
        df = pd.read_csv(args.input)
        contracts_data = []
        
        # Group by contract_id
        for contract_id, group in df.groupby("contract_id"):
            text = " ".join(group["text"].fillna(""))
            contracts_data.append({
                "contract_id": contract_id,
                "text": text,
            })
    else:
        logger.error(f"Input file not found: {args.input}")
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize NER and process
    config = {}
    ner = LegalNER(config)
    results = ner.process_contracts_batch(contracts_data)
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save all entities
    all_entities = []
    for contract_data in results["entities_by_contract"].values():
        all_entities.extend(contract_data["entities"])
    
    ner.save_entities(all_entities, args.output)
    
    # Save summary report
    with open(output_path / "ner_report.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Legal NER Results:")
    print(f"Total contracts: {results['total_contracts']}")
    print(f"Processed: {results['processed']}")
    print(f"Failed: {results['failed']}")
    print(f"Total entities: {results['total_entities']}")
    
    if results["summary_report"]:
        print(f"Entity types: {list(results['summary_report']['entity_types'].keys())}")


if __name__ == "__main__":
    main()

