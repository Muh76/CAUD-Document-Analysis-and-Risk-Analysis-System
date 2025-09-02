"""
Data Pipeline for Contract Analysis System - Phase 1
Handles CUAD ingestion, parsing, clause segmentation, and metadata extraction
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import spacy
from transformers import AutoTokenizer
import re
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContractMetadata:
    """Metadata extracted from contracts"""
    contract_id: str
    contract_type: Optional[str]
    parties: List[str]
    effective_date: Optional[str]
    expiration_date: Optional[str]
    jurisdiction: Optional[str]
    governing_law: Optional[str]
    total_clauses: int
    file_size: int
    processing_timestamp: str

@dataclass
class ClauseSegment:
    """Individual clause segment with metadata"""
    clause_id: str
    text: str
    clause_type: Optional[str]
    start_position: int
    end_position: int
    confidence: float
    entities: List[Dict[str, Any]]
    risk_flags: List[str]

class ContractDataPipeline:
    """Main data pipeline for contract processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Initialize data validation schemas
        self._setup_validation_schemas()
        
    def _setup_validation_schemas(self):
        """Setup data validation schemas using Pandera"""
        import pandera as pa
        from pandera.typing import Series
        
        class ContractSchema(pa.SchemaModel):
            contract_id: Series[str] = pa.Field(unique=True)
            contract_type: Series[str] = pa.Field(nullable=True)
            parties: Series[object] = pa.Field()  # List of strings
            effective_date: Series[str] = pa.Field(nullable=True)
            total_clauses: Series[int] = pa.Field(ge=0)
            file_size: Series[int] = pa.Field(ge=0)
            
        class ClauseSchema(pa.SchemaModel):
            clause_id: Series[str] = pa.Field(unique=True)
            contract_id: Series[str] = pa.Field()
            text: Series[str] = pa.Field(str_len={"min_value": 10})
            clause_type: Series[str] = pa.Field(nullable=True)
            start_position: Series[int] = pa.Field(ge=0)
            end_position: Series[int] = pa.Field(ge=0)
            confidence: Series[float] = pa.Field(ge=0.0, le=1.0)
            
        self.contract_schema = ContractSchema
        self.clause_schema = ClauseSchema
    
    def load_cuad_dataset(self, file_path: str) -> pd.DataFrame:
        """Load and validate CUAD dataset"""
        logger.info(f"Loading CUAD dataset from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract data from CUAD format
            contracts = []
            for item in data['data']:
                contract = {
                    'contract_id': item.get('id', f"contract_{len(contracts)}"),
                    'title': item.get('title', ''),
                    'context': item.get('context', ''),
                    'questions': item.get('questions', []),
                    'answers': item.get('answers', [])
                }
                contracts.append(contract)
            
            df = pd.DataFrame(contracts)
            
            # Validate with Pandera
            validated_df = self.contract_schema.validate(df)
            logger.info(f"Successfully loaded {len(validated_df)} contracts from CUAD")
            
            return validated_df
            
        except Exception as e:
            logger.error(f"Error loading CUAD dataset: {e}")
            raise
    
    def extract_metadata(self, text: str, contract_id: str) -> ContractMetadata:
        """Extract metadata from contract text using NER and patterns"""
        logger.info(f"Extracting metadata for contract {contract_id}")
        
        doc = self.nlp(text)
        
        # Extract parties (PERSON, ORG entities)
        parties = []
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG']:
                parties.append(ent.text.strip())
        
        # Extract dates using patterns
        date_patterns = [
            r'\b(?:effective|commencement|start)\s+(?:date|as\s+of)?\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'\b(?:expiration|end|termination)\s+(?:date)?\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?:effective|commencement)',
        ]
        
        effective_date = None
        expiration_date = None
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if 'effective' in pattern or 'commencement' in pattern:
                    effective_date = matches[0]
                elif 'expiration' in pattern or 'termination' in pattern:
                    expiration_date = matches[0]
        
        # Extract jurisdiction and governing law
        jurisdiction_patterns = [
            r'\b(?:jurisdiction|venue)\s+(?:of|in)\s+([A-Za-z\s,]+?)(?:\.|,|$)',
            r'\b(?:governing\s+law|law\s+governing)\s+([A-Za-z\s,]+?)(?:\.|,|$)',
        ]
        
        jurisdiction = None
        governing_law = None
        
        for pattern in jurisdiction_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if 'jurisdiction' in pattern or 'venue' in pattern:
                    jurisdiction = matches[0].strip()
                elif 'governing' in pattern:
                    governing_law = matches[0].strip()
        
        # Determine contract type based on content
        contract_type = self._classify_contract_type(text)
        
        # Count clauses (approximate)
        clause_patterns = [
            r'\b(?:section|article|clause|paragraph)\s+\d+',
            r'\b\d+\.\s+[A-Z]',
        ]
        
        total_clauses = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                          for pattern in clause_patterns)
        
        return ContractMetadata(
            contract_id=contract_id,
            contract_type=contract_type,
            parties=list(set(parties)),  # Remove duplicates
            effective_date=effective_date,
            expiration_date=expiration_date,
            jurisdiction=jurisdiction,
            governing_law=governing_law,
            total_clauses=total_clauses,
            file_size=len(text.encode('utf-8')),
            processing_timestamp=datetime.now().isoformat()
        )
    
    def _classify_contract_type(self, text: str) -> Optional[str]:
        """Classify contract type based on content"""
        text_lower = text.lower()
        
        contract_types = {
            'nda': ['non-disclosure', 'confidentiality', 'nda', 'non disclosure'],
            'msa': ['master service', 'master agreement', 'msa'],
            'employment': ['employment', 'hire', 'employee', 'staff'],
            'vendor': ['vendor', 'supplier', 'service provider'],
            'license': ['license', 'licensing', 'intellectual property'],
            'lease': ['lease', 'rental', 'tenancy'],
            'partnership': ['partnership', 'joint venture', 'collaboration'],
        }
        
        for contract_type, keywords in contract_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return contract_type.upper()
        
        return None
    
    def segment_clauses(self, text: str, contract_id: str) -> List[ClauseSegment]:
        """Segment contract into individual clauses"""
        logger.info(f"Segmenting clauses for contract {contract_id}")
        
        clauses = []
        
        # Pattern-based clause segmentation
        clause_patterns = [
            r'(?:Section|Article|Clause)\s+(\d+[A-Z]?)[:.]?\s*([^.]*(?:\.|$))',
            r'(\d+[A-Z]?)[:.]\s*([^.]*(?:\.|$))',
            r'([A-Z][A-Z\s]+)[:.]\s*([^.]*(?:\.|$))',
        ]
        
        for pattern in clause_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                clause_header = match.group(1).strip()
                clause_text = match.group(2).strip()
                
                if len(clause_text) > 10:  # Minimum clause length
                    clause_id = f"{contract_id}_clause_{len(clauses)}"
                    
                    # Extract entities from clause
                    doc = self.nlp(clause_text)
                    entities = []
                    for ent in doc.ents:
                        entities.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char
                        })
                    
                    # Identify risk flags
                    risk_flags = self._identify_risk_flags(clause_text)
                    
                    clause = ClauseSegment(
                        clause_id=clause_id,
                        text=clause_text,
                        clause_type=self._classify_clause_type(clause_header, clause_text),
                        start_position=match.start(),
                        end_position=match.end(),
                        confidence=0.8,  # Base confidence
                        entities=entities,
                        risk_flags=risk_flags
                    )
                    
                    clauses.append(clause)
        
        logger.info(f"Segmented {len(clauses)} clauses from contract {contract_id}")
        return clauses
    
    def _classify_clause_type(self, header: str, text: str) -> Optional[str]:
        """Classify clause type based on header and content"""
        header_lower = header.lower()
        text_lower = text.lower()
        
        clause_types = {
            'liability': ['liability', 'indemnification', 'damages', 'warranty'],
            'termination': ['termination', 'cancellation', 'expiration'],
            'confidentiality': ['confidentiality', 'non-disclosure', 'privacy'],
            'payment': ['payment', 'compensation', 'fees', 'pricing'],
            'intellectual_property': ['intellectual property', 'ip', 'copyright', 'patent'],
            'governing_law': ['governing law', 'jurisdiction', 'venue', 'dispute'],
            'force_majeure': ['force majeure', 'act of god', 'unforeseen'],
            'amendment': ['amendment', 'modification', 'change'],
        }
        
        for clause_type, keywords in clause_types.items():
            if any(keyword in header_lower or keyword in text_lower 
                  for keyword in keywords):
                return clause_type
        
        return None
    
    def _identify_risk_flags(self, text: str) -> List[str]:
        """Identify potential risk flags in clause text"""
        risk_flags = []
        text_lower = text.lower()
        
        risk_patterns = {
            'uncapped_liability': [r'\b(?:unlimited|uncapped|no\s+limit)\s+liability'],
            'unilateral_termination': [r'\b(?:terminate|cancel)\s+(?:at\s+will|without\s+cause)'],
            'broad_indemnification': [r'\b(?:indemnify|hold\s+harmless)\s+(?:all|any|every)'],
            'excessive_penalties': [r'\b(?:penalty|liquidated\s+damages)\s+(?:of|in\s+amount)'],
            'unfair_terms': [r'\b(?:unfair|unreasonable|excessive)\s+(?:terms|conditions)'],
        }
        
        for risk_type, patterns in risk_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    risk_flags.append(risk_type)
                    break
        
        return risk_flags
    
    def process_contract(self, text: str, contract_id: str) -> Tuple[ContractMetadata, List[ClauseSegment]]:
        """Process a single contract through the pipeline"""
        logger.info(f"Processing contract {contract_id}")
        
        # Extract metadata
        metadata = self.extract_metadata(text, contract_id)
        
        # Segment clauses
        clauses = self.segment_clauses(text, contract_id)
        
        # Update metadata with actual clause count
        metadata.total_clauses = len(clauses)
        
        return metadata, clauses
    
    def create_data_contracts(self) -> Dict[str, Any]:
        """Create data contracts for validation"""
        return {
            'contract_metadata': {
                'required_fields': ['contract_id', 'parties', 'total_clauses', 'file_size'],
                'optional_fields': ['contract_type', 'effective_date', 'expiration_date', 'jurisdiction'],
                'validation_rules': {
                    'contract_id': 'unique string',
                    'total_clauses': 'non-negative integer',
                    'file_size': 'positive integer'
                }
            },
            'clause_segments': {
                'required_fields': ['clause_id', 'contract_id', 'text', 'start_position', 'end_position'],
                'optional_fields': ['clause_type', 'confidence', 'entities', 'risk_flags'],
                'validation_rules': {
                    'clause_id': 'unique string',
                    'text': 'non-empty string, min length 10',
                    'confidence': 'float between 0 and 1',
                    'start_position': 'non-negative integer',
                    'end_position': 'greater than start_position'
                }
            }
        }
    
    def generate_data_report(self, metadata_list: List[ContractMetadata], 
                           clauses_list: List[List[ClauseSegment]]) -> Dict[str, Any]:
        """Generate comprehensive data report"""
        logger.info("Generating data report")
        
        # Aggregate statistics
        total_contracts = len(metadata_list)
        total_clauses = sum(len(clauses) for clauses in clauses_list)
        
        # Contract type distribution
        contract_types = {}
        for metadata in metadata_list:
            contract_type = metadata.contract_type or 'Unknown'
            contract_types[contract_type] = contract_types.get(contract_type, 0) + 1
        
        # Clause type distribution
        clause_types = {}
        for clauses in clauses_list:
            for clause in clauses:
                clause_type = clause.clause_type or 'Unknown'
                clause_types[clause_type] = clause_types.get(clause_type, 0) + 1
        
        # Risk flag distribution
        risk_flags = {}
        for clauses in clauses_list:
            for clause in clauses:
                for flag in clause.risk_flags:
                    risk_flags[flag] = risk_flags.get(flag, 0) + 1
        
        # Entity distribution
        entity_types = {}
        for clauses in clauses_list:
            for clause in clauses:
                for entity in clause.entities:
                    entity_types[entity['label']] = entity_types.get(entity['label'], 0) + 1
        
        return {
            'summary': {
                'total_contracts': total_contracts,
                'total_clauses': total_clauses,
                'avg_clauses_per_contract': total_clauses / total_contracts if total_contracts > 0 else 0,
                'processing_timestamp': datetime.now().isoformat()
            },
            'contract_types': contract_types,
            'clause_types': clause_types,
            'risk_flags': risk_flags,
            'entity_types': entity_types,
            'data_quality': {
                'contracts_with_metadata': sum(1 for m in metadata_list if m.contract_type),
                'clauses_with_type': sum(1 for clauses in clauses_list 
                                       for clause in clauses if clause.clause_type),
                'clauses_with_entities': sum(1 for clauses in clauses_list 
                                           for clause in clauses if clause.entities)
            }
        }

def main():
    """Main execution function for Phase 1 data pipeline"""
    logger.info("Starting Phase 1 Data Pipeline")
    
    # Configuration
    config = {
        'cuad_file_path': 'CUAD_v1.json',
        'output_dir': 'data/processed',
        'validation_enabled': True,
        'logging_level': 'INFO'
    }
    
    # Initialize pipeline
    pipeline = ContractDataPipeline(config)
    
    # Load CUAD dataset
    cuad_df = pipeline.load_cuad_dataset(config['cuad_file_path'])
    
    # Process contracts
    all_metadata = []
    all_clauses = []
    
    for _, contract in cuad_df.iterrows():
        contract_id = contract['contract_id']
        text = contract['context']
        
        try:
            metadata, clauses = pipeline.process_contract(text, contract_id)
            all_metadata.append(metadata)
            all_clauses.append(clauses)
            
            logger.info(f"Processed contract {contract_id}: {len(clauses)} clauses")
            
        except Exception as e:
            logger.error(f"Error processing contract {contract_id}: {e}")
            continue
    
    # Generate data report
    report = pipeline.generate_data_report(all_metadata, all_clauses)
    
    # Save results
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata_df = pd.DataFrame([vars(m) for m in all_metadata])
    metadata_df.to_csv(output_dir / 'contract_metadata.csv', index=False)
    
    # Save clauses
    all_clauses_flat = []
    for clauses in all_clauses:
        for clause in clauses:
            clause_dict = vars(clause)
            clause_dict['entities'] = json.dumps(clause_dict['entities'])
            clause_dict['risk_flags'] = json.dumps(clause_dict['risk_flags'])
            all_clauses_flat.append(clause_dict)
    
    clauses_df = pd.DataFrame(all_clauses_flat)
    clauses_df.to_csv(output_dir / 'clause_segments.csv', index=False)
    
    # Save report
    with open(output_dir / 'data_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save data contracts
    data_contracts = pipeline.create_data_contracts()
    with open(output_dir / 'data_contracts.json', 'w') as f:
        json.dump(data_contracts, f, indent=2)
    
    logger.info(f"Phase 1 Data Pipeline completed successfully!")
    logger.info(f"Processed {len(all_metadata)} contracts with {sum(len(c) for c in all_clauses)} clauses")
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
