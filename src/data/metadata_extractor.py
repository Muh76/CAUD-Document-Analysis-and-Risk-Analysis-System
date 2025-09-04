"""
Metadata Extraction for Contract Analysis System
Extracts parties, dates, amounts, governing law, and other key metadata
"""

import re
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime
import dateparser
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContractMetadata:
    """Structured contract metadata"""
    contract_id: str
    parties: List[str]
    effective_date: Optional[str]
    expiration_date: Optional[str]
    contract_value: Optional[float]
    currency: Optional[str]
    governing_law: Optional[str]
    jurisdiction: Optional[str]
    contract_type: Optional[str]
    total_clauses: int
    processing_timestamp: str


class MetadataExtractor:
    """Extract metadata from contract text using patterns and NLP"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_database()
        
        # Date patterns
        self.date_patterns = [
            r'\b(?:effective|commencement|start)\s+(?:date|as\s+of)?\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'\b(?:expiration|end|termination)\s+(?:date)?\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?:effective|commencement)',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}',
        ]
        
        # Amount patterns
        self.amount_patterns = [
            r'\$[\d,]+(?:\.\d{2})?',  # $1,000.00
            r'[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP|CAD|AUD)',  # 1,000.00 USD
            r'(?:contract\s+value|total\s+value|amount)\s*:?\s*[\$£€]?[\d,]+(?:\.\d{2})?',
        ]
        
        # Party patterns
        self.party_patterns = [
            r'(?:between|by|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Corporation))',
            r'(?:party|parties)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:hereinafter\s+referred\s+to\s+as)\s+["\']([^"\']+)["\']',
        ]
        
        # Governing law patterns
        self.governing_law_patterns = [
            r'(?:governed\s+by|governing\s+law)\s+(?:the\s+)?laws?\s+of\s+([A-Za-z\s,]+?)(?:\.|,|$)',
            r'(?:jurisdiction|venue)\s+(?:of|in)\s+([A-Za-z\s,]+?)(?:\.|,|$)',
            r'(?:subject\s+to|under)\s+(?:the\s+)?laws?\s+of\s+([A-Za-z\s,]+?)(?:\.|,|$)',
        ]

    def setup_database(self):
        """Setup SQLite database for metadata storage"""
        db_path = Path("data/metadata.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        
        # Create tables
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS contracts (
                    contract_id TEXT PRIMARY KEY,
                    file_name TEXT,
                    file_path TEXT,
                    parties TEXT,
                    effective_date TEXT,
                    expiration_date TEXT,
                    contract_value REAL,
                    currency TEXT,
                    governing_law TEXT,
                    jurisdiction TEXT,
                    contract_type TEXT,
                    total_clauses INTEGER,
                    processing_timestamp TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parties (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_id TEXT,
                    party_name TEXT,
                    party_type TEXT,
                    FOREIGN KEY (contract_id) REFERENCES contracts (contract_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS amounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_id TEXT,
                    amount_type TEXT,
                    amount_value REAL,
                    currency TEXT,
                    context TEXT,
                    FOREIGN KEY (contract_id) REFERENCES contracts (contract_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_id TEXT,
                    date_type TEXT,
                    date_value TEXT,
                    context TEXT,
                    FOREIGN KEY (contract_id) REFERENCES contracts (contract_id)
                )
            """)

    def extract_metadata(self, text: str, contract_id: str) -> ContractMetadata:
        """
        Extract comprehensive metadata from contract text
        
        Args:
            text: Contract text
            contract_id: Contract identifier
            
        Returns:
            ContractMetadata object
        """
        logger.info(f"Extracting metadata for contract {contract_id}")
        
        # Extract parties
        parties = self._extract_parties(text)
        
        # Extract dates
        effective_date = self._extract_effective_date(text)
        expiration_date = self._extract_expiration_date(text)
        
        # Extract amounts
        contract_value, currency = self._extract_contract_value(text)
        
        # Extract governing law and jurisdiction
        governing_law = self._extract_governing_law(text)
        jurisdiction = self._extract_jurisdiction(text)
        
        # Determine contract type
        contract_type = self._classify_contract_type(text)
        
        # Count clauses (approximate)
        total_clauses = self._count_clauses(text)
        
        return ContractMetadata(
            contract_id=contract_id,
            parties=parties,
            effective_date=effective_date,
            expiration_date=expiration_date,
            contract_value=contract_value,
            currency=currency,
            governing_law=governing_law,
            jurisdiction=jurisdiction,
            contract_type=contract_type,
            total_clauses=total_clauses,
            processing_timestamp=datetime.now().isoformat(),
        )

    def _extract_parties(self, text: str) -> List[str]:
        """Extract party names from contract text"""
        parties = set()
        
        # Use party patterns
        for pattern in self.party_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                party = match.strip()
                if len(party) > 3:  # Filter out very short matches
                    parties.add(party)
        
        # Look for common company suffixes
        company_suffixes = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Corporation|Limited|Incorporated))\b'
        matches = re.findall(company_suffixes, text)
        for match in matches:
            parties.add(match.strip())
        
        return list(parties)

    def _extract_effective_date(self, text: str) -> Optional[str]:
        """Extract effective date from contract text"""
        # Look for effective date patterns
        effective_patterns = [
            r'\b(?:effective|commencement|start)\s+(?:date|as\s+of)?\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?:effective|commencement)',
        ]
        
        for pattern in effective_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    date_str = matches[0]
                    parsed_date = dateparser.parse(date_str)
                    if parsed_date:
                        return parsed_date.strftime('%Y-%m-%d')
                except:
                    continue
        
        # Try to parse any date near "effective"
        effective_context = re.findall(r'effective[^.]*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
        if effective_context:
            try:
                parsed_date = dateparser.parse(effective_context[0])
                if parsed_date:
                    return parsed_date.strftime('%Y-%m-%d')
            except:
                pass
        
        return None

    def _extract_expiration_date(self, text: str) -> Optional[str]:
        """Extract expiration date from contract text"""
        # Look for expiration date patterns
        expiration_patterns = [
            r'\b(?:expiration|end|termination)\s+(?:date)?\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'\b(?:expires|ending)\s+(?:on|at)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]
        
        for pattern in expiration_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    date_str = matches[0]
                    parsed_date = dateparser.parse(date_str)
                    if parsed_date:
                        return parsed_date.strftime('%Y-%m-%d')
                except:
                    continue
        
        return None

    def _extract_contract_value(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """Extract contract value and currency"""
        # Look for amount patterns
        for pattern in self.amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                match = matches[0]
                
                # Extract currency
                currency = None
                if '$' in match:
                    currency = 'USD'
                elif '£' in match:
                    currency = 'GBP'
                elif '€' in match:
                    currency = 'EUR'
                elif 'USD' in match.upper():
                    currency = 'USD'
                elif 'EUR' in match.upper():
                    currency = 'EUR'
                elif 'GBP' in match.upper():
                    currency = 'GBP'
                
                # Extract numeric value
                try:
                    # Remove currency symbols and commas
                    value_str = re.sub(r'[^\d.]', '', match)
                    value = float(value_str)
                    return value, currency
                except:
                    continue
        
        return None, None

    def _extract_governing_law(self, text: str) -> Optional[str]:
        """Extract governing law from contract text"""
        for pattern in self.governing_law_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                law = matches[0].strip()
                # Clean up the extracted text
                law = re.sub(r'[^\w\s]', '', law)
                return law
        
        return None

    def _extract_jurisdiction(self, text: str) -> Optional[str]:
        """Extract jurisdiction from contract text"""
        jurisdiction_patterns = [
            r'(?:jurisdiction|venue)\s+(?:of|in)\s+([A-Za-z\s,]+?)(?:\.|,|$)',
            r'(?:courts?\s+of)\s+([A-Za-z\s,]+?)(?:\.|,|$)',
        ]
        
        for pattern in jurisdiction_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                jurisdiction = matches[0].strip()
                jurisdiction = re.sub(r'[^\w\s]', '', jurisdiction)
                return jurisdiction
        
        return None

    def _classify_contract_type(self, text: str) -> str:
        """Classify contract type based on content"""
        text_lower = text.lower()
        
        contract_keywords = {
            "employment": ["employment", "employee", "hire", "termination"],
            "service": ["service", "consulting", "professional"],
            "purchase": ["purchase", "buy", "sale", "vendor"],
            "lease": ["lease", "rental", "tenant", "landlord"],
            "nda": ["non-disclosure", "confidentiality", "nda"],
            "partnership": ["partnership", "joint venture", "collaboration"],
            "licensing": ["license", "licensing", "intellectual property"],
        }
        
        for contract_type, keywords in contract_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return contract_type
        
        return "general"

    def _count_clauses(self, text: str) -> int:
        """Count approximate number of clauses"""
        # Count section headers
        section_patterns = [
            r'\b(?:Section|Article|Clause)\s+\d+',
            r'\b\d+\.\s+[A-Z]',
            r'\b[A-Z]\.\s+[A-Z]',
        ]
        
        total_clauses = 0
        for pattern in section_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            total_clauses += len(matches)
        
        return max(total_clauses, 1)  # At least 1 clause

    def save_metadata(self, metadata: ContractMetadata):
        """Save metadata to SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            # Insert contract metadata
            conn.execute("""
                INSERT OR REPLACE INTO contracts (
                    contract_id, parties, effective_date, expiration_date,
                    contract_value, currency, governing_law, jurisdiction,
                    contract_type, total_clauses, processing_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.contract_id,
                json.dumps(metadata.parties),
                metadata.effective_date,
                metadata.expiration_date,
                metadata.contract_value,
                metadata.currency,
                metadata.governing_law,
                metadata.jurisdiction,
                metadata.contract_type,
                metadata.total_clauses,
                metadata.processing_timestamp,
            ))
            
            # Insert parties
            for party in metadata.parties:
                conn.execute("""
                    INSERT INTO parties (contract_id, party_name, party_type)
                    VALUES (?, ?, ?)
                """, (metadata.contract_id, party, "company"))
            
            # Insert amounts
            if metadata.contract_value:
                conn.execute("""
                    INSERT INTO amounts (contract_id, amount_type, amount_value, currency)
                    VALUES (?, ?, ?, ?)
                """, (metadata.contract_id, "contract_value", metadata.contract_value, metadata.currency))
            
            # Insert dates
            if metadata.effective_date:
                conn.execute("""
                    INSERT INTO dates (contract_id, date_type, date_value)
                    VALUES (?, ?, ?)
                """, (metadata.contract_id, "effective_date", metadata.effective_date))
            
            if metadata.expiration_date:
                conn.execute("""
                    INSERT INTO dates (contract_id, date_type, date_value)
                    VALUES (?, ?, ?)
                """, (metadata.contract_id, "expiration_date", metadata.expiration_date))

    def process_contracts_batch(self, contracts_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process metadata extraction for multiple contracts
        
        Args:
            contracts_data: List of contract data with text
            
        Returns:
            Processing summary
        """
        logger.info(f"Processing metadata for {len(contracts_data)} contracts")
        
        results = {
            "total_contracts": len(contracts_data),
            "processed": 0,
            "failed": 0,
            "metadata_list": [],
            "summary_stats": {},
        }
        
        for contract_data in contracts_data:
            try:
                contract_id = contract_data["contract_id"]
                text = contract_data["text"]
                
                # Extract metadata
                metadata = self.extract_metadata(text, contract_id)
                
                # Save to database
                self.save_metadata(metadata)
                
                # Add to results
                results["metadata_list"].append(metadata)
                results["processed"] += 1
                
                logger.info(f"Extracted metadata for {contract_id}")
                
            except Exception as e:
                results["failed"] += 1
                logger.error(f"Failed to extract metadata for {contract_data.get('contract_id', 'unknown')}: {e}")
        
        # Calculate summary statistics
        if results["metadata_list"]:
            total_parties = sum(len(m.parties) for m in results["metadata_list"])
            contracts_with_dates = sum(1 for m in results["metadata_list"] if m.effective_date)
            contracts_with_values = sum(1 for m in results["metadata_list"] if m.contract_value)
            
            results["summary_stats"] = {
                "avg_parties_per_contract": total_parties / len(results["metadata_list"]),
                "contracts_with_effective_date": contracts_with_dates,
                "contracts_with_contract_value": contracts_with_values,
                "contract_types": {},
            }
            
            # Count contract types
            for metadata in results["metadata_list"]:
                contract_type = metadata.contract_type or "unknown"
                results["summary_stats"]["contract_types"][contract_type] = \
                    results["summary_stats"]["contract_types"].get(contract_type, 0) + 1
        
        logger.info(f"Metadata extraction complete: {results['processed']}/{results['total_contracts']} successful")
        return results


def main():
    """Main function for metadata extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Metadata Extraction Pipeline")
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
    
    # Initialize extractor and process
    config = {}
    extractor = MetadataExtractor(config)
    results = extractor.process_contracts_batch(contracts_data)
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "metadata_extraction_report.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Metadata Extraction Results:")
    print(f"Total contracts: {results['total_contracts']}")
    print(f"Processed: {results['processed']}")
    print(f"Failed: {results['failed']}")
    if results["summary_stats"]:
        print(f"Avg parties per contract: {results['summary_stats']['avg_parties_per_contract']:.2f}")


if __name__ == "__main__":
    main()

