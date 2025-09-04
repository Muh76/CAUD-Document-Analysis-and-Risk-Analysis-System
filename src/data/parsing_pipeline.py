"""
Parsing Pipeline for Contract Analysis System
Handles PDF, DOCX parsing with OCR fallback and clause segmentation
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# PDF parsing
try:
    import pypdf
    from pdfplumber import open as pdfplumber_open
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PDF parsing not available - install pypdf and pdfplumber")

# DOCX parsing
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️ DOCX parsing not available - install python-docx")

# OCR fallback
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR not available - install pytesseract and pdf2image")

from src.config.config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class ContractParser:
    """Comprehensive contract parsing with PDF, DOCX, and OCR support"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_directories()
        
        # Clause segmentation patterns
        self.heading_patterns = [
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS headings
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title Case headings
            r'^\d+\.\s+[A-Z]',  # Numbered sections (1. Title)
            r'^[A-Z]\.\s+[A-Z]',  # Lettered sections (A. Title)
        ]
        
        self.bullet_patterns = [
            r'^\d+\.\d+',  # 1.1, 1.2, etc.
            r'^[•\-*]\s+',  # Bullet points
            r'^\(\d+\)',  # (1), (2), etc.
        ]

    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            Path(DATA_DIR) / "raw",
            Path(DATA_DIR) / "processed",
            Path(DATA_DIR) / "clause_spans",
            Path(DATA_DIR) / "temp",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def parse_contract(self, file_path: str) -> Dict[str, Any]:
        """
        Parse contract file (PDF/DOCX) with OCR fallback
        
        Args:
            file_path: Path to contract file
            
        Returns:
            Dictionary with parsed text and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Contract file not found: {file_path}")
        
        # Determine file type and parse accordingly
        if file_path.suffix.lower() == '.pdf':
            return self._parse_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            return self._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def _parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF file with OCR fallback"""
        logger.info(f"Parsing PDF: {file_path}")
        
        try:
            # Try PDFPlumber first (better for text extraction)
            if PDF_AVAILABLE:
                text = self._extract_text_pdfplumber(file_path)
                if text.strip():
                    return {
                        "text": text,
                        "parsing_method": "pdfplumber",
                        "success": True,
                        "file_path": str(file_path),
                        "file_size": file_path.stat().st_size,
                    }
            
            # Try PyPDF as fallback
            if PDF_AVAILABLE:
                text = self._extract_text_pypdf(file_path)
                if text.strip():
                    return {
                        "text": text,
                        "parsing_method": "pypdf",
                        "success": True,
                        "file_path": str(file_path),
                        "file_size": file_path.stat().st_size,
                    }
            
            # OCR fallback
            if OCR_AVAILABLE:
                text = self._extract_text_ocr(file_path)
                return {
                    "text": text,
                    "parsing_method": "ocr",
                    "success": True,
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                }
            
            raise Exception("No PDF parsing method available")
            
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            return {
                "text": "",
                "parsing_method": "failed",
                "success": False,
                "error": str(e),
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
            }

    def _extract_text_pdfplumber(self, file_path: Path) -> str:
        """Extract text using PDFPlumber"""
        text_parts = []
        
        with pdfplumber_open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        return "\n".join(text_parts)

    def _extract_text_pypdf(self, file_path: Path) -> str:
        """Extract text using PyPDF"""
        text_parts = []
        
        with open(file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        return "\n".join(text_parts)

    def _extract_text_ocr(self, file_path: Path) -> str:
        """Extract text using OCR"""
        logger.info(f"Using OCR for {file_path}")
        
        # Convert PDF to images
        images = convert_from_path(file_path)
        
        text_parts = []
        for i, image in enumerate(images):
            try:
                # OCR the image
                text = pytesseract.image_to_string(image)
                if text.strip():
                    text_parts.append(text)
                logger.info(f"OCR completed for page {i+1}")
            except Exception as e:
                logger.warning(f"OCR failed for page {i+1}: {e}")
        
        return "\n".join(text_parts)

    def _parse_docx(self, file_path: Path) -> Dict[str, Any]:
        """Parse DOCX file"""
        logger.info(f"Parsing DOCX: {file_path}")
        
        try:
            if not DOCX_AVAILABLE:
                raise ImportError("python-docx not available")
            
            doc = Document(file_path)
            text_parts = []
            
            for paragraph in doc.paragraphs:
                text_parts.append(paragraph.text)
            
            text = "\n".join(text_parts)
            
            return {
                "text": text,
                "parsing_method": "docx",
                "success": True,
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
            }
            
        except Exception as e:
            logger.error(f"Error parsing DOCX {file_path}: {e}")
            return {
                "text": "",
                "parsing_method": "failed",
                "success": False,
                "error": str(e),
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
            }

    def segment_clauses(self, text: str, contract_id: str) -> List[Dict[str, Any]]:
        """
        Segment text into clauses using multiple strategies
        
        Args:
            text: Contract text
            contract_id: Contract identifier
            
        Returns:
            List of clause segments with metadata
        """
        logger.info(f"Segmenting clauses for contract {contract_id}")
        
        clauses = []
        lines = text.split('\n')
        current_clause = {"text": "", "start_line": 0, "end_line": 0}
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if line is a heading
            is_heading = any(re.match(pattern, line) for pattern in self.heading_patterns)
            is_bullet = any(re.match(pattern, line) for pattern in self.bullet_patterns)
            
            # Start new clause if heading or bullet found
            if is_heading or is_bullet:
                # Save previous clause if it has content
                if current_clause["text"].strip():
                    clause_data = self._create_clause_segment(
                        current_clause, contract_id, len(clauses)
                    )
                    clauses.append(clause_data)
                
                # Start new clause
                current_clause = {
                    "text": line,
                    "start_line": i,
                    "end_line": i,
                    "clause_type": self._classify_clause_type(line)
                }
            else:
                # Continue current clause
                if current_clause["text"]:
                    current_clause["text"] += "\n" + line
                else:
                    current_clause["text"] = line
                current_clause["end_line"] = i
        
        # Add final clause
        if current_clause["text"].strip():
            clause_data = self._create_clause_segment(
                current_clause, contract_id, len(clauses)
            )
            clauses.append(clause_data)
        
        # Fallback: sentence-based segmentation if no clauses found
        if not clauses:
            clauses = self._segment_by_sentences(text, contract_id)
        
        logger.info(f"Extracted {len(clauses)} clauses from contract {contract_id}")
        return clauses

    def _create_clause_segment(
        self, clause: Dict[str, Any], contract_id: str, clause_index: int
    ) -> Dict[str, Any]:
        """Create standardized clause segment"""
        return {
            "clause_id": f"{contract_id}_clause_{clause_index:03d}",
            "contract_id": contract_id,
            "text": clause["text"].strip(),
            "start_line": clause["start_line"],
            "end_line": clause["end_line"],
            "clause_type": clause.get("clause_type", "unknown"),
            "text_length": len(clause["text"]),
            "confidence": self._calculate_confidence(clause["text"]),
        }

    def _segment_by_sentences(self, text: str, contract_id: str) -> List[Dict[str, Any]]:
        """Fallback: segment by sentences if no clause patterns found"""
        sentences = re.split(r'[.!?]+', text)
        clauses = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 50:  # Only include substantial sentences
                clause_data = {
                    "clause_id": f"{contract_id}_sentence_{i:03d}",
                    "contract_id": contract_id,
                    "text": sentence,
                    "start_line": 0,
                    "end_line": 0,
                    "clause_type": "sentence_segment",
                    "text_length": len(sentence),
                    "confidence": 0.5,  # Lower confidence for sentence segments
                }
                clauses.append(clause_data)
        
        return clauses

    def _classify_clause_type(self, text: str) -> str:
        """Classify clause type based on keywords"""
        text_lower = text.lower()
        
        clause_keywords = {
            "governing_law": ["governing law", "jurisdiction", "venue"],
            "liability": ["liability", "damages", "indemnification"],
            "confidentiality": ["confidential", "non-disclosure", "secret"],
            "termination": ["termination", "terminate", "end"],
            "payment": ["payment", "fee", "price", "cost"],
            "warranty": ["warranty", "warrant", "guarantee"],
            "ip_assignment": ["intellectual property", "ip", "assign"],
            "non_compete": ["non-compete", "noncompete", "competition"],
            "force_majeure": ["force majeure", "act of god"],
        }
        
        for clause_type, keywords in clause_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return clause_type
        
        return "general"

    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for clause extraction"""
        # Simple heuristic: longer, well-formatted text gets higher confidence
        if len(text) < 50:
            return 0.3
        elif len(text) < 200:
            return 0.6
        else:
            return 0.8

    def normalize_text(self, text: str) -> str:
        """Normalize text: remove extra whitespace, standardize formatting"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        
        # Standardize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32)
        
        return text.strip()

    def process_contracts_batch(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Process all contracts in a directory
        
        Args:
            input_dir: Directory containing contract files
            output_dir: Directory to save processed data
            
        Returns:
            Processing summary
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all contract files
        contract_files = []
        for ext in ['*.pdf', '*.docx', '*.doc']:
            contract_files.extend(input_path.glob(ext))
        
        logger.info(f"Found {len(contract_files)} contract files")
        
        # Process each contract
        results = {
            "total_files": len(contract_files),
            "successful_parses": 0,
            "failed_parses": 0,
            "total_clauses": 0,
            "contracts": [],
            "clauses": [],
        }
        
        for file_path in contract_files:
            try:
                # Parse contract
                parse_result = self.parse_contract(str(file_path))
                
                if parse_result["success"]:
                    # Normalize text
                    normalized_text = self.normalize_text(parse_result["text"])
                    
                    # Generate contract ID
                    contract_id = f"contract_{file_path.stem}_{results['successful_parses']:04d}"
                    
                    # Segment clauses
                    clauses = self.segment_clauses(normalized_text, contract_id)
                    
                    # Save contract metadata
                    contract_data = {
                        "contract_id": contract_id,
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "file_size": parse_result["file_size"],
                        "parsing_method": parse_result["parsing_method"],
                        "text_length": len(normalized_text),
                        "n_clauses": len(clauses),
                        "parsed_ok": 1,
                        "processing_timestamp": pd.Timestamp.now().isoformat(),
                    }
                    
                    results["contracts"].append(contract_data)
                    results["clauses"].extend(clauses)
                    results["successful_parses"] += 1
                    results["total_clauses"] += len(clauses)
                    
                    # Save clause spans
                    clause_file = output_path / f"clause_spans_{contract_id}.jsonl"
                    with open(clause_file, 'w') as f:
                        for clause in clauses:
                            f.write(json.dumps(clause) + '\n')
                    
                    logger.info(f"Processed {file_path.name}: {len(clauses)} clauses")
                    
                else:
                    results["failed_parses"] += 1
                    logger.error(f"Failed to parse {file_path.name}: {parse_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                results["failed_parses"] += 1
                logger.error(f"Error processing {file_path.name}: {e}")
        
        # Save summary files
        contracts_df = pd.DataFrame(results["contracts"])
        clauses_df = pd.DataFrame(results["clauses"])
        
        contracts_df.to_csv(output_path / "contracts_index.csv", index=False)
        clauses_df.to_csv(output_path / "clause_spans.csv", index=False)
        
        # Save processing report
        report = {
            "processing_summary": results,
            "file_counts": {
                "total_files": results["total_files"],
                "successful_parses": results["successful_parses"],
                "failed_parses": results["failed_parses"],
            },
            "clause_stats": {
                "total_clauses": results["total_clauses"],
                "avg_clauses_per_contract": results["total_clauses"] / max(results["successful_parses"], 1),
            },
            "parsing_methods": contracts_df["parsing_method"].value_counts().to_dict(),
        }
        
        with open(output_path / "parsing_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Batch processing complete: {results['successful_parses']}/{results['total_files']} successful")
        return report


def main():
    """Main function for parsing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Contract Parsing Pipeline")
    parser.add_argument("--input", default="data/raw", help="Input directory")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--config", default="configs/parse.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        "input_dir": args.input,
        "output_dir": args.output,
        "min_clause_length": 50,
        "confidence_threshold": 0.5,
    }
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize parser and process
    parser = ContractParser(config)
    results = parser.process_contracts_batch(args.input, args.output)
    
    print(f"\n📊 Parsing Results:")
    print(f"Total files: {results['total_files']}")
    print(f"Successful: {results['successful_parses']}")
    print(f"Failed: {results['failed_parses']}")
    print(f"Total clauses: {results['total_clauses']}")


if __name__ == "__main__":
    main()

