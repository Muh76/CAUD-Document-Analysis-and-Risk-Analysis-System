"""
Enhanced text ingestion and chunking utilities for PDF and text processing.
"""

import re
from typing import List, Optional, Dict, Any
from pathlib import Path
from .schemas import TextChunk
from .io import IOUtils
from .pdf_processor import PDFProcessor
from .clause_chunker import ClauseChunker

class TextIngestion:
    """Enhanced text extraction and chunking functionality."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, use_pymupdf: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_processor = PDFProcessor(use_pymupdf=use_pymupdf)
        self.clause_chunker = ClauseChunker(
            max_chunk_size=chunk_size,
            min_chunk_size=20,
            overlap_size=chunk_overlap
        )

    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from various file formats."""
        return IOUtils.extract_text_from_file(file_path)

    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text from PDF with metadata."""
        return self.pdf_processor.extract_text_with_metadata(pdf_path)

    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract text from PDF bytes with metadata."""
        return self.pdf_processor.extract_text_from_bytes(pdf_bytes)

    def chunk_text(self, text: str, page_number: Optional[int] = None) -> List[TextChunk]:
        """Split text into intelligent chunks."""
        return self.clause_chunker.chunk_text(text, page_number)

    def chunk_text_by_sentences(self, text: str, page_number: Optional[int] = None) -> List[TextChunk]:
        """Split text into chunks by sentences."""
        return self.clause_chunker.chunk_by_sentences(text, page_number)

    def process_contract_file(self, file_path: Path) -> List[TextChunk]:
        """Process a contract file and return chunks."""
        suffix = file_path.suffix.lower()

        if suffix == '.pdf':
            # Extract text from PDF with metadata
            pdf_data = self.extract_text_from_pdf(file_path)
            chunks = []

            # Process each page
            for page_data in pdf_data['pages']:
                page_chunks = self.chunk_text(
                    page_data['text'], 
                    page_data['page_number']
                )
                chunks.extend(page_chunks)

            return chunks
        else:
            # Process as text file
            text = self.extract_text_from_file(file_path)
            return self.chunk_text(text)

    def process_contract_bytes(self, file_bytes: bytes, mime_type: str) -> List[TextChunk]:
        """Process contract bytes and return chunks."""
        if mime_type == 'application/pdf':
            # Extract text from PDF bytes
            pdf_data = self.extract_text_from_pdf_bytes(file_bytes)
            chunks = []

            # Process each page
            for page_data in pdf_data['pages']:
                page_chunks = self.chunk_text(
                    page_data['text'], 
                    page_data['page_number']
                )
                chunks.extend(page_chunks)

            return chunks
        elif mime_type == 'text/plain':
            # Process as text
            text = file_bytes.decode('utf-8')
            return self.chunk_text(text)
        else:
            raise ValueError(f"Unsupported MIME type: {mime_type}")

    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Get text statistics."""
        words = text.split()
        sentences = re.split(r'[.!?]+\s+', text)
        paragraphs = text.split('\n\n')

        return {
            "total_characters": len(text),
            "total_words": len(words),
            "total_sentences": len([s for s in sentences if s.strip()]),
            "total_paragraphs": len([p for p in paragraphs if p.strip()]),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "average_sentence_length": len(words) / len(sentences) if sentences else 0
        }

    def validate_text_quality(self, text: str) -> Dict[str, Any]:
        """Validate text quality."""
        stats = self.get_text_statistics(text)

        issues = []
        if stats["total_characters"] < 100:
            issues.append("Text too short")
        if stats["total_words"] < 20:
            issues.append("Too few words")
        if stats["average_word_length"] < 2:
            issues.append("Words too short (possible encoding issue)")
        if stats["average_sentence_length"] > 50:
            issues.append("Sentences too long")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "statistics": stats
        }
