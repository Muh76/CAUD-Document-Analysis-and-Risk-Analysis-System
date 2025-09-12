"""
Intelligent clause chunking for contract text.
"""

import re
from typing import List, Dict, Optional
from .schemas import TextChunk

class ClauseChunker:
    """Intelligent text chunking for contract clauses."""

    def __init__(self, max_chunk_size: int = 500, min_chunk_size: int = 100, overlap_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size

        # Enhanced patterns for clause boundaries - much more robust
        self.clause_patterns = [
            # Numbered clauses with optional subclauses: "1. ", "1.1 ", "1.1.1 ", etc.
            r'(?m)^\s*(\d+(?:\.\d+)*)\.\s+',
            # Article headers: "ARTICLE I", "ARTICLE II", etc.
            r'(?m)^\s*ARTICLE\s+[IVX]+(?:\s*[:\-])?\s*',
            # Section headers: "SECTION 1", "SECTION 2", etc.
            r'(?m)^\s*SECTION\s+\d+(?:\s*[:\-])?\s*',
            # All-caps headers ending with colon or dash
            r'(?m)^\s*[A-Z][A-Z\s&/\-]+[:\-]\s*',
            # Lettered subclauses: "a) ", "b) ", etc.
            r'(?m)^\s*[a-z]\)\s+',
            # Parenthesized subclauses: "(a) ", "(b) ", etc.
            r'(?m)^\s*\([a-z]\)\s+',
            # Roman numeral subclauses: "i) ", "ii) ", etc.
            r'(?m)^\s*[ivx]+\)\s+',
            # Common contract section headers
            r'(?m)^\s*(?:DEFINITIONS?|TERM\s+AND\s+TERMINATION|INDEMNITY|LIMITATION\s+ON\s+LIABILITY|CONFIDENTIAL\s+INFORMATION|GENERAL\s+PROVISIONS|GOVERNING\s+LAW|FORCE\s+MAJEURE|DISCLAIMER\s+OF\s+WARRANTIES|REPRESENTATIONS\s+AND\s+WARRANTIES|COVENANTS|DEFAULT|REMEDIES|ASSIGNMENT|AMENDMENT|WAIVER|SEVERABILITY|ENTIRE\s+AGREEMENT|NOTICES|COUNTERPARTS|HEADINGS|INTERPRETATION|CONSTRUCTION|VALIDITY|ENFORCEABILITY|JURISDICTION|VENUE|ARBITRATION|MEDIATION|DISPUTE\s+RESOLUTION|TERMINATION|EXPIRATION|RENEWAL|EXTENSION|MODIFICATION|SUPPLEMENT|ADDENDUM|EXHIBIT|SCHEDULE|APPENDIX|ATTACHMENT|ANNEX|RIDER|AMENDMENT|SUPPLEMENT|ADDENDUM|EXHIBIT|SCHEDULE|APPENDIX|ATTACHMENT|ANNEX|RIDER)\s*[:\-]?\s*',
        ]

        # Compile patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.clause_patterns]

    def chunk_text(self, text: str, page_number: Optional[int] = None) -> List[TextChunk]:
        """Split text into intelligent chunks using robust clause detection."""
        if not text.strip():
            return []

        # Clean text but preserve structure
        text = self._clean_text_preserve_structure(text)
        
        # Find all clause boundaries
        boundaries = self._find_all_clause_boundaries(text)
        
        # If no boundaries found, fall back to sentence-based chunking
        if not boundaries:
            return self.chunk_by_sentences(text, page_number)
        
        # Create chunks based on boundaries
        chunks = []
        chunk_id = 0
        
        # Sort boundaries and add start/end positions
        boundaries = sorted(set(boundaries))
        if boundaries[0] != 0:
            boundaries.insert(0, 0)
        boundaries.append(len(text))
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_text = text[start:end].strip()
            
            # Skip very short chunks
            if len(chunk_text) < self.min_chunk_size:
                continue
                
            # Truncate very long chunks
            if len(chunk_text) > self.max_chunk_size:
                chunk_text = chunk_text[:self.max_chunk_size] + "..."
            
            chunks.append(TextChunk(
                text=chunk_text,
                start_offset=start,
                end_offset=end,
                page_number=page_number,
                chunk_id=chunk_id
            ))
            chunk_id += 1
        
        return chunks

    def _find_all_clause_boundaries(self, text: str) -> List[int]:
        """Find all clause boundary positions using enhanced patterns."""
        boundaries = []
        
        # Find boundaries using all patterns
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                boundaries.append(match.start())
        
        # Also look for line-based patterns that might be missed
        lines = text.split('\n')
        current_pos = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for numbered clauses at start of line
            if re.match(r'^\s*\d+(?:\.\d+)*\.\s+', line_stripped):
                boundaries.append(current_pos)
            
            # Check for common contract headers
            elif re.match(r'^\s*(?:DEFINITIONS?|TERM\s+AND\s+TERMINATION|INDEMNITY|LIMITATION\s+ON\s+LIABILITY|CONFIDENTIAL\s+INFORMATION|GENERAL\s+PROVISIONS|GOVERNING\s+LAW|FORCE\s+MAJEURE|DISCLAIMER\s+OF\s+WARRANTIES)', line_stripped, re.IGNORECASE):
                boundaries.append(current_pos)
            
            current_pos += len(line) + 1  # +1 for newline
        
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        return boundaries

    def _clean_text_preserve_structure(self, text: str) -> str:
        """Clean text while preserving clause structure."""
        # Remove control characters but keep newlines and structure
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize whitespace but preserve line breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n[ \t]+', '\n', text)  # Remove leading whitespace from lines
        text = re.sub(r'[ \t]+\n', '\n', text)  # Remove trailing whitespace from lines
        
        return text.strip()

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        if len(text) <= self.overlap_size:
            return text

        # Find last sentence within overlap range
        sentences = re.split(r'[.!?]+\s+', text)
        overlap_text = ""

        for sentence in reversed(sentences):
            if len(overlap_text + sentence) <= self.overlap_size:
                overlap_text = sentence + overlap_text
            else:
                break

        return overlap_text

    def chunk_by_sentences(self, text: str, page_number: Optional[int] = None) -> List[TextChunk]:
        """Chunk text by sentences (fallback method)."""
        if not text.strip():
            return []

        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_id = 0

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    start_offset=current_start,
                    end_offset=current_start + len(current_chunk),
                    page_number=page_number,
                    chunk_id=chunk_id
                ))

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
                current_start = current_start + len(current_chunk) - len(overlap_text) - len(sentence)
                chunk_id += 1
            else:
                current_chunk += sentence + " "

        # Add final chunk if it exists
        if current_chunk.strip():
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                start_offset=current_start,
                end_offset=current_start + len(current_chunk),
                page_number=page_number,
                chunk_id=chunk_id
            ))

        return chunks
