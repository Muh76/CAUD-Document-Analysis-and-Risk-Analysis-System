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

        # Patterns for clause boundaries
        self.clause_patterns = [
            r'\n\s*\d+\.\s+',  # Numbered clauses: "1. ", "2. ", etc.
            r'\n\s*ARTICLE\s+[IVX]+\s*[\-:]',  # Article headers: "ARTICLE I -", "ARTICLE II:"
            r'\n\s*SECTION\s+\d+\s*[\-:]',  # Section headers: "SECTION 1 -", "SECTION 2:"
            r'\n\s*[A-Z][A-Z\s]+[\-:]',  # All-caps headers: "LICENSE GRANT:", "TERM AND TERMINATION"
            r'\n\s*[a-z]\)\s+',  # Lettered subclauses: "a) ", "b) ", etc.
            r'\n\s*\([a-z]\)\s+',  # Parenthesized subclauses: "(a) ", "(b) ", etc.
        ]

        # Compile patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.clause_patterns]

    def chunk_text(self, text: str, page_number: Optional[int] = None) -> List[TextChunk]:
        """Split text into intelligent chunks."""
        if not text.strip():
            return []

        # Clean text
        text = self._clean_text(text)

        # Find clause boundaries
        boundaries = self._find_clause_boundaries(text)

        # Create chunks
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_id = 0

        for i, char in enumerate(text):
            current_chunk += char

            # Check if we've hit a boundary or max size
            if i in boundaries or len(current_chunk) >= self.max_chunk_size:
                if len(current_chunk.strip()) >= self.min_chunk_size:
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
                    current_chunk = overlap_text
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    chunk_id += 1

        # Add final chunk if it exists
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                start_offset=current_start,
                end_offset=current_start + len(current_chunk),
                page_number=page_number,
                chunk_id=chunk_id
            ))

        return chunks

    def _find_clause_boundaries(self, text: str) -> List[int]:
        """Find clause boundary positions."""
        boundaries = []

        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                boundaries.append(match.start())

        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))

        return boundaries

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
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
