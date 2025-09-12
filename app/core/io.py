"""
I/O utilities for ID normalization, file handling, and data processing.
"""

import re
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib

class IOUtils:
    """Utility functions for I/O operations."""

    @staticmethod
    def normalize_contract_id(contract_id: str) -> str:
        """Normalize contract ID to a standard format."""
        if not contract_id:
            return f"contract_{uuid.uuid4().hex[:8]}"

        # Remove special characters and normalize
        normalized = re.sub(r'[^a-zA-Z0-9_-]', '_', contract_id)
        normalized = re.sub(r'_+', '_', normalized)  # Collapse multiple underscores
        normalized = normalized.strip('_').lower()

        # Ensure it starts with a letter or number
        if not re.match(r'^[a-zA-Z0-9]', normalized):
            normalized = f"contract_{normalized}"

        # Limit length
        if len(normalized) > 50:
            normalized = normalized[:47] + "_" + hashlib.md5(contract_id.encode()).hexdigest()[:2]

        return normalized

    @staticmethod
    def extract_text_from_file(file_path: Path) -> str:
        """Extract text from various file formats."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif suffix == '.pdf':
            # For now, return placeholder - will implement PDF extraction later
            return f"[PDF content from {file_path.name}]"
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def validate_text_length(text: str, max_length: int = 4000) -> str:
        """Validate and truncate text if necessary."""
        if len(text) > max_length:
            return text[:max_length] + "... [truncated]"
        return text

    @staticmethod
    def generate_clause_id(contract_id: str, clause_index: int) -> str:
        """Generate a unique clause ID."""
        return f"{contract_id}_clause_{clause_index:03d}"

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Preprocess text for model inference."""
        if not text:
            return ""
        
        # Basic text cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        
        # Normalize case (keep original for now, but could lowercase)
        # text = text.lower()
        
        return text.strip()
