"""
PDF processing utilities using PyMuPDF and pdfplumber fallback.
"""

import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import io

class PDFProcessor:
    """PDF text extraction and processing."""

    def __init__(self, use_pymupdf: bool = True):
        self.use_pymupdf = use_pymupdf

    def extract_text_with_metadata(self, pdf_path: Path) -> Dict[str, any]:
        """Extract text with page numbers and offsets."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if self.use_pymupdf:
            return self._extract_with_pymupdf(pdf_path)
        else:
            return self._extract_with_pdfplumber(pdf_path)

    def _extract_with_pymupdf(self, pdf_path: Path) -> Dict[str, any]:
        """Extract text using PyMuPDF (faster)."""
        doc = fitz.open(pdf_path)
        pages_data = []
        full_text = ""

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            if text.strip():
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "start_offset": len(full_text),
                    "end_offset": len(full_text) + len(text)
                })
                full_text += text + "\n"

        doc.close()

        return {
            "full_text": full_text.strip(),
            "pages": pages_data,
            "total_pages": len(doc),
            "method": "pymupdf"
        }

    def _extract_with_pdfplumber(self, pdf_path: Path) -> Dict[str, any]:
        """Extract text using pdfplumber (fallback)."""
        pages_data = []
        full_text = ""

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()

                if text and text.strip():
                    pages_data.append({
                        "page_number": page_num + 1,
                        "text": text,
                        "start_offset": len(full_text),
                        "end_offset": len(full_text) + len(text)
                    })
                    full_text += text + "\n"

        return {
            "full_text": full_text.strip(),
            "pages": pages_data,
            "total_pages": len(pdf.pages),
            "method": "pdfplumber"
        }

    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Dict[str, any]:
        """Extract text from PDF bytes."""
        if self.use_pymupdf:
            return self._extract_bytes_with_pymupdf(pdf_bytes)
        else:
            return self._extract_bytes_with_pdfplumber(pdf_bytes)

    def _extract_bytes_with_pymupdf(self, pdf_bytes: bytes) -> Dict[str, any]:
        """Extract text from bytes using PyMuPDF."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_data = []
        full_text = ""

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            if text.strip():
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "start_offset": len(full_text),
                    "end_offset": len(full_text) + len(text)
                })
                full_text += text + "\n"

        doc.close()

        return {
            "full_text": full_text.strip(),
            "pages": pages_data,
            "total_pages": len(doc),
            "method": "pymupdf_bytes"
        }

    def _extract_bytes_with_pdfplumber(self, pdf_bytes: bytes) -> Dict[str, any]:
        """Extract text from bytes using pdfplumber."""
        pages_data = []
        full_text = ""

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()

                if text and text.strip():
                    pages_data.append({
                        "page_number": page_num + 1,
                        "text": text,
                        "start_offset": len(full_text),
                        "end_offset": len(full_text) + len(text)
                    })
                    full_text += text + "\n"

        return {
            "full_text": full_text.strip(),
            "pages": pages_data,
            "total_pages": len(pdf.pages),
            "method": "pdfplumber_bytes"
        }
