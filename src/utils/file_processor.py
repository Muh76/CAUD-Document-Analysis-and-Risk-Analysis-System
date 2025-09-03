"""
File Processing Utilities for Contract Review System
"""

import os
import io
from typing import Optional, List
import logging
from pathlib import Path
import PyPDF2
from docx import Document
import re

from src.config.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE


class FileProcessor:
    """File processing utilities for contract documents"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.allowed_extensions = ALLOWED_EXTENSIONS
        self.max_file_size = MAX_FILE_SIZE

    def is_valid_file(self, file) -> bool:
        """
        Validate uploaded file

        Args:
            file: Uploaded file object

        Returns:
            True if file is valid, False otherwise
        """
        try:
            # Check file size
            if file.size > self.max_file_size:
                self.logger.warning(f"File too large: {file.size} bytes")
                return False

            # Check file extension
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in self.allowed_extensions:
                self.logger.warning(f"Invalid file extension: {file_extension}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating file: {e}")
            return False

    async def extract_text(self, file) -> str:
        """
        Extract text from uploaded file

        Args:
            file: Uploaded file object

        Returns:
            Extracted text content
        """
        try:
            file_extension = Path(file.filename).suffix.lower()

            if file_extension == ".pdf":
                return await self._extract_from_pdf(file)
            elif file_extension == ".txt":
                return await self._extract_from_txt(file)
            elif file_extension == ".docx":
                return await self._extract_from_docx(file)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            raise

    async def _extract_from_pdf(self, file) -> str:
        """Extract text from PDF file"""
        try:
            # Read PDF content
            content = await file.read()
            pdf_file = io.BytesIO(content)

            # Extract text using PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            return self._clean_text(text)

        except Exception as e:
            self.logger.error(f"Error extracting from PDF: {e}")
            raise

    async def _extract_from_txt(self, file) -> str:
        """Extract text from TXT file"""
        try:
            content = await file.read()

            # Try different encodings
            encodings = ["utf-8", "latin-1", "cp1252"]

            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    return self._clean_text(text)
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, use utf-8 with errors='ignore'
            text = content.decode("utf-8", errors="ignore")
            return self._clean_text(text)

        except Exception as e:
            self.logger.error(f"Error extracting from TXT: {e}")
            raise

    async def _extract_from_docx(self, file) -> str:
        """Extract text from DOCX file"""
        try:
            content = await file.read()
            docx_file = io.BytesIO(content)

            # Extract text using python-docx
            doc = Document(docx_file)
            text = ""

            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

            return self._clean_text(text)

        except Exception as e:
            self.logger.error(f"Error extracting from DOCX: {e}")
            raise

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        try:
            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text)

            # Remove special characters but keep legal terms
            text = re.sub(r"[^\w\s\.\,\;\:\!\?\(\)\[\]\{\}\"\'-]", "", text)

            # Normalize line breaks
            text = text.replace("\n", " ").replace("\r", " ")

            # Remove multiple spaces
            text = re.sub(r" +", " ", text)

            # Strip leading/trailing whitespace
            text = text.strip()

            return text

        except Exception as e:
            self.logger.error(f"Error cleaning text: {e}")
            return text

    def extract_metadata(self, text: str) -> dict:
        """
        Extract metadata from contract text

        Args:
            text: Contract text

        Returns:
            Dictionary of metadata
        """
        metadata = {
            "document_type": self._detect_document_type(text),
            "parties": self._extract_parties(text),
            "dates": self._extract_dates(text),
            "amounts": self._extract_amounts(text),
            "word_count": len(text.split()),
            "character_count": len(text),
        }

        return metadata

    def _detect_document_type(self, text: str) -> str:
        """Detect type of legal document"""
        text_lower = text.lower()

        if any(term in text_lower for term in ["employment", "hire", "employee"]):
            return "employment_agreement"
        elif any(
            term in text_lower for term in ["nda", "non-disclosure", "confidentiality"]
        ):
            return "nda"
        elif any(
            term in text_lower for term in ["service", "consulting", "professional"]
        ):
            return "service_agreement"
        elif any(term in text_lower for term in ["license", "licensing", "permit"]):
            return "license_agreement"
        elif any(term in text_lower for term in ["purchase", "sale", "buy"]):
            return "purchase_agreement"
        else:
            return "general_contract"

    def _extract_parties(self, text: str) -> List[str]:
        """Extract party names from contract"""
        parties = []

        # Common party patterns
        patterns = [
            r"between\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"(\w+\s+Inc\.)",
            r"(\w+\s+LLC)",
            r"(\w+\s+Corp\.)",
            r"(\w+\s+Company)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            parties.extend(matches)

        return list(set(parties))  # Remove duplicates

    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from contract"""
        dates = []

        # Date patterns
        patterns = [
            r"\d{1,2}/\d{1,2}/\d{4}",
            r"\d{4}-\d{2}-\d{2}",
            r"\w+\s+\d{1,2},\s+\d{4}",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)

        return list(set(dates))

    def _extract_amounts(self, text: str) -> List[str]:
        """Extract monetary amounts from contract"""
        amounts = []

        # Amount patterns
        patterns = [
            r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
            r"\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s+dollars",
            r"\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s+USD",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts.extend(matches)

        return list(set(amounts))

    def save_file(self, file, directory: str, filename: Optional[str] = None) -> str:
        """
        Save uploaded file to disk

        Args:
            file: Uploaded file object
            directory: Directory to save file
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        try:
            # Create directory if it doesn't exist
            Path(directory).mkdir(parents=True, exist_ok=True)

            # Generate filename if not provided
            if filename is None:
                filename = file.filename

            # Ensure unique filename
            file_path = Path(directory) / filename
            counter = 1

            while file_path.exists():
                name, ext = Path(filename).stem, Path(filename).suffix
                filename = f"{name}_{counter}{ext}"
                file_path = Path(directory) / filename
                counter += 1

            # Save file
            with open(file_path, "wb") as f:
                content = file.file.read()
                f.write(content)

            self.logger.info(f"File saved: {file_path}")
            return str(file_path)

        except Exception as e:
            self.logger.error(f"Error saving file: {e}")
            raise

    def get_file_info(self, file) -> dict:
        """
        Get information about uploaded file

        Args:
            file: Uploaded file object

        Returns:
            Dictionary with file information
        """
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file.size,
            "size_mb": round(file.size / (1024 * 1024), 2),
        }
