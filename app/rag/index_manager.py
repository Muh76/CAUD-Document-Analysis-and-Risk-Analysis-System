"""
Enhanced RAG index management with versioning, batch processing, and automated jobs.
"""

import os
import json
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd

from app.config.settings import get_settings

# Settings
settings = get_settings()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGIndexManager:
    """Enhanced RAG index management with production features."""

    def __init__(self):
        self.settings = settings
        self.index_dir = Path(settings.rag_index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.index_dir),
            settings=Settings(anonymized_telemetry=False)
        )

    def create_versioned_index(self, data_source: str, version: Optional[str] = None) -> str:
        """Create a new versioned index with timestamp."""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        index_name = f"{settings.rag_collection}_{version}"

        try:
            # Create collection
            collection = self.chroma_client.create_collection(
                name=index_name,
                metadata={"version": version, "created_at": datetime.now().isoformat()}
            )

            # Load and process data
            documents = self._load_documents(data_source)

            # Batch process documents
            self._batch_add_documents(collection, documents)

            # Update manifest
            self._update_manifest(index_name, version, len(documents))

            logger.info(f"Created index {index_name} with {len(documents)} documents")
            return index_name

        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            raise

    def _load_documents(self, data_source: str) -> List[Dict[str, Any]]:
        """Load documents from data source."""
        documents = []

        if data_source.endswith('.json'):
            with open(data_source, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    documents = data
                else:
                    documents = [data]
        elif data_source.endswith('.csv'):
            df = pd.read_csv(data_source)
            documents = df.to_dict('records')
        else:
            # Default: load from directory
            data_path = Path(data_source)
            if data_path.is_dir():
                for file_path in data_path.glob("*.json"):
                    with open(file_path, 'r') as f:
                        docs = json.load(f)
                        if isinstance(docs, list):
                            documents.extend(docs)
                        else:
                            documents.append(docs)

        return documents

    def _batch_add_documents(self, collection, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Add documents to collection in batches."""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Prepare batch data
            texts = [doc.get('text', str(doc)) for doc in batch]
            metadatas = [doc.get('metadata', {}) for doc in batch]
            ids = [f"doc_{i + j}" for j in range(len(batch))]

            # Add to collection
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")

    def _update_manifest(self, index_name: str, version: str, doc_count: int):
        """Update the index manifest."""
        manifest_path = self.index_dir / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {"indices": [], "latest": None}

        # Add new index entry
        index_entry = {
            "name": index_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "document_count": doc_count,
            "status": "active"
        }

        manifest["indices"].append(index_entry)
        manifest["latest"] = index_name

        # Save manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Updated manifest with index {index_name}")

    def get_latest_index(self) -> Optional[str]:
        """Get the latest active index name."""
        manifest_path = self.index_dir / "manifest.json"

        if not manifest_path.exists():
            return None

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        return manifest.get("latest")

    def list_indices(self) -> List[Dict[str, Any]]:
        """List all available indices."""
        manifest_path = self.index_dir / "manifest.json"

        if not manifest_path.exists():
            return []

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        return manifest.get("indices", [])

    def cleanup_old_indices(self, keep_count: int = 5):
        """Clean up old indices, keeping only the most recent ones."""
        indices = self.list_indices()

        if len(indices) <= keep_count:
            return

        # Sort by creation date (newest first)
        indices.sort(key=lambda x: x["created_at"], reverse=True)

        # Keep the most recent ones
        indices_to_keep = indices[:keep_count]
        indices_to_remove = indices[keep_count:]

        # Remove old indices
        for index_info in indices_to_remove:
            try:
                self.chroma_client.delete_collection(index_info["name"])
                logger.info(f"Removed old index: {index_info['name']}")
            except Exception as e:
                logger.error(f"Failed to remove index {index_info['name']}: {e}")

        # Update manifest
        manifest_path = self.index_dir / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        manifest["indices"] = indices_to_keep
        if indices_to_keep:
            manifest["latest"] = indices_to_keep[0]["name"]
        else:
            manifest["latest"] = None

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Cleaned up {len(indices_to_remove)} old indices")

    def rebuild_index(self, data_source: str) -> str:
        """Rebuild the index with new data."""
        logger.info(f"Starting index rebuild from {data_source}")

        # Create new versioned index
        new_index_name = self.create_versioned_index(data_source)

        # Cleanup old indices
        self.cleanup_old_indices()

        logger.info(f"Index rebuild completed: {new_index_name}")
        return new_index_name

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        latest_index = self.get_latest_index()

        if not latest_index:
            return {"status": "no_index", "message": "No index available"}

        try:
            collection = self.chroma_client.get_collection(latest_index)
            count = collection.count()

            return {
                "status": "active",
                "index_name": latest_index,
                "document_count": count,
                "collection_metadata": collection.metadata
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Global instance
index_manager = RAGIndexManager()
