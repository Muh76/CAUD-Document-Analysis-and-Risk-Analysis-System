"""
Build ChromaDB index from safe clause exemplars.
"""

import yaml
import chromadb
from pathlib import Path
from typing import List, Dict, Any
import json
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGIndexBuilder:
    """Build and manage RAG index for safe clauses."""

    def __init__(self, index_dir: str = "rag/index", embedding_model: str = "all-MiniLM-L6-v2"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.collection_name = "safe_clauses"

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.index_dir))

    def load_safe_clauses(self, yaml_path: str) -> List[Dict[str, Any]]:
        """Load safe clauses from YAML file."""
        with open(yaml_path, 'r') as f:
            clauses = yaml.safe_load(f)
        return clauses

    def build_index(self, yaml_path: str):
        """Build ChromaDB index from safe clauses."""
        print("🔍 Loading safe clauses...")
        clauses = self.load_safe_clauses(yaml_path)

        print(f"📋 Found {len(clauses)} safe clauses")

        # Create or get collection
        try:
            collection = self.client.get_collection(self.collection_name)
            print("📁 Using existing collection")
        except:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Safe contract clause exemplars"}
            )
            print("📁 Created new collection")

        # Prepare documents and metadata
        documents = []
        metadatas = []
        ids = []

        for i, clause in enumerate(clauses):
            # Create document text
            doc_text = f"{clause['label']}: {clause['text']}"
            if clause.get('notes'):
                doc_text += f" Notes: {clause['notes']}"

            documents.append(doc_text)
            metadatas.append({
                "label": clause["label"],
                "category": clause.get("category", "general"),
                "risk_level": clause.get("risk_level", "low"),
                "original_text": clause["text"],
                "notes": clause.get("notes", "")
            })
            ids.append(f"clause_{i:03d}")

        # Add to collection
        print("📝 Adding clauses to index...")
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✅ Index built successfully with {len(clauses)} clauses")

        # Save metadata
        metadata_path = self.index_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "total_clauses": len(clauses),
                "embedding_model": "all-MiniLM-L6-v2",
                "collection_name": self.collection_name,
                "categories": list(set(c.get("category", "general") for c in clauses)),
                "risk_levels": list(set(c.get("risk_level", "low") for c in clauses))
            }, f, indent=2)

        print(f"📋 Metadata saved to {metadata_path}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()

            # Get sample metadata
            sample = collection.get(limit=1)
            if sample['metadatas']:
                sample_metadata = sample['metadatas'][0]
            else:
                sample_metadata = {}

            return {
                "collection_name": self.collection_name,
                "total_clauses": count,
                "sample_metadata": sample_metadata,
                "index_path": str(self.index_dir)
            }
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    # Build index
    builder = RAGIndexBuilder()
    yaml_path = "rag/safe_clauses.yaml"

    if Path(yaml_path).exists():
        builder.build_index(yaml_path)

        # Show collection info
        info = builder.get_collection_info()
        print(f"\n📊 Collection Info:")
        print(f"  Total clauses: {info.get('total_clauses', 'Unknown')}")
        print(f"  Index path: {info.get('index_path', 'Unknown')}")
    else:
        print(f"❌ YAML file not found: {yaml_path}")
