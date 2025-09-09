"""
RAG retrieval system for finding similar safe clauses.
"""

import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGRetrieval:
    """Retrieve similar safe clauses using ChromaDB."""

    def __init__(self, index_dir: str = "rag/index", embedding_model: str = "all-MiniLM-L6-v2"):
        self.index_dir = Path(index_dir)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.collection_name = "safe_clauses"

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.index_dir))

        # Load metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load index metadata."""
        metadata_path = self.index_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def search_similar_clauses(self, query_text: str, top_k: int = 5, 
                              category_filter: Optional[str] = None,
                              risk_level_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar safe clauses."""
        try:
            collection = self.client.get_collection(self.collection_name)

            # Prepare where clause for filtering
            where_clause = {}
            if category_filter:
                where_clause["category"] = category_filter
            if risk_level_filter:
                where_clause["risk_level"] = risk_level_filter

            # Search
            results = collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where_clause if where_clause else None
            )

            # Format results
            similar_clauses = []
            for i in range(len(results['documents'][0])):
                similar_clauses.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    "similarity": 1 - results['distances'][0][i]  # Convert distance to similarity
                })

            return similar_clauses

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    def get_clause_suggestions(self, clause_text: str, detected_labels: List[str]) -> List[Dict[str, Any]]:
        """Get suggestions for improving a clause."""
        suggestions = []

        # Search for each detected label
        for label in detected_labels:
            similar_clauses = self.search_similar_clauses(
                f"{label}: {clause_text}",
                top_k=3,
                risk_level_filter="low"  # Prefer low-risk examples
            )

            for clause in similar_clauses:
                suggestions.append({
                    "label": label,
                    "suggestion_type": "safe_example",
                    "original_clause": clause_text,
                    "suggested_clause": clause["metadata"]["original_text"],
                    "similarity": clause["similarity"],
                    "notes": clause["metadata"]["notes"],
                    "risk_level": clause["metadata"]["risk_level"]
                })

        # Sort by similarity
        suggestions.sort(key=lambda x: x["similarity"], reverse=True)

        return suggestions[:5]  # Return top 5 suggestions

    def get_missing_clause_suggestions(self, contract_text: str, 
                                     required_labels: List[str]) -> List[Dict[str, Any]]:
        """Get suggestions for missing clauses."""
        suggestions = []

        for label in required_labels:
            # Search for safe examples of this label
            similar_clauses = self.search_similar_clauses(
                label,
                top_k=2,
                risk_level_filter="low"
            )

            for clause in similar_clauses:
                suggestions.append({
                    "label": label,
                    "suggestion_type": "missing_clause",
                    "suggested_clause": clause["metadata"]["original_text"],
                    "notes": clause["metadata"]["notes"],
                    "risk_level": clause["metadata"]["risk_level"],
                    "category": clause["metadata"]["category"]
                })

        return suggestions

    def get_risk_mitigation_suggestions(self, high_risk_clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get suggestions for mitigating high-risk clauses."""
        suggestions = []

        for clause in high_risk_clauses:
            clause_text = clause.get("text", "")
            detected_labels = clause.get("detected_labels", [])

            # Get suggestions for each high-risk label
            for label in detected_labels:
                similar_clauses = self.search_similar_clauses(
                    f"{label}: {clause_text}",
                    top_k=2,
                    risk_level_filter="low"
                )

                for safe_clause in similar_clauses:
                    suggestions.append({
                        "label": label,
                        "suggestion_type": "risk_mitigation",
                        "original_clause": clause_text,
                        "suggested_clause": safe_clause["metadata"]["original_text"],
                        "risk_reduction": "high",
                        "notes": safe_clause["metadata"]["notes"],
                        "similarity": safe_clause["similarity"]
                    })

        return suggestions

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()

            # Get all metadata to analyze
            all_data = collection.get()
            categories = {}
            risk_levels = {}

            for metadata in all_data['metadatas']:
                category = metadata.get('category', 'general')
                risk_level = metadata.get('risk_level', 'low')

                categories[category] = categories.get(category, 0) + 1
                risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1

            return {
                "total_clauses": count,
                "categories": categories,
                "risk_levels": risk_levels,
                "embedding_model": "all-MiniLM-L6-v2",
                "index_path": str(self.index_dir)
            }
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    # Test retrieval
    retrieval = RAGRetrieval()

    # Get stats
    stats = retrieval.get_collection_stats()
    print(f"📊 Collection Stats:")
    print(f"  Total clauses: {stats.get('total_clauses', 'Unknown')}")
    print(f"  Categories: {stats.get('categories', {})}")
    print(f"  Risk levels: {stats.get('risk_levels', {})}")

    # Test search
    test_query = "liability cap limitation"
    results = retrieval.search_similar_clauses(test_query, top_k=3)
    print(f"\n🔍 Search results for '{test_query}':")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['metadata']['label']} (similarity: {result['similarity']:.3f})")
