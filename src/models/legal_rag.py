"""
Legal RAG System with OpenAI Integration
"""

import openai
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
import os

from src.config.config import OPENAI_API_KEY


class LegalRAGSystem:
    """Retrieval-Augmented Generation system for legal contracts"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

        # Initialize sentence transformer for embeddings
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize ChromaDB (local vector database)
        self.vector_db = chromadb.Client()
        self.collection = self.vector_db.create_collection("legal_clauses")

        # Mock precedent database for demo
        self._initialize_precedent_database()

    def _initialize_precedent_database(self):
        """Initialize with sample precedent clauses"""
        sample_clauses = [
            {
                "text": "Party A shall be liable for all damages up to $100,000",
                "clause_type": "liability",
                "outcome": "favorable",
                "risk_level": "low",
            },
            {
                "text": "Liability shall be limited to direct damages only",
                "clause_type": "liability",
                "outcome": "favorable",
                "risk_level": "low",
            },
            {
                "text": "Party A's liability is capped at the contract value",
                "clause_type": "liability",
                "outcome": "favorable",
                "risk_level": "medium",
            },
            {
                "text": "Either party may terminate this agreement with 30 days written notice",
                "clause_type": "termination",
                "outcome": "favorable",
                "risk_level": "low",
            },
            {
                "text": "This agreement shall be governed by the laws of California",
                "clause_type": "governing_law",
                "outcome": "neutral",
                "risk_level": "low",
            },
            {
                "text": "All intellectual property shall be assigned to Company",
                "clause_type": "ip_assignment",
                "outcome": "unfavorable",
                "risk_level": "high",
            },
            {
                "text": "Confidential information shall be kept secret for 5 years",
                "clause_type": "confidentiality",
                "outcome": "favorable",
                "risk_level": "medium",
            },
        ]

        # Add sample clauses to vector database
        for clause in sample_clauses:
            self.add_precedent(clause["text"], clause["clause_type"], clause["outcome"])

    def add_precedent(self, contract_text: str, clause_type: str, outcome: str):
        """
        Add a precedent contract to the vector database

        Args:
            contract_text: Text of the clause
            clause_type: Type of clause
            outcome: Outcome (favorable/unfavorable/neutral)
        """
        try:
            # Generate embedding
            embedding = self.embedder.encode(contract_text)

            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[contract_text],
                metadatas=[
                    {
                        "clause_type": clause_type,
                        "outcome": outcome,
                        "source": "precedent_database",
                    }
                ],
            )

            self.logger.info(f"Added precedent clause: {clause_type}")

        except Exception as e:
            self.logger.error(f"Error adding precedent: {e}")

    def find_similar_clauses(
        self, query_clause: str, clause_type: str, top_k: int = 5
    ) -> List[Dict]:
        """
        Find similar clauses using vector similarity

        Args:
            query_clause: Query clause text
            clause_type: Type of clause to search for
            top_k: Number of similar clauses to return

        Returns:
            List of similar clauses with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode(query_clause)

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where={"clause_type": clause_type},
            )

            # Format results
            similar_clauses = []
            for i in range(len(results["documents"][0])):
                similar_clauses.append(
                    {
                        "text": results["documents"][0][i],
                        "similarity": results["distances"][0][i],
                        "outcome": results["metadatas"][0][i]["outcome"],
                        "source": results["metadatas"][0][i]["source"],
                    }
                )

            return similar_clauses

        except Exception as e:
            self.logger.error(f"Error finding similar clauses: {e}")
            return []

    def suggest_alternative_wording(
        self, risky_clause: str, clause_type: str
    ) -> List[str]:
        """
        Suggest alternative wording using OpenAI

        Args:
            risky_clause: Original risky clause
            clause_type: Type of clause

        Returns:
            List of alternative suggestions
        """
        try:
            # Find similar clauses first
            similar_clauses = self.find_similar_clauses(
                risky_clause, clause_type, top_k=3
            )

            # Build context from similar clauses
            context = ""
            for clause in similar_clauses:
                context += f"- {clause['text']} (Outcome: {clause['outcome']})\n"

            # Create prompt for OpenAI
            prompt = f"""
            You are a legal expert specializing in contract review and risk mitigation.
            
            Analyze this potentially risky clause:
            "{risky_clause}"
            
            Clause type: {clause_type}
            
            Here are similar clauses with better outcomes:
            {context}
            
            Provide 3 alternative wordings that:
            1. Maintain the original intent
            2. Reduce legal risk
            3. Are more favorable to the client
            4. Follow legal best practices
            
            Format your response as a numbered list of suggestions.
            """

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal expert specializing in contract review and risk mitigation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.3,
            )

            # Parse suggestions
            suggestions_text = response.choices[0].message.content
            suggestions = []

            # Extract numbered suggestions
            lines = suggestions_text.split("\n")
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Remove numbering and clean up
                    suggestion = line.lstrip("0123456789.- ").strip()
                    if suggestion:
                        suggestions.append(suggestion)

            return suggestions[:3]  # Return top 3 suggestions

        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            # Fallback suggestions
            return [
                f"Consider modifying the {clause_type} clause to include reasonable limitations",
                f"Add specific conditions and exceptions to the {clause_type} clause",
                f"Review the {clause_type} clause with legal counsel for risk assessment",
            ]

    def analyze_clause_risk(self, clause_text: str, clause_type: str) -> Dict:
        """
        Analyze risk of a specific clause using OpenAI

        Args:
            clause_text: Text of the clause
            clause_type: Type of clause

        Returns:
            Risk analysis with score and explanation
        """
        try:
            prompt = f"""
            Analyze the risk level of this {clause_type} clause:
            "{clause_text}"
            
            Provide a risk analysis with:
            1. Risk level (low/medium/high/critical)
            2. Risk score (0-1)
            3. Key risk factors
            4. Potential issues
            5. Recommendations
            
            Format as JSON with keys: risk_level, risk_score, risk_factors, issues, recommendations
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal risk analyst. Respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.1,
            )

            # Parse JSON response
            import json

            analysis_text = response.choices[0].message.content
            analysis = json.loads(analysis_text)

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing clause risk: {e}")
            # Fallback analysis
            return {
                "risk_level": "medium",
                "risk_score": 0.5,
                "risk_factors": ["Unable to analyze"],
                "issues": ["Analysis failed"],
                "recommendations": ["Review with legal counsel"],
            }

    def get_precedent_statistics(self) -> Dict:
        """Get statistics about the precedent database"""
        try:
            # Get collection info
            collection_info = self.collection.get()

            # Count by clause type
            clause_types = {}
            outcomes = {}

            for metadata in collection_info["metadatas"]:
                clause_type = metadata["clause_type"]
                outcome = metadata["outcome"]

                clause_types[clause_type] = clause_types.get(clause_type, 0) + 1
                outcomes[outcome] = outcomes.get(outcome, 0) + 1

            return {
                "total_clauses": len(collection_info["metadatas"]),
                "clause_types": clause_types,
                "outcomes": outcomes,
            }

        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {"total_clauses": 0, "clause_types": {}, "outcomes": {}}
