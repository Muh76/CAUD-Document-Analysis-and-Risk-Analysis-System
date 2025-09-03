"""
Enhanced Legal RAG System with OpenAI and Azure Integration
"""

import openai
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
import os

from src.config.config import (
    OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
)


class EnhancedLegalRAGSystem:
    """Enhanced RAG system with OpenAI and Azure integration"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

        # Initialize Azure OpenAI client (if available)
        self.azure_client = None
        if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
            try:
                self.azure_client = openai.AzureOpenAI(
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    api_key=AZURE_OPENAI_API_KEY,
                    api_version="2024-02-15-preview",
                )
                self.logger.info("Azure OpenAI client initialized successfully")
            except Exception as e:
                self.logger.warning(f"Azure OpenAI initialization failed: {e}")

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
        """Add a precedent contract to the vector database"""
        try:
            embedding = self.embedder.encode(contract_text)
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
        """Find similar clauses using vector similarity"""
        try:
            query_embedding = self.embedder.encode(query_clause)
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where={"clause_type": clause_type},
            )

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
        self, risky_clause: str, clause_type: str, use_azure: bool = False
    ) -> List[str]:
        """Suggest alternative wording using OpenAI or Azure"""
        try:
            # Find similar clauses first
            similar_clauses = self.find_similar_clauses(
                risky_clause, clause_type, top_k=3
            )

            # Build context from similar clauses
            context = ""
            for clause in similar_clauses:
                context += f"- {clause['text']} (Outcome: {clause['outcome']})\n"

            # Create prompt
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

            # Choose client based on preference
            if use_azure and self.azure_client:
                client = self.azure_client
                model = "gpt-4"  # Azure deployment name
                self.logger.info("Using Azure OpenAI for suggestions")
            else:
                client = self.openai_client
                model = "gpt-4"
                self.logger.info("Using OpenAI for suggestions")

            # Call API
            response = client.chat.completions.create(
                model=model,
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

            lines = suggestions_text.split("\n")
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    suggestion = line.lstrip("0123456789.- ").strip()
                    if suggestion:
                        suggestions.append(suggestion)

            return suggestions[:3]

        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            return [
                f"Consider modifying the {clause_type} clause to include reasonable limitations",
                f"Add specific conditions and exceptions to the {clause_type} clause",
                f"Review the {clause_type} clause with legal counsel for risk assessment",
            ]

    def analyze_clause_risk(
        self, clause_text: str, clause_type: str, use_azure: bool = False
    ) -> Dict:
        """Analyze risk of a specific clause using OpenAI or Azure"""
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

            # Choose client based on preference
            if use_azure and self.azure_client:
                client = self.azure_client
                model = "gpt-4"
                self.logger.info("Using Azure OpenAI for risk analysis")
            else:
                client = self.openai_client
                model = "gpt-4"
                self.logger.info("Using OpenAI for risk analysis")

            response = client.chat.completions.create(
                model=model,
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

            import json

            analysis_text = response.choices[0].message.content
            analysis = json.loads(analysis_text)

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing clause risk: {e}")
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
            collection_info = self.collection.get()

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

    def get_available_models(self) -> Dict:
        """Get available models and their status"""
        models = {
            "openai": {"available": True, "model": "gpt-4", "status": "Ready"},
            "azure": {
                "available": self.azure_client is not None,
                "model": "gpt-4" if self.azure_client else "Not configured",
                "status": "Ready" if self.azure_client else "Not available",
            },
        }
        return models
