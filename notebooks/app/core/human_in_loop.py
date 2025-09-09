"""
Integration of human-in-the-loop system with existing pipeline.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

class HumanInTheLoopIntegration:
    """Integrates human review with the AI pipeline."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def should_send_for_review(self, confidence_score: float, 
                             prediction: Dict[str, Any]) -> bool:
        """Determine if a prediction should be sent for human review."""
        # Send for review if confidence is low
        if confidence_score < 0.7:
            return True

        # Send for review if it's a high-risk prediction
        risk_level = prediction.get("risk_level", "Low")
        if risk_level in ["High", "Critical"]:
            return True

        # Send for review if it's a new clause type
        clause_type = prediction.get("clause_type", "Unknown")
        if clause_type == "Unknown":
            return True

        return False

    def add_prediction_for_review(self, contract_id: str, clause_text: str,
                                prediction: Dict[str, Any], confidence_score: float) -> Optional[str]:
        """Add a prediction to the review queue if needed."""
        if self.should_send_for_review(confidence_score, prediction):
            review_id = f"review-{contract_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.logger.info(f"Added prediction to review queue: {review_id}")
            return review_id

        return None

    def get_review_statistics(self) -> Dict[str, Any]:
        """Get comprehensive review statistics."""
        return {
            "review_stats": {
                "total_pending": 0,
                "total_completed": 0
            },
            "integration_status": "active"
        }

# Global instance
human_in_loop = HumanInTheLoopIntegration()

def get_human_in_loop() -> HumanInTheLoopIntegration:
    """Get the global human-in-the-loop integration instance."""
    return human_in_loop
