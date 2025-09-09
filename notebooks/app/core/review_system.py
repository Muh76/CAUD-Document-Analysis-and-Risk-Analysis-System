"""
Human-in-the-loop review system for contract analysis.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import logging

class ReviewStatus(Enum):
    """Review status enumeration."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"

class Priority(Enum):
    """Priority levels for reviews."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ReviewItem:
    """Individual review item."""
    id: str
    contract_id: str
    clause_text: str
    ai_prediction: Dict[str, Any]
    confidence_score: float
    priority: Priority
    status: ReviewStatus
    created_at: datetime
    assigned_to: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    reviewer_notes: Optional[str] = None
    human_prediction: Optional[Dict[str, Any]] = None
    feedback_score: Optional[int] = None  # 1-5 scale

@dataclass
class Reviewer:
    """Reviewer information."""
    id: str
    name: str
    email: str
    expertise_areas: List[str]
    max_reviews_per_day: int
    current_reviews: int = 0
    total_reviews: int = 0
    accuracy_score: float = 0.0

class ReviewQueue:
    """Manages the review queue and assignments."""

    def __init__(self, storage_path: str = "app/var/reviews"):
        self.storage_path = storage_path
        self.reviewers: Dict[str, Reviewer] = {}
        self.pending_reviews: List[ReviewItem] = []
        self.completed_reviews: List[ReviewItem] = []
        self.load_data()

    def add_review_item(self, contract_id: str, clause_text: str, 
                       ai_prediction: Dict[str, Any], confidence_score: float,
                       priority: Priority = Priority.MEDIUM) -> str:
        """Add a new item to the review queue."""
        review_id = str(uuid.uuid4())

        review_item = ReviewItem(
            id=review_id,
            contract_id=contract_id,
            clause_text=clause_text,
            ai_prediction=ai_prediction,
            confidence_score=confidence_score,
            priority=priority,
            status=ReviewStatus.PENDING,
            created_at=datetime.now()
        )

        self.pending_reviews.append(review_item)
        self.save_data()

        logging.info(f"Added review item {review_id} with priority {priority.value}")
        return review_id

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "total_pending": len(self.pending_reviews),
            "total_completed": len(self.completed_reviews),
            "by_priority": {
                priority.value: len([r for r in self.pending_reviews 
                                   if r.priority == priority])
                for priority in Priority
            },
            "by_status": {
                status.value: len([r for r in self.pending_reviews 
                                if r.status == status])
                for status in ReviewStatus
            }
        }

    def save_data(self):
        """Save data to storage."""
        import os
        os.makedirs(self.storage_path, exist_ok=True)

        # Save pending reviews
        with open(f"{self.storage_path}/pending_reviews.json", "w") as f:
            json.dump([asdict(item) for item in self.pending_reviews], 
                     f, default=str, indent=2)

        # Save completed reviews
        with open(f"{self.storage_path}/completed_reviews.json", "w") as f:
            json.dump([asdict(item) for item in self.completed_reviews], 
                     f, default=str, indent=2)

    def load_data(self):
        """Load data from storage."""
        import os

        # Load pending reviews
        if os.path.exists(f"{self.storage_path}/pending_reviews.json"):
            with open(f"{self.storage_path}/pending_reviews.json", "r") as f:
                data = json.load(f)
                self.pending_reviews = [
                    ReviewItem(**item) for item in data
                ]

        # Load completed reviews
        if os.path.exists(f"{self.storage_path}/completed_reviews.json"):
            with open(f"{self.storage_path}/completed_reviews.json", "r") as f:
                data = json.load(f)
                self.completed_reviews = [
                    ReviewItem(**item) for item in data
                ]

class ActiveLearningSystem:
    """Active learning system for continuous improvement."""

    def __init__(self, review_queue: ReviewQueue):
        self.review_queue = review_queue
        self.learning_data: List[Dict[str, Any]] = []

    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in human feedback."""
        if not self.review_queue.completed_reviews:
            return {"message": "No completed reviews available for analysis"}

        total_reviews = len(self.review_queue.completed_reviews)
        return {
            "total_reviews": total_reviews,
            "message": "Analysis ready for completed reviews"
        }

    def export_learning_data(self) -> Dict[str, Any]:
        """Export data for model retraining."""
        return {
            "completed_reviews": [asdict(review) for review in self.review_queue.completed_reviews],
            "feedback_analysis": self.analyze_feedback_patterns(),
            "export_timestamp": datetime.now().isoformat()
        }

# Global instances
review_queue = ReviewQueue()
active_learning = ActiveLearningSystem(review_queue)

def get_review_queue() -> ReviewQueue:
    """Get the global review queue instance."""
    return review_queue

def get_active_learning() -> ActiveLearningSystem:
    """Get the global active learning instance."""
    return active_learning
