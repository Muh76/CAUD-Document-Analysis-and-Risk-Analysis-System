"""
Core settings and configuration management for the Contract Analysis Pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any
import json

class Settings:
    """Application settings and configuration."""

    def __init__(self, artifacts_dir: str = None):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else Path("app/artifacts/snapshot_20250909")
        self.model_snapshot = self.artifacts_dir.name
        self.label_map_path = self.artifacts_dir / "label_map.json"
        self.thresholds_path = self.artifacts_dir / "thresholds.json"
        self.models_dir = self.artifacts_dir / "models"

        # Load configuration
        self.label_map = self._load_label_map()
        self.thresholds = self._load_thresholds()

        # Model configuration
        self.num_labels = len(self.label_map)
        self.max_text_length = 4000
        self.chunk_size = 500
        self.chunk_overlap = 50

        # Deterministic settings
        self.random_seed = 42
        self.torch_deterministic = True

    def _load_label_map(self) -> Dict[int, str]:
        """Load label mapping from JSON file."""
        if self.label_map_path.exists():
            with open(self.label_map_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Label map not found: {self.label_map_path}")

    def _load_thresholds(self) -> Dict[str, Any]:
        """Load thresholds from JSON file."""
        if self.thresholds_path.exists():
            with open(self.thresholds_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Thresholds not found: {self.thresholds_path}")

    def get_per_label_threshold(self, label: str) -> float:
        """Get threshold for a specific label."""
        return self.thresholds.get("per_label_thresholds", {}).get(label, 0.5)

    def get_global_threshold(self, threshold_name: str) -> float:
        """Get global threshold value."""
        return self.thresholds.get("global_thresholds", {}).get(threshold_name, 0.5)

    def get_model_path(self, model_name: str) -> Path:
        """Get path to a specific model file."""
        return self.models_dir / model_name
