"""
Structured logging and telemetry for the Contract Analysis Pipeline.
"""

import logging
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import sys

class StructuredLogger:
    """Structured JSON logging for contract analysis."""

    def __init__(self, log_file: Optional[str] = None, log_level: str = "INFO"):
        self.logger = logging.getLogger("contract_analysis")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create formatter
        formatter = logging.Formatter('%(message)s')

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _log(self, level: str, message: str, **kwargs):
        """Log structured message."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logger.log(getattr(logging, level.upper()), json.dumps(log_entry))

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log("DEBUG", message, **kwargs)

class RequestLogger:
    """Log API requests with structured data."""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    def log_request(self, request_id: str, method: str, path: str, 
                   client_ip: str, user_agent: str, **kwargs):
        """Log incoming request."""
        self.logger.info(
            "API request received",
            request_id=request_id,
            method=method,
            path=path,
            client_ip=client_ip,
            user_agent=user_agent,
            **kwargs
        )

    def log_response(self, request_id: str, status_code: int, 
                    latency_ms: int, **kwargs):
        """Log API response."""
        self.logger.info(
            "API response sent",
            request_id=request_id,
            status_code=status_code,
            latency_ms=latency_ms,
            **kwargs
        )

    def log_analysis(self, request_id: str, contract_id: str, 
                    total_clauses: int, high_risk_count: int, 
                    latency_ms: int, **kwargs):
        """Log contract analysis."""
        self.logger.info(
            "Contract analysis completed",
            request_id=request_id,
            contract_id=contract_id,
            total_clauses=total_clauses,
            high_risk_count=high_risk_count,
            latency_ms=latency_ms,
            **kwargs
        )

class MetricsCollector:
    """Collect and store metrics in memory."""

    def __init__(self):
        self.counters = {}
        self.histograms = {}
        self.gauges = {}

    def increment_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        key = f"{name}:{labels or ''}"
        self.counters[key] = self.counters.get(key, 0) + value

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value."""
        key = f"{name}:{labels or ''}"
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge value."""
        key = f"{name}:{labels or ''}"
        self.gauges[key] = value

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": self.counters,
            "histograms": self.histograms,
            "gauges": self.gauges,
            "timestamp": datetime.now().isoformat()
        }

# Global instances
logger = StructuredLogger()
request_logger = RequestLogger(logger)
metrics = MetricsCollector()
