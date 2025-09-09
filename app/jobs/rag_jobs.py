"""
Automated RAG jobs for production data pipeline operations.
"""

import os
import json
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import threading

from app.config.settings import get_settings
from app.rag.index_manager import index_manager

# Settings
settings = get_settings()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGJobs:
    """Automated RAG jobs for production operations."""

    def __init__(self):
        self.settings = settings
        self.jobs_dir = Path("app/jobs")
        self.jobs_dir.mkdir(exist_ok=True)

        # Job status tracking
        self.job_status_file = self.jobs_dir / "job_status.json"
        self._load_job_status()

    def _load_job_status(self):
        """Load job status from file."""
        if self.job_status_file.exists():
            with open(self.job_status_file, 'r') as f:
                self.job_status = json.load(f)
        else:
            self.job_status = {
                "last_index_rebuild": None,
                "last_cleanup": None,
                "failed_jobs": [],
                "job_history": []
            }

    def _save_job_status(self):
        """Save job status to file."""
        with open(self.job_status_file, 'w') as f:
            json.dump(self.job_status, f, indent=2)

    def nightly_index_rebuild(self, data_source: str = "data/processed"):
        """Nightly job to rebuild the RAG index."""
        job_name = "nightly_index_rebuild"
        start_time = datetime.now()

        try:
            logger.info(f"Starting {job_name} at {start_time}")

            # Check if data source exists
            if not Path(data_source).exists():
                logger.warning(f"Data source {data_source} not found, skipping rebuild")
                return

            # Rebuild index
            new_index_name = index_manager.rebuild_index(data_source)

            # Update job status
            self.job_status["last_index_rebuild"] = {
                "timestamp": start_time.isoformat(),
                "index_name": new_index_name,
                "status": "success"
            }

            # Add to job history
            self.job_status["job_history"].append({
                "job_name": job_name,
                "timestamp": start_time.isoformat(),
                "status": "success",
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "details": f"Created index: {new_index_name}"
            })

            logger.info(f"Completed {job_name} successfully")

        except Exception as e:
            logger.error(f"Failed {job_name}: {e}")

            # Update job status
            self.job_status["last_index_rebuild"] = {
                "timestamp": start_time.isoformat(),
                "status": "failed",
                "error": str(e)
            }

            # Add to failed jobs
            self.job_status["failed_jobs"].append({
                "job_name": job_name,
                "timestamp": start_time.isoformat(),
                "error": str(e)
            })

            # Add to job history
            self.job_status["job_history"].append({
                "job_name": job_name,
                "timestamp": start_time.isoformat(),
                "status": "failed",
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "error": str(e)
            })

        finally:
            self._save_job_status()

    def weekly_cleanup(self):
        """Weekly job to clean up old indices and logs."""
        job_name = "weekly_cleanup"
        start_time = datetime.now()

        try:
            logger.info(f"Starting {job_name} at {start_time}")

            # Cleanup old indices
            index_manager.cleanup_old_indices(keep_count=5)

            # Cleanup old job history (keep last 100 entries)
            if len(self.job_status["job_history"]) > 100:
                self.job_status["job_history"] = self.job_status["job_history"][-100:]

            # Cleanup old failed jobs (keep last 50 entries)
            if len(self.job_status["failed_jobs"]) > 50:
                self.job_status["failed_jobs"] = self.job_status["failed_jobs"][-50:]

            # Update job status
            self.job_status["last_cleanup"] = {
                "timestamp": start_time.isoformat(),
                "status": "success"
            }

            # Add to job history
            self.job_status["job_history"].append({
                "job_name": job_name,
                "timestamp": start_time.isoformat(),
                "status": "success",
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "details": "Cleaned up old indices and job history"
            })

            logger.info(f"Completed {job_name} successfully")

        except Exception as e:
            logger.error(f"Failed {job_name}: {e}")

            # Update job status
            self.job_status["last_cleanup"] = {
                "timestamp": start_time.isoformat(),
                "status": "failed",
                "error": str(e)
            }

            # Add to failed jobs
            self.job_status["failed_jobs"].append({
                "job_name": job_name,
                "timestamp": start_time.isoformat(),
                "error": str(e)
            })

            # Add to job history
            self.job_status["job_history"].append({
                "job_name": job_name,
                "timestamp": start_time.isoformat(),
                "status": "failed",
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "error": str(e)
            })

        finally:
            self._save_job_status()

    def health_check(self):
        """Health check job to monitor system status."""
        job_name = "health_check"
        start_time = datetime.now()

        try:
            logger.info(f"Starting {job_name} at {start_time}")

            # Check index status
            index_stats = index_manager.get_index_stats()

            # Check disk space
            disk_usage = self._get_disk_usage()

            # Check for failed jobs
            recent_failures = self._get_recent_failures()

            health_status = {
                "timestamp": start_time.isoformat(),
                "index_status": index_stats,
                "disk_usage": disk_usage,
                "recent_failures": recent_failures,
                "overall_status": "healthy" if index_stats["status"] == "active" else "unhealthy"
            }

            # Save health check results
            health_file = self.jobs_dir / f"health_check_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(health_file, 'w') as f:
                json.dump(health_status, f, indent=2)

            logger.info(f"Completed {job_name} - Status: {health_status['overall_status']}")

        except Exception as e:
            logger.error(f"Failed {job_name}: {e}")

    def _get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage information."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(settings.rag_index_dir)
            return {
                "total_gb": total // (1024**3),
                "used_gb": used // (1024**3),
                "free_gb": free // (1024**3),
                "usage_percent": (used / total) * 100
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_recent_failures(self) -> List[Dict[str, Any]]:
        """Get recent job failures."""
        recent_failures = []
        cutoff_time = datetime.now() - timedelta(hours=24)

        for failure in self.job_status["failed_jobs"]:
            failure_time = datetime.fromisoformat(failure["timestamp"])
            if failure_time > cutoff_time:
                recent_failures.append(failure)

        return recent_failures

    def schedule_jobs(self):
        """Schedule all automated jobs."""
        # Schedule nightly index rebuild at 2 AM
        schedule.every().day.at("02:00").do(self.nightly_index_rebuild)

        # Schedule weekly cleanup on Sundays at 3 AM
        schedule.every().sunday.at("03:00").do(self.weekly_cleanup)

        # Schedule health check every 6 hours
        schedule.every(6).hours.do(self.health_check)

        logger.info("Scheduled all RAG jobs")

    def run_scheduler(self):
        """Run the job scheduler in a separate thread."""
        def scheduler_loop():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        scheduler_thread.start()
        logger.info("Started RAG job scheduler")

# Global instance
rag_jobs = RAGJobs()
