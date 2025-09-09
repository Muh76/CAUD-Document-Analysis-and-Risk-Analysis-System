"""
RAG job runner script for manual execution and testing.
"""

import argparse
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config.settings import get_settings
from app.rag.index_manager import index_manager
from app.jobs.rag_jobs import rag_jobs

def main():
    """Main job runner function."""
    parser = argparse.ArgumentParser(description="RAG Job Runner")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the RAG index")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old indices")
    parser.add_argument("--health-check", action="store_true", help="Run health check")
    parser.add_argument("--list-indices", action="store_true", help="List all indices")
    parser.add_argument("--data-source", default="data/processed", help="Data source for index rebuild")
    parser.add_argument("--schedule", action="store_true", help="Start job scheduler")

    args = parser.parse_args()

    if args.rebuild_index:
        print("🔄 Rebuilding RAG index...")
        try:
            new_index = index_manager.rebuild_index(args.data_source)
            print(f"✅ Index rebuilt successfully: {new_index}")
        except Exception as e:
            print(f"❌ Index rebuild failed: {e}")
            sys.exit(1)

    elif args.cleanup:
        print("🧹 Cleaning up old indices...")
        try:
            index_manager.cleanup_old_indices()
            print("✅ Cleanup completed successfully")
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            sys.exit(1)

    elif args.health_check:
        print("🏥 Running health check...")
        try:
            rag_jobs.health_check()
            stats = index_manager.get_index_stats()
            print(f"✅ Health check completed - Status: {stats['status']}")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            sys.exit(1)

    elif args.list_indices:
        print("📋 Listing all indices...")
        try:
            indices = index_manager.list_indices()
            if not indices:
                print("No indices found")
            else:
                for idx in indices:
                    print(f"  - {idx['name']} (v{idx['version']}) - {idx['document_count']} docs")
        except Exception as e:
            print(f"❌ Failed to list indices: {e}")
            sys.exit(1)

    elif args.schedule:
        print("⏰ Starting job scheduler...")
        try:
            rag_jobs.schedule_jobs()
            rag_jobs.run_scheduler()
            print("✅ Job scheduler started")
            print("Press Ctrl+C to stop...")
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Job scheduler stopped")
        except Exception as e:
            print(f"❌ Job scheduler failed: {e}")
            sys.exit(1)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
