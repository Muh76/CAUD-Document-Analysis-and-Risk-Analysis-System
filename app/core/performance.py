"""
Performance and cost optimization for Contract Analysis System.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from functools import lru_cache
import threading
from collections import defaultdict

class PerformanceMonitor:
    """Performance monitoring and optimization."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()

    def record_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a performance metric."""
        metric_data = {
            'timestamp': datetime.now().isoformat(),
            'value': value,
            'tags': tags or {}
        }
        self.metrics[metric_name].append(metric_data)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {}

        for metric_name, values in self.metrics.items():
            if values:
                recent_values = [v['value'] for v in values[-100:]]  # Last 100 values
                summary[metric_name] = {
                    'count': len(recent_values),
                    'avg': sum(recent_values) / len(recent_values),
                    'min': min(recent_values),
                    'max': max(recent_values),
                    'latest': recent_values[-1] if recent_values else 0
                }

        return summary

    def get_performance_score(self) -> Dict[str, Any]:
        """Calculate overall performance score."""
        summary = self.get_metrics_summary()

        # Calculate scores based on key metrics
        scores = {}

        if 'api_response_time' in summary:
            avg_response_time = summary['api_response_time']['avg']
            if avg_response_time < 1.0:
                scores['response_time'] = 100
            elif avg_response_time < 2.0:
                scores['response_time'] = 80
            elif avg_response_time < 5.0:
                scores['response_time'] = 60
            else:
                scores['response_time'] = 40

        if 'memory_usage' in summary:
            avg_memory = summary['memory_usage']['avg']
            if avg_memory < 100:  # MB
                scores['memory'] = 100
            elif avg_memory < 500:
                scores['memory'] = 80
            elif avg_memory < 1000:
                scores['memory'] = 60
            else:
                scores['memory'] = 40

        overall_score = sum(scores.values()) / len(scores) if scores else 0

        return {
            'overall_score': overall_score,
            'component_scores': scores,
            'recommendations': self._get_optimization_recommendations(summary)
        }

    def _get_optimization_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Get optimization recommendations."""
        recommendations = []

        if 'api_response_time' in summary:
            avg_response_time = summary['api_response_time']['avg']
            if avg_response_time > 2.0:
                recommendations.append("Consider implementing caching for API responses")
                recommendations.append("Optimize database queries")

        if 'memory_usage' in summary:
            avg_memory = summary['memory_usage']['avg']
            if avg_memory > 500:
                recommendations.append("Implement memory pooling")
                recommendations.append("Consider garbage collection optimization")

        return recommendations

class CacheManager:
    """Intelligent caching system."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_times = {}
        self.logger = logging.getLogger(__name__)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            # Check TTL
            if time.time() - self.access_times[key] > self.ttl_seconds:
                self.delete(key)
                return None

            # Update access time
            self.access_times[key] = time.time()
            return self.cache[key]

        return None

    def set(self, key: str, value: Any) -> bool:
        """Set value in cache."""
        try:
            # Evict if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            self.cache[key] = value
            self.access_times[key] = time.time()
            return True
        except Exception as e:
            self.logger.error(f"Failed to set cache key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete cache key {key}: {e}")
            return False

    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return

        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self.delete(lru_key)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': self._calculate_hit_rate(),
            'memory_usage': self._estimate_memory_usage()
        }

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        # Simplified hit rate calculation
        return 0.85  # Mock value

    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in MB."""
        return len(self.cache) * 0.1  # Mock calculation

class BatchProcessor:
    """Batch processing for cost optimization."""

    def __init__(self, batch_size: int = 10, max_wait_time: int = 5):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.processing_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    async def add_request(self, request_data: Dict[str, Any]) -> str:
        """Add request to batch."""
        request_id = f"req_{int(time.time() * 1000)}"

        with self.processing_lock:
            self.pending_requests.append({
                'id': request_id,
                'data': request_data,
                'timestamp': time.time()
            })

        # Check if batch is ready
        if len(self.pending_requests) >= self.batch_size:
            await self._process_batch()

        return request_id

    async def _process_batch(self):
        """Process pending batch."""
        if not self.pending_requests:
            return

        with self.processing_lock:
            batch = self.pending_requests.copy()
            self.pending_requests.clear()

        # Process batch
        results = await self._execute_batch(batch)

        # Store results
        for result in results:
            self._store_result(result)

    async def _execute_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute batch processing."""
        # Mock batch processing
        results = []
        for request in batch:
            result = {
                'id': request['id'],
                'status': 'completed',
                'result': f"Processed {request['data']}",
                'processing_time': 0.1
            }
            results.append(result)

        return results

    def _store_result(self, result: Dict[str, Any]):
        """Store processing result."""
        # Mock result storage
        self.logger.info(f"Stored result for {result['id']}")

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return {
            'pending_requests': len(self.pending_requests),
            'batch_size': self.batch_size,
            'max_wait_time': self.max_wait_time,
            'efficiency_gain': 0.75  # Mock value
        }

class CostOptimizer:
    """Cost optimization utilities."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cost_tracking = defaultdict(float)

    def track_api_cost(self, endpoint: str, cost: float):
        """Track API endpoint costs."""
        self.cost_tracking[f"api_{endpoint}"] += cost

    def track_model_cost(self, model_name: str, cost: float):
        """Track model inference costs."""
        self.cost_tracking[f"model_{model_name}"] += cost

    def track_storage_cost(self, storage_type: str, cost: float):
        """Track storage costs."""
        self.cost_tracking[f"storage_{storage_type}"] += cost

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        total_cost = sum(self.cost_tracking.values())

        return {
            'total_cost': total_cost,
            'cost_breakdown': dict(self.cost_tracking),
            'cost_per_request': total_cost / max(1, self._get_total_requests()),
            'optimization_opportunities': self._get_optimization_opportunities()
        }

    def _get_total_requests(self) -> int:
        """Get total number of requests."""
        return 1000  # Mock value

    def _get_optimization_opportunities(self) -> List[str]:
        """Get cost optimization opportunities."""
        opportunities = []

        if self.cost_tracking.get('api_analyze_contract', 0) > 100:
            opportunities.append("Consider implementing request batching for contract analysis")

        if self.cost_tracking.get('model_inference', 0) > 200:
            opportunities.append("Implement model caching to reduce inference costs")

        if self.cost_tracking.get('storage_index', 0) > 50:
            opportunities.append("Optimize index storage with compression")

        return opportunities

class ResourceManager:
    """Resource management and optimization."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.resource_usage = defaultdict(list)

    def monitor_cpu_usage(self):
        """Monitor CPU usage."""
        # Mock CPU monitoring
        cpu_usage = 45.2  # Mock value
        self.resource_usage['cpu'].append({
            'timestamp': datetime.now().isoformat(),
            'usage': cpu_usage
        })
        return cpu_usage

    def monitor_memory_usage(self):
        """Monitor memory usage."""
        # Mock memory monitoring
        memory_usage = 256.8  # Mock value in MB
        self.resource_usage['memory'].append({
            'timestamp': datetime.now().isoformat(),
            'usage': memory_usage
        })
        return memory_usage

    def monitor_disk_usage(self):
        """Monitor disk usage."""
        # Mock disk monitoring
        disk_usage = 1024.5  # Mock value in MB
        self.resource_usage['disk'].append({
            'timestamp': datetime.now().isoformat(),
            'usage': disk_usage
        })
        return disk_usage

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        summary = {}

        for resource_type, usage_data in self.resource_usage.items():
            if usage_data:
                recent_usage = [u['usage'] for u in usage_data[-10:]]  # Last 10 readings
                summary[resource_type] = {
                    'current': recent_usage[-1] if recent_usage else 0,
                    'average': sum(recent_usage) / len(recent_usage),
                    'peak': max(recent_usage),
                    'trend': 'stable'  # Mock trend analysis
                }

        return summary

    def get_optimization_recommendations(self) -> List[str]:
        """Get resource optimization recommendations."""
        recommendations = []
        summary = self.get_resource_summary()

        if 'cpu' in summary and summary['cpu']['average'] > 80:
            recommendations.append("High CPU usage detected - consider scaling")

        if 'memory' in summary and summary['memory']['average'] > 500:
            recommendations.append("High memory usage - implement memory pooling")

        if 'disk' in summary and summary['disk']['average'] > 2000:
            recommendations.append("High disk usage - implement cleanup procedures")

        return recommendations

# Initialize performance components
performance_monitor = PerformanceMonitor()
cache_manager = CacheManager()
batch_processor = BatchProcessor()
cost_optimizer = CostOptimizer()
resource_manager = ResourceManager()

# Performance optimization manager
class PerformanceOptimizer:
    """Centralized performance optimization management."""

    def __init__(self):
        self.monitor = performance_monitor
        self.cache = cache_manager
        self.batch_processor = batch_processor
        self.cost_optimizer = cost_optimizer
        self.resource_manager = resource_manager
        self.logger = logging.getLogger(__name__)

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'performance_score': self.monitor.get_performance_score(),
            'cache_stats': self.cache.get_cache_stats(),
            'batch_stats': self.batch_processor.get_batch_stats(),
            'cost_summary': self.cost_optimizer.get_cost_summary(),
            'resource_summary': self.resource_manager.get_resource_summary(),
            'recommendations': self._get_combined_recommendations()
        }

    def _get_combined_recommendations(self) -> List[str]:
        """Get combined optimization recommendations."""
        recommendations = []

        # Performance recommendations
        perf_score = self.monitor.get_performance_score()
        if perf_score['overall_score'] < 80:
            recommendations.extend(perf_score['recommendations'])

        # Cost recommendations
        cost_summary = self.cost_optimizer.get_cost_summary()
        recommendations.extend(cost_summary['optimization_opportunities'])

        # Resource recommendations
        recommendations.extend(self.resource_manager.get_optimization_recommendations())

        return list(set(recommendations))  # Remove duplicates

# Initialize performance optimizer
performance_optimizer = PerformanceOptimizer()
