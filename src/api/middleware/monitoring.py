"""
Enterprise Monitoring Middleware

Comprehensive monitoring and observability middleware with:
- Real-time metrics collection and aggregation
- Performance monitoring and profiling
- Health checks and system status
- Error tracking and alerting  
- Custom metrics and business KPIs
- Prometheus-compatible metrics export
- Distributed tracing support
"""

import time
import asyncio
import psutil
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import structlog

from ...core.config import settings


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"


class HealthStatus(str, Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    request_count: int = 0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    queue_size: int = 0


@dataclass 
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Centralized metrics collection and aggregation."""
    
    def __init__(self):
        # Prometheus metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code', 'user_id']
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.active_requests = Gauge(
            'http_requests_active',
            'Currently active HTTP requests'
        )
        
        self.error_count = Counter(
            'http_errors_total', 
            'Total HTTP errors',
            ['method', 'endpoint', 'error_type']
        )
        
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage'
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes', 
            'System memory usage in bytes'
        )
        
        self.rag_query_duration = Histogram(
            'rag_query_duration_seconds',
            'RAG query processing duration',
            ['provider', 'model']
        )
        
        self.rag_query_count = Counter(
            'rag_queries_total',
            'Total RAG queries processed',
            ['provider', 'model', 'status']
        )
        
        self.document_processing_duration = Histogram(
            'document_processing_duration_seconds',
            'Document processing duration',
            ['document_type', 'processing_step']
        )
        
        self.vector_store_operations = Counter(
            'vector_store_operations_total',
            'Vector store operations',
            ['operation', 'store_type', 'status']
        )
        
        # Internal metrics storage
        self._metrics_history: deque = deque(maxlen=1000)
        self._performance_window: deque = deque(maxlen=100)
        self._error_tracking: Dict[str, List[Dict]] = defaultdict(list)
        self._health_checks: Dict[str, HealthCheck] = {}
        
        # Background collection
        self._collection_interval = 10  # seconds
        self._background_task = None
        self._redis_client = None
        
    async def get_redis_client(self):
        """Get Redis client for metrics persistence."""
        if not self._redis_client:
            try:
                self._redis_client = redis.Redis.from_url(
                    settings.cache.redis_url,
                    decode_responses=True
                )
                await self._redis_client.ping()
            except Exception:
                self._redis_client = None
        return self._redis_client
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        user_id: Optional[str] = None
    ):
        """Record HTTP request metrics."""
        # Prometheus metrics
        self.request_count.labels(
            method=method,
            endpoint=endpoint, 
            status_code=status_code,
            user_id=user_id or "anonymous"
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        # Internal tracking
        metric_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration": duration,
            "user_id": user_id,
        }
        self._metrics_history.append(metric_data)
        
        # Track errors
        if status_code >= 400:
            error_type = "client_error" if status_code < 500 else "server_error"
            self.error_count.labels(
                method=method,
                endpoint=endpoint,
                error_type=error_type
            ).inc()
            
            self._error_tracking[endpoint].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status_code": status_code,
                "duration": duration,
                "user_id": user_id,
            })
    
    def record_rag_query(
        self,
        provider: str,
        model: str,
        duration: float,
        status: str = "success"
    ):
        """Record RAG query metrics."""
        self.rag_query_count.labels(
            provider=provider,
            model=model,
            status=status
        ).inc()
        
        self.rag_query_duration.labels(
            provider=provider,
            model=model
        ).observe(duration)
    
    def record_document_processing(
        self,
        document_type: str,
        processing_step: str,
        duration: float
    ):
        """Record document processing metrics."""
        self.document_processing_duration.labels(
            document_type=document_type,
            processing_step=processing_step
        ).observe(duration)
    
    def record_vector_operation(
        self,
        operation: str,
        store_type: str,
        status: str = "success"
    ):
        """Record vector store operation metrics."""
        self.vector_store_operations.labels(
            operation=operation,
            store_type=store_type,
            status=status
        ).inc()
    
    async def collect_system_metrics(self):
        """Collect system-level performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # Memory usage  
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.used)
            
            # Store performance window
            perf_metric = PerformanceMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                active_connections=len(asyncio.all_tasks()),
            )
            self._performance_window.append(perf_metric)
            
            # Persist to Redis if available
            redis_client = await self.get_redis_client()
            if redis_client:
                await redis_client.setex(
                    "metrics:system",
                    300,  # 5 minutes TTL
                    f"{cpu_percent},{memory.percent},{len(asyncio.all_tasks())}"
                )
                
        except Exception as e:
            structlog.get_logger().error("Failed to collect system metrics", error=str(e))
    
    def start_background_collection(self):
        """Start background metrics collection."""
        if self._background_task is None:
            self._background_task = threading.Thread(
                target=self._background_collector,
                daemon=True
            )
            self._background_task.start()
    
    def _background_collector(self):
        """Background thread for periodic metrics collection."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while True:
            try:
                loop.run_until_complete(self.collect_system_metrics())
                time.sleep(self._collection_interval)
            except Exception as e:
                structlog.get_logger().error("Background metrics collection failed", error=str(e))
                time.sleep(self._collection_interval)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        now = datetime.now(timezone.utc)
        
        # Calculate recent performance
        recent_metrics = [m for m in self._metrics_history 
                         if datetime.fromisoformat(m["timestamp"]) > now - timedelta(minutes=5)]
        
        total_requests = len(recent_metrics)
        avg_duration = sum(m["duration"] for m in recent_metrics) / max(total_requests, 1)
        error_count = sum(1 for m in recent_metrics if m["status_code"] >= 400)
        error_rate = (error_count / max(total_requests, 1)) * 100
        
        # Get latest performance data
        latest_perf = self._performance_window[-1] if self._performance_window else PerformanceMetrics()
        
        return {
            "timestamp": now.isoformat(),
            "requests": {
                "total_recent": total_requests,
                "avg_duration_ms": avg_duration * 1000,
                "error_rate_percent": error_rate,
                "errors_count": error_count,
            },
            "system": {
                "cpu_usage_percent": latest_perf.cpu_usage,
                "memory_usage_percent": latest_perf.memory_usage,
                "active_connections": latest_perf.active_connections,
            },
            "health_checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "last_checked": check.timestamp.isoformat(),
                    "duration_ms": check.duration_ms,
                }
                for name, check in self._health_checks.items()
            }
        }


class HealthCheckManager:
    """Manages system health checks."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks: Dict[str, Callable] = {}
        
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.health_checks[name] = check_func
    
    async def run_health_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        start_time = time.time()
        
        try:
            check_func = self.health_checks.get(name)
            if not check_func:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message="Health check not registered",
                    duration_ms=0.0
                )
            
            result = await check_func()
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, HealthCheck):
                result.duration_ms = duration_ms
                return result
            elif isinstance(result, bool):
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="OK" if result else "Check failed",
                    duration_ms=duration_ms
                )
            else:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=str(result),
                    duration_ms=duration_ms
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms
            )
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        for name in self.health_checks.keys():
            results[name] = await self.run_health_check(name)
            self.metrics_collector._health_checks[name] = results[name]
        
        return results
    
    async def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        health_checks = await self.run_all_health_checks()
        
        if not health_checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in health_checks.values()]
        
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Comprehensive monitoring middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics_collector = MetricsCollector()
        self.health_manager = HealthCheckManager(self.metrics_collector)
        
        # Start background collection
        self.metrics_collector.start_background_collection()
        
        # Register default health checks
        self._register_default_health_checks()
        
        # Request tracking
        self._active_requests = set()
        
    def _register_default_health_checks(self):
        """Register default system health checks."""
        
        async def check_memory():
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return HealthCheck("memory", HealthStatus.UNHEALTHY, f"Memory usage: {memory.percent}%")
            elif memory.percent > 80:
                return HealthCheck("memory", HealthStatus.DEGRADED, f"Memory usage: {memory.percent}%")
            return HealthCheck("memory", HealthStatus.HEALTHY, f"Memory usage: {memory.percent}%")
        
        async def check_cpu():
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 90:
                return HealthCheck("cpu", HealthStatus.UNHEALTHY, f"CPU usage: {cpu_percent}%")
            elif cpu_percent > 80:
                return HealthCheck("cpu", HealthStatus.DEGRADED, f"CPU usage: {cpu_percent}%")
            return HealthCheck("cpu", HealthStatus.HEALTHY, f"CPU usage: {cpu_percent}%")
        
        async def check_redis():
            try:
                redis_client = await self.metrics_collector.get_redis_client()
                if redis_client:
                    await redis_client.ping()
                    return HealthCheck("redis", HealthStatus.HEALTHY, "Redis connection OK")
                else:
                    return HealthCheck("redis", HealthStatus.DEGRADED, "Redis not configured")
            except Exception as e:
                return HealthCheck("redis", HealthStatus.UNHEALTHY, f"Redis error: {str(e)}")
        
        self.health_manager.register_health_check("memory", check_memory)
        self.health_manager.register_health_check("cpu", check_cpu)
        self.health_manager.register_health_check("redis", check_redis)
    
    async def dispatch(self, request: Request, call_next):
        """Process request through monitoring middleware."""
        start_time = time.time()
        request_id = f"{id(request)}_{start_time}"
        
        # Track active requests
        self._active_requests.add(request_id)
        self.metrics_collector.active_requests.inc()
        
        # Get user info if available
        user_id = getattr(request.state, "user", None)
        user_id = user_id.user_id if user_id else None
        
        try:
            # Handle special monitoring endpoints
            if request.url.path == "/api/metrics":
                return await self._handle_metrics_endpoint()
            elif request.url.path == "/api/health":
                return await self._handle_health_endpoint()
            
            # Process normal request
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics_collector.record_request(
                method=request.method,
                endpoint=self._normalize_endpoint(request.url.path),
                status_code=response.status_code,
                duration=duration,
                user_id=user_id
            )
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            self.metrics_collector.record_request(
                method=request.method,
                endpoint=self._normalize_endpoint(request.url.path),
                status_code=500,
                duration=duration,
                user_id=user_id
            )
            raise e
        
        finally:
            # Clean up tracking
            self._active_requests.discard(request_id)
            self.metrics_collector.active_requests.dec()
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics grouping."""
        # Replace IDs and UUIDs with placeholders
        import re
        
        # Replace UUIDs
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{uuid}',
            path,
            flags=re.IGNORECASE
        )
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        return path
    
    async def _handle_metrics_endpoint(self) -> Response:
        """Handle Prometheus metrics endpoint."""
        metrics_data = generate_latest()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    
    async def _handle_health_endpoint(self) -> JSONResponse:
        """Handle comprehensive health check endpoint."""
        overall_status = await self.health_manager.get_overall_health()
        health_checks = await self.health_manager.run_all_health_checks()
        metrics_summary = self.metrics_collector.get_metrics_summary()
        
        return JSONResponse(
            status_code=200 if overall_status == HealthStatus.HEALTHY else 503,
            content={
                "status": overall_status.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": {
                    name: {
                        "status": check.status.value,
                        "message": check.message,
                        "duration_ms": check.duration_ms,
                        "metadata": check.metadata,
                    }
                    for name, check in health_checks.items()
                },
                "metrics": metrics_summary,
                "uptime_seconds": time.time() - self.metrics_collector._metrics_history[0].get("timestamp", time.time()) if self.metrics_collector._metrics_history else 0,
            }
        )


# Global instances
metrics_collector = MetricsCollector()
health_manager = HealthCheckManager(metrics_collector)