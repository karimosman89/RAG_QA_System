"""
Enterprise Middleware Package

Comprehensive middleware stack for the RAG platform including:
- Authentication and authorization (JWT, API keys, RBAC)
- Rate limiting and DDoS protection
- Comprehensive logging and audit trails
- Real-time monitoring and metrics collection
- Performance tracking and optimization
"""

from .auth import (
    AuthMiddleware,
    AuthenticatedUser,
    UserRole,
    Permission,
    JWTManager,
    APIKeyManager,
    get_current_user,
    require_permission,
    require_role,
    jwt_manager,
    api_key_manager,
)

from .rate_limit import (
    RateLimitMiddleware,
    RateLimiter,
    rate_limiter,
)

from .logging import (
    LoggingMiddleware,
    AuditLogger,
    SecurityLogger,
    PerformanceLogger,
    audit_logger,
    security_logger,
    performance_logger,
)

from .monitoring import (
    MonitoringMiddleware,
    MetricsCollector,
    HealthCheckManager,
    HealthStatus,
    PerformanceMetrics,
    metrics_collector,
    health_manager,
)

__all__ = [
    # Authentication
    "AuthMiddleware",
    "AuthenticatedUser", 
    "UserRole",
    "Permission",
    "JWTManager",
    "APIKeyManager",
    "get_current_user",
    "require_permission",
    "require_role",
    "jwt_manager",
    "api_key_manager",
    
    # Rate Limiting
    "RateLimitMiddleware",
    "RateLimiter",
    "rate_limiter",
    
    # Logging
    "LoggingMiddleware",
    "AuditLogger",
    "SecurityLogger", 
    "PerformanceLogger",
    "audit_logger",
    "security_logger",
    "performance_logger",
    
    # Monitoring
    "MonitoringMiddleware",
    "MetricsCollector",
    "HealthCheckManager",
    "HealthStatus",
    "PerformanceMetrics",
    "metrics_collector",
    "health_manager",
]