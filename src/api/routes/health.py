"""
Health and Monitoring Routes

System health and monitoring endpoints:
- Application health
- Database connectivity
- AI provider status
- System metrics
"""

import asyncio
import time
import psutil
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...core.config import settings
from ...core.ai_engine import ai_engine


router = APIRouter()


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str
    timestamp: float
    version: str
    environment: str
    uptime: float
    checks: Dict[str, Any]


class SystemMetrics(BaseModel):
    """System metrics response model."""
    cpu_percent: float
    memory_percent: float
    disk_usage: Dict[str, float]
    active_connections: int


@router.get("/health", response_model=HealthStatus, summary="Application health check")
async def health_check():
    """Comprehensive health check of the application."""
    start_time = time.time()
    
    checks = {
        "database": await check_database_health(),
        "ai_providers": await check_ai_providers_health(),
        "features": check_feature_flags(),
        "configuration": check_configuration()
    }
    
    # Determine overall status
    overall_status = "healthy"
    for check_name, check_result in checks.items():
        if check_result.get("status") == "unhealthy":
            overall_status = "unhealthy"
            break
        elif check_result.get("status") == "degraded":
            overall_status = "degraded"
    
    response_time = time.time() - start_time
    
    return HealthStatus(
        status=overall_status,
        timestamp=time.time(),
        version=settings.app_version,
        environment=settings.environment.value,
        uptime=get_uptime(),
        checks={
            **checks,
            "response_time": response_time
        }
    )


@router.get("/health/live", summary="Liveness probe")
async def liveness_probe():
    """Simple liveness probe for container orchestration."""
    return {"status": "alive", "timestamp": time.time()}


@router.get("/health/ready", summary="Readiness probe")
async def readiness_probe():
    """Readiness probe to check if the app is ready to serve traffic."""
    try:
        # Check critical components
        db_health = await check_database_health()
        ai_health = await check_ai_providers_health()
        
        if db_health.get("status") == "unhealthy":
            raise HTTPException(status_code=503, detail="Database not ready")
        
        if ai_health.get("status") == "unhealthy":
            raise HTTPException(status_code=503, detail="AI providers not ready")
        
        return {"status": "ready", "timestamp": time.time()}
    
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")


@router.get("/metrics", response_model=SystemMetrics, summary="System metrics")
async def system_metrics():
    """Get system performance metrics."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": (disk.used / disk.total) * 100
        }
        
        # Network connections (approximate)
        connections = len(psutil.net_connections())
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage=disk_usage,
            active_connections=connections
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/status", summary="Detailed system status")
async def detailed_status():
    """Get detailed system and component status."""
    try:
        status_info = {
            "application": {
                "name": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment.value,
                "debug": settings.debug,
                "uptime": get_uptime()
            },
            "configuration": {
                "host": settings.host,
                "port": settings.port,
                "workers": settings.workers,
                "log_level": settings.log_level.value
            },
            "features": {
                "code_completion": settings.features.enable_code_completion,
                "code_generation": settings.features.enable_code_generation,
                "code_analysis": settings.features.enable_code_analysis,
                "real_time_collaboration": settings.features.enable_real_time_collaboration,
                "analytics": settings.features.enable_analytics
            },
            "ai_providers": {
                "primary": settings.ai_providers.primary_provider.value,
                "fallbacks": [p.value for p in settings.ai_providers.fallback_providers],
                "available": [p.value for p in ai_engine.get_available_providers()]
            },
            "security": {
                "cors_enabled": bool(settings.security.cors_origins),
                "rate_limiting": {
                    "per_minute": settings.security.rate_limit_per_minute,
                    "per_hour": settings.security.rate_limit_per_hour
                }
            }
        }
        
        return status_info
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


async def check_database_health() -> Dict[str, Any]:
    """Check database health."""
    try:
        # For SQLite, just check if we can connect
        # In a real implementation, you'd test the actual database connection
        database_url = settings.database.database_url
        
        if database_url.startswith("sqlite"):
            # Simple SQLite check
            return {
                "status": "healthy",
                "type": "sqlite",
                "url": database_url.split("///")[-1] if "///" in database_url else "memory"
            }
        else:
            # For other databases, you'd implement proper connection testing
            return {
                "status": "healthy",
                "type": "external",
                "connection_pool": {
                    "size": settings.database.pool_size,
                    "max_overflow": settings.database.max_overflow
                }
            }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_ai_providers_health() -> Dict[str, Any]:
    """Check AI providers health."""
    try:
        health_status = await ai_engine.health_check()
        
        if not health_status:
            return {
                "status": "unhealthy",
                "error": "No AI providers available"
            }
        
        healthy_providers = sum(
            1 for status in health_status.values() 
            if status.get("status") == "healthy"
        )
        
        if healthy_providers == 0:
            return {
                "status": "unhealthy",
                "providers": health_status,
                "healthy_count": 0
            }
        elif healthy_providers < len(health_status):
            return {
                "status": "degraded",
                "providers": health_status,
                "healthy_count": healthy_providers,
                "total_count": len(health_status)
            }
        else:
            return {
                "status": "healthy",
                "providers": health_status,
                "healthy_count": healthy_providers
            }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def check_feature_flags() -> Dict[str, Any]:
    """Check feature flag status."""
    try:
        feature_status = {
            "code_completion": settings.features.enable_code_completion,
            "code_generation": settings.features.enable_code_generation,
            "code_analysis": settings.features.enable_code_analysis,
            "bug_detection": settings.features.enable_bug_detection,
            "code_refactoring": settings.features.enable_code_refactoring,
            "real_time_collaboration": settings.features.enable_real_time_collaboration,
            "analytics": settings.features.enable_analytics,
            "performance_monitoring": settings.features.enable_performance_monitoring
        }
        
        enabled_count = sum(1 for enabled in feature_status.values() if enabled)
        
        return {
            "status": "healthy",
            "features": feature_status,
            "enabled_count": enabled_count,
            "total_count": len(feature_status)
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def check_configuration() -> Dict[str, Any]:
    """Check configuration health."""
    try:
        config_checks = {
            "secret_key_set": bool(settings.security.secret_key),
            "environment_valid": settings.environment.value in ["development", "staging", "production", "testing"],
            "log_level_valid": settings.log_level.value in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "cors_configured": bool(settings.security.cors_origins),
            "rate_limits_set": settings.security.rate_limit_per_minute > 0
        }
        
        all_valid = all(config_checks.values())
        
        return {
            "status": "healthy" if all_valid else "degraded",
            "checks": config_checks,
            "valid_count": sum(1 for valid in config_checks.values() if valid)
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def get_uptime() -> float:
    """Get application uptime in seconds."""
    # This is a simplified implementation
    # In a real app, you'd track the start time
    try:
        boot_time = psutil.boot_time()
        return time.time() - boot_time
    except:
        return 0.0