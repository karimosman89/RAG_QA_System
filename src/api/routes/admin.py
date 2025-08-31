"""
Admin API Routes

Enterprise administration interface with:
- User management and role assignment
- System configuration and settings
- Analytics dashboard and reporting
- Resource monitoring and optimization
- Audit logs and security monitoring
- Multi-tenant management
- API key management
- System health and diagnostics
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    BackgroundTasks,
    status
)
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy.orm import Session

from ...core.config import settings
from ...core.ai_engine import ai_engine
from ..middleware.auth import (
    get_current_user,
    require_permission,
    require_role,
    AuthenticatedUser,
    UserRole,
    Permission,
    JWTManager,
    APIKeyManager,
    jwt_manager,
    api_key_manager
)
from ..middleware.monitoring import metrics_collector, health_manager


router = APIRouter()


# Enums
class SystemStatus(str, Enum):
    """System status levels."""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class UserStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class AuditEventType(str, Enum):
    """Types of audit events."""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    ROLE_CHANGED = "role_changed"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    SYSTEM_CONFIG_CHANGED = "system_config_changed"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_DELETED = "document_deleted"
    QUERY_PROCESSED = "query_processed"


# Pydantic Models
class UserCreate(BaseModel):
    """User creation model."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.USER
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserUpdate(BaseModel):
    """User update model."""
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    role: Optional[UserRole] = None
    status: Optional[UserStatus] = None
    tenant_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    username: str
    full_name: str
    role: UserRole
    status: UserStatus
    tenant_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime]
    login_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class APIKeyCreate(BaseModel):
    """API key creation model."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    expires_in_days: Optional[int] = Field(None, ge=1, le=3650)
    permissions: List[Permission] = Field(default_factory=list)
    rate_limit: Optional[int] = Field(None, ge=1, le=10000)


class APIKeyResponse(BaseModel):
    """API key response model."""
    id: str
    name: str
    description: Optional[str]
    key_preview: str  # Only first/last few characters
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int
    permissions: List[Permission]
    rate_limit: Optional[int]
    status: str


class SystemConfig(BaseModel):
    """System configuration model."""
    max_upload_size_mb: int = Field(default=100, ge=1, le=1000)
    max_documents_per_user: int = Field(default=1000, ge=1, le=10000)
    default_embedding_model: str = "text-embedding-ada-002"
    default_llm_model: str = "gpt-3.5-turbo"
    enable_user_registration: bool = True
    require_email_verification: bool = True
    session_timeout_minutes: int = Field(default=1440, ge=5, le=10080)  # 1 day default
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)
    enable_analytics: bool = True
    maintenance_mode: bool = False
    custom_settings: Dict[str, Any] = Field(default_factory=dict)


class AuditLog(BaseModel):
    """Audit log entry model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType
    user_id: Optional[str]
    tenant_id: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str]
    user_agent: Optional[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    severity: str = "info"  # info, warning, error, critical


class SystemMetrics(BaseModel):
    """System metrics summary."""
    timestamp: datetime
    total_users: int
    active_users_24h: int
    total_documents: int
    documents_processed_24h: int
    total_queries: int
    queries_processed_24h: int
    avg_response_time_ms: float
    error_rate_percent: float
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    active_connections: int


class TenantInfo(BaseModel):
    """Multi-tenant information."""
    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    user_count: int
    document_count: int
    storage_used_bytes: int
    storage_limit_bytes: int
    is_active: bool
    settings: Dict[str, Any] = Field(default_factory=dict)


class AdminDashboardStats(BaseModel):
    """Admin dashboard statistics."""
    overview: SystemMetrics
    recent_audit_logs: List[AuditLog]
    top_users: List[Dict[str, Any]]
    system_health: Dict[str, Any]
    resource_usage: Dict[str, Any]
    recent_errors: List[Dict[str, Any]]


# Admin Services
class UserManager:
    """User management service."""
    
    def __init__(self):
        self.password_hasher = None  # Would integrate with password hashing library
    
    async def create_user(self, user_data: UserCreate, admin_user: AuthenticatedUser) -> UserResponse:
        """Create a new user account."""
        # This would integrate with the database
        user_id = str(uuid.uuid4())
        
        # Create user record
        user = UserResponse(
            id=user_id,
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            role=user_data.role,
            status=UserStatus.ACTIVE,
            tenant_id=user_data.tenant_id or admin_user.tenant_id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_login=None,
            metadata=user_data.metadata
        )
        
        # Log audit event
        await self._log_audit_event(
            AuditEventType.USER_CREATED,
            admin_user.user_id,
            user_id,
            f"Created user {user_data.username} with role {user_data.role.value}",
            admin_user.tenant_id
        )
        
        return user
    
    async def get_users(
        self,
        limit: int = 50,
        offset: int = 0,
        role_filter: Optional[UserRole] = None,
        status_filter: Optional[UserStatus] = None,
        tenant_id: Optional[str] = None
    ) -> List[UserResponse]:
        """Get list of users with filtering."""
        # This would query the database
        return []
    
    async def update_user(
        self,
        user_id: str,
        updates: UserUpdate,
        admin_user: AuthenticatedUser
    ) -> UserResponse:
        """Update user account."""
        # This would update the database record
        
        # Log audit event for role changes
        if updates.role:
            await self._log_audit_event(
                AuditEventType.ROLE_CHANGED,
                admin_user.user_id,
                user_id,
                f"Changed user role to {updates.role.value}",
                admin_user.tenant_id
            )
        
        # Return updated user (placeholder)
        return UserResponse(
            id=user_id,
            email="placeholder@example.com",
            username="placeholder",
            full_name="Placeholder User",
            role=updates.role or UserRole.USER,
            status=updates.status or UserStatus.ACTIVE,
            tenant_id=updates.tenant_id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_login=None
        )
    
    async def delete_user(self, user_id: str, admin_user: AuthenticatedUser):
        """Delete user account."""
        # This would remove from database
        
        await self._log_audit_event(
            AuditEventType.USER_DELETED,
            admin_user.user_id,
            user_id,
            f"Deleted user account",
            admin_user.tenant_id
        )
    
    async def _log_audit_event(
        self,
        event_type: AuditEventType,
        actor_user_id: str,
        resource_id: Optional[str],
        description: str,
        tenant_id: Optional[str]
    ):
        """Log audit event."""
        audit_log = AuditLog(
            event_type=event_type,
            user_id=actor_user_id,
            tenant_id=tenant_id,
            resource_id=resource_id,
            description=description
        )
        # This would store in audit log database
        pass


class SystemConfigManager:
    """System configuration management."""
    
    def __init__(self):
        self._config_cache = {}
    
    async def get_config(self) -> SystemConfig:
        """Get current system configuration."""
        return SystemConfig(
            max_upload_size_mb=settings.files.max_upload_size,
            default_embedding_model="text-embedding-ada-002",
            default_llm_model="gpt-3.5-turbo",
            enable_user_registration=True,
            require_email_verification=True,
            session_timeout_minutes=1440,
            rate_limit_per_minute=60,
            enable_analytics=True,
            maintenance_mode=False
        )
    
    async def update_config(
        self,
        config_updates: Dict[str, Any],
        admin_user: AuthenticatedUser
    ) -> SystemConfig:
        """Update system configuration."""
        current_config = await self.get_config()
        
        # Apply updates
        for key, value in config_updates.items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)
        
        # Log configuration change
        audit_log = AuditLog(
            event_type=AuditEventType.SYSTEM_CONFIG_CHANGED,
            user_id=admin_user.user_id,
            tenant_id=admin_user.tenant_id,
            description=f"Updated system configuration: {list(config_updates.keys())}",
            metadata=config_updates
        )
        
        # This would persist changes to database/config store
        
        return current_config


# Initialize services
user_manager = UserManager()
config_manager = SystemConfigManager()


# API Routes - User Management
@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    admin_user: AuthenticatedUser = Depends(require_role(UserRole.ADMIN))
):
    """Create a new user account."""
    return await user_manager.create_user(user_data, admin_user)


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    role: Optional[UserRole] = Query(None),
    status: Optional[UserStatus] = Query(None),
    tenant_id: Optional[str] = Query(None),
    admin_user: AuthenticatedUser = Depends(require_permission(Permission.ADMIN_USERS))
):
    """List users with filtering and pagination."""
    return await user_manager.get_users(
        limit=limit,
        offset=offset,
        role_filter=role,
        status_filter=status,
        tenant_id=tenant_id
    )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    admin_user: AuthenticatedUser = Depends(require_permission(Permission.ADMIN_USERS))
):
    """Get user by ID."""
    # This would query the database
    raise HTTPException(status_code=404, detail="User not found")


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    updates: UserUpdate,
    admin_user: AuthenticatedUser = Depends(require_permission(Permission.ADMIN_USERS))
):
    """Update user account."""
    return await user_manager.update_user(user_id, updates, admin_user)


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    admin_user: AuthenticatedUser = Depends(require_role(UserRole.ADMIN))
):
    """Delete user account."""
    await user_manager.delete_user(user_id, admin_user)
    return {"message": f"User {user_id} deleted successfully"}


# API Routes - API Key Management
@router.post("/api-keys", response_model=Dict[str, Any])
async def create_api_key(
    key_data: APIKeyCreate,
    admin_user: AuthenticatedUser = Depends(require_permission(Permission.ADMIN_SYSTEM))
):
    """Create a new API key."""
    # Generate API key
    api_key = api_key_manager.generate_api_key()
    
    # Create user object for API key
    api_user = AuthenticatedUser(
        user_id=str(uuid.uuid4()),
        role=UserRole.API_USER,
        permissions=set(key_data.permissions),
        tenant_id=admin_user.tenant_id
    )
    
    # Store API key
    key_hash = await api_key_manager.store_api_key(
        api_key=api_key,
        user=api_user,
        name=key_data.name,
        expires_in_days=key_data.expires_in_days
    )
    
    return {
        "api_key": api_key,  # Return full key only once
        "key_id": key_hash,
        "name": key_data.name,
        "permissions": [p.value for p in key_data.permissions],
        "expires_in_days": key_data.expires_in_days,
        "message": "Store this API key securely. It will not be shown again."
    }


@router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    admin_user: AuthenticatedUser = Depends(require_permission(Permission.ADMIN_SYSTEM))
):
    """List API keys."""
    # This would query API keys from Redis/database
    return []


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    admin_user: AuthenticatedUser = Depends(require_permission(Permission.ADMIN_SYSTEM))
):
    """Revoke an API key."""
    # This would remove the API key from Redis
    return {"message": f"API key {key_id} revoked successfully"}


# API Routes - System Configuration
@router.get("/config", response_model=SystemConfig)
async def get_system_config(
    admin_user: AuthenticatedUser = Depends(require_permission(Permission.ADMIN_SYSTEM))
):
    """Get system configuration."""
    return await config_manager.get_config()


@router.put("/config", response_model=SystemConfig)
async def update_system_config(
    config_updates: Dict[str, Any],
    admin_user: AuthenticatedUser = Depends(require_role(UserRole.ADMIN))
):
    """Update system configuration."""
    return await config_manager.update_config(config_updates, admin_user)


# API Routes - Analytics and Monitoring
@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics(
    admin_user: AuthenticatedUser = Depends(require_permission(Permission.READ_ANALYTICS))
):
    """Get system metrics and performance data."""
    metrics_summary = metrics_collector.get_metrics_summary()
    
    return SystemMetrics(
        timestamp=datetime.now(timezone.utc),
        total_users=0,  # Would query from database
        active_users_24h=0,
        total_documents=0,
        documents_processed_24h=0,
        total_queries=0,
        queries_processed_24h=0,
        avg_response_time_ms=metrics_summary["requests"]["avg_duration_ms"],
        error_rate_percent=metrics_summary["requests"]["error_rate_percent"],
        cpu_usage_percent=metrics_summary["system"]["cpu_usage_percent"],
        memory_usage_percent=metrics_summary["system"]["memory_usage_percent"],
        disk_usage_percent=0.0,  # Would get from system monitoring
        active_connections=metrics_summary["system"]["active_connections"]
    )


@router.get("/health/detailed")
async def get_detailed_health(
    admin_user: AuthenticatedUser = Depends(require_permission(Permission.ADMIN_SYSTEM))
):
    """Get detailed system health information."""
    health_checks = await health_manager.run_all_health_checks()
    overall_status = await health_manager.get_overall_health()
    
    return {
        "overall_status": overall_status.value,
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
        "system_info": {
            "version": settings.app_version,
            "environment": settings.environment.value,
            "uptime_seconds": 0,  # Would calculate actual uptime
        }
    }


@router.get("/audit-logs", response_model=List[AuditLog])
async def get_audit_logs(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    event_type: Optional[AuditEventType] = Query(None),
    user_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    admin_user: AuthenticatedUser = Depends(require_permission(Permission.READ_ANALYTICS))
):
    """Get audit logs with filtering."""
    # This would query audit logs from database
    return []


@router.get("/dashboard", response_model=AdminDashboardStats)
async def get_admin_dashboard(
    admin_user: AuthenticatedUser = Depends(require_permission(Permission.READ_ANALYTICS))
):
    """Get admin dashboard statistics."""
    metrics_summary = metrics_collector.get_metrics_summary()
    health_checks = await health_manager.run_all_health_checks()
    
    # Create overview metrics
    overview = SystemMetrics(
        timestamp=datetime.now(timezone.utc),
        total_users=0,
        active_users_24h=0,
        total_documents=0,
        documents_processed_24h=0,
        total_queries=0,
        queries_processed_24h=0,
        avg_response_time_ms=metrics_summary["requests"]["avg_duration_ms"],
        error_rate_percent=metrics_summary["requests"]["error_rate_percent"],
        cpu_usage_percent=metrics_summary["system"]["cpu_usage_percent"],
        memory_usage_percent=metrics_summary["system"]["memory_usage_percent"],
        disk_usage_percent=0.0,
        active_connections=metrics_summary["system"]["active_connections"]
    )
    
    return AdminDashboardStats(
        overview=overview,
        recent_audit_logs=[],
        top_users=[],
        system_health=metrics_summary["health_checks"],
        resource_usage={
            "cpu": metrics_summary["system"]["cpu_usage_percent"],
            "memory": metrics_summary["system"]["memory_usage_percent"],
            "connections": metrics_summary["system"]["active_connections"],
        },
        recent_errors=[]
    )


# API Routes - Multi-tenant Management
@router.get("/tenants", response_model=List[TenantInfo])
async def list_tenants(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    admin_user: AuthenticatedUser = Depends(require_role(UserRole.ADMIN))
):
    """List all tenants (super admin only)."""
    # This would query tenant information from database
    return []


@router.post("/tenants", response_model=TenantInfo)
async def create_tenant(
    tenant_name: str,
    description: Optional[str] = None,
    admin_user: AuthenticatedUser = Depends(require_role(UserRole.ADMIN))
):
    """Create a new tenant."""
    tenant_id = str(uuid.uuid4())
    
    tenant = TenantInfo(
        id=tenant_id,
        name=tenant_name,
        description=description,
        created_at=datetime.now(timezone.utc),
        user_count=0,
        document_count=0,
        storage_used_bytes=0,
        storage_limit_bytes=10 * 1024 * 1024 * 1024,  # 10GB default
        is_active=True
    )
    
    # This would store in database
    
    return tenant


# Maintenance and Control
@router.post("/maintenance/enable")
async def enable_maintenance_mode(
    admin_user: AuthenticatedUser = Depends(require_role(UserRole.ADMIN))
):
    """Enable maintenance mode."""
    # This would set a global flag to reject non-admin requests
    return {"message": "Maintenance mode enabled"}


@router.post("/maintenance/disable") 
async def disable_maintenance_mode(
    admin_user: AuthenticatedUser = Depends(require_role(UserRole.ADMIN))
):
    """Disable maintenance mode."""
    return {"message": "Maintenance mode disabled"}


@router.post("/cache/clear")
async def clear_system_cache(
    cache_type: str = Query("all", regex="^(all|redis|embeddings|documents)$"),
    admin_user: AuthenticatedUser = Depends(require_role(UserRole.ADMIN))
):
    """Clear system caches."""
    # This would clear various system caches
    return {"message": f"Cache cleared: {cache_type}"}