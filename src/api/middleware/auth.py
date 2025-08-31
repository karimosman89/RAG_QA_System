"""
Enterprise Authentication Middleware

Comprehensive authentication and authorization middleware with:
- JWT token validation and refresh
- API key authentication
- Role-based access control (RBAC)
- Multi-tenancy support
- Session management
- Security audit logging
"""

import jwt
import time
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Set
from enum import Enum

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis.asyncio as redis
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from ...core.config import settings


class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    USER = "user" 
    GUEST = "guest"
    API_USER = "api_user"
    DEVELOPER = "developer"


class Permission(str, Enum):
    """System permissions."""
    READ_DOCUMENTS = "read:documents"
    WRITE_DOCUMENTS = "write:documents"
    DELETE_DOCUMENTS = "delete:documents"
    READ_ANALYTICS = "read:analytics"
    WRITE_ANALYTICS = "write:analytics"
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    API_ACCESS = "api:access"


# Role-based permissions mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: {
        Permission.READ_DOCUMENTS,
        Permission.WRITE_DOCUMENTS, 
        Permission.DELETE_DOCUMENTS,
        Permission.READ_ANALYTICS,
        Permission.WRITE_ANALYTICS,
        Permission.ADMIN_USERS,
        Permission.ADMIN_SYSTEM,
        Permission.API_ACCESS,
    },
    UserRole.USER: {
        Permission.READ_DOCUMENTS,
        Permission.WRITE_DOCUMENTS,
    },
    UserRole.GUEST: {
        Permission.READ_DOCUMENTS,
    },
    UserRole.API_USER: {
        Permission.READ_DOCUMENTS,
        Permission.WRITE_DOCUMENTS,
        Permission.API_ACCESS,
    },
    UserRole.DEVELOPER: {
        Permission.READ_DOCUMENTS,
        Permission.WRITE_DOCUMENTS,
        Permission.READ_ANALYTICS,
        Permission.API_ACCESS,
    },
}


class AuthenticatedUser:
    """Represents an authenticated user."""
    
    def __init__(
        self,
        user_id: str,
        email: Optional[str] = None,
        username: Optional[str] = None,
        role: UserRole = UserRole.GUEST,
        permissions: Optional[Set[Permission]] = None,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.user_id = user_id
        self.email = email
        self.username = username
        self.role = role
        self.permissions = permissions or ROLE_PERMISSIONS.get(role, set())
        self.tenant_id = tenant_id
        self.session_id = session_id
        self.expires_at = expires_at
        self.metadata = metadata or {}
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        return any(perm in self.permissions for perm in permissions)
    
    def is_expired(self) -> bool:
        """Check if user session is expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "username": self.username,
            "role": self.role.value,
            "permissions": [perm.value for perm in self.permissions],
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }


class JWTManager:
    """JWT token manager for authentication."""
    
    def __init__(self):
        self.secret_key = settings.security.jwt_secret_key
        self.algorithm = settings.security.jwt_algorithm
        self.access_token_expire = settings.security.access_token_expire_minutes
        self.refresh_token_expire = settings.security.refresh_token_expire_days
    
    def create_access_token(self, user: AuthenticatedUser) -> str:
        """Create JWT access token."""
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire)
        
        payload = {
            "sub": user.user_id,
            "email": user.email,
            "username": user.username,
            "role": user.role.value,
            "permissions": [perm.value for perm in user.permissions],
            "tenant_id": user.tenant_id,
            "session_id": user.session_id,
            "exp": expires_at,
            "iat": datetime.now(timezone.utc),
            "type": "access",
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user: AuthenticatedUser) -> str:
        """Create JWT refresh token."""
        expires_at = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire)
        
        payload = {
            "sub": user.user_id,
            "session_id": user.session_id,
            "exp": expires_at,
            "iat": datetime.now(timezone.utc),
            "type": "refresh",
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def extract_user_from_token(self, token: str) -> Optional[AuthenticatedUser]:
        """Extract user information from JWT token."""
        payload = self.verify_token(token)
        if not payload or payload.get("type") != "access":
            return None
        
        permissions = {Permission(perm) for perm in payload.get("permissions", [])}
        expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        
        return AuthenticatedUser(
            user_id=payload["sub"],
            email=payload.get("email"),
            username=payload.get("username"),
            role=UserRole(payload.get("role", "guest")),
            permissions=permissions,
            tenant_id=payload.get("tenant_id"),
            session_id=payload.get("session_id"),
            expires_at=expires_at,
        )


class APIKeyManager:
    """API key manager for service-to-service authentication."""
    
    def __init__(self):
        self.redis_client = None
        self.key_prefix = "api_key:"
        self.salt = settings.security.jwt_secret_key.encode()
    
    async def get_redis_client(self):
        """Get Redis client instance."""
        if not self.redis_client:
            self.redis_client = redis.Redis.from_url(
                settings.cache.redis_url,
                decode_responses=True
            )
        return self.redis_client
    
    def generate_api_key(self) -> str:
        """Generate a new API key."""
        return secrets.token_urlsafe(32)
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key_hash = base64.urlsafe_b64encode(kdf.derive(api_key.encode()))
        return key_hash.decode()
    
    async def store_api_key(
        self,
        api_key: str,
        user: AuthenticatedUser,
        name: Optional[str] = None,
        expires_in_days: Optional[int] = None
    ) -> str:
        """Store API key with associated user information."""
        redis_client = await self.get_redis_client()
        key_hash = self.hash_api_key(api_key)
        
        key_data = {
            "user_id": user.user_id,
            "role": user.role.value,
            "permissions": ",".join([perm.value for perm in user.permissions]),
            "tenant_id": user.tenant_id or "",
            "name": name or f"API Key {datetime.now().strftime('%Y-%m-%d')}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_used": "",
            "usage_count": "0",
        }
        
        # Store with expiration if specified
        if expires_in_days:
            await redis_client.setex(
                f"{self.key_prefix}{key_hash}",
                expires_in_days * 24 * 3600,
                "|".join([f"{k}:{v}" for k, v in key_data.items()])
            )
        else:
            await redis_client.set(
                f"{self.key_prefix}{key_hash}",
                "|".join([f"{k}:{v}" for k, v in key_data.items()])
            )
        
        return key_hash
    
    async def verify_api_key(self, api_key: str) -> Optional[AuthenticatedUser]:
        """Verify API key and return associated user."""
        redis_client = await self.get_redis_client()
        key_hash = self.hash_api_key(api_key)
        
        key_data_str = await redis_client.get(f"{self.key_prefix}{key_hash}")
        if not key_data_str:
            return None
        
        # Parse key data
        key_data = {}
        for item in key_data_str.split("|"):
            if ":" in item:
                k, v = item.split(":", 1)
                key_data[k] = v
        
        # Update usage statistics
        usage_count = int(key_data.get("usage_count", "0")) + 1
        key_data["usage_count"] = str(usage_count)
        key_data["last_used"] = datetime.now(timezone.utc).isoformat()
        
        await redis_client.set(
            f"{self.key_prefix}{key_hash}",
            "|".join([f"{k}:{v}" for k, v in key_data.items()])
        )
        
        # Create user object
        permissions = set()
        if key_data.get("permissions"):
            permissions = {Permission(perm) for perm in key_data["permissions"].split(",")}
        
        return AuthenticatedUser(
            user_id=key_data["user_id"],
            role=UserRole(key_data.get("role", "api_user")),
            permissions=permissions,
            tenant_id=key_data.get("tenant_id") if key_data.get("tenant_id") else None,
            metadata={
                "api_key_name": key_data.get("name"),
                "usage_count": usage_count,
                "last_used": key_data["last_used"],
            }
        )


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for the RAG platform."""
    
    def __init__(self, app):
        super().__init__(app)
        self.jwt_manager = JWTManager()
        self.api_key_manager = APIKeyManager()
        
        # Paths that don't require authentication
        self.public_paths = {
            "/",
            "/health",
            "/api/docs",
            "/api/redoc",
            "/api/openapi.json",
            "/api/auth/login",
            "/api/auth/register",
            "/api/auth/forgot-password",
            "/static",
        }
        
        # Paths that require specific permissions
        self.protected_paths = {
            "/api/admin": [Permission.ADMIN_SYSTEM],
            "/api/analytics": [Permission.READ_ANALYTICS],
            "/api/documents/delete": [Permission.DELETE_DOCUMENTS],
        }
    
    def is_public_path(self, path: str) -> bool:
        """Check if path is public (no authentication required)."""
        return any(path.startswith(public_path) for public_path in self.public_paths)
    
    def get_required_permissions(self, path: str, method: str) -> List[Permission]:
        """Get required permissions for a given path and method."""
        # Check exact path matches
        for protected_path, permissions in self.protected_paths.items():
            if path.startswith(protected_path):
                return permissions
        
        # Default permissions based on method
        if method in ["POST", "PUT", "PATCH"]:
            return [Permission.WRITE_DOCUMENTS]
        elif method == "DELETE":
            return [Permission.DELETE_DOCUMENTS]
        else:  # GET, HEAD, OPTIONS
            return [Permission.READ_DOCUMENTS]
    
    async def extract_token_from_request(self, request: Request) -> Optional[str]:
        """Extract authentication token from request."""
        # Check Authorization header (Bearer token)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        # Check API key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key
        
        # Check query parameter (for WebSocket connections)
        token = request.query_params.get("token")
        if token:
            return token
        
        return None
    
    async def authenticate_user(self, token: str) -> Optional[AuthenticatedUser]:
        """Authenticate user using token (JWT or API key)."""
        # Try JWT authentication first
        user = self.jwt_manager.extract_user_from_token(token)
        if user and not user.is_expired():
            return user
        
        # Try API key authentication
        try:
            user = await self.api_key_manager.verify_api_key(token)
            if user:
                return user
        except Exception as e:
            # Log authentication error but don't expose details
            print(f"API key authentication error: {e}")
        
        return None
    
    async def dispatch(self, request: Request, call_next):
        """Process request through authentication middleware."""
        path = request.url.path
        method = request.method
        
        # Skip authentication for public paths
        if self.is_public_path(path):
            return await call_next(request)
        
        # Extract authentication token
        token = await self.extract_token_from_request(request)
        if not token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Authentication required",
                    "message": "Missing authentication token"
                }
            )
        
        # Authenticate user
        user = await self.authenticate_user(token)
        if not user:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Invalid authentication",
                    "message": "Invalid or expired token"
                }
            )
        
        # Check permissions
        required_permissions = self.get_required_permissions(path, method)
        if required_permissions and not user.has_any_permission(required_permissions):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "Insufficient permissions",
                    "message": f"Required permissions: {[p.value for p in required_permissions]}"
                }
            )
        
        # Add user to request state
        request.state.user = user
        
        # Add security headers
        response = await call_next(request)
        response.headers["X-User-ID"] = user.user_id
        if user.tenant_id:
            response.headers["X-Tenant-ID"] = user.tenant_id
        
        return response


# Dependency for FastAPI route injection
async def get_current_user(request: Request) -> AuthenticatedUser:
    """Dependency to get current authenticated user in route handlers."""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user


async def require_permission(permission: Permission):
    """Dependency factory to require specific permissions."""
    async def check_permission(user: AuthenticatedUser = Depends(get_current_user)):
        if not user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required permission: {permission.value}"
            )
        return user
    return check_permission


async def require_role(role: UserRole):
    """Dependency factory to require specific role."""
    async def check_role(user: AuthenticatedUser = Depends(get_current_user)):
        if user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {role.value}"
            )
        return user
    return check_role


# Global instances
jwt_manager = JWTManager()
api_key_manager = APIKeyManager()