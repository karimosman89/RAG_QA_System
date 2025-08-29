"""
Rate Limiting Middleware

Implement rate limiting to prevent API abuse:
- Per-IP rate limiting
- Per-user rate limiting
- Different limits for different endpoints
"""

import time
import asyncio
from typing import Dict, Tuple
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ...core.config import settings


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests: Dict[str, list] = {}
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> Tuple[bool, int]:
        """Check if a request is allowed and return remaining requests."""
        async with self.lock:
            now = time.time()
            
            # Initialize or clean old requests
            if key not in self.requests:
                self.requests[key] = []
            
            # Remove requests outside the window
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if now - req_time < window_seconds
            ]
            
            # Check if request is allowed
            if len(self.requests[key]) < max_requests:
                self.requests[key].append(now)
                remaining = max_requests - len(self.requests[key])
                return True, remaining
            else:
                remaining = 0
                return False, remaining
    
    async def time_until_reset(self, key: str, window_seconds: int) -> int:
        """Get time until rate limit resets."""
        if key not in self.requests or not self.requests[key]:
            return 0
        
        oldest_request = min(self.requests[key])
        reset_time = oldest_request + window_seconds
        return max(0, int(reset_time - time.time()))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.rate_limiter = RateLimiter()
        
        # Rate limit configurations
        self.default_limits = {
            "requests_per_minute": settings.security.rate_limit_per_minute,
            "requests_per_hour": settings.security.rate_limit_per_hour,
        }
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/ai/": {
                "requests_per_minute": 20,  # Lower limit for AI endpoints
                "requests_per_hour": 200,
            },
            "/ws/": {
                "requests_per_minute": 100,  # Higher limit for WebSocket
                "requests_per_hour": 1000,
            }
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process the request with rate limiting."""
        # Skip rate limiting for certain paths
        if self._should_skip_rate_limiting(request):
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Get rate limits for this endpoint
        limits = self._get_limits_for_path(request.url.path)
        
        # Check rate limits
        allowed_minute, remaining_minute = await self.rate_limiter.is_allowed(
            f"{client_id}:minute",
            limits["requests_per_minute"],
            60
        )
        
        allowed_hour, remaining_hour = await self.rate_limiter.is_allowed(
            f"{client_id}:hour", 
            limits["requests_per_hour"],
            3600
        )
        
        # If rate limit exceeded
        if not allowed_minute:
            reset_time = await self.rate_limiter.time_until_reset(f"{client_id}:minute", 60)
            return self._create_rate_limit_response(
                "Rate limit exceeded - too many requests per minute",
                reset_time,
                limits["requests_per_minute"],
                0
            )
        
        if not allowed_hour:
            reset_time = await self.rate_limiter.time_until_reset(f"{client_id}:hour", 3600)
            return self._create_rate_limit_response(
                "Rate limit exceeded - too many requests per hour",
                reset_time,
                limits["requests_per_hour"], 
                0
            )
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit-Minute"] = str(limits["requests_per_minute"])
        response.headers["X-RateLimit-Remaining-Minute"] = str(remaining_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(limits["requests_per_hour"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(remaining_hour)
        
        return response
    
    def _should_skip_rate_limiting(self, request: Request) -> bool:
        """Check if rate limiting should be skipped for this request."""
        skip_paths = [
            "/health",
            "/api/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/static/",
            "/favicon.ico"
        ]
        
        path = request.url.path
        return any(path.startswith(skip_path) for skip_path in skip_paths)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get user ID from JWT token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In a real implementation, decode the JWT and get user ID
            token = auth_header.split(" ")[1]
            # For now, use token as identifier
            return f"user:{hash(token) % 10000}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        
        # Check for forwarded IP headers (for reverse proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            client_ip = real_ip
        
        return f"ip:{client_ip}"
    
    def _get_limits_for_path(self, path: str) -> Dict[str, int]:
        """Get rate limits for a specific path."""
        # Check for endpoint-specific limits
        for endpoint_prefix, limits in self.endpoint_limits.items():
            if path.startswith(endpoint_prefix):
                return limits
        
        # Return default limits
        return self.default_limits
    
    def _create_rate_limit_response(self, message: str, reset_time: int, 
                                  limit: int, remaining: int) -> Response:
        """Create a rate limit exceeded response."""
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(time.time()) + reset_time),
            "Retry-After": str(reset_time),
            "Content-Type": "application/json"
        }
        
        body = {
            "error": "Rate limit exceeded",
            "message": message,
            "retry_after": reset_time
        }
        
        import json
        return Response(
            content=json.dumps(body),
            status_code=429,
            headers=headers
        )