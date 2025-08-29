"""
Logging Middleware

Enhanced request/response logging with:
- Request/response timing
- User identification
- Error tracking
- Performance monitoring
"""

import time
import json
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...core.config import settings


logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced logging middleware for request/response monitoring."""
    
    def __init__(self, app):
        super().__init__(app)
        self.sensitive_headers = {
            "authorization", 
            "cookie", 
            "x-api-key",
            "x-auth-token"
        }
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with enhanced logging."""
        start_time = time.time()
        
        # Extract request information
        request_info = self._extract_request_info(request)
        
        # Log incoming request
        if settings.log_level in ["DEBUG"]:
            logger.debug(f"Incoming request: {json.dumps(request_info, indent=2)}")
        else:
            logger.info(f"{request_info['method']} {request_info['path']} - {request_info['client_ip']}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Extract response information
            response_info = self._extract_response_info(response, process_time)
            
            # Log response
            self._log_response(request_info, response_info)
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            error_info = {
                "error": str(e),
                "error_type": type(e).__name__,
                "process_time": process_time
            }
            
            logger.error(f"Request failed: {json.dumps({**request_info, **error_info}, indent=2)}")
            
            # Re-raise the exception
            raise
    
    def _extract_request_info(self, request: Request) -> dict:
        """Extract relevant information from the request."""
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            client_ip = real_ip
        
        # Get user agent
        user_agent = request.headers.get("User-Agent", "unknown")
        
        # Get request ID (if available)
        request_id = request.headers.get("X-Request-ID")
        
        # Filter sensitive headers
        headers = {}
        for name, value in request.headers.items():
            if name.lower() not in self.sensitive_headers:
                headers[name] = value
            else:
                headers[name] = "[REDACTED]"
        
        # Get query parameters
        query_params = dict(request.query_params)
        
        request_info = {
            "timestamp": time.time(),
            "method": request.method,
            "path": str(request.url.path),
            "full_url": str(request.url),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "content_type": request.headers.get("Content-Type"),
            "content_length": request.headers.get("Content-Length"),
            "query_params": query_params,
            "path_params": dict(request.path_params),
        }
        
        # Add request ID if available
        if request_id:
            request_info["request_id"] = request_id
        
        # Add headers in debug mode
        if settings.debug:
            request_info["headers"] = headers
        
        return request_info
    
    def _extract_response_info(self, response: Response, process_time: float) -> dict:
        """Extract relevant information from the response."""
        response_info = {
            "status_code": response.status_code,
            "content_type": response.headers.get("Content-Type"),
            "content_length": response.headers.get("Content-Length"),
            "process_time": round(process_time, 4)
        }
        
        # Add response headers in debug mode
        if settings.debug:
            response_info["headers"] = dict(response.headers)
        
        return response_info
    
    def _log_response(self, request_info: dict, response_info: dict):
        """Log the response with appropriate level."""
        status_code = response_info["status_code"]
        process_time = response_info["process_time"]
        
        # Create log message
        log_data = {
            **request_info,
            **response_info
        }
        
        # Determine log level based on status code and processing time
        if status_code >= 500:
            # Server errors
            logger.error(f"Server error response: {json.dumps(log_data, indent=2)}")
        elif status_code >= 400:
            # Client errors
            if settings.debug:
                logger.warning(f"Client error response: {json.dumps(log_data, indent=2)}")
            else:
                logger.warning(f"{request_info['method']} {request_info['path']} - {status_code} - {process_time}s")
        elif process_time > 5.0:
            # Slow requests
            logger.warning(f"Slow request: {json.dumps(log_data, indent=2)}")
        elif process_time > 1.0:
            # Medium requests
            logger.info(f"Request completed: {request_info['method']} {request_info['path']} - {status_code} - {process_time}s")
        else:
            # Fast requests
            if settings.debug:
                logger.debug(f"Request completed: {json.dumps(log_data, indent=2)}")
            else:
                logger.info(f"{request_info['method']} {request_info['path']} - {status_code} - {process_time}s")
        
        # Log performance metrics for monitoring
        if settings.features.enable_performance_monitoring:
            self._log_performance_metrics(request_info, response_info)
    
    def _log_performance_metrics(self, request_info: dict, response_info: dict):
        """Log performance metrics for monitoring systems."""
        metrics = {
            "metric_type": "request_performance",
            "path": request_info["path"],
            "method": request_info["method"],
            "status_code": response_info["status_code"],
            "process_time": response_info["process_time"],
            "timestamp": request_info["timestamp"],
            "client_ip": request_info["client_ip"]
        }
        
        # Log as structured data for monitoring
        logger.info(f"METRICS: {json.dumps(metrics)}")
        
        # Emit warnings for performance issues
        process_time = response_info["process_time"]
        if process_time > 10.0:
            logger.warning(f"PERFORMANCE ALERT: Very slow request - {process_time}s for {request_info['method']} {request_info['path']}")
        elif process_time > 5.0:
            logger.warning(f"PERFORMANCE WARNING: Slow request - {process_time}s for {request_info['method']} {request_info['path']}")
        
        # Log error rates
        if response_info["status_code"] >= 500:
            logger.error(f"ERROR RATE: Server error for {request_info['method']} {request_info['path']}")
        elif response_info["status_code"] >= 400:
            logger.warning(f"ERROR RATE: Client error for {request_info['method']} {request_info['path']}")


# Additional utility for structured logging
class StructuredLogger:
    """Utility class for structured logging."""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
    
    def log_event(self, event_type: str, data: dict, level: str = "INFO"):
        """Log a structured event."""
        log_data = {
            "event_type": event_type,
            "timestamp": time.time(),
            **data
        }
        
        message = f"EVENT: {json.dumps(log_data)}"
        
        if level == "DEBUG":
            self.logger.debug(message)
        elif level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "CRITICAL":
            self.logger.critical(message)
    
    def log_ai_request(self, provider: str, model: str, task_type: str, 
                      success: bool, response_time: float, tokens_used: int = None):
        """Log AI API request."""
        self.log_event("ai_request", {
            "provider": provider,
            "model": model,
            "task_type": task_type,
            "success": success,
            "response_time": response_time,
            "tokens_used": tokens_used
        })
    
    def log_user_action(self, user_id: str, action: str, details: dict = None):
        """Log user action."""
        self.log_event("user_action", {
            "user_id": user_id,
            "action": action,
            "details": details or {}
        })
    
    def log_error(self, error_type: str, error_message: str, 
                  context: dict = None, stack_trace: str = None):
        """Log application error."""
        self.log_event("application_error", {
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "stack_trace": stack_trace
        }, level="ERROR")