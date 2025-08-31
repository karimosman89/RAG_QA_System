"""
FastAPI Server for AI-Assisted Coding Environment

Modern web server with:
- RESTful API endpoints
- WebSocket real-time communication
- Authentication and authorization
- Rate limiting and security
- Health monitoring
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from ..core.config import settings
from ..core.ai_engine import ai_engine
from .routes import ai, websocket, auth, health, documents, queries, admin
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.auth import AuthMiddleware
from .middleware.monitoring import MonitoringMiddleware


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.value),
    format=settings.log_format,
    filename=settings.log_file
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment.value}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Initialize AI engine
    logger.info("Initializing AI engine...")
    ai_engine.initialize_clients()
    
    # Health check
    health_status = await ai_engine.health_check()
    logger.info(f"AI providers health: {health_status}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.description,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/api/docs" if not settings.is_production else None,
        redoc_url="/api/redoc" if not settings.is_production else None,
    )
    
    # Configure middleware
    setup_middleware(app)
    
    # Configure routes
    setup_routes(app)
    
    # Configure static files and templates
    setup_static_and_templates(app)
    
    return app


def setup_middleware(app: FastAPI):
    """Set up application middleware."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=settings.security.cors_allow_credentials,
        allow_methods=settings.security.cors_allow_methods,
        allow_headers=settings.security.cors_allow_headers,
    )
    
    # Trusted host middleware
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.security.allowed_hosts
        )
    
    # Custom middleware (order matters - added in reverse execution order)
    app.add_middleware(MonitoringMiddleware)  # Last to execute, first to setup
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware)  # First to execute, last to setup


def setup_routes(app: FastAPI):
    """Set up application routes."""
    
    # API routes
    app.include_router(
        auth.router,
        prefix="/api/auth",
        tags=["authentication"]
    )
    
    app.include_router(
        ai.router,
        prefix="/api/ai",
        tags=["ai"]
    )
    
    app.include_router(
        websocket.router,
        prefix="/ws",
        tags=["websocket"]
    )
    
    app.include_router(
        health.router,
        prefix="/api",
        tags=["health"]
    )
    
    app.include_router(
        documents.router,
        prefix="/api/documents",
        tags=["documents"]
    )
    
    app.include_router(
        queries.router,
        prefix="/api/queries", 
        tags=["queries"]
    )
    
    app.include_router(
        admin.router,
        prefix="/api/admin",
        tags=["admin"]
    )


def setup_static_and_templates(app: FastAPI):
    """Set up static files and templates."""
    
    # Static files
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    
    # Templates
    templates = Jinja2Templates(directory=settings.templates_dir)
    
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root(request: Request):
        """Serve the main application page."""
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "app_name": settings.app_name,
                "version": settings.app_version,
                "debug": settings.debug
            }
        )
    
    @app.get("/health", include_in_schema=False)
    async def health_check():
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment.value
        }


# Create application instance
app = create_app()


def run_server():
    """Run the development server."""
    uvicorn.run(
        "src.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload and settings.is_development,
        workers=settings.workers if settings.is_production else 1,
        log_level=settings.log_level.value.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    run_server()