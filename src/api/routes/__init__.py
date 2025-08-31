"""
API Routes Package

Comprehensive API route collection for the enterprise RAG platform:
- Authentication and authorization routes
- Document management and processing
- Query processing and conversation management
- WebSocket real-time communication
- Administrative functions and analytics
- Health monitoring and diagnostics
"""

from . import ai
from . import websocket
from . import auth
from . import health
from . import documents
from . import queries
from . import admin

__all__ = [
    "ai",
    "websocket", 
    "auth",
    "health",
    "documents",
    "queries",
    "admin",
]