"""
Core Package

Core functionality for the enterprise RAG platform including:
- Enterprise configuration management
- Multi-provider AI engine for code processing
- Advanced RAG engine with vector search and embeddings
- Document processing and chunking strategies
- Monitoring and health checks
"""

from .config import (
    settings,
    Environment,
    LogLevel,
    AIProvider,
    DatabaseConfig,
)

from .ai_engine import (
    ai_engine,
    AIEngine,
    TaskType,
    AIRequest,
    AIResponse,
)

from .rag_engine import (
    rag_engine,
    RAGEngine,
    Document,
    DocumentChunk,
    RAGResponse,
    ConversationContext,
    LLMProvider,
    EmbeddingProvider,
    VectorStoreType,
    ChunkingStrategy,
    initialize_rag_engine,
    health_check as rag_health_check,
)

__all__ = [
    # Configuration
    "settings",
    "Environment", 
    "LogLevel",
    "AIProvider",
    "DatabaseConfig",
    
    # AI Engine
    "ai_engine",
    "AIEngine",
    "TaskType",
    "AIRequest", 
    "AIResponse",
    
    # RAG Engine
    "rag_engine",
    "RAGEngine",
    "Document",
    "DocumentChunk", 
    "RAGResponse",
    "ConversationContext",
    "LLMProvider",
    "EmbeddingProvider",
    "VectorStoreType",
    "ChunkingStrategy",
    "initialize_rag_engine",
    "rag_health_check",
]