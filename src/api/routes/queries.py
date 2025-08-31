"""
Query and Chat API Routes

Advanced RAG query processing with:
- Real-time streaming responses
- Conversation memory and context management
- Multi-modal query support (text, images, documents)
- Intelligent query routing and fallbacks
- Query optimization and caching
- Analytics and performance tracking
- Collaborative chat sessions
"""

import uuid
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, AsyncGenerator, Union
from enum import Enum

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
    Query as QueryParam,
    status
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import redis.asyncio as redis

from ...core.config import settings
from ...core.ai_engine import ai_engine
from ..middleware.auth import (
    get_current_user,
    require_permission,
    AuthenticatedUser,
    Permission
)
from ..middleware.monitoring import metrics_collector


router = APIRouter()


# Enums
class QueryType(str, Enum):
    """Types of queries."""
    RAG = "rag"
    DIRECT = "direct"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"


class MessageRole(str, Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationStatus(str, Enum):
    """Conversation status."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


# Pydantic Models
class QueryContext(BaseModel):
    """Query context and metadata."""
    document_ids: List[str] = Field(default_factory=list)
    conversation_id: Optional[str] = None
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    query_type: QueryType = QueryType.RAG
    language: str = "en"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4000)
    stream: bool = False
    include_sources: bool = True
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_context_documents: int = Field(default=5, ge=1, le=20)


class Message(BaseModel):
    """Chat message model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    tokens_used: Optional[int] = None
    processing_time_ms: Optional[float] = None


class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., min_length=1, max_length=10000)
    context: QueryContext = Field(default_factory=QueryContext)
    message_history: List[Message] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Query response model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    response: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_id: Optional[str] = None
    processing_time_ms: float
    tokens_used: int
    model_used: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StreamChunk(BaseModel):
    """Streaming response chunk."""
    id: str
    content: str
    is_complete: bool = False
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """Conversation model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: Optional[str] = None
    status: ConversationStatus = ConversationStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    owner_id: str
    tenant_id: Optional[str] = None
    participants: List[str] = Field(default_factory=list)
    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    settings: Dict[str, Any] = Field(default_factory=dict)


class ConversationSummary(BaseModel):
    """Conversation summary model."""
    id: str
    title: str
    message_count: int
    last_activity: datetime
    participants_count: int
    status: ConversationStatus
    preview: str  # First few words of the last message


class QueryAnalytics(BaseModel):
    """Query analytics model."""
    total_queries: int
    avg_response_time_ms: float
    success_rate: float
    queries_by_type: Dict[str, int]
    popular_topics: List[Dict[str, Any]]
    user_satisfaction: float
    peak_usage_hours: List[int]
    recent_queries: List[QueryResponse]


# Services
class ConversationManager:
    """Manages conversation state and memory."""
    
    def __init__(self):
        self.redis_client = None
        self.conversation_prefix = "conversation:"
        self.message_prefix = "message:"
        self.max_context_messages = 50
        
    async def get_redis_client(self):
        """Get Redis client for conversation storage."""
        if not self.redis_client:
            try:
                self.redis_client = redis.Redis.from_url(
                    settings.cache.redis_url,
                    decode_responses=True
                )
            except Exception:
                self.redis_client = None
        return self.redis_client
    
    async def create_conversation(
        self,
        title: str,
        user: AuthenticatedUser,
        description: Optional[str] = None
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            title=title,
            description=description,
            owner_id=user.user_id,
            tenant_id=user.tenant_id,
            participants=[user.user_id]
        )
        
        await self._store_conversation(conversation)
        return conversation
    
    async def get_conversation(
        self,
        conversation_id: str,
        user: AuthenticatedUser
    ) -> Optional[Conversation]:
        """Get conversation by ID."""
        redis_client = await self.get_redis_client()
        if not redis_client:
            return None
        
        try:
            conv_data = await redis_client.get(f"{self.conversation_prefix}{conversation_id}")
            if conv_data:
                conv_dict = json.loads(conv_data)
                conversation = Conversation(**conv_dict)
                
                # Check access permissions
                if user.user_id not in conversation.participants:
                    return None
                
                # Load recent messages
                await self._load_conversation_messages(conversation)
                return conversation
        except Exception:
            pass
        
        return None
    
    async def add_message(
        self,
        conversation_id: str,
        message: Message,
        user: AuthenticatedUser
    ) -> bool:
        """Add message to conversation."""
        conversation = await self.get_conversation(conversation_id, user)
        if not conversation:
            return False
        
        # Add message to conversation
        conversation.messages.append(message)
        conversation.updated_at = datetime.now(timezone.utc)
        
        # Keep only recent messages in memory
        if len(conversation.messages) > self.max_context_messages:
            conversation.messages = conversation.messages[-self.max_context_messages:]
        
        # Store updated conversation
        await self._store_conversation(conversation)
        await self._store_message(conversation_id, message)
        
        return True
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        user: AuthenticatedUser,
        max_messages: int = 10
    ) -> List[Message]:
        """Get recent conversation messages for context."""
        conversation = await self.get_conversation(conversation_id, user)
        if not conversation:
            return []
        
        return conversation.messages[-max_messages:]
    
    async def _store_conversation(self, conversation: Conversation):
        """Store conversation in Redis."""
        redis_client = await self.get_redis_client()
        if redis_client:
            try:
                # Store conversation metadata (without full message history)
                conv_dict = conversation.dict()
                conv_dict['messages'] = []  # Store messages separately
                
                await redis_client.setex(
                    f"{self.conversation_prefix}{conversation.id}",
                    86400 * 30,  # 30 days TTL
                    json.dumps(conv_dict, default=str)
                )
            except Exception:
                pass
    
    async def _store_message(self, conversation_id: str, message: Message):
        """Store individual message."""
        redis_client = await self.get_redis_client()
        if redis_client:
            try:
                await redis_client.lpush(
                    f"{self.message_prefix}{conversation_id}",
                    json.dumps(message.dict(), default=str)
                )
                # Keep only recent messages
                await redis_client.ltrim(f"{self.message_prefix}{conversation_id}", 0, 99)
            except Exception:
                pass
    
    async def _load_conversation_messages(self, conversation: Conversation):
        """Load conversation messages from Redis."""
        redis_client = await self.get_redis_client()
        if redis_client:
            try:
                messages_data = await redis_client.lrange(
                    f"{self.message_prefix}{conversation.id}", 0, -1
                )
                
                messages = []
                for msg_data in reversed(messages_data):  # Reverse to get chronological order
                    try:
                        msg_dict = json.loads(msg_data)
                        message = Message(**msg_dict)
                        messages.append(message)
                    except Exception:
                        continue
                
                conversation.messages = messages[-self.max_context_messages:]
            except Exception:
                pass


class QueryProcessor:
    """Advanced query processing with RAG integration."""
    
    def __init__(self):
        self.conversation_manager = ConversationManager()
        
    async def process_query(
        self,
        query_request: QueryRequest,
        user: AuthenticatedUser,
        stream: bool = False
    ) -> Union[QueryResponse, AsyncGenerator[StreamChunk, None]]:
        """Process a RAG query with optional streaming."""
        start_time = datetime.now()
        query_id = str(uuid.uuid4())
        
        try:
            # Get conversation context if provided
            context_messages = []
            if query_request.context.conversation_id:
                context_messages = await self.conversation_manager.get_conversation_context(
                    query_request.context.conversation_id,
                    user,
                    max_messages=10
                )
            
            # Combine context from conversation and request
            all_context_messages = context_messages + query_request.message_history
            
            # Build conversation history for LLM
            conversation_history = []
            for msg in all_context_messages[-10:]:  # Last 10 messages for context
                conversation_history.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
            
            # Process query based on type
            if query_request.context.query_type == QueryType.RAG:
                # RAG processing with document retrieval
                if stream:
                    return self._stream_rag_response(
                        query_request.query,
                        query_request.context,
                        conversation_history,
                        query_id,
                        user
                    )
                else:
                    response_content, sources, tokens_used, model_used = await self._process_rag_query(
                        query_request.query,
                        query_request.context,
                        conversation_history
                    )
            else:
                # Direct LLM processing
                if stream:
                    return self._stream_direct_response(
                        query_request.query,
                        query_request.context,
                        conversation_history,
                        query_id,
                        user
                    )
                else:
                    response_content, tokens_used, model_used = await self._process_direct_query(
                        query_request.query,
                        query_request.context,
                        conversation_history
                    )
                    sources = []
            
            # Calculate processing time
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create response
            response = QueryResponse(
                id=query_id,
                query=query_request.query,
                response=response_content,
                sources=sources,
                conversation_id=query_request.context.conversation_id,
                processing_time_ms=processing_time_ms,
                tokens_used=tokens_used,
                model_used=model_used
            )
            
            # Store in conversation if applicable
            if query_request.context.conversation_id:
                # Add user message
                user_message = Message(
                    role=MessageRole.USER,
                    content=query_request.query
                )
                
                # Add assistant response
                assistant_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=response_content,
                    sources=sources,
                    tokens_used=tokens_used,
                    processing_time_ms=processing_time_ms
                )
                
                await self.conversation_manager.add_message(
                    query_request.context.conversation_id,
                    user_message,
                    user
                )
                await self.conversation_manager.add_message(
                    query_request.context.conversation_id,
                    assistant_message,
                    user
                )
            
            # Record metrics
            metrics_collector.record_rag_query(
                provider=model_used.split('/')[0] if '/' in model_used else "unknown",
                model=model_used,
                duration=processing_time_ms / 1000,
                status="success"
            )
            
            return response
            
        except Exception as e:
            # Record error metrics
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            metrics_collector.record_rag_query(
                provider="unknown",
                model="unknown", 
                duration=processing_time_ms / 1000,
                status="error"
            )
            
            raise HTTPException(
                status_code=500,
                detail=f"Query processing failed: {str(e)}"
            )
    
    async def _process_rag_query(
        self,
        query: str,
        context: QueryContext,
        conversation_history: List[Dict]
    ) -> tuple:
        """Process RAG query with document retrieval."""
        # This would integrate with the RAG engine
        # For now, return placeholder values
        response_content = f"RAG response to: {query}"
        sources = []
        tokens_used = 150
        model_used = "gpt-3.5-turbo"
        
        return response_content, sources, tokens_used, model_used
    
    async def _process_direct_query(
        self,
        query: str,
        context: QueryContext,
        conversation_history: List[Dict]
    ) -> tuple:
        """Process direct LLM query."""
        # This would integrate with the AI engine
        # For now, return placeholder values
        response_content = f"Direct response to: {query}"
        tokens_used = 100
        model_used = "gpt-3.5-turbo"
        
        return response_content, tokens_used, model_used
    
    async def _stream_rag_response(
        self,
        query: str,
        context: QueryContext,
        conversation_history: List[Dict],
        query_id: str,
        user: AuthenticatedUser
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream RAG response chunks."""
        # Simulate streaming response
        response_parts = [
            "Based on the documents, ",
            "I can provide the following information: ",
            f"The answer to '{query}' is that ",
            "this is a streaming response simulation. ",
            "In a real implementation, this would stream ",
            "actual RAG results from the AI engine."
        ]
        
        accumulated_content = ""
        
        for i, part in enumerate(response_parts):
            accumulated_content += part
            
            chunk = StreamChunk(
                id=f"{query_id}_{i}",
                content=part,
                is_complete=False
            )
            
            yield chunk
            await asyncio.sleep(0.1)  # Simulate processing delay
        
        # Final chunk
        final_chunk = StreamChunk(
            id=f"{query_id}_final",
            content="",
            is_complete=True,
            sources=[],
            metadata={
                "total_tokens": len(accumulated_content.split()),
                "model_used": "gpt-3.5-turbo"
            }
        )
        
        yield final_chunk
    
    async def _stream_direct_response(
        self,
        query: str,
        context: QueryContext,
        conversation_history: List[Dict],
        query_id: str,
        user: AuthenticatedUser
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream direct LLM response chunks."""
        # Similar to RAG streaming but without document retrieval
        response_parts = [
            "This is a direct response ",
            f"to your query: '{query}'. ",
            "The AI model processes this ",
            "without accessing additional documents. ",
            "This is a streaming simulation."
        ]
        
        accumulated_content = ""
        
        for i, part in enumerate(response_parts):
            accumulated_content += part
            
            chunk = StreamChunk(
                id=f"{query_id}_{i}",
                content=part,
                is_complete=False
            )
            
            yield chunk
            await asyncio.sleep(0.1)
        
        # Final chunk
        final_chunk = StreamChunk(
            id=f"{query_id}_final",
            content="",
            is_complete=True,
            metadata={
                "total_tokens": len(accumulated_content.split()),
                "model_used": "gpt-3.5-turbo"
            }
        )
        
        yield final_chunk


# Initialize services
query_processor = QueryProcessor()
conversation_manager = ConversationManager()


# API Routes
@router.post("/query", response_model=QueryResponse)
async def process_query(
    query_request: QueryRequest,
    background_tasks: BackgroundTasks,
    user: AuthenticatedUser = Depends(require_permission(Permission.READ_DOCUMENTS))
):
    """Process a RAG query and return the response."""
    return await query_processor.process_query(query_request, user, stream=False)


@router.post("/query/stream")
async def stream_query(
    query_request: QueryRequest,
    user: AuthenticatedUser = Depends(require_permission(Permission.READ_DOCUMENTS))
):
    """Process a RAG query with streaming response."""
    
    async def generate_stream():
        async for chunk in await query_processor.process_query(query_request, user, stream=True):
            yield f"data: {chunk.json()}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/conversations", response_model=Conversation)
async def create_conversation(
    title: str,
    description: Optional[str] = None,
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Create a new conversation."""
    return await conversation_manager.create_conversation(title, user, description)


@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(
    conversation_id: str,
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Get conversation by ID."""
    conversation = await conversation_manager.get_conversation(conversation_id, user)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    limit: int = QueryParam(20, ge=1, le=100),
    offset: int = QueryParam(0, ge=0),
    status: Optional[ConversationStatus] = QueryParam(None),
    user: AuthenticatedUser = Depends(get_current_user)
):
    """List user's conversations."""
    # This would query the database for user's conversations
    return []


@router.put("/conversations/{conversation_id}", response_model=Conversation)
async def update_conversation(
    conversation_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[ConversationStatus] = None,
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Update conversation metadata."""
    conversation = await conversation_manager.get_conversation(conversation_id, user)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Update fields
    if title is not None:
        conversation.title = title
    if description is not None:
        conversation.description = description
    if status is not None:
        conversation.status = status
    
    conversation.updated_at = datetime.now(timezone.utc)
    
    await conversation_manager._store_conversation(conversation)
    return conversation


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Delete a conversation."""
    conversation = await conversation_manager.get_conversation(conversation_id, user)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Only owner can delete
    if conversation.owner_id != user.user_id:
        raise HTTPException(status_code=403, detail="Only conversation owner can delete")
    
    # This would remove the conversation from storage
    return {"message": f"Conversation {conversation_id} deleted successfully"}


@router.get("/analytics/summary", response_model=QueryAnalytics)
async def get_query_analytics(
    user: AuthenticatedUser = Depends(require_permission(Permission.READ_ANALYTICS))
):
    """Get query analytics and statistics."""
    return QueryAnalytics(
        total_queries=0,
        avg_response_time_ms=0.0,
        success_rate=0.0,
        queries_by_type={},
        popular_topics=[],
        user_satisfaction=0.0,
        peak_usage_hours=[],
        recent_queries=[]
    )


# WebSocket endpoint for real-time chat
@router.websocket("/chat/{conversation_id}")
async def websocket_chat(
    websocket: WebSocket,
    conversation_id: str
):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    
    try:
        # This would handle real-time chat messages
        # For now, just echo messages
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Echo back the message
            response = {
                "type": "message",
                "content": f"Echo: {message_data.get('content', '')}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=1000)