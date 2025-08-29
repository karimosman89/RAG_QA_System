"""
AI API Routes

RESTful endpoints for AI-powered coding assistance:
- Code completion
- Code generation  
- Code analysis
- Bug detection
- Code refactoring
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ...core.ai_engine import ai_engine, TaskType
from ...core.config import settings


logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class CodeCompletionRequest(BaseModel):
    """Request model for code completion."""
    content: str = Field(..., description="Code to complete", min_length=1, max_length=10000)
    language: str = Field(default="python", description="Programming language")
    cursor_position: Optional[int] = Field(default=None, description="Cursor position in code")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class CodeGenerationRequest(BaseModel):
    """Request model for code generation."""
    description: str = Field(..., description="Natural language description", min_length=5, max_length=5000)
    language: str = Field(default="python", description="Programming language")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    style: Optional[str] = Field(default="clean", description="Code style preference")


class CodeAnalysisRequest(BaseModel):
    """Request model for code analysis."""
    content: str = Field(..., description="Code to analyze", min_length=1, max_length=50000)
    language: str = Field(default="python", description="Programming language")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis")
    include_suggestions: bool = Field(default=True, description="Include improvement suggestions")


class AIResponse(BaseModel):
    """Standard AI response model."""
    success: bool = Field(..., description="Whether the request was successful")
    content: str = Field(..., description="Generated/processed content")
    provider: str = Field(..., description="AI provider used")
    model: str = Field(..., description="AI model used")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Token usage information")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class HealthResponse(BaseModel):
    """AI engine health response."""
    status: str = Field(..., description="Overall health status")
    providers: Dict[str, Any] = Field(..., description="Individual provider health")
    available_models: Dict[str, str] = Field(..., description="Available models by provider")


# Dependency to check if AI features are enabled
def check_ai_features_enabled():
    """Check if AI features are enabled."""
    if not any([
        settings.features.enable_code_completion,
        settings.features.enable_code_generation,
        settings.features.enable_code_analysis
    ]):
        raise HTTPException(
            status_code=503,
            detail="AI features are currently disabled"
        )


@router.post("/complete", 
             response_model=AIResponse,
             summary="Complete code",
             description="Get AI-powered code completion suggestions")
async def complete_code(
    request: CodeCompletionRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_ai_features_enabled)
):
    """Complete code using AI."""
    if not settings.features.enable_code_completion:
        raise HTTPException(
            status_code=503,
            detail="Code completion is currently disabled"
        )
    
    try:
        logger.info(f"Code completion request for {request.language}")
        
        # Process with AI engine
        response = await ai_engine.complete_code(
            content=request.content,
            language=request.language,
            context=request.context
        )
        
        # Log usage for analytics (background task)
        background_tasks.add_task(
            log_ai_usage,
            task_type=TaskType.CODE_COMPLETION,
            provider=response.provider.value,
            success=response.success,
            usage=response.usage
        )
        
        if not response.success:
            raise HTTPException(
                status_code=500,
                detail=f"Code completion failed: {response.error}"
            )
        
        return AIResponse(
            success=response.success,
            content=response.content,
            provider=response.provider.value,
            model=response.model,
            usage=response.usage,
            metadata={"task_type": "code_completion"},
            error=response.error
        )
        
    except Exception as e:
        logger.error(f"Code completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate",
             response_model=AIResponse,
             summary="Generate code",
             description="Generate code from natural language description")
async def generate_code(
    request: CodeGenerationRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_ai_features_enabled)
):
    """Generate code from description."""
    if not settings.features.enable_code_generation:
        raise HTTPException(
            status_code=503,
            detail="Code generation is currently disabled"
        )
    
    try:
        logger.info(f"Code generation request for {request.language}: {request.description[:100]}...")
        
        # Process with AI engine
        response = await ai_engine.generate_code(
            description=request.description,
            language=request.language,
            context=request.context
        )
        
        # Log usage
        background_tasks.add_task(
            log_ai_usage,
            task_type=TaskType.CODE_GENERATION,
            provider=response.provider.value,
            success=response.success,
            usage=response.usage
        )
        
        if not response.success:
            raise HTTPException(
                status_code=500,
                detail=f"Code generation failed: {response.error}"
            )
        
        return AIResponse(
            success=response.success,
            content=response.content,
            provider=response.provider.value,
            model=response.model,
            usage=response.usage,
            metadata={
                "task_type": "code_generation",
                "description_length": len(request.description),
                "style": request.style
            },
            error=response.error
        )
        
    except Exception as e:
        logger.error(f"Code generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze",
             response_model=AIResponse,
             summary="Analyze code",
             description="Analyze code for bugs, performance, and improvements")
async def analyze_code(
    request: CodeAnalysisRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_ai_features_enabled)
):
    """Analyze code for issues and improvements."""
    if not settings.features.enable_code_analysis:
        raise HTTPException(
            status_code=503,
            detail="Code analysis is currently disabled"
        )
    
    try:
        logger.info(f"Code analysis request for {request.language}")
        
        # Process with AI engine
        response = await ai_engine.analyze_code(
            content=request.content,
            language=request.language
        )
        
        # Log usage
        background_tasks.add_task(
            log_ai_usage,
            task_type=TaskType.CODE_ANALYSIS,
            provider=response.provider.value,
            success=response.success,
            usage=response.usage
        )
        
        if not response.success:
            raise HTTPException(
                status_code=500,
                detail=f"Code analysis failed: {response.error}"
            )
        
        return AIResponse(
            success=response.success,
            content=response.content,
            provider=response.provider.value,
            model=response.model,
            usage=response.usage,
            metadata={
                "task_type": "code_analysis",
                "analysis_type": request.analysis_type,
                "code_length": len(request.content)
            },
            error=response.error
        )
        
    except Exception as e:
        logger.error(f"Code analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refactor",
             response_model=AIResponse,
             summary="Refactor code",
             description="Refactor code for improved quality and performance")
async def refactor_code(
    content: str,
    language: str = "python",
    instructions: str = "Improve code quality and performance",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: None = Depends(check_ai_features_enabled)
):
    """Refactor code based on instructions."""
    try:
        logger.info(f"Code refactoring request for {language}")
        
        response = await ai_engine.refactor_code(
            content=content,
            language=language,
            instructions=instructions
        )
        
        background_tasks.add_task(
            log_ai_usage,
            task_type=TaskType.CODE_REFACTORING,
            provider=response.provider.value,
            success=response.success,
            usage=response.usage
        )
        
        if not response.success:
            raise HTTPException(
                status_code=500,
                detail=f"Code refactoring failed: {response.error}"
            )
        
        return AIResponse(
            success=response.success,
            content=response.content,
            provider=response.provider.value,
            model=response.model,
            usage=response.usage,
            metadata={
                "task_type": "code_refactoring",
                "instructions": instructions
            }
        )
        
    except Exception as e:
        logger.error(f"Code refactoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/document",
             response_model=AIResponse,
             summary="Generate documentation",
             description="Generate documentation for code")
async def document_code(
    content: str,
    language: str = "python",
    style: str = "comprehensive",
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Generate documentation for code."""
    try:
        logger.info(f"Documentation generation request for {language}")
        
        response = await ai_engine.generate_documentation(
            content=content,
            language=language
        )
        
        background_tasks.add_task(
            log_ai_usage,
            task_type=TaskType.DOCUMENTATION,
            provider=response.provider.value,
            success=response.success,
            usage=response.usage
        )
        
        if not response.success:
            raise HTTPException(
                status_code=500,
                detail=f"Documentation generation failed: {response.error}"
            )
        
        return AIResponse(
            success=response.success,
            content=response.content,
            provider=response.provider.value,
            model=response.model,
            usage=response.usage,
            metadata={
                "task_type": "documentation",
                "style": style
            }
        )
        
    except Exception as e:
        logger.error(f"Documentation generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health",
            response_model=HealthResponse,
            summary="AI engine health",
            description="Check the health status of all AI providers")
async def ai_health():
    """Get AI engine health status."""
    try:
        health_status = await ai_engine.health_check()
        
        # Determine overall status
        overall_status = "healthy"
        if not health_status:
            overall_status = "no_providers"
        elif all(provider.get("status") != "healthy" for provider in health_status.values()):
            overall_status = "unhealthy"
        elif any(provider.get("status") != "healthy" for provider in health_status.values()):
            overall_status = "degraded"
        
        # Get available models
        available_models = {}
        for provider_name, status in health_status.items():
            if status.get("status") == "healthy":
                available_models[provider_name] = status.get("model", "unknown")
        
        return HealthResponse(
            status=overall_status,
            providers=health_status,
            available_models=available_models
        )
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/providers",
            summary="List AI providers",
            description="Get list of available AI providers and their configuration")
async def list_providers():
    """List available AI providers."""
    try:
        available_providers = ai_engine.get_available_providers()
        
        provider_info = {}
        for provider in available_providers:
            config = settings.get_ai_provider_config(provider)
            provider_info[provider.value] = {
                "model": config.get("model", "unknown"),
                "available": True
            }
        
        return {
            "primary_provider": settings.ai_providers.primary_provider.value,
            "fallback_providers": [p.value for p in settings.ai_providers.fallback_providers],
            "available_providers": provider_info,
            "total_providers": len(available_providers)
        }
        
    except Exception as e:
        logger.error(f"Provider list error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def log_ai_usage(task_type: TaskType, provider: str, success: bool, usage: Dict[str, Any]):
    """Log AI usage for analytics (background task)."""
    if not settings.features.enable_analytics:
        return
    
    try:
        # Log usage statistics
        logger.info(f"AI Usage - Task: {task_type.value}, Provider: {provider}, Success: {success}, Usage: {usage}")
        
        # Here you could store to database, send to analytics service, etc.
        
    except Exception as e:
        logger.error(f"Failed to log AI usage: {str(e)}")