"""
Advanced AI Engine for Code Processing

Multi-provider AI engine supporting OpenAI, Anthropic, and Google AI for:
- Code completion and suggestions
- Code generation from natural language
- Code analysis and bug detection
- Code refactoring and optimization
- Documentation generation
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import openai
import anthropic
import google.generativeai as genai
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from .config import settings, AIProvider, get_ai_config


logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of AI tasks."""
    CODE_COMPLETION = "code_completion"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    BUG_DETECTION = "bug_detection"
    CODE_REFACTORING = "code_refactoring"
    DOCUMENTATION = "documentation"
    CODE_REVIEW = "code_review"
    TEST_GENERATION = "test_generation"


@dataclass
class AIRequest:
    """AI request structure."""
    task_type: TaskType
    content: str
    language: str = "python"
    context: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class AIResponse:
    """AI response structure."""
    success: bool
    content: str
    provider: AIProvider
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RateLimiter:
    """Simple rate limiter for AI requests."""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
    
    async def acquire(self) -> bool:
        """Acquire a rate limit slot."""
        now = time.time()
        
        # Remove old requests outside the window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.window_seconds]
        
        # Check if we can make a request
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def time_until_next_request(self) -> float:
        """Time until next request is allowed."""
        if not self.requests:
            return 0.0
        
        oldest_request = min(self.requests)
        return max(0.0, self.window_seconds - (time.time() - oldest_request))


class OpenAIClient:
    """OpenAI API client wrapper."""
    
    def __init__(self, api_key: str, model: str = "gpt-4", **kwargs):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.config = kwargs
        self.rate_limiter = RateLimiter(
            max_requests=settings.ai_providers.requests_per_minute
        )
    
    async def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> AIResponse:
        """Make a request to OpenAI API."""
        try:
            # Rate limiting
            if not await self.rate_limiter.acquire():
                wait_time = self.rate_limiter.time_until_next_request()
                logger.warning(f"Rate limit hit, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                await self.rate_limiter.acquire()
            
            # Merge configuration
            request_config = {**self.config, **kwargs}
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **request_config
            )
            
            content = response.choices[0].message.content
            usage = response.usage.model_dump() if response.usage else None
            
            return AIResponse(
                success=True,
                content=content,
                provider=AIProvider.OPENAI,
                model=self.model,
                usage=usage
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return AIResponse(
                success=False,
                content="",
                provider=AIProvider.OPENAI,
                model=self.model,
                error=str(e)
            )
    
    async def complete_code(self, request: AIRequest) -> AIResponse:
        """Complete code using OpenAI."""
        messages = [
            {
                "role": "system",
                "content": f"You are an expert {request.language} programmer. Complete the given code naturally and efficiently. Only provide the completion, no explanations."
            },
            {
                "role": "user",
                "content": f"Complete this {request.language} code:\n\n{request.content}"
            }
        ]
        
        return await self._make_request(
            messages, 
            temperature=0.2,
            max_tokens=500
        )
    
    async def generate_code(self, request: AIRequest) -> AIResponse:
        """Generate code from description."""
        context_info = ""
        if request.context:
            context_info = f"\nContext: {json.dumps(request.context, indent=2)}"
        
        messages = [
            {
                "role": "system",
                "content": f"You are an expert {request.language} programmer. Generate clean, efficient, well-documented code with proper error handling."
            },
            {
                "role": "user",
                "content": f"Generate {request.language} code for: {request.content}{context_info}"
            }
        ]
        
        return await self._make_request(
            messages,
            temperature=0.3,
            max_tokens=2000
        )
    
    async def analyze_code(self, request: AIRequest) -> AIResponse:
        """Analyze code for issues and improvements."""
        messages = [
            {
                "role": "system",
                "content": f"You are a code review expert. Analyze the provided {request.language} code for bugs, performance issues, security vulnerabilities, and suggest improvements."
            },
            {
                "role": "user",
                "content": f"Analyze this {request.language} code:\n\n{request.content}"
            }
        ]
        
        return await self._make_request(
            messages,
            temperature=0.1,
            max_tokens=1500
        )


class AnthropicClient:
    """Anthropic Claude API client wrapper."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", **kwargs):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.config = kwargs
        self.rate_limiter = RateLimiter(
            max_requests=settings.ai_providers.requests_per_minute
        )
    
    async def _make_request(self, system: str, user_message: str, **kwargs) -> AIResponse:
        """Make a request to Anthropic API."""
        try:
            # Rate limiting
            if not await self.rate_limiter.acquire():
                wait_time = self.rate_limiter.time_until_next_request()
                await asyncio.sleep(wait_time)
                await self.rate_limiter.acquire()
            
            request_config = {**self.config, **kwargs}
            
            response = await self.client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user_message}],
                **request_config
            )
            
            content = response.content[0].text
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            
            return AIResponse(
                success=True,
                content=content,
                provider=AIProvider.ANTHROPIC,
                model=self.model,
                usage=usage
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            return AIResponse(
                success=False,
                content="",
                provider=AIProvider.ANTHROPIC,
                model=self.model,
                error=str(e)
            )
    
    async def complete_code(self, request: AIRequest) -> AIResponse:
        """Complete code using Claude."""
        system = f"You are an expert {request.language} programmer. Complete the given code naturally and efficiently. Only provide the completion, no explanations."
        user_message = f"Complete this {request.language} code:\n\n{request.content}"
        
        return await self._make_request(
            system, user_message,
            max_tokens=500,
            temperature=0.2
        )
    
    async def generate_code(self, request: AIRequest) -> AIResponse:
        """Generate code from description."""
        context_info = ""
        if request.context:
            context_info = f"\nContext: {json.dumps(request.context, indent=2)}"
        
        system = f"You are an expert {request.language} programmer. Generate clean, efficient, well-documented code with proper error handling."
        user_message = f"Generate {request.language} code for: {request.content}{context_info}"
        
        return await self._make_request(
            system, user_message,
            max_tokens=2000,
            temperature=0.3
        )
    
    async def analyze_code(self, request: AIRequest) -> AIResponse:
        """Analyze code for issues and improvements."""
        system = f"You are a code review expert. Analyze the provided {request.language} code for bugs, performance issues, security vulnerabilities, and suggest improvements."
        user_message = f"Analyze this {request.language} code:\n\n{request.content}"
        
        return await self._make_request(
            system, user_message,
            max_tokens=1500,
            temperature=0.1
        )


class GoogleAIClient:
    """Google Generative AI client wrapper."""
    
    def __init__(self, api_key: str, model: str = "gemini-pro", **kwargs):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)
        self.config = kwargs
        self.rate_limiter = RateLimiter(
            max_requests=settings.ai_providers.requests_per_minute
        )
    
    async def _make_request(self, prompt: str, **kwargs) -> AIResponse:
        """Make a request to Google AI API."""
        try:
            # Rate limiting
            if not await self.rate_limiter.acquire():
                wait_time = self.rate_limiter.time_until_next_request()
                await asyncio.sleep(wait_time)
                await self.rate_limiter.acquire()
            
            # Generate content
            response = await asyncio.to_thread(
                self.model.generate_content, 
                prompt,
                generation_config=genai.types.GenerationConfig(**{**self.config, **kwargs})
            )
            
            content = response.text
            
            return AIResponse(
                success=True,
                content=content,
                provider=AIProvider.GOOGLE,
                model=self.model_name,
                usage={"input_tokens": len(prompt.split()), "output_tokens": len(content.split())}
            )
            
        except Exception as e:
            logger.error(f"Google AI API error: {str(e)}")
            return AIResponse(
                success=False,
                content="",
                provider=AIProvider.GOOGLE,
                model=self.model_name,
                error=str(e)
            )
    
    async def complete_code(self, request: AIRequest) -> AIResponse:
        """Complete code using Gemini."""
        prompt = f"""You are an expert {request.language} programmer. Complete the given code naturally and efficiently. Only provide the completion, no explanations.

Complete this {request.language} code:

{request.content}"""
        
        return await self._make_request(
            prompt,
            temperature=0.2,
            max_output_tokens=500
        )
    
    async def generate_code(self, request: AIRequest) -> AIResponse:
        """Generate code from description."""
        context_info = ""
        if request.context:
            context_info = f"\nContext: {json.dumps(request.context, indent=2)}"
        
        prompt = f"""You are an expert {request.language} programmer. Generate clean, efficient, well-documented code with proper error handling.

Generate {request.language} code for: {request.content}{context_info}"""
        
        return await self._make_request(
            prompt,
            temperature=0.3,
            max_output_tokens=2000
        )
    
    async def analyze_code(self, request: AIRequest) -> AIResponse:
        """Analyze code for issues and improvements."""
        prompt = f"""You are a code review expert. Analyze the provided {request.language} code for bugs, performance issues, security vulnerabilities, and suggest improvements.

Analyze this {request.language} code:

{request.content}"""
        
        return await self._make_request(
            prompt,
            temperature=0.1,
            max_output_tokens=1500
        )


class AIEngine:
    """Main AI engine with multi-provider support and intelligent fallback."""
    
    def __init__(self):
        self.clients: Dict[AIProvider, Union[OpenAIClient, AnthropicClient, GoogleAIClient]] = {}
        self.initialize_clients()
    
    def initialize_clients(self):
        """Initialize all available AI clients."""
        # OpenAI
        if settings.ai_providers.openai_api_key:
            try:
                self.clients[AIProvider.OPENAI] = OpenAIClient(
                    api_key=settings.ai_providers.openai_api_key,
                    model=settings.ai_providers.openai_model,
                    temperature=settings.ai_providers.openai_temperature,
                    max_tokens=settings.ai_providers.openai_max_tokens,
                    timeout=settings.ai_providers.openai_timeout
                )
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        # Anthropic
        if settings.ai_providers.anthropic_api_key:
            try:
                self.clients[AIProvider.ANTHROPIC] = AnthropicClient(
                    api_key=settings.ai_providers.anthropic_api_key,
                    model=settings.ai_providers.anthropic_model,
                    temperature=settings.ai_providers.anthropic_temperature,
                    max_tokens=settings.ai_providers.anthropic_max_tokens
                )
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
        
        # Google AI
        if settings.ai_providers.google_api_key:
            try:
                self.clients[AIProvider.GOOGLE] = GoogleAIClient(
                    api_key=settings.ai_providers.google_api_key,
                    model=settings.ai_providers.google_model,
                    temperature=settings.ai_providers.google_temperature
                )
                logger.info("Google AI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google AI client: {e}")
    
    def get_available_providers(self) -> List[AIProvider]:
        """Get list of available providers."""
        return list(self.clients.keys())
    
    async def _try_providers(self, request: AIRequest, task_method: str) -> AIResponse:
        """Try providers in order with fallback."""
        # Get provider order (primary + fallbacks)
        provider_order = [settings.ai_providers.primary_provider]
        for provider in settings.ai_providers.fallback_providers:
            if provider not in provider_order:
                provider_order.append(provider)
        
        # Try each provider
        last_error = None
        for provider in provider_order:
            if provider not in self.clients:
                continue
            
            try:
                client = self.clients[provider]
                method = getattr(client, task_method)
                response = await method(request)
                
                if response.success:
                    logger.info(f"Successfully processed request with {provider}")
                    return response
                else:
                    logger.warning(f"Provider {provider} failed: {response.error}")
                    last_error = response.error
                    
            except Exception as e:
                logger.error(f"Error with provider {provider}: {e}")
                last_error = str(e)
                continue
        
        # All providers failed
        return AIResponse(
            success=False,
            content="",
            provider=AIProvider.OPENAI,  # Default
            model="unknown",
            error=f"All providers failed. Last error: {last_error}"
        )
    
    async def complete_code(self, content: str, language: str = "python", 
                          context: Optional[Dict[str, Any]] = None) -> AIResponse:
        """Complete code with intelligent fallback."""
        request = AIRequest(
            task_type=TaskType.CODE_COMPLETION,
            content=content,
            language=language,
            context=context
        )
        
        return await self._try_providers(request, "complete_code")
    
    async def generate_code(self, description: str, language: str = "python",
                          context: Optional[Dict[str, Any]] = None) -> AIResponse:
        """Generate code from description."""
        request = AIRequest(
            task_type=TaskType.CODE_GENERATION,
            content=description,
            language=language,
            context=context
        )
        
        return await self._try_providers(request, "generate_code")
    
    async def analyze_code(self, content: str, language: str = "python",
                         context: Optional[Dict[str, Any]] = None) -> AIResponse:
        """Analyze code for issues and improvements."""
        request = AIRequest(
            task_type=TaskType.CODE_ANALYSIS,
            content=content,
            language=language,
            context=context
        )
        
        return await self._try_providers(request, "analyze_code")
    
    async def generate_documentation(self, content: str, language: str = "python") -> AIResponse:
        """Generate documentation for code."""
        request = AIRequest(
            task_type=TaskType.DOCUMENTATION,
            content=f"Generate comprehensive documentation for this {language} code:\n\n{content}",
            language=language
        )
        
        return await self._try_providers(request, "generate_code")
    
    async def generate_tests(self, content: str, language: str = "python") -> AIResponse:
        """Generate unit tests for code."""
        request = AIRequest(
            task_type=TaskType.TEST_GENERATION,
            content=f"Generate comprehensive unit tests for this {language} code:\n\n{content}",
            language=language
        )
        
        return await self._try_providers(request, "generate_code")
    
    async def refactor_code(self, content: str, language: str = "python", 
                          instructions: str = "Improve code quality and performance") -> AIResponse:
        """Refactor code based on instructions."""
        request = AIRequest(
            task_type=TaskType.CODE_REFACTORING,
            content=f"Refactor this {language} code following these instructions: {instructions}\n\nCode:\n{content}",
            language=language
        )
        
        return await self._try_providers(request, "generate_code")
    
    async def review_code(self, content: str, language: str = "python") -> AIResponse:
        """Perform detailed code review."""
        request = AIRequest(
            task_type=TaskType.CODE_REVIEW,
            content=content,
            language=language
        )
        
        return await self._try_providers(request, "analyze_code")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all AI providers."""
        health_status = {}
        
        for provider, client in self.clients.items():
            try:
                # Simple test request
                test_request = AIRequest(
                    task_type=TaskType.CODE_COMPLETION,
                    content="print('hello')",
                    language="python"
                )
                
                start_time = time.time()
                response = await client.complete_code(test_request)
                response_time = time.time() - start_time
                
                health_status[provider.value] = {
                    "status": "healthy" if response.success else "unhealthy",
                    "response_time": response_time,
                    "model": response.model,
                    "error": response.error
                }
                
            except Exception as e:
                health_status[provider.value] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return health_status


# Global AI engine instance
ai_engine = AIEngine()

# Convenience functions
async def complete_code(content: str, language: str = "python") -> AIResponse:
    """Complete code using the AI engine."""
    return await ai_engine.complete_code(content, language)

async def generate_code(description: str, language: str = "python") -> AIResponse:
    """Generate code using the AI engine."""
    return await ai_engine.generate_code(description, language)

async def analyze_code(content: str, language: str = "python") -> AIResponse:
    """Analyze code using the AI engine."""
    return await ai_engine.analyze_code(content, language)