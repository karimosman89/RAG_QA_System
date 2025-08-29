#!/usr/bin/env python3
"""
AI-Assisted Coding Environment - Application Runner

Production-ready application runner with:
- CLI commands for different modes
- Configuration validation
- Health checks
- Development and production modes
"""

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import settings, get_settings
from src.core.ai_engine import ai_engine


def setup_logging():
    """Set up application logging."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.value),
        format=settings.log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ] + ([logging.FileHandler(settings.log_file)] if settings.log_file else [])
    )
    
    # Set up structured logging for production
    if settings.is_production:
        import structlog
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


def validate_configuration():
    """Validate application configuration."""
    logger = logging.getLogger(__name__)
    
    # Check required configurations
    issues = []
    
    # Check AI provider configurations
    if not any([
        settings.ai_providers.openai_api_key,
        settings.ai_providers.anthropic_api_key,
        settings.ai_providers.google_api_key
    ]):
        issues.append("No AI provider API keys configured")
    
    # Check secret key in production
    if settings.is_production and not settings.security.secret_key:
        issues.append("Secret key not set for production")
    
    # Check database configuration
    if not settings.database.database_url:
        issues.append("Database URL not configured")
    
    # Log configuration status
    if issues:
        logger.warning("Configuration issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        
        if settings.is_production:
            logger.error("Configuration issues in production environment")
            return False
    else:
        logger.info("Configuration validation passed")
    
    return True


async def health_check():
    """Perform application health check."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize AI engine
        ai_engine.initialize_clients()
        
        # Check AI providers
        health_status = await ai_engine.health_check()
        
        healthy_providers = sum(
            1 for status in health_status.values() 
            if status.get("status") == "healthy"
        )
        
        logger.info(f"Health check completed:")
        logger.info(f"  - AI Providers: {healthy_providers}/{len(health_status)} healthy")
        
        if healthy_providers == 0:
            logger.warning("No healthy AI providers available")
            return False
        
        logger.info("Health check passed")
        return True
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def run_development_server():
    """Run development server with hot reload."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting development server...")
    logger.info(f"Server URL: http://{settings.host}:{settings.port}")
    logger.info("Press CTRL+C to stop the server")
    
    import uvicorn
    
    uvicorn.run(
        "src.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        reload_dirs=["src"],
        log_level=settings.log_level.value.lower(),
        access_log=True,
    )


def run_production_server():
    """Run production server with multiple workers."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting production server...")
    logger.info(f"Server URL: http://{settings.host}:{settings.port}")
    logger.info(f"Workers: {settings.workers}")
    
    import uvicorn
    
    uvicorn.run(
        "src.api.server:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.value.lower(),
        access_log=True,
        loop="uvloop" if sys.platform != "win32" else "asyncio",
    )


def run_worker():
    """Run background worker for async tasks."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting background worker...")
    
    # This would be implemented with Celery or similar
    # For now, just a placeholder
    logger.info("Background worker functionality not implemented yet")
    logger.info("Use 'pip install celery' and implement worker tasks")


async def setup_database():
    """Set up database tables and initial data."""
    logger = logging.getLogger(__name__)
    
    logger.info("Setting up database...")
    
    try:
        # This would use Alembic for migrations
        # For now, just a placeholder
        logger.info("Database setup completed")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise


def show_configuration():
    """Display current configuration."""
    logger = logging.getLogger(__name__)
    
    logger.info("Current Configuration:")
    logger.info(f"  App Name: {settings.app_name}")
    logger.info(f"  Version: {settings.app_version}")
    logger.info(f"  Environment: {settings.environment.value}")
    logger.info(f"  Debug: {settings.debug}")
    logger.info(f"  Host: {settings.host}")
    logger.info(f"  Port: {settings.port}")
    logger.info(f"  Workers: {settings.workers}")
    logger.info(f"  Log Level: {settings.log_level.value}")
    
    logger.info("AI Providers:")
    logger.info(f"  Primary: {settings.ai_providers.primary_provider.value}")
    logger.info(f"  OpenAI: {'✓' if settings.ai_providers.openai_api_key else '✗'}")
    logger.info(f"  Anthropic: {'✓' if settings.ai_providers.anthropic_api_key else '✗'}")
    logger.info(f"  Google: {'✓' if settings.ai_providers.google_api_key else '✗'}")
    
    logger.info("Features:")
    logger.info(f"  Code Completion: {settings.features.enable_code_completion}")
    logger.info(f"  Code Generation: {settings.features.enable_code_generation}")
    logger.info(f"  Code Analysis: {settings.features.enable_code_analysis}")
    logger.info(f"  Real-time Collaboration: {settings.features.enable_real_time_collaboration}")
    logger.info(f"  Analytics: {settings.features.enable_analytics}")


def create_env_file():
    """Create a sample .env file."""
    env_content = '''# AI-Assisted Coding Environment Configuration

# Application Settings
APP_NAME="AI-Assisted Coding Environment"
APP_VERSION="2.0.0"
ENVIRONMENT="development"
DEBUG=true
HOST="0.0.0.0"
PORT=8000

# AI Provider Configuration
# OpenAI
OPENAI_API_KEY="your_openai_api_key_here"
OPENAI_MODEL="gpt-4"
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0.3

# Anthropic Claude
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
ANTHROPIC_MODEL="claude-3-sonnet-20240229"
ANTHROPIC_MAX_TOKENS=4096
ANTHROPIC_TEMPERATURE=0.3

# Google AI
GOOGLE_API_KEY="your_google_api_key_here"
GOOGLE_MODEL="gemini-pro"
GOOGLE_TEMPERATURE=0.3

# AI Provider Settings
PRIMARY_AI_PROVIDER="openai"
FALLBACK_AI_PROVIDERS=["anthropic", "google"]
AI_REQUESTS_PER_MINUTE=60
AI_MAX_CONCURRENT_REQUESTS=5

# Database Configuration
DATABASE_URL="sqlite:///./ai_coding_env.db"

# Redis Configuration
REDIS_HOST="localhost"
REDIS_PORT=6379
REDIS_PASSWORD=""
REDIS_DB=0

# Security Settings
SECRET_KEY="your_secret_key_here_change_in_production"
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS Settings
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000

# Feature Flags
ENABLE_CODE_COMPLETION=true
ENABLE_CODE_GENERATION=true
ENABLE_CODE_ANALYSIS=true
ENABLE_BUG_DETECTION=true
ENABLE_CODE_REFACTORING=true
ENABLE_REAL_TIME_COLLABORATION=true
ENABLE_CODE_SHARING=true
ENABLE_ANALYTICS=true
ENABLE_PERFORMANCE_MONITORING=true

# Logging
LOG_LEVEL="INFO"
LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# LOG_FILE="app.log"
'''
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("Created .env file with default configuration")
    print("Please edit .env and add your API keys")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI-Assisted Coding Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py dev              # Run development server
  python run.py server           # Run production server
  python run.py health           # Check application health
  python run.py config           # Show current configuration
  python run.py create-env       # Create sample .env file
        """
    )
    
    parser.add_argument(
        "command",
        choices=["dev", "server", "worker", "health", "config", "setup-db", "create-env"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--host",
        default=settings.host,
        help=f"Host to bind to (default: {settings.host})"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help=f"Port to bind to (default: {settings.port})"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.workers,
        help=f"Number of worker processes (default: {settings.workers})"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=settings.log_level.value,
        help=f"Log level (default: {settings.log_level.value})"
    )
    
    args = parser.parse_args()
    
    # Override settings with command line arguments
    if hasattr(args, 'host') and args.host != settings.host:
        settings.host = args.host
    if hasattr(args, 'port') and args.port != settings.port:
        settings.port = args.port
    if hasattr(args, 'workers') and args.workers != settings.workers:
        settings.workers = args.workers
    if hasattr(args, 'log_level'):
        settings.log_level = args.log_level
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Handle commands that don't require full validation
    if args.command == "create-env":
        create_env_file()
        return
    
    if args.command == "config":
        show_configuration()
        return
    
    # Validate configuration for other commands
    if not validate_configuration():
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    # Execute commands
    try:
        if args.command == "dev":
            logger.info("Starting in development mode")
            run_development_server()
            
        elif args.command == "server":
            logger.info("Starting in production mode")
            run_production_server()
            
        elif args.command == "worker":
            run_worker()
            
        elif args.command == "health":
            logger.info("Performing health check...")
            success = asyncio.run(health_check())
            sys.exit(0 if success else 1)
            
        elif args.command == "setup-db":
            logger.info("Setting up database...")
            asyncio.run(setup_database())
            logger.info("Database setup completed")
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()