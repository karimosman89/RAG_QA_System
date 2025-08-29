"""
Enterprise Configuration Management System

Centralized configuration for the AI-Assisted Coding Environment with support for:
- Multiple AI providers (OpenAI, Anthropic, Google)
- Database configurations
- Security settings
- Feature flags
- Environment-specific settings
"""

import os
import secrets
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from enum import Enum


class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AIProvider(str, Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    # Database URL and connection settings
    database_url: str = Field(
        default="sqlite:///./ai_coding_env.db",
        env="DATABASE_URL",
        description="Database connection URL"
    )
    
    # Connection pool settings
    pool_size: int = Field(default=5, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=10, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")
    
    # Query settings
    echo_sql: bool = Field(default=False, env="DB_ECHO_SQL")
    
    class Config:
        env_prefix = "DB_"


class RedisConfig(BaseSettings):
    """Redis configuration for caching and sessions."""
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    ssl: bool = Field(default=False, env="REDIS_SSL")
    
    # Connection settings
    max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")
    
    @property
    def url(self) -> str:
        """Generate Redis URL."""
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


class AIProviderConfig(BaseSettings):
    """AI provider configuration."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=4096, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.3, env="OPENAI_TEMPERATURE")
    openai_timeout: int = Field(default=60, env="OPENAI_TIMEOUT")
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    anthropic_max_tokens: int = Field(default=4096, env="ANTHROPIC_MAX_TOKENS")
    anthropic_temperature: float = Field(default=0.3, env="ANTHROPIC_TEMPERATURE")
    
    # Google AI Configuration
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    google_model: str = Field(default="gemini-pro", env="GOOGLE_MODEL")
    google_temperature: float = Field(default=0.3, env="GOOGLE_TEMPERATURE")
    
    # Provider priority and fallback
    primary_provider: AIProvider = Field(default=AIProvider.OPENAI, env="PRIMARY_AI_PROVIDER")
    fallback_providers: List[AIProvider] = Field(
        default=[AIProvider.ANTHROPIC, AIProvider.GOOGLE],
        env="FALLBACK_AI_PROVIDERS"
    )
    
    # Rate limiting
    requests_per_minute: int = Field(default=60, env="AI_REQUESTS_PER_MINUTE")
    max_concurrent_requests: int = Field(default=5, env="AI_MAX_CONCURRENT_REQUESTS")


class SecurityConfig(BaseSettings):
    """Security configuration."""
    
    # JWT Configuration
    secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        env="SECRET_KEY"
    )
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    
    # API Security
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        env="CORS_ALLOW_METHODS"
    )
    cors_allow_headers: List[str] = Field(
        default=["*"],
        env="CORS_ALLOW_HEADERS"
    )
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")


class FeatureFlags(BaseSettings):
    """Feature flag configuration."""
    
    # AI Features
    enable_code_completion: bool = Field(default=True, env="ENABLE_CODE_COMPLETION")
    enable_code_generation: bool = Field(default=True, env="ENABLE_CODE_GENERATION")
    enable_code_analysis: bool = Field(default=True, env="ENABLE_CODE_ANALYSIS")
    enable_bug_detection: bool = Field(default=True, env="ENABLE_BUG_DETECTION")
    enable_code_refactoring: bool = Field(default=True, env="ENABLE_CODE_REFACTORING")
    
    # Collaboration Features
    enable_real_time_collaboration: bool = Field(default=True, env="ENABLE_REAL_TIME_COLLABORATION")
    enable_code_sharing: bool = Field(default=True, env="ENABLE_CODE_SHARING")
    enable_project_templates: bool = Field(default=True, env="ENABLE_PROJECT_TEMPLATES")
    
    # Analytics and Monitoring
    enable_analytics: bool = Field(default=True, env="ENABLE_ANALYTICS")
    enable_performance_monitoring: bool = Field(default=True, env="ENABLE_PERFORMANCE_MONITORING")
    enable_error_tracking: bool = Field(default=True, env="ENABLE_ERROR_TRACKING")
    
    # Enterprise Features
    enable_sso: bool = Field(default=False, env="ENABLE_SSO")
    enable_audit_logging: bool = Field(default=True, env="ENABLE_AUDIT_LOGGING")
    enable_advanced_permissions: bool = Field(default=False, env="ENABLE_ADVANCED_PERMISSIONS")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application Info
    app_name: str = Field(default="AI-Assisted Coding Environment", env="APP_NAME")
    app_version: str = Field(default="2.0.0", env="APP_VERSION")
    description: str = Field(
        default="Enterprise-grade AI-powered development environment",
        env="APP_DESCRIPTION"
    )
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    reload: bool = Field(default=False, env="RELOAD")
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Paths
    static_dir: str = Field(default="static", env="STATIC_DIR")
    templates_dir: str = Field(default="src/templates", env="TEMPLATES_DIR")
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    ai_providers: AIProviderConfig = Field(default_factory=AIProviderConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        """Validate environment setting."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator("log_level", pre=True)
    def validate_log_level(cls, v):
        """Validate log level setting."""
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing."""
        return self.environment == Environment.TESTING
    
    def get_ai_provider_config(self, provider: AIProvider) -> Dict[str, Any]:
        """Get configuration for specific AI provider."""
        configs = {
            AIProvider.OPENAI: {
                "api_key": self.ai_providers.openai_api_key,
                "model": self.ai_providers.openai_model,
                "max_tokens": self.ai_providers.openai_max_tokens,
                "temperature": self.ai_providers.openai_temperature,
                "timeout": self.ai_providers.openai_timeout,
            },
            AIProvider.ANTHROPIC: {
                "api_key": self.ai_providers.anthropic_api_key,
                "model": self.ai_providers.anthropic_model,
                "max_tokens": self.ai_providers.anthropic_max_tokens,
                "temperature": self.ai_providers.anthropic_temperature,
            },
            AIProvider.GOOGLE: {
                "api_key": self.ai_providers.google_api_key,
                "model": self.ai_providers.google_model,
                "temperature": self.ai_providers.google_temperature,
            },
        }
        return configs.get(provider, {})
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            """Customize settings sources priority."""
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )


# Global settings instance
settings = Settings()

# Convenience functions
def get_settings() -> Settings:
    """Get application settings."""
    return settings

def get_database_url() -> str:
    """Get database URL."""
    return settings.database.database_url

def get_redis_url() -> str:
    """Get Redis URL."""
    return settings.redis.url

def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled."""
    return getattr(settings.features, f"enable_{feature}", False)

def get_ai_config(provider: AIProvider = None) -> Dict[str, Any]:
    """Get AI provider configuration."""
    if provider is None:
        provider = settings.ai_providers.primary_provider
    return settings.get_ai_provider_config(provider)