"""
Application Configuration
=========================
Centralized configuration using Pydantic settings management.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # App Info
    APP_NAME: str = Field(default="TumorHeal API", description="Application name")
    DEBUG: bool = Field(default=False, description="Debug mode")
    VERSION: str = Field(default="2.0.0", description="API version")
    
    # Server
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", description="Secret key for JWT")
    ALLOWED_HOSTS: List[str] = Field(default=["*"], description="Allowed hosts for TrustedHostMiddleware")
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000"
        ],
        description="CORS allowed origins"
    )
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://user:password@localhost:5432/tumorheal",
        description="Database connection URL"
    )
    SUPABASE_URL: Optional[str] = Field(default=None, description="Supabase project URL")
    SUPABASE_KEY: Optional[str] = Field(default=None, description="Supabase anon key")
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    
    # JWT Settings
    JWT_SECRET_KEY: str = Field(default="jwt-secret-key-change-in-production", description="JWT secret")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token expiration")
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    
    # Payment Gateway Keys
    STRIPE_SECRET_KEY: Optional[str] = Field(default=None, description="Stripe secret key")
    STRIPE_PUBLISHABLE_KEY: Optional[str] = Field(default=None, description="Stripe publishable key")
    STRIPE_WEBHOOK_SECRET: Optional[str] = Field(default=None, description="Stripe webhook secret")
    
    PAYPAL_CLIENT_ID: Optional[str] = Field(default=None, description="PayPal client ID")
    PAYPAL_CLIENT_SECRET: Optional[str] = Field(default=None, description="PayPal client secret")
    PAYPAL_MODE: str = Field(default="sandbox", description="PayPal mode (sandbox/live)")
    PAYPAL_WEBHOOK_ID: Optional[str] = Field(default=None, description="PayPal webhook ID")
    
    # Quantum encryption
    QUANTUM_MASTER_KEY: str = Field(
        default="quantum-master-key-change-in-production",
        description="Master key for quantum encryption"
    )
    
    # M-Pesa (Safaricom)
    MPESA_CONSUMER_KEY: Optional[str] = Field(default=None, description="M-Pesa consumer key")
    MPESA_CONSUMER_SECRET: Optional[str] = Field(default=None, description="M-Pesa consumer secret")
    MPESA_SHORTCODE: Optional[str] = Field(default=None, description="M-Pesa business shortcode")
    MPESA_PASSKEY: Optional[str] = Field(default=None, description="M-Pesa Lipa Na M-Pesa passkey")
    MPESA_CALLBACK_URL: Optional[str] = Field(default=None, description="M-Pesa callback URL")
    
    # Email
    SMTP_HOST: str = Field(default="smtp.gmail.com", description="SMTP host")
    SMTP_PORT: int = Field(default=587, description="SMTP port")
    SMTP_USER: Optional[str] = Field(default=None, description="SMTP username")
    SMTP_PASSWORD: Optional[str] = Field(default=None, description="SMTP password")
    SMTP_FROM_EMAIL: str = Field(default="noreply@tumorheal.com", description="From email address")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Rate limit requests per window")
    RATE_LIMIT_WINDOW: int = Field(default=60, description="Rate limit window in seconds")
    
    # File Upload
    MAX_UPLOAD_SIZE_MB: int = Field(default=10, description="Max file upload size in MB")
    UPLOAD_DIR: str = Field(default="uploads", description="Upload directory")
    
    # AI/ML Settings
    ML_MODEL_PATH: str = Field(default="models", description="ML model directory")
    USE_GPU: bool = Field(default=False, description="Enable GPU acceleration")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
