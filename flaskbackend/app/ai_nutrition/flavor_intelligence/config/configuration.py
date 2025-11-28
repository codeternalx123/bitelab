"""
Configuration System for Flavor Intelligence Pipeline
===================================================

This module provides comprehensive configuration management for the entire
Automated Flavor Intelligence Pipeline. It handles environment-specific
settings, model parameters, database connections, API configurations,
and deployment settings.

Key Features:
- Hierarchical configuration management (defaults, environment, runtime)
- Environment-specific configuration profiles (dev, staging, production)
- Configuration validation and type checking
- Secrets management and encryption
- Dynamic configuration updates
- Configuration templates and generators
- Monitoring and logging configuration
- Model hyperparameter management
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Union, Any, Type, get_type_hints
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
import logging
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import base64
from cryptography.fernet import Fernet
import tempfile
import shutil
from urllib.parse import urlparse
import re

# Configuration validation
from pydantic import BaseModel, Field, validator, root_validator
import marshmallow as ma
from marshmallow import Schema, fields as ma_fields, validate, post_load

# Environment and system
import socket
import platform
import psutil


class Environment(str, Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """Database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    NEO4J = "neo4j"
    REDIS = "redis"
    MONGODB = "mongodb"


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    
    # Connection details
    host: str = "localhost"
    port: int = 5432
    database: str = "flavordb"
    username: str = "user"
    password: str = "password"
    
    # Connection pool settings
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: int = 30
    pool_timeout: int = 30
    
    # SSL settings
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    # Query settings
    query_timeout: int = 300
    statement_timeout: int = 600
    
    # Additional options
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def get_connection_url(self, db_type: DatabaseType = DatabaseType.POSTGRESQL) -> str:
        """Generate connection URL"""
        
        if db_type == DatabaseType.POSTGRESQL:
            base_url = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif db_type == DatabaseType.MYSQL:
            base_url = f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif db_type == DatabaseType.SQLITE:
            base_url = f"sqlite:///{self.database}"
        elif db_type == DatabaseType.NEO4J:
            base_url = f"bolt://{self.host}:{self.port}"
        elif db_type == DatabaseType.REDIS:
            base_url = f"redis://{self.host}:{self.port}"
        elif db_type == DatabaseType.MONGODB:
            base_url = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        # Add additional parameters
        if self.additional_params:
            params = "&".join([f"{k}={v}" for k, v in self.additional_params.items()])
            base_url += f"?{params}"
        
        return base_url


@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    
    # Model architecture
    model_type: str = "multimodal_encoder"
    embedding_dim: int = 512
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
    num_attention_heads: int = 8
    num_transformer_layers: int = 6
    dropout_rate: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Loss function weights
    sensory_loss_weight: float = 1.0
    similarity_loss_weight: float = 0.5
    classification_loss_weight: float = 0.3
    reconstruction_loss_weight: float = 0.2
    
    # Data processing
    max_sequence_length: int = 128
    num_negative_samples: int = 5
    augmentation_probability: float = 0.3
    validation_split: float = 0.2
    
    # Model paths
    model_save_dir: str = "models"
    checkpoint_dir: str = "checkpoints"
    pretrained_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Hardware settings
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    num_workers: int = 4
    
    # Monitoring
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 5
    early_stopping_patience: int = 15
    monitor_metric: str = "val_loss"
    
    def validate_config(self) -> bool:
        """Validate model configuration"""
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        return True


@dataclass
class APIConfig:
    """API service configuration"""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    worker_class: str = "uvicorn.workers.UvicornWorker"
    
    # Security
    enable_auth: bool = False
    secret_key: str = "your-secret-key-here"
    api_keys: List[str] = field(default_factory=list)
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst: int = 10
    
    # Request/response settings
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 300
    response_compression: bool = True
    
    # Monitoring and logging
    enable_metrics: bool = True
    metrics_port: int = 9090
    access_log: bool = True
    log_level: LogLevel = LogLevel.INFO
    
    # Documentation
    docs_enabled: bool = True
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"


@dataclass
class ScrapingConfig:
    """Data scraping configuration"""
    
    # Rate limiting
    requests_per_second: float = 5.0
    concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    backoff_factor: float = 2.0
    
    # Data quality
    min_ingredients_per_recipe: int = 3
    max_ingredients_per_recipe: int = 50
    min_nutrition_completeness: float = 0.3
    validate_data_quality: bool = True
    
    # File handling
    chunk_size: int = 1000
    compression_enabled: bool = True
    cache_duration_hours: int = 24
    output_directory: str = "data/scraped"
    
    # API keys and credentials
    usda_api_key: Optional[str] = None
    spoonacular_api_key: Optional[str] = None
    openfoodfacts_user_agent: str = "FlavorIntelligence/1.0"
    
    # Data sources
    enabled_sources: List[str] = field(default_factory=lambda: [
        "openfoodfacts", "usda_fdc", "flavordb", "recipe1m"
    ])
    
    # Processing options
    filter_duplicate_entries: bool = True
    normalize_ingredient_names: bool = True
    backup_enabled: bool = True
    progress_reporting: bool = True


@dataclass
class CacheConfig:
    """Caching configuration"""
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Cache behavior
    default_ttl_seconds: int = 3600
    max_memory_mb: int = 1024
    eviction_policy: str = "allkeys-lru"
    
    # Cache keys
    key_prefix: str = "flavor_intel"
    key_separator: str = ":"
    
    # Cache regions
    profile_cache_ttl: int = 7200
    similarity_cache_ttl: int = 3600
    recipe_cache_ttl: int = 1800
    graph_query_cache_ttl: int = 900


@dataclass
class LoggingConfig:
    """Logging configuration"""
    
    # Basic settings
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Output settings
    console_enabled: bool = True
    file_enabled: bool = True
    file_path: str = "logs/flavor_intelligence.log"
    
    # Rotation settings
    max_file_size: str = "10MB"
    backup_count: int = 5
    rotation_time: str = "midnight"
    
    # Structured logging
    structured_logging: bool = True
    json_format: bool = False
    
    # Logger-specific levels
    logger_levels: Dict[str, str] = field(default_factory=lambda: {
        "uvicorn": "INFO",
        "sqlalchemy": "WARNING",
        "neo4j": "WARNING",
        "transformers": "ERROR"
    })
    
    # Performance logging
    slow_query_threshold: float = 1.0
    log_sql_queries: bool = False


@dataclass 
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    
    # Prometheus metrics
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"
    
    # Health checks
    health_check_enabled: bool = True
    health_check_path: str = "/health"
    health_check_interval: int = 30
    
    # Performance metrics
    response_time_buckets: List[float] = field(default_factory=lambda: [
        0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])
    
    # Alerting
    alert_enabled: bool = False
    alert_webhook_url: Optional[str] = None
    error_rate_threshold: float = 0.05
    response_time_threshold: float = 2.0
    
    # System monitoring
    monitor_system_resources: bool = True
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0


@dataclass
class SecurityConfig:
    """Security configuration"""
    
    # Authentication
    auth_enabled: bool = False
    auth_type: str = "bearer"  # bearer, basic, oauth
    jwt_secret: str = "your-jwt-secret"
    jwt_expiry_hours: int = 24
    
    # API keys
    api_key_header: str = "X-API-Key"
    api_keys: List[str] = field(default_factory=list)
    
    # Encryption
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None
    
    # Request validation
    validate_requests: bool = True
    max_request_size: int = 10 * 1024 * 1024
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 10


@dataclass
class FlavorIntelligenceConfig:
    """Main configuration class for the entire system"""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    version: str = "1.0.0"
    
    # Database configurations
    postgres: DatabaseConfig = field(default_factory=DatabaseConfig)
    neo4j: DatabaseConfig = field(default_factory=lambda: DatabaseConfig(port=7687))
    redis: DatabaseConfig = field(default_factory=lambda: DatabaseConfig(port=6379))
    
    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Data directories
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    cache_dir: str = "cache"
    
    # Feature flags
    enable_ml_inference: bool = True
    enable_graph_queries: bool = True
    enable_batch_processing: bool = True
    enable_data_scraping: bool = True
    
    # Performance settings
    max_workers: int = 4
    memory_limit_gb: int = 8
    cpu_limit_percent: int = 80
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._validate_config()
        self._create_directories()
        self._setup_logging()
    
    def _validate_config(self):
        """Validate configuration values"""
        
        # Validate model configuration
        self.model.validate_config()
        
        # Validate directories
        for dir_attr in ['data_dir', 'models_dir', 'logs_dir', 'cache_dir']:
            dir_path = getattr(self, dir_attr)
            if not isinstance(dir_path, str) or not dir_path:
                raise ValueError(f"{dir_attr} must be a non-empty string")
        
        # Validate resource limits
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        
        if self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")
        
        if not 0 < self.cpu_limit_percent <= 100:
            raise ValueError("cpu_limit_percent must be between 1 and 100")
    
    def _create_directories(self):
        """Create necessary directories"""
        for dir_attr in ['data_dir', 'models_dir', 'logs_dir', 'cache_dir']:
            dir_path = Path(getattr(self, dir_attr))
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        logging.basicConfig(
            level=getattr(logging, self.logging.level.value),
            format=self.logging.format,
            datefmt=self.logging.date_format
        )
    
    def get_database_url(self, db_name: str) -> str:
        """Get database connection URL"""
        if db_name == "postgres":
            return self.postgres.get_connection_url(DatabaseType.POSTGRESQL)
        elif db_name == "neo4j":
            return self.neo4j.get_connection_url(DatabaseType.NEO4J)
        elif db_name == "redis":
            return self.redis.get_connection_url(DatabaseType.REDIS)
        else:
            raise ValueError(f"Unknown database: {db_name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to file"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'w') as f:
                yaml.dump(self.to_dict(), f, indent=2, default_flow_style=False)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml")


class ConfigurationManager:
    """Configuration management system"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.encryption_key = None
        
        # Configuration cache
        self._config_cache = {}
        self._config_templates = {}
        
        # Initialize encryption if needed
        self._setup_encryption()
    
    def _setup_encryption(self):
        """Setup configuration encryption"""
        key_file = self.config_dir / ".encryption_key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                self.encryption_key = f.read()
        else:
            # Generate new encryption key
            self.encryption_key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(self.encryption_key)
            
            # Set restrictive permissions
            key_file.chmod(0o600)
    
    def create_default_config(self, environment: Environment = Environment.DEVELOPMENT) -> FlavorIntelligenceConfig:
        """Create default configuration for environment"""
        
        config = FlavorIntelligenceConfig(environment=environment)
        
        # Environment-specific adjustments
        if environment == Environment.DEVELOPMENT:
            config.debug = True
            config.logging.level = LogLevel.DEBUG
            config.api.reload = True
            config.monitoring.prometheus_enabled = False
            
        elif environment == Environment.TESTING:
            config.debug = True
            config.logging.level = LogLevel.DEBUG
            config.postgres.database = "flavordb_test"
            config.cache.redis_db = 1
            
        elif environment == Environment.STAGING:
            config.debug = False
            config.logging.level = LogLevel.INFO
            config.security.auth_enabled = True
            config.monitoring.alert_enabled = True
            
        elif environment == Environment.PRODUCTION:
            config.debug = False
            config.logging.level = LogLevel.WARNING
            config.security.auth_enabled = True
            config.security.cors_origins = ["https://yourdomain.com"]
            config.monitoring.alert_enabled = True
            config.api.workers = 4
        
        return config
    
    def load_config(self, config_path: Union[str, Path], 
                   environment: Optional[Environment] = None) -> FlavorIntelligenceConfig:
        """Load configuration from file"""
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load base configuration
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported configuration file format")
        
        # Apply environment-specific overrides
        if environment:
            env_config_path = config_path.parent / f"{config_path.stem}_{environment.value}{config_path.suffix}"
            if env_config_path.exists():
                self.logger.info(f"Loading environment-specific config: {env_config_path}")
                
                if env_config_path.suffix.lower() == '.json':
                    with open(env_config_path, 'r') as f:
                        env_config_dict = json.load(f)
                else:
                    with open(env_config_path, 'r') as f:
                        env_config_dict = yaml.safe_load(f)
                
                # Merge configurations
                config_dict = self._deep_merge_dicts(config_dict, env_config_dict)
        
        # Apply environment variable overrides
        config_dict = self._apply_env_overrides(config_dict)
        
        # Convert to configuration object
        config = self._dict_to_config(config_dict)
        
        # Cache the configuration
        self._config_cache[str(config_path)] = config
        
        return config
    
    def save_config(self, config: FlavorIntelligenceConfig, 
                   file_path: Union[str, Path], encrypt: bool = False):
        """Save configuration to file"""
        
        config_dict = config.to_dict()
        
        if encrypt:
            config_dict = self._encrypt_sensitive_data(config_dict)
        
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, indent=2, default_flow_style=False)
        else:
            raise ValueError("Unsupported file format")
        
        self.logger.info(f"Configuration saved to: {file_path}")
    
    def create_config_template(self, template_name: str, 
                             base_config: FlavorIntelligenceConfig):
        """Create configuration template"""
        
        template_path = self.config_dir / f"template_{template_name}.yaml"
        
        # Create template with placeholders
        config_dict = base_config.to_dict()
        template_dict = self._create_template_placeholders(config_dict)
        
        with open(template_path, 'w') as f:
            yaml.dump(template_dict, f, indent=2, default_flow_style=False)
        
        self._config_templates[template_name] = template_dict
        
        self.logger.info(f"Configuration template created: {template_path}")
    
    def load_from_template(self, template_name: str, 
                          variables: Dict[str, Any]) -> FlavorIntelligenceConfig:
        """Load configuration from template with variables"""
        
        template_path = self.config_dir / f"template_{template_name}.yaml"
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Replace variables in template
        for key, value in variables.items():
            template_content = template_content.replace(f"${{{key}}}", str(value))
        
        # Parse the processed template
        config_dict = yaml.safe_load(template_content)
        
        return self._dict_to_config(config_dict)
    
    def validate_config(self, config: FlavorIntelligenceConfig) -> List[str]:
        """Validate configuration and return list of issues"""
        
        issues = []
        
        try:
            # Test database connections
            self._validate_database_config(config.postgres, "PostgreSQL", issues)
            self._validate_database_config(config.neo4j, "Neo4j", issues)
            self._validate_database_config(config.redis, "Redis", issues)
            
            # Validate model configuration
            try:
                config.model.validate_config()
            except ValueError as e:
                issues.append(f"Model configuration error: {e}")
            
            # Validate API configuration
            if config.api.port < 1024 and os.geteuid() != 0:
                issues.append("API port < 1024 requires root privileges")
            
            # Validate directories
            for dir_attr in ['data_dir', 'models_dir', 'logs_dir', 'cache_dir']:
                dir_path = Path(getattr(config, dir_attr))
                if not os.access(dir_path.parent, os.W_OK):
                    issues.append(f"No write permission for {dir_attr}: {dir_path}")
            
            # Validate system resources
            system_memory_gb = psutil.virtual_memory().total / (1024**3)
            if config.memory_limit_gb > system_memory_gb * 0.8:
                issues.append(f"Memory limit ({config.memory_limit_gb}GB) exceeds 80% of system memory")
            
        except Exception as e:
            issues.append(f"Configuration validation error: {e}")
        
        return issues
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for configuration optimization"""
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'disk_space_gb': round(shutil.disk_usage('/').free / (1024**3), 2),
            'hostname': socket.gethostname(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor()
        }
    
    def optimize_config_for_system(self, config: FlavorIntelligenceConfig) -> FlavorIntelligenceConfig:
        """Optimize configuration based on system resources"""
        
        system_info = self.get_system_info()
        
        # Optimize based on CPU count
        cpu_count = system_info['cpu_count']
        config.max_workers = min(cpu_count, 8)  # Cap at 8 workers
        config.model.num_workers = min(cpu_count // 2, 4)
        config.api.workers = min(cpu_count // 2, 4) if config.environment == Environment.PRODUCTION else 1
        
        # Optimize based on memory
        memory_gb = system_info['memory_gb']
        if memory_gb < 4:
            # Low memory system
            config.model.batch_size = 16
            config.cache.max_memory_mb = 256
            config.memory_limit_gb = 2
        elif memory_gb < 8:
            # Medium memory system
            config.model.batch_size = 32
            config.cache.max_memory_mb = 512
            config.memory_limit_gb = 4
        else:
            # High memory system
            config.model.batch_size = 64
            config.cache.max_memory_mb = 1024
            config.memory_limit_gb = min(memory_gb * 0.6, 16)
        
        # Optimize scraping settings
        if memory_gb < 4:
            config.scraping.concurrent_requests = 5
            config.scraping.chunk_size = 500
        else:
            config.scraping.concurrent_requests = 10
            config.scraping.chunk_size = 1000
        
        self.logger.info("Configuration optimized for system resources")
        
        return config
    
    def _deep_merge_dicts(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config_dict: Dict) -> Dict:
        """Apply environment variable overrides"""
        
        env_prefix = "FLAVOR_INTEL_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Convert environment variable name to config path
                config_path = key[len(env_prefix):].lower().split('_')
                
                # Navigate to the correct nested position
                current = config_dict
                for path_part in config_path[:-1]:
                    if path_part not in current:
                        current[path_part] = {}
                    current = current[path_part]
                
                # Set the value with type conversion
                final_key = config_path[-1]
                current[final_key] = self._convert_env_value(value)
        
        return config_dict
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable value to appropriate type"""
        
        # Boolean conversion
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _dict_to_config(self, config_dict: Dict) -> FlavorIntelligenceConfig:
        """Convert dictionary to configuration object"""
        
        # Create nested configuration objects
        if 'postgres' in config_dict:
            config_dict['postgres'] = DatabaseConfig(**config_dict['postgres'])
        
        if 'neo4j' in config_dict:
            config_dict['neo4j'] = DatabaseConfig(**config_dict['neo4j'])
        
        if 'redis' in config_dict:
            config_dict['redis'] = DatabaseConfig(**config_dict['redis'])
        
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        
        if 'api' in config_dict:
            config_dict['api'] = APIConfig(**config_dict['api'])
        
        if 'scraping' in config_dict:
            config_dict['scraping'] = ScrapingConfig(**config_dict['scraping'])
        
        if 'cache' in config_dict:
            config_dict['cache'] = CacheConfig(**config_dict['cache'])
        
        if 'logging' in config_dict:
            config_dict['logging'] = LoggingConfig(**config_dict['logging'])
        
        if 'monitoring' in config_dict:
            config_dict['monitoring'] = MonitoringConfig(**config_dict['monitoring'])
        
        if 'security' in config_dict:
            config_dict['security'] = SecurityConfig(**config_dict['security'])
        
        # Convert environment string to enum
        if 'environment' in config_dict and isinstance(config_dict['environment'], str):
            config_dict['environment'] = Environment(config_dict['environment'])
        
        return FlavorIntelligenceConfig(**config_dict)
    
    def _validate_database_config(self, db_config: DatabaseConfig, 
                                 db_name: str, issues: List[str]):
        """Validate database configuration"""
        
        try:
            # Check if host is reachable
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((db_config.host, db_config.port))
            if result != 0:
                issues.append(f"{db_name} connection failed: {db_config.host}:{db_config.port}")
            sock.close()
            
        except Exception as e:
            issues.append(f"{db_name} validation error: {e}")
    
    def _encrypt_sensitive_data(self, config_dict: Dict) -> Dict:
        """Encrypt sensitive configuration data"""
        
        if not self.encryption_key:
            return config_dict
        
        cipher = Fernet(self.encryption_key)
        
        # List of sensitive fields to encrypt
        sensitive_fields = [
            'password', 'secret_key', 'api_key', 'jwt_secret'
        ]
        
        def encrypt_recursive(data):
            if isinstance(data, dict):
                result = {}
                for key, value in data.items():
                    if any(field in key.lower() for field in sensitive_fields):
                        if isinstance(value, str):
                            encrypted_value = cipher.encrypt(value.encode()).decode()
                            result[key] = f"encrypted:{encrypted_value}"
                        else:
                            result[key] = value
                    else:
                        result[key] = encrypt_recursive(value)
                return result
            elif isinstance(data, list):
                return [encrypt_recursive(item) for item in data]
            else:
                return data
        
        return encrypt_recursive(config_dict)
    
    def _create_template_placeholders(self, config_dict: Dict) -> Dict:
        """Create template with placeholders for common values"""
        
        placeholder_mappings = {
            'localhost': '${HOST}',
            'password': '${DATABASE_PASSWORD}',
            'your-secret-key': '${SECRET_KEY}',
            'your-api-key': '${API_KEY}',
            'development': '${ENVIRONMENT}'
        }
        
        def replace_with_placeholders(data):
            if isinstance(data, dict):
                result = {}
                for key, value in data.items():
                    result[key] = replace_with_placeholders(value)
                return result
            elif isinstance(data, list):
                return [replace_with_placeholders(item) for item in data]
            elif isinstance(data, str):
                for original, placeholder in placeholder_mappings.items():
                    if original in data:
                        data = data.replace(original, placeholder)
                return data
            else:
                return data
        
        return replace_with_placeholders(config_dict)


# Configuration factory functions

def create_development_config() -> FlavorIntelligenceConfig:
    """Create development configuration"""
    manager = ConfigurationManager()
    return manager.create_default_config(Environment.DEVELOPMENT)


def create_production_config() -> FlavorIntelligenceConfig:
    """Create production configuration"""
    manager = ConfigurationManager()
    config = manager.create_default_config(Environment.PRODUCTION)
    return manager.optimize_config_for_system(config)


def load_config_from_env(config_dir: str = "config") -> FlavorIntelligenceConfig:
    """Load configuration based on environment variables"""
    
    manager = ConfigurationManager(config_dir)
    
    # Determine environment
    env_name = os.getenv("FLAVOR_INTEL_ENV", "development")
    environment = Environment(env_name)
    
    # Try to load from file first
    config_file = Path(config_dir) / f"config_{env_name}.yaml"
    
    if config_file.exists():
        config = manager.load_config(config_file, environment)
    else:
        # Create default configuration
        config = manager.create_default_config(environment)
        
        # Save default config for future use
        manager.save_config(config, config_file)
    
    # Optimize for current system
    config = manager.optimize_config_for_system(config)
    
    return config


# Export key components
__all__ = [
    'Environment', 'LogLevel', 'DatabaseType', 'DatabaseConfig',
    'ModelConfig', 'APIConfig', 'ScrapingConfig', 'CacheConfig',
    'LoggingConfig', 'MonitoringConfig', 'SecurityConfig',
    'FlavorIntelligenceConfig', 'ConfigurationManager',
    'create_development_config', 'create_production_config',
    'load_config_from_env'
]