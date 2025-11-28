"""
Configuration Management for Food Knowledge Graph
==============================================

Centralized configuration management for the comprehensive
food knowledge graph system with environment-specific settings,
API configurations, database connections, and ML model parameters.

Author: Wellomex AI Nutrition Team
Version: 1.0.0
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Neo4j database configuration"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create config from environment variables"""
        return cls(
            uri=os.getenv('NEO4J_URI', cls.uri),
            username=os.getenv('NEO4J_USERNAME', cls.username), 
            password=os.getenv('NEO4J_PASSWORD', cls.password),
            database=os.getenv('NEO4J_DATABASE', cls.database),
            max_connection_lifetime=int(os.getenv('NEO4J_MAX_LIFETIME', cls.max_connection_lifetime)),
            max_connection_pool_size=int(os.getenv('NEO4J_POOL_SIZE', cls.max_connection_pool_size)),
            connection_timeout=int(os.getenv('NEO4J_TIMEOUT', cls.connection_timeout))
        )

@dataclass
class RedisConfig:
    """Redis cache configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 100
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    decode_responses: bool = True
    
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Create config from environment variables"""
        return cls(
            host=os.getenv('REDIS_HOST', cls.host),
            port=int(os.getenv('REDIS_PORT', cls.port)),
            password=os.getenv('REDIS_PASSWORD'),
            db=int(os.getenv('REDIS_DB', cls.db)),
            max_connections=int(os.getenv('REDIS_MAX_CONNECTIONS', cls.max_connections)),
            socket_timeout=int(os.getenv('REDIS_SOCKET_TIMEOUT', cls.socket_timeout)),
            socket_connect_timeout=int(os.getenv('REDIS_CONNECT_TIMEOUT', cls.socket_connect_timeout))
        )

@dataclass
class APIClientConfig:
    """External API client configuration"""
    # USDA FoodData Central
    usda_api_key: Optional[str] = None
    usda_base_url: str = "https://api.nal.usda.gov/fdc/v1"
    usda_rate_limit: int = 3600  # requests per hour
    
    # Open Food Facts
    off_base_url: str = "https://world.openfoodfacts.org/api/v2"
    off_user_agent: str = "FoodKnowledgeGraph/1.0"
    off_rate_limit: int = 10000  # requests per hour
    
    # Nutritionix
    nutritionix_app_id: Optional[str] = None
    nutritionix_api_key: Optional[str] = None
    nutritionix_base_url: str = "https://trackapi.nutritionix.com/v2"
    nutritionix_rate_limit: int = 500  # requests per day
    
    # Spoonacular  
    spoonacular_api_key: Optional[str] = None
    spoonacular_base_url: str = "https://api.spoonacular.com"
    spoonacular_rate_limit: int = 150  # requests per day
    
    # Request configuration
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    @classmethod
    def from_env(cls) -> 'APIClientConfig':
        """Create config from environment variables"""
        return cls(
            usda_api_key=os.getenv('USDA_API_KEY'),
            usda_rate_limit=int(os.getenv('USDA_RATE_LIMIT', cls.usda_rate_limit)),
            
            off_user_agent=os.getenv('OFF_USER_AGENT', cls.off_user_agent),
            off_rate_limit=int(os.getenv('OFF_RATE_LIMIT', cls.off_rate_limit)),
            
            nutritionix_app_id=os.getenv('NUTRITIONIX_APP_ID'),
            nutritionix_api_key=os.getenv('NUTRITIONIX_API_KEY'),
            nutritionix_rate_limit=int(os.getenv('NUTRITIONIX_RATE_LIMIT', cls.nutritionix_rate_limit)),
            
            spoonacular_api_key=os.getenv('SPOONACULAR_API_KEY'),
            spoonacular_rate_limit=int(os.getenv('SPOONACULAR_RATE_LIMIT', cls.spoonacular_rate_limit)),
            
            request_timeout=int(os.getenv('API_REQUEST_TIMEOUT', cls.request_timeout)),
            max_retries=int(os.getenv('API_MAX_RETRIES', cls.max_retries)),
            retry_delay=float(os.getenv('API_RETRY_DELAY', cls.retry_delay))
        )

@dataclass
class MLConfig:
    """Machine Learning model configuration"""
    # Model paths and settings
    model_cache_dir: str = "./models/cache"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    classification_model: str = "distilbert-base-uncased"
    
    # Embedding configuration
    embedding_dimension: int = 384
    max_sequence_length: int = 512
    batch_size: int = 32
    
    # Similarity thresholds
    similarity_threshold: float = 0.7
    substitution_threshold: float = 0.8
    cultural_similarity_threshold: float = 0.6
    
    # Training configuration
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda
    
    @classmethod
    def from_env(cls) -> 'MLConfig':
        """Create config from environment variables"""
        return cls(
            model_cache_dir=os.getenv('ML_MODEL_CACHE_DIR', cls.model_cache_dir),
            embedding_model=os.getenv('ML_EMBEDDING_MODEL', cls.embedding_model),
            classification_model=os.getenv('ML_CLASSIFICATION_MODEL', cls.classification_model),
            
            embedding_dimension=int(os.getenv('ML_EMBEDDING_DIM', cls.embedding_dimension)),
            max_sequence_length=int(os.getenv('ML_MAX_SEQ_LENGTH', cls.max_sequence_length)),
            batch_size=int(os.getenv('ML_BATCH_SIZE', cls.batch_size)),
            
            similarity_threshold=float(os.getenv('ML_SIMILARITY_THRESHOLD', cls.similarity_threshold)),
            substitution_threshold=float(os.getenv('ML_SUBSTITUTION_THRESHOLD', cls.substitution_threshold)),
            cultural_similarity_threshold=float(os.getenv('ML_CULTURAL_THRESHOLD', cls.cultural_similarity_threshold)),
            
            learning_rate=float(os.getenv('ML_LEARNING_RATE', cls.learning_rate)),
            num_epochs=int(os.getenv('ML_NUM_EPOCHS', cls.num_epochs)),
            warmup_steps=int(os.getenv('ML_WARMUP_STEPS', cls.warmup_steps)),
            
            device=os.getenv('ML_DEVICE', cls.device)
        )

@dataclass  
class CacheConfig:
    """Caching configuration"""
    # Cache TTL settings (in seconds)
    food_details_ttl: int = 3600  # 1 hour
    search_results_ttl: int = 1800  # 30 minutes  
    api_response_ttl: int = 7200  # 2 hours
    ml_embedding_ttl: int = 86400  # 24 hours
    cultural_analysis_ttl: int = 43200  # 12 hours
    seasonal_prediction_ttl: int = 21600  # 6 hours
    
    # Cache size limits
    max_cache_size: int = 10000
    max_memory_usage: str = "512MB"
    
    # Cache key prefixes
    food_prefix: str = "food:"
    search_prefix: str = "search:"
    api_prefix: str = "api:"
    ml_prefix: str = "ml:"
    cultural_prefix: str = "cultural:"
    seasonal_prefix: str = "seasonal:"
    
    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Create config from environment variables"""
        return cls(
            food_details_ttl=int(os.getenv('CACHE_FOOD_TTL', cls.food_details_ttl)),
            search_results_ttl=int(os.getenv('CACHE_SEARCH_TTL', cls.search_results_ttl)),
            api_response_ttl=int(os.getenv('CACHE_API_TTL', cls.api_response_ttl)),
            ml_embedding_ttl=int(os.getenv('CACHE_ML_TTL', cls.ml_embedding_ttl)),
            cultural_analysis_ttl=int(os.getenv('CACHE_CULTURAL_TTL', cls.cultural_analysis_ttl)),
            seasonal_prediction_ttl=int(os.getenv('CACHE_SEASONAL_TTL', cls.seasonal_prediction_ttl)),
            
            max_cache_size=int(os.getenv('CACHE_MAX_SIZE', cls.max_cache_size)),
            max_memory_usage=os.getenv('CACHE_MAX_MEMORY', cls.max_memory_usage)
        )

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Create config from environment variables"""
        return cls(
            level=os.getenv('LOG_LEVEL', cls.level),
            format=os.getenv('LOG_FORMAT', cls.format),
            file_path=os.getenv('LOG_FILE_PATH'),
            max_file_size=int(os.getenv('LOG_MAX_FILE_SIZE', cls.max_file_size)),
            backup_count=int(os.getenv('LOG_BACKUP_COUNT', cls.backup_count))
        )

@dataclass
class SecurityConfig:
    """Security configuration"""
    # API security
    enable_api_key_auth: bool = False
    api_keys: List[str] = field(default_factory=list)
    
    # Rate limiting
    enable_rate_limiting: bool = True
    default_rate_limit: str = "1000/hour"
    
    # CORS settings
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    allowed_headers: List[str] = field(default_factory=lambda: ["*"])
    
    @classmethod
    def from_env(cls) -> 'SecurityConfig':
        """Create config from environment variables"""
        api_keys_str = os.getenv('API_KEYS', '')
        api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()] if api_keys_str else []
        
        origins_str = os.getenv('ALLOWED_ORIGINS', '*')
        origins = [origin.strip() for origin in origins_str.split(',') if origin.strip()] if origins_str != '*' else ['*']
        
        return cls(
            enable_api_key_auth=os.getenv('ENABLE_API_KEY_AUTH', 'false').lower() == 'true',
            api_keys=api_keys,
            enable_rate_limiting=os.getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true',
            default_rate_limit=os.getenv('DEFAULT_RATE_LIMIT', cls.default_rate_limit),
            allowed_origins=origins
        )

@dataclass
class FoodKnowledgeConfig:
    """Main configuration class combining all config sections"""
    
    # Environment
    environment: str = "development"  # development, staging, production
    debug: bool = False
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api_clients: APIClientConfig = field(default_factory=APIClientConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Data processing settings
    max_concurrent_requests: int = 100
    batch_processing_size: int = 1000
    sync_interval_hours: int = 24
    
    # Quality thresholds
    min_data_quality_score: float = 0.7
    max_missing_nutrients: int = 5
    require_country_data: bool = False
    
    @classmethod
    def from_env(cls) -> 'FoodKnowledgeConfig':
        """Create complete configuration from environment variables"""
        return cls(
            environment=os.getenv('ENVIRONMENT', 'development'),
            debug=os.getenv('DEBUG', 'false').lower() == 'true',
            
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            api_clients=APIClientConfig.from_env(),
            ml=MLConfig.from_env(),
            cache=CacheConfig.from_env(),
            logging=LoggingConfig.from_env(),
            security=SecurityConfig.from_env(),
            
            max_concurrent_requests=int(os.getenv('MAX_CONCURRENT_REQUESTS', 100)),
            batch_processing_size=int(os.getenv('BATCH_PROCESSING_SIZE', 1000)),
            sync_interval_hours=int(os.getenv('SYNC_INTERVAL_HOURS', 24)),
            
            min_data_quality_score=float(os.getenv('MIN_DATA_QUALITY_SCORE', 0.7)),
            max_missing_nutrients=int(os.getenv('MAX_MISSING_NUTRIENTS', 5)),
            require_country_data=os.getenv('REQUIRE_COUNTRY_DATA', 'false').lower() == 'true'
        )
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Validate database config
        if not self.database.uri:
            issues.append("Database URI is required")
        if not self.database.username:
            issues.append("Database username is required")
        if not self.database.password:
            issues.append("Database password is required")
            
        # Validate Redis config
        if not self.redis.host:
            issues.append("Redis host is required")
        if self.redis.port <= 0 or self.redis.port > 65535:
            issues.append("Redis port must be between 1 and 65535")
            
        # Validate ML config
        if self.ml.similarity_threshold < 0 or self.ml.similarity_threshold > 1:
            issues.append("ML similarity threshold must be between 0 and 1")
        if self.ml.batch_size <= 0:
            issues.append("ML batch size must be positive")
            
        # Validate cache config
        if self.cache.food_details_ttl <= 0:
            issues.append("Cache TTL values must be positive")
            
        # Validate processing settings
        if self.max_concurrent_requests <= 0:
            issues.append("Max concurrent requests must be positive")
        if self.batch_processing_size <= 0:
            issues.append("Batch processing size must be positive")
            
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'database': {
                'uri': self.database.uri,
                'username': self.database.username,
                'database': self.database.database,
                'max_connection_pool_size': self.database.max_connection_pool_size,
                'connection_timeout': self.database.connection_timeout
            },
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'db': self.redis.db,
                'max_connections': self.redis.max_connections
            },
            'api_clients': {
                'usda_configured': bool(self.api_clients.usda_api_key),
                'nutritionix_configured': bool(self.api_clients.nutritionix_api_key),
                'spoonacular_configured': bool(self.api_clients.spoonacular_api_key),
                'request_timeout': self.api_clients.request_timeout,
                'max_retries': self.api_clients.max_retries
            },
            'ml': {
                'embedding_model': self.ml.embedding_model,
                'embedding_dimension': self.ml.embedding_dimension,
                'device': self.ml.device,
                'similarity_threshold': self.ml.similarity_threshold
            },
            'cache': {
                'food_details_ttl': self.cache.food_details_ttl,
                'search_results_ttl': self.cache.search_results_ttl,
                'max_cache_size': self.cache.max_cache_size
            },
            'processing': {
                'max_concurrent_requests': self.max_concurrent_requests,
                'batch_processing_size': self.batch_processing_size,
                'sync_interval_hours': self.sync_interval_hours
            },
            'quality': {
                'min_data_quality_score': self.min_data_quality_score,
                'max_missing_nutrients': self.max_missing_nutrients,
                'require_country_data': self.require_country_data
            }
        }

# Global configuration instance
_config: Optional[FoodKnowledgeConfig] = None

def get_config() -> FoodKnowledgeConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = FoodKnowledgeConfig.from_env()
        
        # Validate configuration
        issues = _config.validate()
        if issues:
            logger.warning(f"Configuration validation issues: {issues}")
    
    return _config

def reload_config() -> FoodKnowledgeConfig:
    """Reload configuration from environment"""
    global _config
    _config = None
    return get_config()

def setup_logging(config: Optional[LoggingConfig] = None):
    """Setup logging based on configuration"""
    if config is None:
        config = get_config().logging
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format=config.format
    )
    
    # Add file handler if specified
    if config.file_path:
        from logging.handlers import RotatingFileHandler
        
        # Ensure log directory exists
        log_dir = Path(config.file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(logging.Formatter(config.format))
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Logging configured: level={config.level}, file={config.file_path}")

# Initialize logging on import
setup_logging()