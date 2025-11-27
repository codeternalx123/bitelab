from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import time
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from typing import Generator, Optional
import logging
import threading

logger = logging.getLogger(__name__)

# Attempt to import missing modules with logging
try:
    from app.core.security import decode_access_token as decode_token
    logger.info("Successfully imported decode_access_token from app.core.security")
except ImportError as e:
    logger.error(f"Failed to import decode_access_token from app.core.security: {e}")

try:
    from app.core.database import get_db
    logger.info("Successfully imported get_db from app.core.database")
except ImportError as e:
    logger.error(f"Failed to import get_db from app.core.database: {e}")

try:
    from app.models.database import User
    logger.info("Successfully imported User from app.models.database")
except ImportError as e:
    logger.error(f"Failed to import User from app.models.database: {e}")

from app.core.config import settings

logger = logging.getLogger(__name__)

security = HTTPBearer()

# Try to connect to Redis, but don't fail if it's not available
try:
    redis = Redis.from_url(settings.REDIS_URL)
    redis.ping()  # Test connection
    REDIS_AVAILABLE = True
    logger.info("Redis connected successfully")
except (RedisConnectionError, Exception) as e:
    redis = None
    REDIS_AVAILABLE = False
    # Redis not available - silently disable rate limiting

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    try:
        token = credentials.credentials
        payload = decode_token(token)
        user = db.query(User).filter(User.email == payload['sub']).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not user.is_active:
            raise HTTPException(status_code=400, detail="Inactive user")
        return user
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403, detail="Not enough privileges"
        )
    return current_user

def rate_limit(request: Request):
    """Rate limiting function - skips if Redis is not available"""
    if not REDIS_AVAILABLE or redis is None:
        # Skip rate limiting if Redis is not available
        return
    
    try:
        client = request.client.host
        key = f"rate_limit:{client}"
        current = redis.get(key)
        
        if current is not None and int(current) > settings.RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail="Too many requests"
            )
        
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)
        pipe.execute()
    except RedisConnectionError as e:
        # Silently continue without rate limiting
        return
    except Exception as e:
        logger.error(f"Unexpected error in rate_limit: {e}")
        return

# Singleton instance for the colorimetry engine
try:
    ComputationalColorimetryEngine
    logger.info("ComputationalColorimetryEngine class is available")
except NameError:
    logger.error("ComputationalColorimetryEngine class is not defined")
    ComputationalColorimetryEngine = None  # Placeholder to avoid NameError

_colorimetry_engine_instance: Optional[ComputationalColorimetryEngine] = None
_engine_lock = threading.Lock()

def get_colorimetry_engine() -> ComputationalColorimetryEngine:
    """
    Dependency provider for the ComputationalColorimetryEngine.

    Creates a singleton instance of the engine to ensure that its state,
    including the loaded database and calibration, is preserved across requests.
    """
    global _colorimetry_engine_instance
    with _engine_lock:
        if _colorimetry_engine_instance is None:
            logger.info("Creating singleton instance of ComputationalColorimetryEngine.")
            if ComputationalColorimetryEngine is None:
                logger.error("Cannot create ComputationalColorimetryEngine: class not defined")
                raise RuntimeError("ComputationalColorimetryEngine class not available")
            _colorimetry_engine_instance = ComputationalColorimetryEngine()
    return _colorimetry_engine_instance

# Singleton instance for the fusion engine
try:
    UnifiedFusionEngine
    logger.info("UnifiedFusionEngine class is available")
except NameError:
    logger.error("UnifiedFusionEngine class is not defined")
    UnifiedFusionEngine = None  # Placeholder to avoid NameError

_fusion_engine_instance: Optional[UnifiedFusionEngine] = None
_fusion_lock = threading.Lock()

def get_fusion_engine() -> UnifiedFusionEngine:
    """
    Dependency provider for the UnifiedFusionEngine.

    Creates a singleton instance to preserve its configuration and loaded databases.
    """
    global _fusion_engine_instance
    with _fusion_lock:
        if _fusion_engine_instance is None:
            logger.info("Creating singleton instance of UnifiedFusionEngine.")
            if UnifiedFusionEngine is None:
                logger.error("Cannot create UnifiedFusionEngine: class not defined")
                raise RuntimeError("UnifiedFusionEngine class not available")
            _fusion_engine_instance = UnifiedFusionEngine()
    return _fusion_engine_instance
