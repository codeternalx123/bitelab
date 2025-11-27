from functools import wraps
from typing import Callable, Any
import time
import asyncio
from app.core.logging import get_logger

logger = get_logger(__name__)

def validate_input(f: Callable) -> Callable:
    """Decorator for input validation"""
    @wraps(f)
    async def wrapper(*args, **kwargs):
        # Get instance and method name
        instance = args[0]
        method_name = f.__name__
        
        try:
            # Validate inputs if preprocessor exists
            if hasattr(instance, 'preprocessor'):
                for key, value in kwargs.items():
                    if hasattr(value, '__dict__'):
                        validation_result = \
                            await instance.preprocessor.validate_data(
                                value.__dict__,
                                key
                            )
                        if not validation_result.is_valid:
                            raise ValueError(
                                f"Validation failed for {key}: "
                                f"{validation_result.errors}"
                            )
            
            return await f(*args, **kwargs)
            
        except Exception as e:
            logger.error(
                f"Validation error in {method_name}: {str(e)}"
            )
            raise
            
    return wrapper

def cache_result(ttl: int = 3600):
    """Decorator for caching function results"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        async def wrapper(*args, **kwargs):
            # Get instance and method name
            instance = args[0]
            method_name = f.__name__
            
            try:
                # Generate cache key
                key_parts = [
                    method_name,
                    *(str(arg) for arg in args[1:]),
                    *(f"{k}:{v}" for k, v in kwargs.items())
                ]
                cache_key = ":".join(key_parts)
                
                # Check cache if available
                if hasattr(instance, 'cache'):
                    cached_result = await instance.cache.get(
                        cache_key
                    )
                    if cached_result:
                        if hasattr(instance, 'metrics'):
                            instance.metrics.record_cache_hit()
                        return cached_result
                    
                    if hasattr(instance, 'metrics'):
                        instance.metrics.record_cache_miss()
                
                # Execute function
                result = await f(*args, **kwargs)
                
                # Store in cache if available
                if hasattr(instance, 'cache'):
                    await instance.cache.set(
                        cache_key,
                        result,
                        ttl
                    )
                
                return result
                
            except Exception as e:
                logger.error(
                    f"Cache error in {method_name}: {str(e)}"
                )
                # Execute function without caching
                return await f(*args, **kwargs)
                
        return wrapper
    return decorator

def monitor_performance(f: Callable) -> Callable:
    """Decorator for monitoring function performance"""
    @wraps(f)
    async def wrapper(*args, **kwargs):
        # Get instance and method name
        instance = args[0]
        method_name = f.__name__
        
        try:
            start_time = time.time()
            
            # Execute function
            result = await f(*args, **kwargs)
            
            duration = time.time() - start_time
            
            # Record metrics if available
            if hasattr(instance, 'metrics'):
                instance.metrics.record_response_time(
                    method_name,
                    duration
                )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Performance monitoring error in {method_name}: "
                f"{str(e)}"
            )
            raise
            
    return wrapper

def handle_errors(f: Callable) -> Callable:
    """Decorator for consistent error handling"""
    @wraps(f)
    async def wrapper(*args, **kwargs):
        # Get instance and method name
        instance = args[0]
        method_name = f.__name__
        
        try:
            return await f(*args, **kwargs)
            
        except Exception as e:
            # Log error
            logger.error(
                f"Error in {method_name}: {str(e)}",
                exc_info=True
            )
            
            # Record metric if available
            if hasattr(instance, 'metrics'):
                instance.metrics.record_error(
                    method_name,
                    str(e)
                )
            
            # Raise appropriate exception
            raise
            
    return wrapper

def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0
):
    """Decorator for retrying failed operations"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        async def wrapper(*args, **kwargs):
            # Get instance and method name
            instance = args[0]
            method_name = f.__name__
            
            for attempt in range(max_retries):
                try:
                    return await f(*args, **kwargs)
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Final retry failed for {method_name}: "
                            f"{str(e)}"
                        )
                        raise
                        
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for "
                        f"{method_name}: {str(e)}"
                    )
                    
                    # Wait before retrying
                    await asyncio.sleep(delay * (attempt + 1))
                    
        return wrapper
    return decorator

def batch_process(batch_size: int = 100):
    """Decorator for batch processing"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        async def wrapper(*args, **kwargs):
            # Get data to process
            data = kwargs.get('data', [])
            if not data:
                return await f(*args, **kwargs)
                
            # Process in batches
            results = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                kwargs['data'] = batch
                batch_result = await f(*args, **kwargs)
                results.extend(batch_result)
                
            return results
            
        return wrapper
    return decorator