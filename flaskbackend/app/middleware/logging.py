"""
Request Logging Middleware
===========================
Logs all incoming requests and outgoing responses.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging
import time
from typing import Callable

logger = logging.getLogger("tumorheal")


class RequestMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests and responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process each request and log details
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint
        
        Returns:
            Response from endpoint
        """
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response
            logger.info(
                f"Response: {response.status_code} "
                f"for {request.method} {request.url.path} "
                f"({duration_ms:.2f}ms)"
            )
            
            return response
            
        except Exception as e:
            # Log error
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Error processing {request.method} {request.url.path} "
                f"({duration_ms:.2f}ms): {str(e)}"
            )
            raise
