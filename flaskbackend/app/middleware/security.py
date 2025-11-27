"""
Security Middleware
===================
Security headers and SQL injection prevention middleware.
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import Callable
import re
import logging

logger = logging.getLogger("tumorheal")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Add security headers to response
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint
        
        Returns:
            Response with security headers
        """
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


class SQLInjectionMiddleware(BaseHTTPMiddleware):
    """Middleware to detect and block potential SQL injection attempts"""
    
    # Common SQL injection patterns
    SQL_PATTERNS = [
        r"(\bunion\b.*\bselect\b)",
        r"(\bselect\b.*\bfrom\b)",
        r"(\binsert\b.*\binto\b)",
        r"(\bupdate\b.*\bset\b)",
        r"(\bdelete\b.*\bfrom\b)",
        r"(\bdrop\b.*\btable\b)",
        r"(\bexec\b.*\()",
        r"(\bor\b.*=.*)",
        r"(';.*--)",
        r"(--.*$)",
        r"(/\*.*\*/)",
    ]
    
    def __init__(self, app):
        super().__init__(app)
        self.sql_regex = re.compile("|".join(self.SQL_PATTERNS), re.IGNORECASE)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check request for SQL injection attempts
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint
        
        Returns:
            Response or raises HTTPException if SQL injection detected
        """
        # Check query parameters
        for key, value in request.query_params.items():
            if self._contains_sql_injection(value):
                logger.warning(
                    f"Potential SQL injection detected in query param '{key}': {value}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Invalid request parameters"
                )
        
        # Check URL path
        if self._contains_sql_injection(request.url.path):
            logger.warning(
                f"Potential SQL injection detected in URL path: {request.url.path}"
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid request path"
            )
        
        response = await call_next(request)
        return response
    
    def _contains_sql_injection(self, text: str) -> bool:
        """
        Check if text contains potential SQL injection patterns
        
        Args:
            text: Text to check
        
        Returns:
            True if SQL injection pattern detected
        """
        if not text:
            return False
        
        return bool(self.sql_regex.search(str(text)))
