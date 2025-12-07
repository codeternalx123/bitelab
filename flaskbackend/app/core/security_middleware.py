"""
Security Middleware for Flask/FastAPI Integration
=================================================

Integrates quantum security system into web application middleware.
Provides automatic request validation, encryption, and threat detection.

Author: BiteLab Security Team
Version: 1.0.0
"""

from functools import wraps
from typing import Optional, Callable, Any
from flask import request, jsonify, g
import logging

from .quantum_security import get_security_system, ThreatLevel

logger = logging.getLogger(__name__)


class SecurityMiddleware:
    """
    Flask/FastAPI middleware for automatic security enforcement
    
    Features:
    - Automatic request validation
    - IP blocking
    - Rate limiting
    - Input sanitization
    - Threat logging
    """
    
    def __init__(self, app=None):
        self.app = app
        self.security = get_security_system()
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize with Flask app"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        
        logger.info("SecurityMiddleware initialized")
    
    def before_request(self):
        """Execute before each request"""
        # Get client IP
        client_ip = request.remote_addr
        
        # Check if IP is blocked
        if self.security.threat_detector.is_ip_blocked(client_ip):
            logger.warning(f"Blocked IP attempted access: {client_ip}")
            return jsonify({
                'error': 'Access denied',
                'message': 'Your IP has been blocked due to suspicious activity'
            }), 403
        
        # Rate limiting
        if self.security.threat_detector.check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded: {client_ip}")
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': 'Too many requests. Please try again later.'
            }), 429
        
        # Validate request data
        if request.method in ['POST', 'PUT', 'PATCH']:
            try:
                # Get request data
                if request.is_json:
                    data = request.get_json()
                else:
                    data = request.form.to_dict()
                
                # Validate each field
                for key, value in data.items():
                    if isinstance(value, str):
                        is_valid, sanitized = self.security.validate_input(
                            value, source_ip=client_ip
                        )
                        
                        if not is_valid:
                            logger.warning(f"Malicious input detected in field '{key}': {client_ip}")
                            return jsonify({
                                'error': 'Invalid input',
                                'message': f'Field "{key}" contains invalid or malicious content'
                            }), 400
            
            except Exception as e:
                logger.error(f"Request validation error: {e}")
        
        # Store request start time
        g.request_start_time = g.get('request_start_time', None)
    
    def after_request(self, response):
        """Execute after each request"""
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        
        return response


def require_security(threat_level: ThreatLevel = ThreatLevel.MEDIUM):
    """
    Decorator for endpoint security enforcement
    
    Usage:
        @app.route('/api/sensitive')
        @require_security(ThreatLevel.HIGH)
        def sensitive_endpoint():
            return {'data': 'protected'}
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            security = get_security_system()
            client_ip = request.remote_addr
            
            # Enhanced validation for high-security endpoints
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                # Require authentication token
                auth_header = request.headers.get('Authorization')
                if not auth_header:
                    return jsonify({
                        'error': 'Unauthorized',
                        'message': 'Authentication required'
                    }), 401
                
                # Validate token format
                if not auth_header.startswith('Bearer '):
                    return jsonify({
                        'error': 'Invalid token',
                        'message': 'Invalid authentication token format'
                    }), 401
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


def validate_file_upload(allowed_types: Optional[list] = None):
    """
    Decorator for secure file upload endpoints
    
    Usage:
        @app.route('/api/upload', methods=['POST'])
        @validate_file_upload(['image/jpeg', 'image/png'])
        def upload_file():
            file = request.files['file']
            return {'message': 'File uploaded'}
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            security = get_security_system()
            client_ip = request.remote_addr
            
            if 'file' not in request.files:
                return jsonify({
                    'error': 'No file',
                    'message': 'No file provided in request'
                }), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    'error': 'Empty filename',
                    'message': 'No file selected'
                }), 400
            
            # Read file data
            file_data = file.read()
            file.seek(0)  # Reset file pointer
            
            # Validate file
            is_safe, safe_filename = security.validate_file_upload(
                file_data,
                file.filename,
                file.content_type or 'application/octet-stream',
                source_ip=client_ip
            )
            
            if not is_safe:
                logger.warning(f"Malicious file upload blocked: {file.filename} from {client_ip}")
                return jsonify({
                    'error': 'Malicious file',
                    'message': 'File upload blocked due to security concerns'
                }), 400
            
            # Check allowed types
            if allowed_types and file.content_type not in allowed_types:
                return jsonify({
                    'error': 'Invalid file type',
                    'message': f'File type {file.content_type} not allowed'
                }), 400
            
            # Store safe filename in request context
            g.safe_filename = safe_filename
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


# Example Flask routes with security
def setup_secure_routes(app):
    """
    Setup example secure routes
    
    Call this in your Flask app initialization:
        from app.core.security_middleware import setup_secure_routes
        setup_secure_routes(app)
    """
    
    @app.route('/api/security/stats', methods=['GET'])
    @require_security(ThreatLevel.MEDIUM)
    def get_security_stats():
        """Get security system statistics"""
        security = get_security_system()
        stats = security.get_security_stats()
        return jsonify(stats)
    
    @app.route('/api/security/threats', methods=['GET'])
    @require_security(ThreatLevel.HIGH)
    def get_recent_threats():
        """Get recent security threats"""
        security = get_security_system()
        count = request.args.get('count', 50, type=int)
        threats = security.get_recent_threats(count)
        return jsonify({'threats': threats})
    
    @app.route('/api/security/encrypt', methods=['POST'])
    @require_security(ThreatLevel.HIGH)
    def encrypt_data():
        """Encrypt sensitive data"""
        security = get_security_system()
        data = request.json.get('data')
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        encrypted = security.encrypt(str(data))
        return jsonify({'encrypted': encrypted})
    
    @app.route('/api/security/decrypt', methods=['POST'])
    @require_security(ThreatLevel.CRITICAL)
    def decrypt_data():
        """Decrypt data"""
        security = get_security_system()
        encrypted_data = request.json.get('encrypted')
        
        if not encrypted_data:
            return jsonify({'error': 'No encrypted data provided'}), 400
        
        try:
            decrypted = security.decrypt(encrypted_data)
            return jsonify({'decrypted': decrypted})
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return jsonify({'error': 'Decryption failed'}), 400
    
    logger.info("Secure routes configured")
