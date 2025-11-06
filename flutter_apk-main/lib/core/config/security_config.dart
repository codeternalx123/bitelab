// import 'package:injectable/injectable.dart';
import 'environment_config.dart';

// @singleton
class SecurityConfig {
  final EnvironmentConfig _envConfig;

  SecurityConfig(this._envConfig);

  // SSL/TLS Configuration
  bool get sslPinningEnabled => _envConfig.security['sslPinning'] as bool;
  bool get certificateVerificationEnabled =>
      _envConfig.security['certificateVerification'] as bool;

  // Biometric Authentication
  bool get requireBiometrics => _envConfig.security['requireBiometrics'] as bool;
  Duration get sessionTimeout =>
      _envConfig.security['sessionTimeout'] as Duration;

  // Password Security
  int get maxPasswordAttempts =>
      _envConfig.security['maxPasswordAttempts'] as int;
  Duration get lockoutDuration =>
      _envConfig.security['lockoutDuration'] as Duration;

  // Quantum-Safe Cryptography
  bool get quantumSafeEnabled => _envConfig.quantum['enabled'] as bool;
  String get quantumAlgorithm => _envConfig.quantum['algorithm'] as String;
  int get quantumKeySize => _envConfig.quantum['keySize'] as int;

  // Token Management
  Duration get accessTokenTTL => const Duration(minutes: 15);
  Duration get refreshTokenTTL => const Duration(days: 30);
  bool get rotateRefreshTokens => true;
  int get refreshTokenRotationLimit => 5;

  // API Security
  Map<String, List<String>> get endpointPermissions => {
        'GET': [
          '/api/v1/user/profile',
          '/api/v1/scans/history',
          '/api/v1/reports',
        ],
        'POST': [
          '/api/v1/scans',
          '/api/v1/reports',
          '/api/v1/payments',
        ],
        'PUT': [
          '/api/v1/user/profile',
          '/api/v1/reports/{id}',
        ],
        'DELETE': [
          '/api/v1/reports/{id}',
          '/api/v1/scans/{id}',
        ],
      };

  // Rate Limiting
  Map<String, Map<String, dynamic>> get rateLimits =>
      _envConfig.rateLimits;

  // Input Validation Rules
  Map<String, Map<String, dynamic>> get validationRules => {
        'email': {
          'type': 'string',
          'format': 'email',
          'maxLength': 255,
          'required': true,
        },
        'password': {
          'type': 'string',
          'minLength': 12,
          'maxLength': 128,
          'pattern':
              r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]+$',
          'required': true,
        },
        'phoneNumber': {
          'type': 'string',
          'pattern': r'^\+[1-9]\d{1,14}$',
          'required': false,
        },
        'medicalRecordNumber': {
          'type': 'string',
          'pattern': r'^[A-Z]{2}\d{8}$',
          'required': true,
        },
      };

  // Data Encryption
  Map<String, String> get encryptionKeys => {
    'data': _generateKey(32),
    'files': _generateKey(32),
    'tokens': _generateKey(32),
  };

  String _generateKey(int length) {
    // TODO: Implement secure key generation
    return 'secure_key_$length';
  }

  // Sensitive Data Fields
  Set<String> get sensitiveFields => {
    'password',
    'creditCardNumber',
    'ssn',
    'medicalRecordNumber',
    'diagnosis',
    'treatment',
    'medications',
  };

  // Security Headers
  Map<String, String> get securityHeaders => {
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Content-Security-Policy': "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data: https:; "
        "connect-src 'self' https:;",
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Permissions-Policy': 'geolocation=(), camera=(), microphone=()',
  };

  // JWT Configuration
  Map<String, dynamic> get jwtConfig => {
    'issuer': 'tumorheal.com',
    'audience': 'tumorheal-app',
    'algorithm': 'RS256',
    'accessTokenTTL': accessTokenTTL.inSeconds,
    'refreshTokenTTL': refreshTokenTTL.inSeconds,
    'rotateRefreshTokens': rotateRefreshTokens,
    'refreshTokenRotationLimit': refreshTokenRotationLimit,
  };

  // Session Management
  Map<String, dynamic> get sessionConfig => {
    'timeout': sessionTimeout.inSeconds,
    'maxConcurrentSessions': 3,
    'enforceUniqueDevices': true,
    'trackDeviceInfo': true,
    'requireMfa': _envConfig.isProduction,
  };

  // Recovery and Reset
  Map<String, Duration> get recoveryTimeouts => {
    'passwordReset': const Duration(hours: 1),
    'emailVerification': const Duration(hours: 24),
    'mfaReset': const Duration(minutes: 10),
    'accountLockout': lockoutDuration,
  };

  // Audit Logging
  Set<String> get auditEvents => {
    'user.login',
    'user.logout',
    'user.password_change',
    'user.mfa_enable',
    'user.mfa_disable',
    'data.access',
    'data.modify',
    'data.delete',
    'scan.create',
    'scan.view',
    'scan.delete',
    'report.create',
    'report.view',
    'report.modify',
    'report.delete',
    'payment.initiate',
    'payment.complete',
    'payment.fail',
  };

  // Error Messages
  Map<String, String> get securityErrorMessages => {
    'invalidCredentials': 'Invalid email or password',
    'accountLocked': 'Account locked due to too many failed attempts',
    'sessionExpired': 'Your session has expired, please login again',
    'unauthorized': 'You are not authorized to perform this action',
    'invalidToken': 'Invalid or expired token',
    'biometricsRequired': 'Biometric authentication is required',
    'mfaRequired': 'Multi-factor authentication is required',
    'invalidMfaCode': 'Invalid authentication code',
  };

  // Device Trust
  Map<String, dynamic> get deviceTrustConfig => {
    'requireDeviceRegistration': _envConfig.isProduction,
    'maxDevicesPerUser': 5,
    'trustDuration': const Duration(days: 30),
    'requireLocationVerification': false,
    'allowedCountries': ['US', 'CA', 'GB', 'AU', 'NZ'],
  };
}