import 'package:dio/dio.dart';
// import 'package:injectable/injectable.dart';
import 'environment_config.dart';

// @singleton
class NetworkConfig {
  final EnvironmentConfig _envConfig;

  NetworkConfig(this._envConfig);

  BaseOptions get dioOptions => BaseOptions(
        baseUrl: _envConfig.apiBaseUrl,
        connectTimeout: _envConfig.timeouts['connect'] as Duration,
        receiveTimeout: _envConfig.timeouts['read'] as Duration,
        sendTimeout: _envConfig.timeouts['write'] as Duration,
        validateStatus: (status) => status! >= 200 && status < 300,
        headers: _getDefaultHeaders(),
      );

  Map<String, String> _getDefaultHeaders() {
    return {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-Client-Version': '1.0.0', // TODO: Get from package info
      'X-Platform': 'flutter',
      'X-Flavor': _envConfig.flavor,
    };
  }

  bool get shouldVerifyCertificate =>
      _envConfig.security['certificateVerification'] as bool;
      
  bool get shouldPinSSL => _envConfig.security['sslPinning'] as bool;

  List<String> get allowedHosts => [
        Uri.parse(_envConfig.apiBaseUrl).host,
        Uri.parse(_envConfig.cdnBaseUrl).host,
        Uri.parse(_envConfig.aiModelEndpoint).host,
      ];

  Map<String, String> get certificateHashes => {
        // Production certificate hashes
        'api.tumorheal.com':
            'sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=',
        'cdn.tumorheal.com':
            'sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=',
        'ai.tumorheal.com':
            'sha256/CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC=',

        // Staging certificate hashes
        'api-staging.tumorheal.com':
            'sha256/DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD=',
        'cdn-staging.tumorheal.com':
            'sha256/EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE=',
        'ai-staging.tumorheal.com':
            'sha256/FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF=',

        // Development certificate hashes (for testing)
        'api-dev.tumorheal.com':
            'sha256/GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG=',
        'cdn-dev.tumorheal.com':
            'sha256/HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH=',
        'ai-dev.tumorheal.com':
            'sha256/IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII=',
      };

  // Rate limiting configuration
  Map<String, dynamic> get apiRateLimit =>
      _envConfig.rateLimits['api'] as Map<String, dynamic>;
      
  Map<String, dynamic> get scannerRateLimit =>
      _envConfig.rateLimits['scanner'] as Map<String, dynamic>;
      
  Map<String, dynamic> get paymentRateLimit =>
      _envConfig.rateLimits['payment'] as Map<String, dynamic>;

  // Retry configuration
  int get maxRetries => _envConfig.maxRetries;
  
  // WebSocket configuration
  String get websocketUrl => _envConfig.websocketUrl;
  Duration get websocketPingInterval => const Duration(seconds: 30);
  Duration get websocketReconnectDelay => const Duration(seconds: 5);
  int get websocketMaxReconnectAttempts => 5;

  // Cache configuration
  Duration get cacheDuration => _envConfig.cacheDuration;
  int get maxCacheSize => _envConfig.cache['maxSize'] as int;
  int get maxCacheEntries => _envConfig.cache['maxEntries'] as int;
  Duration get cacheCleanupInterval =>
      _envConfig.cache['cleanupInterval'] as Duration;

  // Sync configuration
  bool get syncEnabled => _envConfig.sync['enabled'] as bool;
  Duration get syncInterval => _envConfig.sync['interval'] as Duration;
  int get syncBatchSize => _envConfig.sync['maxBatchSize'] as int;

  // Quantum-safe configuration
  bool get quantumSafeEnabled => _envConfig.quantum['enabled'] as bool;
  String get quantumAlgorithm => _envConfig.quantum['algorithm'] as String;
  int get quantumKeySize => _envConfig.quantum['keySize'] as int;
}