// import 'package:injectable/injectable.dart';

// @singleton
class EnvironmentConfig {
  final String flavor;
  final Map<String, dynamic> _config;

  EnvironmentConfig({
    required this.flavor,
  }) : _config = _getConfigForFlavor(flavor);

  static Map<String, dynamic> _getConfigForFlavor(String flavor) {
    switch (flavor) {
      case 'prod':
        return {
          'apiBaseUrl': 'https://api.tumorheal.com/v1',
          'websocketUrl': 'wss://ws.tumorheal.com',
          'cdnBaseUrl': 'https://cdn.tumorheal.com',
          'aiModelEndpoint': 'https://ai.tumorheal.com',
          'enableLogging': false,
          'logLevel': 'error',
          'analyticsEnabled': true,
          'crashlyticsEnabled': true,
          'performanceMonitoringEnabled': true,
          'cacheDuration': const Duration(hours: 1),
          'maxRetries': 3,
          'timeouts': {
            'connect': const Duration(seconds: 30),
            'read': const Duration(seconds: 30),
            'write': const Duration(seconds: 30),
          },
          'rateLimits': {
            'api': {
              'maxRequests': 60,
              'perWindow': const Duration(minutes: 1),
            },
            'scanner': {
              'maxScans': 10,
              'perWindow': const Duration(minutes: 1),
            },
            'payment': {
              'maxAttempts': 5,
              'perWindow': const Duration(minutes: 15),
            },
          },
          'security': {
            'sslPinning': true,
            'certificateVerification': true,
            'requireBiometrics': true,
            'sessionTimeout': const Duration(minutes: 30),
            'maxPasswordAttempts': 5,
            'lockoutDuration': const Duration(minutes: 15),
          },
          'quantum': {
            'enabled': true,
            'algorithm': 'CRYSTALS-Kyber',
            'keySize': 1024,
          },
          'cache': {
            'maxSize': 100 * 1024 * 1024, // 100 MB
            'maxEntries': 1000,
            'cleanupInterval': const Duration(hours: 24),
          },
          'sync': {
            'enabled': true,
            'interval': const Duration(minutes: 15),
            'maxBatchSize': 100,
          },
        };

      case 'staging':
        return {
          'apiBaseUrl': 'https://api-staging.tumorheal.com/v1',
          'websocketUrl': 'wss://ws-staging.tumorheal.com',
          'cdnBaseUrl': 'https://cdn-staging.tumorheal.com',
          'aiModelEndpoint': 'https://ai-staging.tumorheal.com',
          'enableLogging': true,
          'logLevel': 'debug',
          'analyticsEnabled': true,
          'crashlyticsEnabled': true,
          'performanceMonitoringEnabled': true,
          'cacheDuration': const Duration(minutes: 30),
          'maxRetries': 5,
          'timeouts': {
            'connect': const Duration(seconds: 45),
            'read': const Duration(seconds: 45),
            'write': const Duration(seconds: 45),
          },
          'rateLimits': {
            'api': {
              'maxRequests': 120,
              'perWindow': const Duration(minutes: 1),
            },
            'scanner': {
              'maxScans': 20,
              'perWindow': const Duration(minutes: 1),
            },
            'payment': {
              'maxAttempts': 10,
              'perWindow': const Duration(minutes: 15),
            },
          },
          'security': {
            'sslPinning': true,
            'certificateVerification': true,
            'requireBiometrics': false,
            'sessionTimeout': const Duration(hours: 1),
            'maxPasswordAttempts': 10,
            'lockoutDuration': const Duration(minutes: 5),
          },
          'quantum': {
            'enabled': true,
            'algorithm': 'CRYSTALS-Kyber',
            'keySize': 1024,
          },
          'cache': {
            'maxSize': 200 * 1024 * 1024, // 200 MB
            'maxEntries': 2000,
            'cleanupInterval': const Duration(hours: 12),
          },
          'sync': {
            'enabled': true,
            'interval': const Duration(minutes: 5),
            'maxBatchSize': 200,
          },
        };

      default: // Development
        return {
          'apiBaseUrl': 'https://api-dev.tumorheal.com/v1',
          'websocketUrl': 'wss://ws-dev.tumorheal.com',
          'cdnBaseUrl': 'https://cdn-dev.tumorheal.com',
          'aiModelEndpoint': 'https://ai-dev.tumorheal.com',
          'enableLogging': true,
          'logLevel': 'verbose',
          'analyticsEnabled': false,
          'crashlyticsEnabled': false,
          'performanceMonitoringEnabled': false,
          'cacheDuration': const Duration(minutes: 5),
          'maxRetries': 10,
          'timeouts': {
            'connect': const Duration(minutes: 1),
            'read': const Duration(minutes: 1),
            'write': const Duration(minutes: 1),
          },
          'rateLimits': {
            'api': {
              'maxRequests': 1000,
              'perWindow': const Duration(minutes: 1),
            },
            'scanner': {
              'maxScans': 100,
              'perWindow': const Duration(minutes: 1),
            },
            'payment': {
              'maxAttempts': 100,
              'perWindow': const Duration(minutes: 15),
            },
          },
          'security': {
            'sslPinning': false,
            'certificateVerification': false,
            'requireBiometrics': false,
            'sessionTimeout': const Duration(hours: 4),
            'maxPasswordAttempts': 1000,
            'lockoutDuration': const Duration(seconds: 30),
          },
          'quantum': {
            'enabled': true,
            'algorithm': 'CRYSTALS-Kyber',
            'keySize': 512,
          },
          'cache': {
            'maxSize': 500 * 1024 * 1024, // 500 MB
            'maxEntries': 5000,
            'cleanupInterval': const Duration(hours: 6),
          },
          'sync': {
            'enabled': true,
            'interval': const Duration(minutes: 1),
            'maxBatchSize': 500,
          },
        };
    }
  }

  bool get isProduction => flavor == 'prod';
  bool get isStaging => flavor == 'staging';
  bool get isDevelopment => flavor == 'dev';

  String get apiBaseUrl => _config['apiBaseUrl'] as String;
  String get websocketUrl => _config['websocketUrl'] as String;
  String get cdnBaseUrl => _config['cdnBaseUrl'] as String;
  String get aiModelEndpoint => _config['aiModelEndpoint'] as String;

  bool get enableLogging => _config['enableLogging'] as bool;
  String get logLevel => _config['logLevel'] as String;
  bool get analyticsEnabled => _config['analyticsEnabled'] as bool;
  bool get crashlyticsEnabled => _config['crashlyticsEnabled'] as bool;
  bool get performanceMonitoringEnabled =>
      _config['performanceMonitoringEnabled'] as bool;

  Duration get cacheDuration => _config['cacheDuration'] as Duration;
  int get maxRetries => _config['maxRetries'] as int;

  Map<String, Duration> get timeouts =>
      _config['timeouts'] as Map<String, Duration>;

  Map<String, Map<String, dynamic>> get rateLimits =>
      _config['rateLimits'] as Map<String, Map<String, dynamic>>;

  Map<String, dynamic> get security => _config['security'] as Map<String, dynamic>;
  Map<String, dynamic> get quantum => _config['quantum'] as Map<String, dynamic>;
  Map<String, dynamic> get cache => _config['cache'] as Map<String, dynamic>;
  Map<String, dynamic> get sync => _config['sync'] as Map<String, dynamic>;

  T getConfig<T>(String key) => _config[key] as T;
}