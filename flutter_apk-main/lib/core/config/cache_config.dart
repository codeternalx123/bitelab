// import 'package:injectable/injectable.dart';
import 'environment_config.dart';

// @singleton
class CacheConfig {
  final EnvironmentConfig _envConfig;

  CacheConfig(this._envConfig);

  // General cache settings
  int get maxCacheSize => _envConfig.cache['maxSize'] as int;
  int get maxEntries => _envConfig.cache['maxEntries'] as int;
  Duration get cleanupInterval =>
      _envConfig.cache['cleanupInterval'] as Duration;
  Duration get defaultTtl => _envConfig.cacheDuration;

  // Cache strategies
  bool shouldCacheResponse(String url) {
    final uri = Uri.parse(url);
    return _getCacheableEndpoints().contains(uri.path);
  }

  Duration getTtlForEndpoint(String url) {
    final uri = Uri.parse(url);
    return _getEndpointTtl()[uri.path] ?? defaultTtl;
  }

  Set<String> _getCacheableEndpoints() => {
        '/api/v1/user/profile',
        '/api/v1/analytics/summary',
        '/api/v1/scans/history',
        '/api/v1/reports/templates',
        '/api/v1/plans',
        // Add more cacheable endpoints
      };

  Map<String, Duration> _getEndpointTtl() => {
        '/api/v1/user/profile': const Duration(hours: 1),
        '/api/v1/analytics/summary': const Duration(minutes: 15),
        '/api/v1/scans/history': const Duration(minutes: 30),
        '/api/v1/reports/templates': const Duration(hours: 24),
        '/api/v1/plans': const Duration(hours: 6),
        // Add more endpoint-specific TTLs
      };

  // Cache invalidation rules
  bool shouldInvalidateOn(String url, String method) {
    if (method.toUpperCase() != 'GET') {
      final invalidationRules = _getInvalidationRules();
      final matchingRules = invalidationRules[method.toUpperCase()] ?? {};
      return matchingRules.any((pattern) => url.contains(pattern));
    }
    return false;
  }

  Map<String, Set<String>> _getInvalidationRules() => {
        'POST': {
          '/api/v1/user/profile',
          '/api/v1/scans',
          '/api/v1/reports',
        },
        'PUT': {
          '/api/v1/user/profile',
          '/api/v1/reports',
        },
        'PATCH': {
          '/api/v1/user/profile',
          '/api/v1/reports',
        },
        'DELETE': {
          '/api/v1/user/profile',
          '/api/v1/reports',
          '/api/v1/scans',
        },
      };

  // Cache prioritization
  double getPriority(String url) {
    final uri = Uri.parse(url);
    return _getCachePriorities()[uri.path] ?? 1.0;
  }

  Map<String, double> _getCachePriorities() => {
        '/api/v1/user/profile': 2.0, // High priority
        '/api/v1/analytics/summary': 1.5,
        '/api/v1/scans/history': 1.2,
        '/api/v1/reports/templates': 1.0, // Normal priority
        '/api/v1/plans': 0.8, // Lower priority
      };

  // Cache compression settings
  bool shouldCompressData(String url, int dataSize) {
    // Only compress data above 50KB
    return dataSize > 50 * 1024;
  }

  // Cache encryption settings
  bool shouldEncryptData(String url) {
    final uri = Uri.parse(url);
    return _getSensitiveEndpoints().contains(uri.path);
  }

  Set<String> _getSensitiveEndpoints() => {
        '/api/v1/user/profile',
        '/api/v1/payments',
        '/api/v1/medical-records',
        // Add more sensitive endpoints
      };

  // Cache prefetching rules
  bool shouldPrefetch(String url) {
    final uri = Uri.parse(url);
    return _getPrefetchEndpoints().contains(uri.path);
  }

  Set<String> _getPrefetchEndpoints() => {
        '/api/v1/plans',
        '/api/v1/reports/templates',
        // Add more prefetch endpoints
      };

  // Cache versioning
  String getCacheVersion(String url) {
    final uri = Uri.parse(url);
    return _getCacheVersions()[uri.path] ?? '1.0.0';
  }

  Map<String, String> _getCacheVersions() => {
        '/api/v1/user/profile': '1.0.0',
        '/api/v1/analytics/summary': '1.1.0',
        '/api/v1/scans/history': '1.0.1',
        '/api/v1/reports/templates': '2.0.0',
        '/api/v1/plans': '1.2.0',
      };

  // Cache synchronization settings
  bool get syncEnabled => _envConfig.sync['enabled'] as bool;
  Duration get syncInterval => _envConfig.sync['interval'] as Duration;
  int get syncBatchSize => _envConfig.sync['maxBatchSize'] as int;

  // Cache monitoring and metrics
  Map<String, dynamic> getCacheMetrics() => {
        'maxSize': maxCacheSize,
        'maxEntries': maxEntries,
        'cleanupInterval': cleanupInterval.inSeconds,
        'defaultTtl': defaultTtl.inSeconds,
        'syncEnabled': syncEnabled,
        'syncInterval': syncInterval.inSeconds,
        'syncBatchSize': syncBatchSize,
        'totalEndpoints': _getCacheableEndpoints().length,
        'sensitiveEndpoints': _getSensitiveEndpoints().length,
        'prefetchEndpoints': _getPrefetchEndpoints().length,
      };
}