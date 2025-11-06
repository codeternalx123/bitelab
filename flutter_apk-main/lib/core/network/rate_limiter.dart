import 'dart:async';
import 'package:injectable/injectable.dart';
import '../logging/app_logger.dart';

@singleton
class RateLimiter {
  final AppLogger _logger;
  final Map<String, DateTime> _lastRequestTime = {};
  final Duration _minInterval;

  RateLimiter({
    required AppLogger logger,
    Duration minInterval = const Duration(milliseconds: 100),
  })  : _logger = logger,
        _minInterval = minInterval;

  Future<void> checkRateLimit(String endpoint) async {
    final now = DateTime.now();
    final lastRequest = _lastRequestTime[endpoint];

    if (lastRequest != null) {
      final timeSinceLastRequest = now.difference(lastRequest);
      if (timeSinceLastRequest < _minInterval) {
        final waitTime = _minInterval - timeSinceLastRequest;
        _logger.d('Rate limiting $endpoint, waiting for $waitTime');
        await Future.delayed(waitTime);
      }
    }

    _lastRequestTime[endpoint] = now;
  }

  void reset(String endpoint) {
    _lastRequestTime.remove(endpoint);
  }
}