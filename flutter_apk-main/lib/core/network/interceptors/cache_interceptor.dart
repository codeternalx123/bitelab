import 'package:dio/dio.dart';
import 'package:injectable/injectable.dart';

import '../../cache/cache_manager.dart';
import '../../logging/app_logger.dart';

@singleton
class CacheInterceptor extends Interceptor {
  final AppLogger _logger;
  final CacheManager _cache;

  CacheInterceptor({
    required AppLogger logger,
    required CacheManager cache,
  })  : _logger = logger,
        _cache = cache;

  @override
  Future<void> onRequest(
    RequestOptions options,
    RequestInterceptorHandler handler,
  ) async {
    // Only cache GET requests
    if (options.method != 'GET') {
      return handler.next(options);
    }

    try {
      final key = _generateCacheKey(options);
      final cachedResponse = await _cache.get(key);
      if (cachedResponse != null) {
        _logger.debug('Cache hit for key: $key');
        return handler.resolve(
          Response(
            requestOptions: options,
            data: cachedResponse,
            statusCode: 200,
          ),
        );
      }
      _logger.debug('Cache miss for key: $key');
      handler.next(options);
    } catch (e, stack) {
      _logger.error('Cache interceptor error', e, stack);
      handler.next(options);
    }
  }

  @override
  void onResponse(Response response, ResponseInterceptorHandler handler) {
    // Only cache successful GET requests
    if (response.requestOptions.method == 'GET' &&
        response.statusCode == 200 &&
        response.data != null) {
      try {
        final key = _generateCacheKey(response.requestOptions);
        _cache.set(key, response.data);
        _logger.debug('Cached response for key: $key');
      } catch (e, stack) {
        _logger.error('Error caching response', e, stack);
      }
    }
    handler.next(response);
  }

  String _generateCacheKey(RequestOptions options) {
    final buffer = StringBuffer();
    buffer.write(options.method);
    buffer.write(':');
    buffer.write(options.uri.toString());
    
    if (options.queryParameters.isNotEmpty) {
      buffer.write('?');
      buffer.write(options.queryParameters.entries
          .map((e) => '${e.key}=${e.value}')
          .join('&'));
    }

    return buffer.toString();
  }
}
