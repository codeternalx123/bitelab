import 'package:dio/dio.dart';
import 'package:injectable/injectable.dart';

import '../../logging/app_logger.dart';

@singleton
class RetryInterceptor extends Interceptor {
  final AppLogger _logger;
  final Dio _dio;
  static const int maxRetries = 3;
  static const Duration retryDelay = Duration(seconds: 1);

  RetryInterceptor({
    required AppLogger logger,
    required Dio dio,
  })  : _logger = logger,
        _dio = dio;

  @override
  Future<void> onError(
    DioException err,
    ErrorInterceptorHandler handler,
  ) async {
    var extra = err.requestOptions.extra;
    final retryCount = (extra['retryCount'] as int?) ?? 0;

    if (_shouldRetry(err) && retryCount < maxRetries) {
      try {
        extra = Map.from(extra);
        extra['retryCount'] = retryCount + 1;

        // Add exponential backoff delay
        final delay = Duration(milliseconds: (retryDelay.inMilliseconds * (1 << retryCount)));
        await Future.delayed(delay);

        _logger.debug(
          'Retrying request (${retryCount + 1}/$maxRetries) after $delay: ${err.requestOptions.uri}',
        );

        final options = Options(
          method: err.requestOptions.method,
          headers: err.requestOptions.headers,
          extra: extra,
        );

        final response = await _dio.request(
          err.requestOptions.path,
          data: err.requestOptions.data,
          queryParameters: err.requestOptions.queryParameters,
          options: options,
        );

        handler.resolve(response);
        return;
      } catch (e) {
        _logger.error('Retry attempt failed', e);
      }
    }

    handler.next(err);
  }

  bool _shouldRetry(DioException error) {
    // Retry on connection errors and server errors (5xx)
    return error.type == DioExceptionType.connectionError ||
        error.response?.statusCode == null ||
        (error.response!.statusCode! >= 500 && error.response!.statusCode! < 600);
  }
}
