import 'package:dio/dio.dart';
import 'package:injectable/injectable.dart';

import '../../logging/app_logger.dart';

@singleton
class ErrorInterceptor extends Interceptor {
  final AppLogger _logger;

  ErrorInterceptor({
    required AppLogger logger,
  }) : _logger = logger;

  @override
  Future<void> onError(
    DioException err,
    ErrorInterceptorHandler handler,
  ) async {
    final statusCode = err.response?.statusCode;
    final method = err.requestOptions.method;
    final path = err.requestOptions.path;

    String message = 'Network request failed';
    if (err.response?.data is Map) {
      message = err.response?.data['message'] ?? message;
    }

    _logger.error(
      'API Error: [$method] $path (Status: $statusCode) - $message',
      err,
      err.stackTrace,
    );

    // Standardize error format
    if (err.response != null) {
      err.response!.data = {
        'message': message,
        'statusCode': statusCode,
        'error': err.error.toString(),
      };
    }

    handler.next(err);
  }
}
