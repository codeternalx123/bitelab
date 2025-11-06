import 'package:dio/dio.dart';
import 'package:injectable/injectable.dart';

import '../../logging/app_logger.dart';

@singleton
class LoggingInterceptor extends Interceptor {
  final AppLogger _logger;

  LoggingInterceptor({
    required AppLogger logger,
  }) : _logger = logger;

  @override
  void onRequest(
    RequestOptions options,
    RequestInterceptorHandler handler,
  ) {
    final requestId = DateTime.now().millisecondsSinceEpoch;
    options.extra['requestId'] = requestId;

    _logger.debug('''
    🌐 Request [$requestId]
    ${options.method} ${options.uri}
    Headers: ${options.headers}
    Query: ${options.queryParameters}
    Data: ${options.data}
    ''');

    handler.next(options);
  }

  @override
  void onResponse(
    Response response,
    ResponseInterceptorHandler handler,
  ) {
    final requestId = response.requestOptions.extra['requestId'];

    _logger.debug('''
    ✅ Response [$requestId]
    ${response.requestOptions.method} ${response.requestOptions.uri}
    Status: ${response.statusCode}
    Headers: ${response.headers}
    Data: ${response.data}
    ''');

    handler.next(response);
  }

  @override
  void onError(
    DioException err,
    ErrorInterceptorHandler handler,
  ) {
    final requestId = err.requestOptions.extra['requestId'];

    _logger.error('''
    ❌ Error [$requestId]
    ${err.requestOptions.method} ${err.requestOptions.uri}
    Status: ${err.response?.statusCode}
    Type: ${err.type}
    Message: ${err.message}
    Error: ${err.error}
    ''', err, err.stackTrace);

    handler.next(err);
  }
}
