import 'package:dio/dio.dart';
import 'package:injectable/injectable.dart';

import '../../logging/app_logger.dart';
import '../../storage/secure_storage_service.dart';

@singleton
class AuthInterceptor extends Interceptor {
  final AppLogger _logger;
  final SecureStorageService _secureStorage;

  AuthInterceptor({
    required AppLogger logger,
    required SecureStorageService secureStorage,
  })  : _logger = logger,
        _secureStorage = secureStorage;

  @override
  Future<void> onRequest(
    RequestOptions options,
    RequestInterceptorHandler handler,
  ) async {
    try {
      final token = await _secureStorage.getToken();
      if (token != null) {
        options.headers['Authorization'] = 'Bearer $token';
      }
      handler.next(options);
    } catch (e, stack) {
      _logger.error('Auth interceptor error', e, stack);
      handler.reject(
        DioException(
          requestOptions: options,
          error: e,
          stackTrace: stack,
        ),
      );
    }
  }

  @override
  Future<void> onResponse(
    Response response,
    ResponseInterceptorHandler handler,
  ) async {
    // Check for authentication errors
    if (response.statusCode == 401) {
      // Handle token refresh or logout
      await _handleAuthError();
    }
    handler.next(response);
  }

  @override
  Future<void> onError(
    DioException err,
    ErrorInterceptorHandler handler,
  ) async {
    if (err.response?.statusCode == 401) {
      // Handle token refresh or logout
      await _handleAuthError();
    }
    handler.next(err);
  }

  Future<void> _handleAuthError() async {
    try {
      // Clear stored credentials on auth error
      await _secureStorage.deleteAll();
      _logger.info('Cleared credentials due to authentication error');
    } catch (e, stack) {
      _logger.error('Error handling auth error', e, stack);
    }
  }
}
