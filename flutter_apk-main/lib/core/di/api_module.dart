import 'package:injectable/injectable.dart';
import 'package:dio/dio.dart';
import '../network/api_service.dart';
import '../network/interceptors/auth_interceptor.dart' as auth;
import '../network/interceptors/cache_interceptor.dart' as cache;
import '../network/interceptors/error_interceptor.dart' as error;
import '../network/interceptors/retry_interceptor.dart' as retry;
import '../network/interceptors/logging_interceptor.dart' as logging;
import '../storage/secure_storage_service.dart';
import '../logging/app_logger.dart';
import '../cache/cache_manager.dart';
import '../network/network_manager.dart';
import '../network/request_queue_manager.dart';
import '../network/rate_limiter.dart' as rate_limiter;

@module
abstract class ApiModule {
  @singleton
  Dio getDio(
    AppLogger logger,
    NetworkManager networkManager,
    CacheManager cacheManager,
    SecureStorageService secureStorage,
  ) {
    final dio = Dio(BaseOptions(
      baseUrl: 'https://api.tumorheal.com/v1',
      connectTimeout: const Duration(seconds: 30),
      receiveTimeout: const Duration(seconds: 30),
      sendTimeout: const Duration(seconds: 30),
      validateStatus: (status) => status != null && status < 500,
    ));

    // Initialize interceptors
    final authInterceptor = auth.AuthInterceptor(
      logger: logger,
      secureStorage: secureStorage,
    );

    final cacheInterceptor = cache.CacheInterceptor(
      logger: logger,
      cache: cacheManager,
    );

    final retryInterceptor = retry.RetryInterceptor(
      logger: logger,
      dio: dio,
    );

    final errorInterceptor = error.ErrorInterceptor(
      logger: logger,
    );

    final loggingInterceptor = logging.LoggingInterceptor(
      logger: logger,
    );

    // Add interceptors in order of execution
    dio.interceptors.addAll([
      authInterceptor,
      cacheInterceptor,
      retryInterceptor,
      errorInterceptor,
      loggingInterceptor,
    ]);

    return dio;
  }

  @singleton
  ApiService getApiService(
    Dio dio,
    AppLogger logger,
    NetworkManager networkManager,
    CacheManager cacheManager,
  ) {
    return ApiService(
      dio: dio,
      logger: logger,
      networkManager: networkManager,
      cacheManager: cacheManager,
    );
  }

  @singleton
  auth.AuthInterceptor getAuthInterceptor(
    AppLogger logger,
    SecureStorageService secureStorage,
  ) {
    return auth.AuthInterceptor(
      logger: logger,
      secureStorage: secureStorage,
    );
  }

  @singleton
  cache.CacheInterceptor getCacheInterceptor(
    AppLogger logger,
    CacheManager cacheManager,
  ) {
    return cache.CacheInterceptor(
      logger: logger,
      cache: cacheManager,
    );
  }

  @singleton
  retry.RetryInterceptor getRetryInterceptor(
    AppLogger logger,
    Dio dio,
  ) {
    return retry.RetryInterceptor(
      logger: logger,
      dio: dio,
    );
  }

  @singleton
  error.ErrorInterceptor getErrorInterceptor(
    AppLogger logger,
  ) {
    return error.ErrorInterceptor(
      logger: logger,
    );
  }

  @singleton
  logging.LoggingInterceptor getLoggingInterceptor(
    AppLogger logger,
  ) {
    return logging.LoggingInterceptor(
      logger: logger,
    );
  }

  @singleton
  RequestQueueManager getRequestQueueManager(
    AppLogger logger,
  ) {
    return RequestQueueManager(logger);
  }

  @singleton
  rate_limiter.RateLimiter getRateLimiter(
    AppLogger logger,
  ) {
    logger.debug('Creating RateLimiter with logger: $logger');
    return rate_limiter.RateLimiter(logger: logger);
  }
}