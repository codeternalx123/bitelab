import 'dart:async';
import 'dart:io';
import 'package:dio/dio.dart';
import 'package:injectable/injectable.dart';
import 'package:retry/retry.dart';
import '../error/app_error.dart';
import '../logging/app_logger.dart';
import '../network/network_manager.dart';
import '../cache/cache_manager.dart';
import '../result/result.dart';
import 'interceptors/logging_interceptor.dart' as logging;

@singleton
class ApiService {
  final Dio _dio;
  final AppLogger _logger;
  final NetworkManager _networkManager;
  final CacheManager _cacheManager;
  final _requestQueue = <Future>[];
  final _rateLimiter = RateLimiter();

  ApiService({
    required Dio dio,
    required AppLogger logger,
    required NetworkManager networkManager,
    required CacheManager cacheManager,
  })  : _dio = dio,
        _logger = logger,
        _networkManager = networkManager,
        _cacheManager = cacheManager {
    _setupInterceptors();
  }

  void _setupInterceptors() {
    _dio.interceptors.addAll([
      RetryInterceptor(
        dio: _dio,
        logger: _logger,
        retries: 3,
      ),
      RateLimitInterceptor(
        limiter: _rateLimiter,
        logger: _logger,
      ),
      CacheInterceptor(
        cache: _cacheManager,
        logger: _logger,
      ),
      logging.LoggingInterceptor(logger: _logger),
    ]);
  }

  Future<Result<T>> request<T>({
    required String endpoint,
    required HttpMethod method,
    Map<String, dynamic>? queryParams,
    dynamic data,
    Map<String, String>? headers,
    Duration? timeout,
    bool useCache = false,
    Duration? cacheExpiration,
    bool forceRefresh = false,
    RetryConfig? retryConfig,
    RateLimitConfig? rateLimitConfig,
  }) async {
    try {
      // Check network connectivity
      if (!await _networkManager.isConnected) {
        return Result.failure(
          const AppError.network(
            message: 'No internet connection',
          ),
        );
      }

      // Check cache if enabled
      if (useCache && !forceRefresh) {
        final cachedData = await _cacheManager.get<T>(endpoint);
        if (cachedData != null) {
          return Result.success(cachedData);
        }
      }

      // Queue the request
      final completer = Completer<Result<T>>();
      _executeRequest<T>(
        endpoint: endpoint,
        method: method,
        queryParams: queryParams,
        data: data,
        headers: headers,
        timeout: timeout,
        useCache: useCache,
        cacheExpiration: cacheExpiration,
        retryConfig: retryConfig,
        rateLimitConfig: rateLimitConfig,
      ).then((result) {
        completer.complete(result);
        _requestQueue.remove(completer.future);
      });

      _requestQueue.add(completer.future);
      return completer.future;
    } catch (e, stack) {
      _logger.error('API request failed', e, stack);
      return Result.failure(
        AppError.unexpected(
          message: 'Failed to execute API request',
          error: e,
          stackTrace: stack,
        ),
      );
    }
  }

  Future<Result<T>> _executeRequest<T>({
    required String endpoint,
    required HttpMethod method,
    Map<String, dynamic>? queryParams,
    dynamic data,
    Map<String, String>? headers,
    Duration? timeout,
    bool useCache = false,
    Duration? cacheExpiration,
    RetryConfig? retryConfig,
    RateLimitConfig? rateLimitConfig,
  }) async {
    try {
      // Apply rate limiting if configured
      if (rateLimitConfig != null) {
        await _rateLimiter.acquire(
          endpoint,
          rateLimitConfig.maxRequests,
          rateLimitConfig.window,
        );
      }

      // Execute request with retry logic
      final response = await retry(
        () => _dio.request(
          endpoint,
          queryParameters: queryParams,
          data: data,
          options: Options(
            method: method.toString(),
            headers: headers,
            sendTimeout: timeout,
            receiveTimeout: timeout,
          ),
        ),
        retryIf: (e) => _shouldRetry(e),
        maxAttempts: retryConfig?.maxAttempts ?? 3,
        delayFactor: retryConfig?.delayFactor ?? const Duration(milliseconds: 200),
      );

      // Cache successful response if enabled
      if (useCache) {
        await _cacheManager.put(
          endpoint,
          response.data,
          expiration: cacheExpiration,
        );
      }

      return Result.success(response.data as T);
    } on DioException catch (e, stack) {
      return Result.failure(_handleDioError(e, stack));
    } catch (e, stack) {
      return Result.failure(
        AppError.unexpected(
          message: 'Request failed unexpectedly',
          error: e,
          stackTrace: stack,
        ),
      );
    }
  }

  bool _shouldRetry(Exception e) {
    if (e is DioException) {
      return e.type == DioExceptionType.connectionTimeout ||
             e.type == DioExceptionType.receiveTimeout ||
             e.type == DioExceptionType.sendTimeout ||
             (e.error is SocketException);
    }
    return false;
  }

  AppError _handleDioError(DioException e, StackTrace stack) {
    switch (e.type) {
      case DioExceptionType.connectionTimeout:
      case DioExceptionType.sendTimeout:
      case DioExceptionType.receiveTimeout:
        return AppError.network(
          message: 'Request timed out',
          endpoint: e.requestOptions.path,
        );
      case DioExceptionType.badResponse:
        final statusCode = e.response?.statusCode;
        if (statusCode == 401) {
          return const AppError.unauthorized(
            message: 'Authentication required',
            needsReauthentication: true,
          );
        } else if (statusCode == 403) {
          return const AppError.unauthorized(
            message: 'Access denied',
          );
        } else if (statusCode == 422) {
          return AppError.validation(
            message: 'Validation failed',
            fieldErrors: _parseValidationErrors(e.response?.data),
          );
        }
        return AppError.api(
          message: e.response?.data?['message'] ?? 'API error occurred',
          code: e.response?.data?['code']?.toString(),
        );
      case DioExceptionType.cancel:
        return const AppError.api(
          message: 'Request was cancelled',
        );
      default:
        return AppError.unexpected(
          message: 'An unexpected error occurred',
          error: e,
          stackTrace: stack,
        );
    }
  }

  Map<String, List<String>>? _parseValidationErrors(dynamic data) {
    if (data == null || data['errors'] == null) return null;
    
    final errors = <String, List<String>>{};
    (data['errors'] as Map<String, dynamic>).forEach((key, value) {
      if (value is List) {
        errors[key] = value.map((e) => e.toString()).toList();
      } else {
        errors[key] = [value.toString()];
      }
    });
    return errors;
  }

  Future<Result<Map<String, dynamic>>> signOut() async {
    return request<Map<String, dynamic>>(
      endpoint: 'auth/signout',
      method: HttpMethod.post,
    );
  }

  Future<Result<Map<String, dynamic>>> refreshToken(String token) async {
    return request<Map<String, dynamic>>(
      endpoint: 'auth/refresh',
      method: HttpMethod.post,
      data: {'token': token},
    );
  }

  Future<Result<Map<String, dynamic>>> updateUser(dynamic user) async {
    return request<Map<String, dynamic>>(
      endpoint: 'users/me',
      method: HttpMethod.put,
      data: user.toJson(),
    );
  }

  Future<Result<Map<String, dynamic>>> analyzeFoodImage({
    required String imagePath,
  }) async {
    return request<Map<String, dynamic>>(
      endpoint: 'scanner/analyze',
      method: HttpMethod.post,
      data: {'image_path': imagePath},
    );
  }
}

enum HttpMethod {
  get,
  post,
  put,
  patch,
  delete
}

class RetryConfig {
  final int maxAttempts;
  final Duration delayFactor;
  final Duration maxDelay;

  const RetryConfig({
    this.maxAttempts = 3,
    this.delayFactor = const Duration(milliseconds: 200),
    this.maxDelay = const Duration(seconds: 5),
  });
}

class RateLimitConfig {
  final int maxRequests;
  final Duration window;

  const RateLimitConfig({
    this.maxRequests = 60,
    this.window = const Duration(minutes: 1),
  });
}

class RateLimiter {
  final _timestamps = <String, List<DateTime>>{};

  Future<void> acquire(
    String endpoint,
    int maxRequests,
    Duration window,
  ) async {
    final now = DateTime.now();
    final timestamps = _timestamps[endpoint] ??= [];

    // Remove old timestamps
    timestamps.removeWhere(
      (timestamp) => now.difference(timestamp) > window,
    );

    // Check if we're at the limit
    if (timestamps.length >= maxRequests) {
      final oldestTimestamp = timestamps.first;
      final waitTime = window - now.difference(oldestTimestamp);
      if (waitTime.isNegative) {
        timestamps.removeAt(0);
      } else {
        await Future.delayed(waitTime);
        return acquire(endpoint, maxRequests, window);
      }
    }

    timestamps.add(now);
  }
}

class RetryInterceptor extends Interceptor {
  final Dio dio;
  final AppLogger logger;
  final int retries;

  RetryInterceptor({
    required this.dio,
    required this.logger,
    this.retries = 3,
  });

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) {
    var extra = err.requestOptions.extra;
    var attemptCount = (extra['attemptCount'] as int?) ?? 0;

    if (attemptCount < retries && _shouldRetry(err)) {
      attemptCount++;
      logger.info('Retrying request (attempt $attemptCount of $retries)');

      final opts = Options(
        method: err.requestOptions.method,
        headers: err.requestOptions.headers,
      );
      
      extra['attemptCount'] = attemptCount;

      final Future<Response<dynamic>> retryRequest;

      retryRequest = dio.request(
        err.requestOptions.path,
        data: err.requestOptions.data,
        queryParameters: err.requestOptions.queryParameters,
        options: opts..extra = extra,
      );

      retryRequest.then((response) => handler.resolve(response));
      return;
    }
    
    return super.onError(err, handler);
  }

  bool _shouldRetry(DioException error) {
    return error.type == DioExceptionType.connectionTimeout ||
           error.type == DioExceptionType.receiveTimeout ||
           error.type == DioExceptionType.sendTimeout ||
           (error.error is SocketException);
  }
}

class RateLimitInterceptor extends Interceptor {
  final RateLimiter limiter;
  final AppLogger logger;

  RateLimitInterceptor({
    required this.limiter,
    required this.logger,
  });

  @override
  Future<void> onRequest(
    RequestOptions options,
    RequestInterceptorHandler handler,
  ) async {
    final rateLimitConfig = options.extra['rateLimitConfig'] as RateLimitConfig?;
    
    if (rateLimitConfig != null) {
      await limiter.acquire(
        options.path,
        rateLimitConfig.maxRequests,
        rateLimitConfig.window,
      );
    }

    return super.onRequest(options, handler);
  }
}

class CacheInterceptor extends Interceptor {
  final CacheManager cache;
  final AppLogger logger;

  CacheInterceptor({
    required this.cache,
    required this.logger,
  });

  @override
  Future<void> onRequest(
    RequestOptions options,
    RequestInterceptorHandler handler,
  ) async {
    if (!options.extra.containsKey('useCache') ||
        !options.extra['useCache']) {
      return super.onRequest(options, handler);
    }

    final cacheKey = _generateCacheKey(options);
    final cachedResponse = await cache.get(cacheKey);

    if (cachedResponse != null && !options.extra['forceRefresh']) {
      logger.debug('Returning cached response for ${options.path}');
      return handler.resolve(
        Response(
          requestOptions: options,
          data: cachedResponse,
          statusCode: 200,
        ),
      );
    }

    return super.onRequest(options, handler);
  }

  @override
  void onResponse(Response response, ResponseInterceptorHandler handler) {
    if (!response.requestOptions.extra.containsKey('useCache') ||
        !response.requestOptions.extra['useCache']) {
      return super.onResponse(response, handler);
    }

    final cacheKey = _generateCacheKey(response.requestOptions);
    final expiration = response.requestOptions.extra['cacheExpiration'];

    cache.put(
      cacheKey,
      response.data,
      expiration: expiration,
    );

    return super.onResponse(response, handler);
  }

  String _generateCacheKey(RequestOptions options) {
    return '${options.method}:${options.path}:${options.queryParameters.toString()}';
  }
}