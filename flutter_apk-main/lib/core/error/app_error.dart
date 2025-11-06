import 'package:freezed_annotation/freezed_annotation.dart';

part 'app_error.freezed.dart';

@freezed
class AppError with _$AppError {
  const factory AppError.network({
    required String message,
    String? endpoint,
  }) = _NetworkError;

  const factory AppError.api({
    required String message,
    String? code,
  }) = _ApiError;

  const factory AppError.unauthorized({
    String? message,
    @Default(false) bool needsReauthentication,
  }) = _UnauthorizedError;

  const factory AppError.validation({
    required String message,
    Map<String, List<String>>? fieldErrors,
  }) = _ValidationError;

  const factory AppError.security({
    required String message,
    String? type,
  }) = _SecurityError;

  const factory AppError.quantum({
    required String message,
  }) = _QuantumError;

  const factory AppError.unexpected({
    required String message,
    Object? error,
    StackTrace? stackTrace,
  }) = _UnexpectedError;

  const AppError._();

  Map<String, List<String>> toJson() => {
    'error': [map(
      network: (e) => e.message,
      api: (e) => e.message,
      unauthorized: (e) => e.message ?? 'Unauthorized',
      validation: (e) => e.message,
      security: (e) => e.message,
      quantum: (e) => e.message,
      unexpected: (e) => e.message,
    )],
  };

  String get userFriendlyMessage => map(
        network: (_) => 'Connection error. Please check your internet connection and try again.',
        api: (value) => value.message,
        unauthorized: (value) => value.message ?? 'Your session has expired. Please sign in again.',
        validation: (value) => value.message,
        security: (value) => value.message,
        quantum: (_) => 'Quantum security verification failed. Please try again.',
        unexpected: (_) => 'An unexpected error occurred. Please try again.',
      );

  String get messageForLogs => map(
        network: (value) => value.message,
        api: (value) => value.message,
        unauthorized: (value) => value.message ?? 'Unauthorized',
        validation: (value) => value.message,
        security: (value) => value.message,
        quantum: (value) => value.message,
        unexpected: (value) => value.message,
      );

  bool get isRecoverable => map(
        network: (_) => true,
        api: (e) => e.code != 'fatal',
        unauthorized: (e) => !e.needsReauthentication,
        validation: (_) => true,
        security: (e) => e.type != 'critical',
        quantum: (_) => true,
        unexpected: (_) => false,
      );

  bool get requiresReauthentication => maybeMap(
        unauthorized: (e) => e.needsReauthentication,
        orElse: () => false,
      );
}