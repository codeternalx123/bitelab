import 'package:freezed_annotation/freezed_annotation.dart';

part 'error_models.freezed.dart';
part 'error_models.g.dart';

@freezed
class ApiError with _$ApiError {
  const ApiError._();

  const factory ApiError({
    required String code,
    required String message,
    String? target,
    List<ApiErrorDetail>? details,
    Map<String, dynamic>? metadata,
  }) = _ApiError;

  factory ApiError.fromJson(Map<String, dynamic> json) =>
      _$ApiErrorFromJson(json);

  bool get isTransient =>
      code.startsWith('transient.') || _transientCodes.contains(code);

  bool get isAuthError =>
      code.startsWith('auth.') ||
      code.contains('unauthorized') ||
      code.contains('forbidden');

  bool get isValidationError =>
      code.startsWith('validation.') || details?.isNotEmpty == true;

  bool get isNetworkError =>
      code.startsWith('network.') ||
      code.contains('timeout') ||
      code.contains('connection');

  bool get isServerError => code.startsWith('server.');

  bool get isClientError => code.startsWith('client.');

  bool get requiresReauthentication =>
      code == 'auth.session.expired' ||
      code == 'auth.token.invalid' ||
      code == 'auth.requires_recent_login';

  Map<String, List<String>> get validationErrors {
    if (!isValidationError || details == null) return {};

    final errors = <String, List<String>>{};
    for (final detail in details!) {
      if (detail.target != null) {
        errors[detail.target!] = [detail.message];
      }
    }
    return errors;
  }

  String get userFriendlyMessage {
    if (isTransient) {
      return 'Please try again in a moment.';
    } else if (isAuthError) {
      if (requiresReauthentication) {
        return 'Your session has expired. Please sign in again.';
      }
      return 'Authentication failed. Please check your credentials.';
    } else if (isValidationError) {
      return 'Please check your input and try again.';
    } else if (isNetworkError) {
      return 'Connection error. Please check your internet connection.';
    } else if (isServerError) {
      return 'Server error. Please try again later.';
    } else if (isClientError) {
      return 'An error occurred in the app. Please try again.';
    }
    return message;
  }

  static const _transientCodes = {
    'rate_limit_exceeded',
    'service_unavailable',
    'gateway_timeout',
    'temporarily_unavailable',
    'too_many_requests',
  };
}

@freezed
class ApiErrorDetail with _$ApiErrorDetail {
  const ApiErrorDetail._();

  const factory ApiErrorDetail({
    required String code,
    required String message,
    String? target,
    Map<String, dynamic>? metadata,
  }) = _ApiErrorDetail;

  factory ApiErrorDetail.fromJson(Map<String, dynamic> json) =>
      _$ApiErrorDetailFromJson(json);
}

@freezed
class ErrorResponse with _$ErrorResponse {
  const ErrorResponse._();

  const factory ErrorResponse({
    required String requestId,
    required DateTime timestamp,
    required ApiError error,
    Map<String, dynamic>? debugInfo,
  }) = _ErrorResponse;

  factory ErrorResponse.fromJson(Map<String, dynamic> json) =>
      _$ErrorResponseFromJson(json);

  Duration get age => DateTime.now().difference(timestamp);

  Map<String, dynamic> toDebugReport() => {
        'requestId': requestId,
        'timestamp': timestamp.toIso8601String(),
        'error': {
          'code': error.code,
          'message': error.message,
          'target': error.target,
          'details': error.details?.map((d) => {
                'code': d.code,
                'message': d.message,
                'target': d.target,
              }).toList(),
        },
        'debugInfo': debugInfo,
      };
}

class ValidationErrorDetail {
  final String field;
  final String message;
  final String? code;
  final dynamic invalidValue;

  ValidationErrorDetail({
    required this.field,
    required this.message,
    this.code,
    this.invalidValue,
  });

  factory ValidationErrorDetail.fromJson(Map<String, dynamic> json) {
    return ValidationErrorDetail(
      field: json['field'] as String,
      message: json['message'] as String,
      code: json['code'] as String?,
      invalidValue: json['invalidValue'],
    );
  }

  Map<String, dynamic> toJson() => {
        'field': field,
        'message': message,
        if (code != null) 'code': code,
        if (invalidValue != null) 'invalidValue': invalidValue,
      };
}

class ValidationError implements Exception {
  final String message;
  final List<ValidationErrorDetail> details;

  ValidationError(this.message, this.details);

  Map<String, List<String>> get fieldErrors {
    return {
      for (var detail in details)
        detail.field: [detail.message]
    };
  }

  factory ValidationError.fromJson(Map<String, dynamic> json) {
    return ValidationError(
      json['message'] as String,
      (json['details'] as List)
          .map((detail) =>
              ValidationErrorDetail.fromJson(detail as Map<String, dynamic>))
          .toList(),
    );
  }

  Map<String, dynamic> toJson() => {
        'message': message,
        'details': details.map((detail) => detail.toJson()).toList(),
      };
}