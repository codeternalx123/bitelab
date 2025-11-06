import 'package:json_annotation/json_annotation.dart';

part 'api_error.g.dart';

@JsonSerializable()
class ApiError {
  final ErrorDetail detail;

  ApiError({required this.detail});

  factory ApiError.fromJson(Map<String, dynamic> json) =>
      _$ApiErrorFromJson(json);

  Map<String, dynamic> toJson() => _$ApiErrorToJson(this);
}

@JsonSerializable()
class ErrorDetail {
  final String message;
  final String code;
  final Map<String, dynamic>? params;

  ErrorDetail({
    required this.message,
    required this.code,
    this.params,
  });

  factory ErrorDetail.fromJson(Map<String, dynamic> json) =>
      _$ErrorDetailFromJson(json);

  Map<String, dynamic> toJson() => _$ErrorDetailToJson(this);
}

class ApiException implements Exception {
  final String message;
  final String code;
  final Map<String, dynamic>? params;

  ApiException({
    required this.message,
    required this.code,
    this.params,
  });

  factory ApiException.fromError(ApiError error) {
    return ApiException(
      message: error.detail.message,
      code: error.detail.code,
      params: error.detail.params,
    );
  }

  @override
  String toString() => message;
}

// Payment-specific exceptions
class PaymentException extends ApiException {
  PaymentException({
    required super.message,
    required super.code,
    super.params,
  });

  static bool isPaymentError(String code) => code.startsWith('P');
}

// Authentication-specific exceptions
class AuthException extends ApiException {
  AuthException({
    required super.message,
    required super.code,
    super.params,
  });

  static bool isAuthError(String code) => code.startsWith('A');
}

// Security-specific exceptions
class SecurityException extends ApiException {
  SecurityException({
    required super.message,
    required super.code,
    super.params,
  });

  static bool isSecurityError(String code) => code.startsWith('S');
}