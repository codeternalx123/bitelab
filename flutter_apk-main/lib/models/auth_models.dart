import 'package:freezed_annotation/freezed_annotation.dart';
import 'user_models.dart';

part 'auth_models.g.dart';
part 'auth_models.freezed.dart';

@freezed
abstract class SignInRequest with _$SignInRequest {
  const factory SignInRequest({
    required String email,
    required String password,
    Map<String, dynamic>? metadata,
  }) = _SignInRequest;

  const SignInRequest._();

  factory SignInRequest.fromJson(Map<String, dynamic> json) =>
      _$SignInRequestFromJson(json);
  
  ValidationResult validate() {
    final errors = <String, List<String>>{};

    if (email.isEmpty) {
      errors['email'] = ['Email is required'];
    } else if (!RegExp(r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$').hasMatch(email)) {
      errors['email'] = ['Invalid email format'];
    }

    if (password.isEmpty) {
      errors['password'] = ['Password is required'];
    } else if (password.length < 8) {
      errors['password'] = ['Password must be at least 8 characters'];
    }

    return errors.isEmpty
        ? const ValidationResult.success()
        : ValidationResult.failure(errors);
  }
}

@freezed
abstract class SignInResponse with _$SignInResponse {
  const factory SignInResponse({
    @JsonKey(name: 'access_token') required String accessToken,
    @JsonKey(name: 'refresh_token') required String refreshToken,
    required User user,
    @JsonKey(name: 'expires_in') required int expiresIn,
    @JsonKey(name: 'token_type') required String tokenType,
  }) = _SignInResponse;

  factory SignInResponse.fromJson(Map<String, dynamic> json) =>
      _$SignInResponseFromJson(json);
}

@freezed
class ValidationResult with _$ValidationResult {
  const factory ValidationResult.success() = _ValidationSuccess;
  const factory ValidationResult.failure(Map<String, List<String>> errors) = _ValidationFailure;
}

@freezed
abstract class AuthResponse with _$AuthResponse {
  const factory AuthResponse({
    @JsonKey(name: 'access_token') required String accessToken,
    @JsonKey(name: 'refresh_token') required String refreshToken,
    @JsonKey(name: 'expires_at') required DateTime expiresAt,
    required User user,
  }) = _AuthResponse;

  factory AuthResponse.fromJson(Map<String, dynamic> json) =>
      _$AuthResponseFromJson(json);
}

@freezed
class AuthState with _$AuthState {
  const factory AuthState.initial() = _Initial;
  const factory AuthState.loading() = _Loading;
  const factory AuthState.authenticated(User user) = _Authenticated;
  const factory AuthState.unauthenticated() = _Unauthenticated;
  const factory AuthState.error(Map<String, List<String>> errors) = _Error;
}