import '../../models/auth_models.dart';
import '../../models/user_models.dart';
import '../error/app_error.dart';
import '../network/api_service.dart';
import '../result/result.dart';

class AuthService {
  final ApiService _apiService;

  AuthService({required ApiService apiService}) : _apiService = apiService;

  Future<Result<AuthResponse>> signIn(SignInRequest request) async {
    try {
      final result = await _apiService.request<Map<String, dynamic>>(
        endpoint: 'auth/signin',
        method: HttpMethod.post,
        data: request.toJson(),
      );
      
      return result.when(
        success: (data) => Result.success(AuthResponse.fromJson(data)),
        failure: (error) => Result.failure(error),
      );
    } catch (e) {
      return Result.failure(AppError.network(message: e.toString()));
    }
  }

  Future<Result<void>> signOut() async {
    try {
      final result = await _apiService.request<Map<String, dynamic>>(
        endpoint: 'auth/signout',
        method: HttpMethod.post,
      );
      return result.when(
        success: (_) => const Result.success(null),
        failure: (error) => Result.failure(error),
      );
    } catch (e) {
      return Result.failure(AppError.network(message: e.toString()));
    }
  }

  Future<Result<User>> getCurrentUser() async {
    try {
      final result = await _apiService.request<Map<String, dynamic>>(
        endpoint: 'auth/me',
        method: HttpMethod.get,
      );
      
      return result.when(
        success: (data) => Result.success(User.fromJson(data)),
        failure: (error) => Result.failure(error),
      );
    } catch (e) {
      return Result.failure(AppError.network(message: e.toString()));
    }
  }

  Future<Result<AuthResponse>> refreshToken(String refreshToken) async {
    try {
      final result = await _apiService.request<Map<String, dynamic>>(
        endpoint: 'auth/refresh',
        method: HttpMethod.post,
        data: {'refresh_token': refreshToken},
      );
      
      return result.when(
        success: (data) => Result.success(AuthResponse.fromJson(data)),
        failure: (error) => Result.failure(error),
      );
    } catch (e) {
      return Result.failure(AppError.network(message: e.toString()));
    }
  }

  Future<Result<User>> updateUser(User userData) async {
    try {
      final result = await _apiService.request<Map<String, dynamic>>(
        endpoint: 'auth/update',
        method: HttpMethod.put,
        data: userData.toJson(),
      );

      return result.when(
        success: (data) => Result.success(User.fromJson(data)),
        failure: (error) => Result.failure(error),
      );
    } catch (e) {
      return Result.failure(AppError.network(message: e.toString()));
    }
  }
}