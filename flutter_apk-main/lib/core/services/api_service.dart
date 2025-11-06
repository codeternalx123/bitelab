import '../error/app_error.dart';
import '../result/result.dart';

enum HttpMethod {
  get,
  post,
  put,
  delete,
  patch
}

class ApiService {
  final String baseUrl;
  
  ApiService({required this.baseUrl});

  Future<Result<T>> request<T>({
    required String endpoint,
    required HttpMethod method,
    Map<String, dynamic>? data,
    Map<String, String>? headers,
  }) async {
    try {
      // Implement actual HTTP request here
      throw UnimplementedError();
    } catch (e) {
      return Result.failure(AppError.network(message: e.toString()));
    }
  }

  Future<Result<Map<String, dynamic>>> signOut() async {
    return request(
      endpoint: 'auth/signout',
      method: HttpMethod.post,
    );
  }

  Future<Result<Map<String, dynamic>>> refreshToken(String token) async {
    return request(
      endpoint: 'auth/refresh',
      method: HttpMethod.post,
      data: {'token': token},
    );
  }

  Future<Result<Map<String, dynamic>>> updateUser(dynamic user) async {
    return request(
      endpoint: 'users/me',
      method: HttpMethod.put,
      data: user.toJson(),
    );
  }

  Future<Result<Map<String, dynamic>>> analyzeFoodImage({
    required String imagePath,
  }) async {
    return request(
      endpoint: 'scanner/analyze',
      method: HttpMethod.post,
      data: {'image_path': imagePath},
    );
  }
}
