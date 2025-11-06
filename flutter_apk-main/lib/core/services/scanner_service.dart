import 'package:dio/dio.dart';
import '../error/app_error.dart';
import '../network/api_service.dart';
import '../result/result.dart';

class ScannerService {
  final ApiService _apiService;

  ScannerService({required ApiService apiService}) : _apiService = apiService;

  Future<Result<Map<String, dynamic>>> analyzeFoodImage(String imagePath) async {
    try {
      final formData = FormData.fromMap({
        'image': await MultipartFile.fromFile(imagePath),
      });

      final result = await _apiService.request<Map<String, dynamic>>(
        endpoint: 'scanner/analyze-food',
        method: HttpMethod.post,
        data: formData,
      );
      return result;
    } catch (e) {
      return Result.failure(AppError.network(message: e.toString()));
    }
  }

  Future<Result<Map<String, dynamic>>> getNutritionInfo(String foodId) async {
    try {
      final result = await _apiService.request<Map<String, dynamic>>(
        endpoint: 'scanner/nutrition/$foodId',
        method: HttpMethod.get,
      );
      return result;
    } catch (e) {
      return Result.failure(AppError.network(message: e.toString()));
    }
  }
}