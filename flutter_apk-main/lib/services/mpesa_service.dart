import '../core/network/api_service.dart';
import '../core/result/result.dart';
import '../models/mpesa_models.dart';
import '../core/error/app_error.dart';
import '../config/api_config.dart';

class MpesaService {
  final ApiService _apiService;

  MpesaService({required ApiService apiService}) : _apiService = apiService;

  Future<Result<MpesaPaymentResponse>> initiatePayment(MpesaPaymentRequest request) async {
    try {
      final result = await _apiService.request<Map<String, dynamic>>(
        endpoint: ApiConfig.mpesaInitiate,
        method: HttpMethod.post,
        data: request.toJson(),
      );

      return result.when(
        success: (data) => Result.success(MpesaPaymentResponse.fromJson(data)),
        failure: (error) => Result.failure(error),
      );
    } catch (e) {
      return Result.failure(AppError.network(message: e.toString()));
    }
  }

  Future<Result<MpesaPaymentResult>> checkPaymentStatus(String checkoutRequestId) async {
    try {
      final result = await _apiService.request<Map<String, dynamic>>(
        endpoint: ApiConfig.mpesaStatus(checkoutRequestId),
        method: HttpMethod.get,
      );

      return result.when(
        success: (data) => Result.success(MpesaPaymentResult.fromJson(data)),
        failure: (error) => Result.failure(error),
      );
    } catch (e) {
      return Result.failure(AppError.network(message: e.toString()));
    }
  }
}