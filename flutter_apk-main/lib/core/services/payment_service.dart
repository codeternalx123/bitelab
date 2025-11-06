import '../error/app_error.dart';
import '../network/api_service.dart';
import '../result/result.dart';
import '../../models/payment_models.dart';

class PaymentService {
  final ApiService _apiService;

  PaymentService({required ApiService apiService}) : _apiService = apiService;

  Future<Result<Payment>> processPayment(PaymentRequest request) async {
    try {
      final result = await _apiService.request<Map<String, dynamic>>(
        endpoint: 'payments/process',
        method: HttpMethod.post,
        data: request.toJson(),
      );

      return result.when(
        success: (data) => Result.success(Payment.fromJson(data)),
        failure: (error) => Result.failure(error),
      );
    } catch (e) {
      return Result.failure(AppError.network(message: e.toString()));
    }
  }

  Future<Result<bool>> verifyPayment(
    String paymentId,
    Map<String, dynamic> securedPayment,
  ) async {
    try {
      final result = await _apiService.request<Map<String, dynamic>>(
        endpoint: 'payments/verify/$paymentId',
        method: HttpMethod.post,
        data: {'secured_payment': securedPayment},
      );

      return result.when(
        success: (data) => Result.success(data['is_verified'] == true),
        failure: (error) => Result.failure(error),
      );
    } catch (e) {
      return Result.failure(AppError.network(message: e.toString()));
    }
  }

  Future<Result<Map<String, dynamic>>> analyzePaymentRisk(PaymentRequest request) async {
    try {
      final result = await _apiService.request<Map<String, dynamic>>(
        endpoint: 'payments/analyze-risk',
        method: HttpMethod.post,
        data: request.toJson(),
      );
      return result;
    } catch (e) {
      return Result.failure(AppError.network(message: e.toString()));
    }
  }

  Future<Result<Payment>> payWithCard({
    required int amountCents,
    required String currency,
  }) async {
    try {
      final paymentMethodResult = await _apiService.request<String>(
        endpoint: 'payment-methods',
        method: HttpMethod.post,
        data: {
          'type': 'card',
          'amount': amountCents,
          'currency': currency,
        },
      );

      return paymentMethodResult.when(
        success: (paymentMethodId) async {
          final paymentRequest = PaymentRequest(
            amount: amountCents / 100,
            currency: currency,
            paymentMethodId: paymentMethodId,
            provider: 'stripe',
          );

          return processPayment(paymentRequest);
        },
        failure: (error) => Result.failure(error),
      );
    } catch (e) {
      return Result.failure(AppError.network(message: e.toString()));
    }
  }
}