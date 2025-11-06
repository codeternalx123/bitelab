import 'api_service.dart';
import '../models/payment_models.dart';
import '../core/error/payment_exception.dart';

class PaymentService {
  final ApiService _api;

  PaymentService({required ApiService apiService}) : _api = apiService;

  Future<Payment> processPayment({
    required double amount,
    required String currency,
    required String paymentMethodId,
    required String provider,
    String? subscriptionId,
  }) async {
    final payment = await _api.processPayment({
      'amount': amount,
      'currency': currency,
      'payment_method_id': paymentMethodId,
      'provider': provider,
      if (subscriptionId != null) 'subscription_id': subscriptionId,
    });

    final riskAnalysis = await analyzePaymentRisk(PaymentRequest(
      amount: amount,
      currency: currency,
      paymentMethodId: paymentMethodId,
      provider: provider,
    ));

    if (riskAnalysis['risk_analysis']['is_fraudulent'] == true) {
      throw PaymentException(
        message: 'Payment flagged as potentially fraudulent',
        code: 'FRAUD_DETECTED',
      );
    }

    return payment;
  }

  Future<bool> verifyPayment(String paymentId, Map<String, dynamic> verificationData) async {
    final response = await _api.verifyPayment(paymentId, verificationData);
    return response['status'] == 'verified';
  }

  Future<Map<String, dynamic>> analyzePaymentRisk(PaymentRequest request) async {
    return _api.analyzePaymentRisk(request.toJson());
  }

  Future<Payment> payWithCard({
    required int amountCents,
    required String currency,
  }) async {
    final paymentMethodId = await _api.createPaymentMethod({
      'type': 'card',
      'amount': amountCents,
      'currency': currency,
    });

    return processPayment(
      amount: amountCents / 100,
      currency: currency,
      paymentMethodId: paymentMethodId,
      provider: 'stripe',
    );
  }
}
