import 'package:freezed_annotation/freezed_annotation.dart';

part 'payment_models.freezed.dart';
part 'payment_models.g.dart';

@freezed
class PaymentRequest with _$PaymentRequest {
  const factory PaymentRequest({
    required String cardNumber,
    required String expiryDate,
    required String cvv,
    required String cardHolderName,
    required double amount,
    String? currency,
    String? description,
  }) = _PaymentRequest;

  factory PaymentRequest.fromJson(Map<String, dynamic> json) =>
      _$PaymentRequestFromJson(json);
}

@freezed
class PaymentResponse with _$PaymentResponse {
  const factory PaymentResponse({
    required String transactionId,
    required String status,
    required DateTime timestamp,
    String? message,
    Map<String, dynamic>? additionalData,
  }) = _PaymentResponse;

  factory PaymentResponse.fromJson(Map<String, dynamic> json) =>
      _$PaymentResponseFromJson(json);
}

@freezed
class PaymentStatus with _$PaymentStatus {
  const factory PaymentStatus.success(PaymentResponse response) = PaymentSuccess;
  const factory PaymentStatus.failure(String message) = PaymentFailure;
  const factory PaymentStatus.loading() = PaymentLoading;
  const factory PaymentStatus.initial() = PaymentInitial;
}

class PaymentResult {
  final FraudAnalysis fraudAnalysis;
  
  PaymentResult({
    required this.fraudAnalysis,
  });
}

class FraudAnalysis {
  final bool isFraudulent;
  
  FraudAnalysis({
    this.isFraudulent = false,
  });
}