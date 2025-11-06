import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../core/services/payment_service.dart';
import 'api_service_provider.dart';

final paymentServiceProvider = Provider<PaymentService>((ref) {
  final apiService = ref.watch(apiServiceProvider);
  return PaymentService(apiService: apiService);
});