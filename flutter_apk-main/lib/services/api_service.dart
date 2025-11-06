import 'package:dio/dio.dart';
import '../config/keys.dart';
import '../models/payment_models.dart';

class ApiService {
  final Dio _dio = Dio(BaseOptions(baseUrl: Keys.API_BASE));

  ApiService() {
    _dio.options.connectTimeout = const Duration(milliseconds: 15000);
    _dio.options.receiveTimeout = const Duration(milliseconds: 15000);
  }

  Future<Response> post(String path, Map<String, dynamic> data, {Map<String, dynamic>? headers}) async {
    return _dio.post(path, data: data, options: Options(headers: headers));
  }

  Future<Response> get(String path, {Map<String, dynamic>? params, Map<String, dynamic>? headers}) async {
    return _dio.get(path, queryParameters: params, options: Options(headers: headers));
  }

  Future<Payment> processPayment(Map<String, dynamic> data) async {
    final response = await post('/api/v1/payments', data);
    return Payment.fromJson(response.data);
  }

  Future<Map<String, dynamic>> verifyPayment(String paymentId, Map<String, dynamic> data) async {
    final response = await post('/api/v1/payments/$paymentId/verify', data);
    return response.data;
  }

  Future<Map<String, dynamic>> analyzePaymentRisk(Map<String, dynamic> data) async {
    final response = await post('/api/v1/payments/risk-analysis', data);
    return response.data;
  }

  Future<String> createPaymentMethod(Map<String, dynamic> data) async {
    final response = await post('/api/v1/payment-methods', data);
    return response.data['id'];
  }
}
