class ApiConfig {
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://localhost:8000',  // FastAPI default port
  );

  static const Map<String, String> defaultHeaders = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  };

  // M-Pesa API endpoints
  static const String mpesaInitiate = '/api/v1/payments/mpesa/initiate';
  static const String mpesaCallback = '/api/v1/payments/mpesa/callback';
  static String mpesaStatus(String checkoutRequestId) => 
      '/api/v1/payments/mpesa/status/$checkoutRequestId';
}