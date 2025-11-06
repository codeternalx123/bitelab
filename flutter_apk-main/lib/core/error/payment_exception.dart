class PaymentException implements Exception {
  final String message;
  final String? code;
  final dynamic details;

  PaymentException({
    required this.message,
    this.code,
    this.details,
  });

  @override
  String toString() {
    if (code != null) {
      return 'PaymentException: $message (Code: $code)';
    }
    return 'PaymentException: $message';
  }
}