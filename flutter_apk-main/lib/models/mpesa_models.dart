class MpesaPaymentRequest {
  final String phoneNumber;
  final double amount;
  final String accountReference;
  final String description;
  final String? callbackUrl;

  const MpesaPaymentRequest({
    required this.phoneNumber,
    required this.amount,
    required this.accountReference,
    required this.description,
    this.callbackUrl,
  });

  Map<String, dynamic> toJson() => {
    'phone_number': phoneNumber,
    'amount': amount,
    'account_reference': accountReference,
    'description': description,
    if (callbackUrl != null) 'callback_url': callbackUrl,
  };

  factory MpesaPaymentRequest.fromJson(Map<String, dynamic> json) {
    return MpesaPaymentRequest(
      phoneNumber: json['phone_number'] as String,
      amount: json['amount'] as double,
      accountReference: json['account_reference'] as String,
      description: json['description'] as String,
      callbackUrl: json['callback_url'] as String?,
    );
  }
}

class MpesaPaymentResponse {
  final String merchantRequestId;
  final String checkoutRequestId;
  final String responseCode;
  final String responseDescription;
  final String customerMessage;

  const MpesaPaymentResponse({
    required this.merchantRequestId,
    required this.checkoutRequestId,
    required this.responseCode,
    required this.responseDescription,
    required this.customerMessage,
  });

  Map<String, dynamic> toJson() => {
    'merchant_request_id': merchantRequestId,
    'checkout_request_id': checkoutRequestId,
    'response_code': responseCode,
    'response_description': responseDescription,
    'customer_message': customerMessage,
  };

  factory MpesaPaymentResponse.fromJson(Map<String, dynamic> json) {
    return MpesaPaymentResponse(
      merchantRequestId: json['merchant_request_id'] as String,
      checkoutRequestId: json['checkout_request_id'] as String,
      responseCode: json['response_code'] as String,
      responseDescription: json['response_description'] as String,
      customerMessage: json['customer_message'] as String,
    );
  }
}

class MpesaPaymentResult {
  final String transactionId;
  final String resultCode;
  final String resultDescription;
  final String? mpesaReceiptNumber;
  final DateTime? transactionDate;
  final double? amount;
  final String? phoneNumber;

  const MpesaPaymentResult({
    required this.transactionId,
    required this.resultCode,
    required this.resultDescription,
    this.mpesaReceiptNumber,
    this.transactionDate,
    this.amount,
    this.phoneNumber,
  });

  Map<String, dynamic> toJson() => {
    'transaction_id': transactionId,
    'result_code': resultCode,
    'result_description': resultDescription,
    if (mpesaReceiptNumber != null) 'mpesa_receipt_number': mpesaReceiptNumber,
    if (transactionDate != null) 'transaction_date': transactionDate!.toIso8601String(),
    if (amount != null) 'amount': amount,
    if (phoneNumber != null) 'phone_number': phoneNumber,
  };

  factory MpesaPaymentResult.fromJson(Map<String, dynamic> json) {
    return MpesaPaymentResult(
      transactionId: json['transaction_id'] as String,
      resultCode: json['result_code'] as String,
      resultDescription: json['result_description'] as String,
      mpesaReceiptNumber: json['mpesa_receipt_number'] as String?,
      transactionDate: json['transaction_date'] != null 
          ? DateTime.parse(json['transaction_date'] as String)
          : null,
      amount: json['amount'] as double?,
      phoneNumber: json['phone_number'] as String?,
    );
  }
}