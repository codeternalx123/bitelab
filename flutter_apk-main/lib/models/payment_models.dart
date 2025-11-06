import 'package:freezed_annotation/freezed_annotation.dart';

part 'payment_models.freezed.dart';
part 'payment_models.g.dart';

@freezed
class PaymentRequest with _$PaymentRequest {
  const factory PaymentRequest({
    required double amount,
    required String currency,
    required String paymentMethodId,
    required String provider,
    String? subscriptionId,
    PaymentMetadata? metadata,
    @Default(false) bool requiresRiskAnalysis,
  }) = _PaymentRequest;

  factory PaymentRequest.fromJson(Map<String, dynamic> json) => _$PaymentRequestFromJson(json);
}

@freezed
class PaymentResponse with _$PaymentResponse {
  const factory PaymentResponse({
    required String id,
    required String status,
    required double amount,
    required String currency,
    required String provider,
    required String createdAt,
    required String providerPaymentId,
    required SecuredPayment securedPayment,
    required FraudAnalysis fraudAnalysis,
  }) = _PaymentResponse;

  factory PaymentResponse.fromJson(Map<String, dynamic> json) => _$PaymentResponseFromJson(json);
}

@freezed
class DeviceInfo with _$DeviceInfo {
  const factory DeviceInfo({
    required String os,
    required String model,
    required String browser,
    required String screenResolution,
    required String timezone,
    String? deviceId,
  }) = _DeviceInfo;

  factory DeviceInfo.fromJson(Map<String, dynamic> json) =>
      _$DeviceInfoFromJson(json);
}

class Location {
  final double latitude;
  final double longitude;
  final String? country;
  final String? city;
  final String? ip;

  const Location({
    required this.latitude,
    required this.longitude,
    this.country,
    this.city,
    this.ip,
  });

  factory Location.fromJson(Map<String, dynamic> json) => Location(
    latitude: json['latitude'] as double,
    longitude: json['longitude'] as double,
    country: json['country'] as String?,
    city: json['city'] as String?,
    ip: json['ip'] as String?,
  );

  Map<String, dynamic> toJson() => {
    'latitude': latitude,
    'longitude': longitude,
    if (country != null) 'country': country,
    if (city != null) 'city': city,
    if (ip != null) 'ip': ip,
  };
}

class NetworkInfo {
  final String ipAddress;
  final bool isVpn;
  final bool isProxy;
  final String? isp;
  final String? connectionType;

  const NetworkInfo({
    required this.ipAddress,
    required this.isVpn,
    required this.isProxy,
    this.isp,
    this.connectionType,
  });

  factory NetworkInfo.fromJson(Map<String, dynamic> json) => NetworkInfo(
    ipAddress: json['ip_address'] as String,
    isVpn: json['is_vpn'] as bool,
    isProxy: json['is_proxy'] as bool,
    isp: json['isp'] as String?,
    connectionType: json['connection_type'] as String?,
  );

  Map<String, dynamic> toJson() => {
    'ip_address': ipAddress,
    'is_vpn': isVpn,
    'is_proxy': isProxy,
    if (isp != null) 'isp': isp,
    if (connectionType != null) 'connection_type': connectionType,
  };
}

enum ValidationStatus { success, failure }

class ValidationResult {
  final ValidationStatus status;
  final Map<String, List<String>> errors;

  const ValidationResult.success() : status = ValidationStatus.success, errors = const {};
  const ValidationResult.failure(this.errors) : status = ValidationStatus.failure;

  bool get isSuccess => status == ValidationStatus.success;
  bool get isFailure => status == ValidationStatus.failure;
}

class PaymentMetadata {
  final DeviceInfo deviceInfo;
  final Location location;
  final NetworkInfo networkInfo;
  final Map<String, dynamic>? customData;

  const PaymentMetadata({
    required this.deviceInfo,
    required this.location,
    required this.networkInfo,
    this.customData,
  });

  factory PaymentMetadata.fromJson(Map<String, dynamic> json) => PaymentMetadata(
    deviceInfo: DeviceInfo.fromJson(json['device_info'] as Map<String, dynamic>),
    location: Location.fromJson(json['location'] as Map<String, dynamic>),
    networkInfo: NetworkInfo.fromJson(json['network_info'] as Map<String, dynamic>),
    customData: json['custom_data'] as Map<String, dynamic>?,
  );

  Map<String, dynamic> toJson() => {
    'device_info': deviceInfo.toJson(),
    'location': location.toJson(),
    'network_info': networkInfo.toJson(),
    if (customData != null) 'custom_data': customData,
  };
}



class SecuredPayment {
  final String encryptedData;
  final String encryptedKey;
  final String mac;
  final String timestamp;

  const SecuredPayment({
    required this.encryptedData,
    required this.encryptedKey,
    required this.mac,
    required this.timestamp,
  });

  factory SecuredPayment.fromJson(Map<String, dynamic> json) => SecuredPayment(
    encryptedData: json['encrypted_data'] as String,
    encryptedKey: json['encrypted_key'] as String,
    mac: json['mac'] as String,
    timestamp: json['timestamp'] as String,
  );

  Map<String, dynamic> toJson() => {
    'encrypted_data': encryptedData,
    'encrypted_key': encryptedKey,
    'mac': mac,
    'timestamp': timestamp,
  };
}

class FeatureImportance {
  final double amount;
  final double frequency;
  final double timePattern;
  final double locationRisk;
  final double deviceRisk;

  const FeatureImportance({
    required this.amount,
    required this.frequency,
    required this.timePattern,
    required this.locationRisk,
    required this.deviceRisk,
  });

  factory FeatureImportance.fromJson(Map<String, dynamic> json) => FeatureImportance(
    amount: json['amount'] as double,
    frequency: json['frequency'] as double,
    timePattern: json['time_pattern'] as double,
    locationRisk: json['location_risk'] as double,
    deviceRisk: json['device_risk'] as double,
  );

  Map<String, dynamic> toJson() => {
    'amount': amount,
    'frequency': frequency,
    'time_pattern': timePattern,
    'location_risk': locationRisk,
    'device_risk': deviceRisk,
  };
}

class FraudAnalysis {
  final bool isFraudulent;
  final double fraudProbability;
  final String riskLevel;
  final FeatureImportance featureImportance;

  const FraudAnalysis({
    required this.isFraudulent,
    required this.fraudProbability,
    required this.riskLevel,
    required this.featureImportance,
  });

  factory FraudAnalysis.fromJson(Map<String, dynamic> json) => FraudAnalysis(
    isFraudulent: json['is_fraudulent'] as bool,
    fraudProbability: json['fraud_probability'] as double,
    riskLevel: json['risk_level'] as String,
    featureImportance: FeatureImportance.fromJson(json['feature_importance'] as Map<String, dynamic>),
  );

  Map<String, dynamic> toJson() => {
    'is_fraudulent': isFraudulent,
    'fraud_probability': fraudProbability,
    'risk_level': riskLevel,
    'feature_importance': featureImportance.toJson(),
  };
}

class Payment {
  final String id;
  final String status;
  final double amount;
  final String currency;
  final String provider;
  final String createdAt;
  final String providerPaymentId;
  final SecuredPayment securedPayment;
  final FraudAnalysis fraudAnalysis;

  const Payment({
    required this.id,
    required this.status,
    required this.amount,
    required this.currency,
    required this.provider,
    required this.createdAt,
    required this.providerPaymentId,
    required this.securedPayment,
    required this.fraudAnalysis,
  });

  factory Payment.fromJson(Map<String, dynamic> json) => Payment(
    id: json['id'] as String,
    status: json['status'] as String,
    amount: json['amount'] as double,
    currency: json['currency'] as String,
    provider: json['provider'] as String,
    createdAt: json['created_at'] as String,
    providerPaymentId: json['provider_payment_id'] as String,
    securedPayment: SecuredPayment.fromJson(json['secured_payment'] as Map<String, dynamic>),
    fraudAnalysis: FraudAnalysis.fromJson(json['fraud_analysis'] as Map<String, dynamic>),
  );

  Map<String, dynamic> toJson() => {
    'id': id,
    'status': status,
    'amount': amount,
    'currency': currency,
    'provider': provider,
    'created_at': createdAt,
    'provider_payment_id': providerPaymentId,
    'secured_payment': securedPayment.toJson(),
    'fraud_analysis': fraudAnalysis.toJson(),
  };
}