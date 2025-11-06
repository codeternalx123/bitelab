// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'payment_models.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$PaymentRequestImpl _$$PaymentRequestImplFromJson(Map<String, dynamic> json) =>
    _$PaymentRequestImpl(
      amount: (json['amount'] as num).toDouble(),
      currency: json['currency'] as String,
      paymentMethodId: json['paymentMethodId'] as String,
      provider: json['provider'] as String,
      subscriptionId: json['subscriptionId'] as String?,
      metadata: json['metadata'] == null
          ? null
          : PaymentMetadata.fromJson(json['metadata'] as Map<String, dynamic>),
      requiresRiskAnalysis: json['requiresRiskAnalysis'] as bool? ?? false,
    );

Map<String, dynamic> _$$PaymentRequestImplToJson(
        _$PaymentRequestImpl instance) =>
    <String, dynamic>{
      'amount': instance.amount,
      'currency': instance.currency,
      'paymentMethodId': instance.paymentMethodId,
      'provider': instance.provider,
      'subscriptionId': instance.subscriptionId,
      'metadata': instance.metadata,
      'requiresRiskAnalysis': instance.requiresRiskAnalysis,
    };

_$PaymentResponseImpl _$$PaymentResponseImplFromJson(
        Map<String, dynamic> json) =>
    _$PaymentResponseImpl(
      id: json['id'] as String,
      status: json['status'] as String,
      amount: (json['amount'] as num).toDouble(),
      currency: json['currency'] as String,
      provider: json['provider'] as String,
      createdAt: json['createdAt'] as String,
      providerPaymentId: json['providerPaymentId'] as String,
      securedPayment: SecuredPayment.fromJson(
          json['securedPayment'] as Map<String, dynamic>),
      fraudAnalysis:
          FraudAnalysis.fromJson(json['fraudAnalysis'] as Map<String, dynamic>),
    );

Map<String, dynamic> _$$PaymentResponseImplToJson(
        _$PaymentResponseImpl instance) =>
    <String, dynamic>{
      'id': instance.id,
      'status': instance.status,
      'amount': instance.amount,
      'currency': instance.currency,
      'provider': instance.provider,
      'createdAt': instance.createdAt,
      'providerPaymentId': instance.providerPaymentId,
      'securedPayment': instance.securedPayment,
      'fraudAnalysis': instance.fraudAnalysis,
    };

_$DeviceInfoImpl _$$DeviceInfoImplFromJson(Map<String, dynamic> json) =>
    _$DeviceInfoImpl(
      os: json['os'] as String,
      model: json['model'] as String,
      browser: json['browser'] as String,
      screenResolution: json['screenResolution'] as String,
      timezone: json['timezone'] as String,
      deviceId: json['deviceId'] as String?,
    );

Map<String, dynamic> _$$DeviceInfoImplToJson(_$DeviceInfoImpl instance) =>
    <String, dynamic>{
      'os': instance.os,
      'model': instance.model,
      'browser': instance.browser,
      'screenResolution': instance.screenResolution,
      'timezone': instance.timezone,
      'deviceId': instance.deviceId,
    };
