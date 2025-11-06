// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'payment_models.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
    'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models');

PaymentRequest _$PaymentRequestFromJson(Map<String, dynamic> json) {
  return _PaymentRequest.fromJson(json);
}

/// @nodoc
mixin _$PaymentRequest {
  double get amount => throw _privateConstructorUsedError;
  String get currency => throw _privateConstructorUsedError;
  String get paymentMethodId => throw _privateConstructorUsedError;
  String get provider => throw _privateConstructorUsedError;
  String? get subscriptionId => throw _privateConstructorUsedError;
  PaymentMetadata? get metadata => throw _privateConstructorUsedError;
  bool get requiresRiskAnalysis => throw _privateConstructorUsedError;

  /// Serializes this PaymentRequest to a JSON map.
  Map<String, dynamic> toJson() => throw _privateConstructorUsedError;

  /// Create a copy of PaymentRequest
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $PaymentRequestCopyWith<PaymentRequest> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $PaymentRequestCopyWith<$Res> {
  factory $PaymentRequestCopyWith(
          PaymentRequest value, $Res Function(PaymentRequest) then) =
      _$PaymentRequestCopyWithImpl<$Res, PaymentRequest>;
  @useResult
  $Res call(
      {double amount,
      String currency,
      String paymentMethodId,
      String provider,
      String? subscriptionId,
      PaymentMetadata? metadata,
      bool requiresRiskAnalysis});
}

/// @nodoc
class _$PaymentRequestCopyWithImpl<$Res, $Val extends PaymentRequest>
    implements $PaymentRequestCopyWith<$Res> {
  _$PaymentRequestCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of PaymentRequest
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? amount = null,
    Object? currency = null,
    Object? paymentMethodId = null,
    Object? provider = null,
    Object? subscriptionId = freezed,
    Object? metadata = freezed,
    Object? requiresRiskAnalysis = null,
  }) {
    return _then(_value.copyWith(
      amount: null == amount
          ? _value.amount
          : amount // ignore: cast_nullable_to_non_nullable
              as double,
      currency: null == currency
          ? _value.currency
          : currency // ignore: cast_nullable_to_non_nullable
              as String,
      paymentMethodId: null == paymentMethodId
          ? _value.paymentMethodId
          : paymentMethodId // ignore: cast_nullable_to_non_nullable
              as String,
      provider: null == provider
          ? _value.provider
          : provider // ignore: cast_nullable_to_non_nullable
              as String,
      subscriptionId: freezed == subscriptionId
          ? _value.subscriptionId
          : subscriptionId // ignore: cast_nullable_to_non_nullable
              as String?,
      metadata: freezed == metadata
          ? _value.metadata
          : metadata // ignore: cast_nullable_to_non_nullable
              as PaymentMetadata?,
      requiresRiskAnalysis: null == requiresRiskAnalysis
          ? _value.requiresRiskAnalysis
          : requiresRiskAnalysis // ignore: cast_nullable_to_non_nullable
              as bool,
    ) as $Val);
  }
}

/// @nodoc
abstract class _$$PaymentRequestImplCopyWith<$Res>
    implements $PaymentRequestCopyWith<$Res> {
  factory _$$PaymentRequestImplCopyWith(_$PaymentRequestImpl value,
          $Res Function(_$PaymentRequestImpl) then) =
      __$$PaymentRequestImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call(
      {double amount,
      String currency,
      String paymentMethodId,
      String provider,
      String? subscriptionId,
      PaymentMetadata? metadata,
      bool requiresRiskAnalysis});
}

/// @nodoc
class __$$PaymentRequestImplCopyWithImpl<$Res>
    extends _$PaymentRequestCopyWithImpl<$Res, _$PaymentRequestImpl>
    implements _$$PaymentRequestImplCopyWith<$Res> {
  __$$PaymentRequestImplCopyWithImpl(
      _$PaymentRequestImpl _value, $Res Function(_$PaymentRequestImpl) _then)
      : super(_value, _then);

  /// Create a copy of PaymentRequest
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? amount = null,
    Object? currency = null,
    Object? paymentMethodId = null,
    Object? provider = null,
    Object? subscriptionId = freezed,
    Object? metadata = freezed,
    Object? requiresRiskAnalysis = null,
  }) {
    return _then(_$PaymentRequestImpl(
      amount: null == amount
          ? _value.amount
          : amount // ignore: cast_nullable_to_non_nullable
              as double,
      currency: null == currency
          ? _value.currency
          : currency // ignore: cast_nullable_to_non_nullable
              as String,
      paymentMethodId: null == paymentMethodId
          ? _value.paymentMethodId
          : paymentMethodId // ignore: cast_nullable_to_non_nullable
              as String,
      provider: null == provider
          ? _value.provider
          : provider // ignore: cast_nullable_to_non_nullable
              as String,
      subscriptionId: freezed == subscriptionId
          ? _value.subscriptionId
          : subscriptionId // ignore: cast_nullable_to_non_nullable
              as String?,
      metadata: freezed == metadata
          ? _value.metadata
          : metadata // ignore: cast_nullable_to_non_nullable
              as PaymentMetadata?,
      requiresRiskAnalysis: null == requiresRiskAnalysis
          ? _value.requiresRiskAnalysis
          : requiresRiskAnalysis // ignore: cast_nullable_to_non_nullable
              as bool,
    ));
  }
}

/// @nodoc
@JsonSerializable()
class _$PaymentRequestImpl implements _PaymentRequest {
  const _$PaymentRequestImpl(
      {required this.amount,
      required this.currency,
      required this.paymentMethodId,
      required this.provider,
      this.subscriptionId,
      this.metadata,
      this.requiresRiskAnalysis = false});

  factory _$PaymentRequestImpl.fromJson(Map<String, dynamic> json) =>
      _$$PaymentRequestImplFromJson(json);

  @override
  final double amount;
  @override
  final String currency;
  @override
  final String paymentMethodId;
  @override
  final String provider;
  @override
  final String? subscriptionId;
  @override
  final PaymentMetadata? metadata;
  @override
  @JsonKey()
  final bool requiresRiskAnalysis;

  @override
  String toString() {
    return 'PaymentRequest(amount: $amount, currency: $currency, paymentMethodId: $paymentMethodId, provider: $provider, subscriptionId: $subscriptionId, metadata: $metadata, requiresRiskAnalysis: $requiresRiskAnalysis)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$PaymentRequestImpl &&
            (identical(other.amount, amount) || other.amount == amount) &&
            (identical(other.currency, currency) ||
                other.currency == currency) &&
            (identical(other.paymentMethodId, paymentMethodId) ||
                other.paymentMethodId == paymentMethodId) &&
            (identical(other.provider, provider) ||
                other.provider == provider) &&
            (identical(other.subscriptionId, subscriptionId) ||
                other.subscriptionId == subscriptionId) &&
            (identical(other.metadata, metadata) ||
                other.metadata == metadata) &&
            (identical(other.requiresRiskAnalysis, requiresRiskAnalysis) ||
                other.requiresRiskAnalysis == requiresRiskAnalysis));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode => Object.hash(
      runtimeType,
      amount,
      currency,
      paymentMethodId,
      provider,
      subscriptionId,
      metadata,
      requiresRiskAnalysis);

  /// Create a copy of PaymentRequest
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$PaymentRequestImplCopyWith<_$PaymentRequestImpl> get copyWith =>
      __$$PaymentRequestImplCopyWithImpl<_$PaymentRequestImpl>(
          this, _$identity);

  @override
  Map<String, dynamic> toJson() {
    return _$$PaymentRequestImplToJson(
      this,
    );
  }
}

abstract class _PaymentRequest implements PaymentRequest {
  const factory _PaymentRequest(
      {required final double amount,
      required final String currency,
      required final String paymentMethodId,
      required final String provider,
      final String? subscriptionId,
      final PaymentMetadata? metadata,
      final bool requiresRiskAnalysis}) = _$PaymentRequestImpl;

  factory _PaymentRequest.fromJson(Map<String, dynamic> json) =
      _$PaymentRequestImpl.fromJson;

  @override
  double get amount;
  @override
  String get currency;
  @override
  String get paymentMethodId;
  @override
  String get provider;
  @override
  String? get subscriptionId;
  @override
  PaymentMetadata? get metadata;
  @override
  bool get requiresRiskAnalysis;

  /// Create a copy of PaymentRequest
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$PaymentRequestImplCopyWith<_$PaymentRequestImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

PaymentResponse _$PaymentResponseFromJson(Map<String, dynamic> json) {
  return _PaymentResponse.fromJson(json);
}

/// @nodoc
mixin _$PaymentResponse {
  String get id => throw _privateConstructorUsedError;
  String get status => throw _privateConstructorUsedError;
  double get amount => throw _privateConstructorUsedError;
  String get currency => throw _privateConstructorUsedError;
  String get provider => throw _privateConstructorUsedError;
  String get createdAt => throw _privateConstructorUsedError;
  String get providerPaymentId => throw _privateConstructorUsedError;
  SecuredPayment get securedPayment => throw _privateConstructorUsedError;
  FraudAnalysis get fraudAnalysis => throw _privateConstructorUsedError;

  /// Serializes this PaymentResponse to a JSON map.
  Map<String, dynamic> toJson() => throw _privateConstructorUsedError;

  /// Create a copy of PaymentResponse
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $PaymentResponseCopyWith<PaymentResponse> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $PaymentResponseCopyWith<$Res> {
  factory $PaymentResponseCopyWith(
          PaymentResponse value, $Res Function(PaymentResponse) then) =
      _$PaymentResponseCopyWithImpl<$Res, PaymentResponse>;
  @useResult
  $Res call(
      {String id,
      String status,
      double amount,
      String currency,
      String provider,
      String createdAt,
      String providerPaymentId,
      SecuredPayment securedPayment,
      FraudAnalysis fraudAnalysis});
}

/// @nodoc
class _$PaymentResponseCopyWithImpl<$Res, $Val extends PaymentResponse>
    implements $PaymentResponseCopyWith<$Res> {
  _$PaymentResponseCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of PaymentResponse
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? id = null,
    Object? status = null,
    Object? amount = null,
    Object? currency = null,
    Object? provider = null,
    Object? createdAt = null,
    Object? providerPaymentId = null,
    Object? securedPayment = null,
    Object? fraudAnalysis = null,
  }) {
    return _then(_value.copyWith(
      id: null == id
          ? _value.id
          : id // ignore: cast_nullable_to_non_nullable
              as String,
      status: null == status
          ? _value.status
          : status // ignore: cast_nullable_to_non_nullable
              as String,
      amount: null == amount
          ? _value.amount
          : amount // ignore: cast_nullable_to_non_nullable
              as double,
      currency: null == currency
          ? _value.currency
          : currency // ignore: cast_nullable_to_non_nullable
              as String,
      provider: null == provider
          ? _value.provider
          : provider // ignore: cast_nullable_to_non_nullable
              as String,
      createdAt: null == createdAt
          ? _value.createdAt
          : createdAt // ignore: cast_nullable_to_non_nullable
              as String,
      providerPaymentId: null == providerPaymentId
          ? _value.providerPaymentId
          : providerPaymentId // ignore: cast_nullable_to_non_nullable
              as String,
      securedPayment: null == securedPayment
          ? _value.securedPayment
          : securedPayment // ignore: cast_nullable_to_non_nullable
              as SecuredPayment,
      fraudAnalysis: null == fraudAnalysis
          ? _value.fraudAnalysis
          : fraudAnalysis // ignore: cast_nullable_to_non_nullable
              as FraudAnalysis,
    ) as $Val);
  }
}

/// @nodoc
abstract class _$$PaymentResponseImplCopyWith<$Res>
    implements $PaymentResponseCopyWith<$Res> {
  factory _$$PaymentResponseImplCopyWith(_$PaymentResponseImpl value,
          $Res Function(_$PaymentResponseImpl) then) =
      __$$PaymentResponseImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call(
      {String id,
      String status,
      double amount,
      String currency,
      String provider,
      String createdAt,
      String providerPaymentId,
      SecuredPayment securedPayment,
      FraudAnalysis fraudAnalysis});
}

/// @nodoc
class __$$PaymentResponseImplCopyWithImpl<$Res>
    extends _$PaymentResponseCopyWithImpl<$Res, _$PaymentResponseImpl>
    implements _$$PaymentResponseImplCopyWith<$Res> {
  __$$PaymentResponseImplCopyWithImpl(
      _$PaymentResponseImpl _value, $Res Function(_$PaymentResponseImpl) _then)
      : super(_value, _then);

  /// Create a copy of PaymentResponse
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? id = null,
    Object? status = null,
    Object? amount = null,
    Object? currency = null,
    Object? provider = null,
    Object? createdAt = null,
    Object? providerPaymentId = null,
    Object? securedPayment = null,
    Object? fraudAnalysis = null,
  }) {
    return _then(_$PaymentResponseImpl(
      id: null == id
          ? _value.id
          : id // ignore: cast_nullable_to_non_nullable
              as String,
      status: null == status
          ? _value.status
          : status // ignore: cast_nullable_to_non_nullable
              as String,
      amount: null == amount
          ? _value.amount
          : amount // ignore: cast_nullable_to_non_nullable
              as double,
      currency: null == currency
          ? _value.currency
          : currency // ignore: cast_nullable_to_non_nullable
              as String,
      provider: null == provider
          ? _value.provider
          : provider // ignore: cast_nullable_to_non_nullable
              as String,
      createdAt: null == createdAt
          ? _value.createdAt
          : createdAt // ignore: cast_nullable_to_non_nullable
              as String,
      providerPaymentId: null == providerPaymentId
          ? _value.providerPaymentId
          : providerPaymentId // ignore: cast_nullable_to_non_nullable
              as String,
      securedPayment: null == securedPayment
          ? _value.securedPayment
          : securedPayment // ignore: cast_nullable_to_non_nullable
              as SecuredPayment,
      fraudAnalysis: null == fraudAnalysis
          ? _value.fraudAnalysis
          : fraudAnalysis // ignore: cast_nullable_to_non_nullable
              as FraudAnalysis,
    ));
  }
}

/// @nodoc
@JsonSerializable()
class _$PaymentResponseImpl implements _PaymentResponse {
  const _$PaymentResponseImpl(
      {required this.id,
      required this.status,
      required this.amount,
      required this.currency,
      required this.provider,
      required this.createdAt,
      required this.providerPaymentId,
      required this.securedPayment,
      required this.fraudAnalysis});

  factory _$PaymentResponseImpl.fromJson(Map<String, dynamic> json) =>
      _$$PaymentResponseImplFromJson(json);

  @override
  final String id;
  @override
  final String status;
  @override
  final double amount;
  @override
  final String currency;
  @override
  final String provider;
  @override
  final String createdAt;
  @override
  final String providerPaymentId;
  @override
  final SecuredPayment securedPayment;
  @override
  final FraudAnalysis fraudAnalysis;

  @override
  String toString() {
    return 'PaymentResponse(id: $id, status: $status, amount: $amount, currency: $currency, provider: $provider, createdAt: $createdAt, providerPaymentId: $providerPaymentId, securedPayment: $securedPayment, fraudAnalysis: $fraudAnalysis)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$PaymentResponseImpl &&
            (identical(other.id, id) || other.id == id) &&
            (identical(other.status, status) || other.status == status) &&
            (identical(other.amount, amount) || other.amount == amount) &&
            (identical(other.currency, currency) ||
                other.currency == currency) &&
            (identical(other.provider, provider) ||
                other.provider == provider) &&
            (identical(other.createdAt, createdAt) ||
                other.createdAt == createdAt) &&
            (identical(other.providerPaymentId, providerPaymentId) ||
                other.providerPaymentId == providerPaymentId) &&
            (identical(other.securedPayment, securedPayment) ||
                other.securedPayment == securedPayment) &&
            (identical(other.fraudAnalysis, fraudAnalysis) ||
                other.fraudAnalysis == fraudAnalysis));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode => Object.hash(runtimeType, id, status, amount, currency,
      provider, createdAt, providerPaymentId, securedPayment, fraudAnalysis);

  /// Create a copy of PaymentResponse
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$PaymentResponseImplCopyWith<_$PaymentResponseImpl> get copyWith =>
      __$$PaymentResponseImplCopyWithImpl<_$PaymentResponseImpl>(
          this, _$identity);

  @override
  Map<String, dynamic> toJson() {
    return _$$PaymentResponseImplToJson(
      this,
    );
  }
}

abstract class _PaymentResponse implements PaymentResponse {
  const factory _PaymentResponse(
      {required final String id,
      required final String status,
      required final double amount,
      required final String currency,
      required final String provider,
      required final String createdAt,
      required final String providerPaymentId,
      required final SecuredPayment securedPayment,
      required final FraudAnalysis fraudAnalysis}) = _$PaymentResponseImpl;

  factory _PaymentResponse.fromJson(Map<String, dynamic> json) =
      _$PaymentResponseImpl.fromJson;

  @override
  String get id;
  @override
  String get status;
  @override
  double get amount;
  @override
  String get currency;
  @override
  String get provider;
  @override
  String get createdAt;
  @override
  String get providerPaymentId;
  @override
  SecuredPayment get securedPayment;
  @override
  FraudAnalysis get fraudAnalysis;

  /// Create a copy of PaymentResponse
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$PaymentResponseImplCopyWith<_$PaymentResponseImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

DeviceInfo _$DeviceInfoFromJson(Map<String, dynamic> json) {
  return _DeviceInfo.fromJson(json);
}

/// @nodoc
mixin _$DeviceInfo {
  String get os => throw _privateConstructorUsedError;
  String get model => throw _privateConstructorUsedError;
  String get browser => throw _privateConstructorUsedError;
  String get screenResolution => throw _privateConstructorUsedError;
  String get timezone => throw _privateConstructorUsedError;
  String? get deviceId => throw _privateConstructorUsedError;

  /// Serializes this DeviceInfo to a JSON map.
  Map<String, dynamic> toJson() => throw _privateConstructorUsedError;

  /// Create a copy of DeviceInfo
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $DeviceInfoCopyWith<DeviceInfo> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $DeviceInfoCopyWith<$Res> {
  factory $DeviceInfoCopyWith(
          DeviceInfo value, $Res Function(DeviceInfo) then) =
      _$DeviceInfoCopyWithImpl<$Res, DeviceInfo>;
  @useResult
  $Res call(
      {String os,
      String model,
      String browser,
      String screenResolution,
      String timezone,
      String? deviceId});
}

/// @nodoc
class _$DeviceInfoCopyWithImpl<$Res, $Val extends DeviceInfo>
    implements $DeviceInfoCopyWith<$Res> {
  _$DeviceInfoCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of DeviceInfo
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? os = null,
    Object? model = null,
    Object? browser = null,
    Object? screenResolution = null,
    Object? timezone = null,
    Object? deviceId = freezed,
  }) {
    return _then(_value.copyWith(
      os: null == os
          ? _value.os
          : os // ignore: cast_nullable_to_non_nullable
              as String,
      model: null == model
          ? _value.model
          : model // ignore: cast_nullable_to_non_nullable
              as String,
      browser: null == browser
          ? _value.browser
          : browser // ignore: cast_nullable_to_non_nullable
              as String,
      screenResolution: null == screenResolution
          ? _value.screenResolution
          : screenResolution // ignore: cast_nullable_to_non_nullable
              as String,
      timezone: null == timezone
          ? _value.timezone
          : timezone // ignore: cast_nullable_to_non_nullable
              as String,
      deviceId: freezed == deviceId
          ? _value.deviceId
          : deviceId // ignore: cast_nullable_to_non_nullable
              as String?,
    ) as $Val);
  }
}

/// @nodoc
abstract class _$$DeviceInfoImplCopyWith<$Res>
    implements $DeviceInfoCopyWith<$Res> {
  factory _$$DeviceInfoImplCopyWith(
          _$DeviceInfoImpl value, $Res Function(_$DeviceInfoImpl) then) =
      __$$DeviceInfoImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call(
      {String os,
      String model,
      String browser,
      String screenResolution,
      String timezone,
      String? deviceId});
}

/// @nodoc
class __$$DeviceInfoImplCopyWithImpl<$Res>
    extends _$DeviceInfoCopyWithImpl<$Res, _$DeviceInfoImpl>
    implements _$$DeviceInfoImplCopyWith<$Res> {
  __$$DeviceInfoImplCopyWithImpl(
      _$DeviceInfoImpl _value, $Res Function(_$DeviceInfoImpl) _then)
      : super(_value, _then);

  /// Create a copy of DeviceInfo
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? os = null,
    Object? model = null,
    Object? browser = null,
    Object? screenResolution = null,
    Object? timezone = null,
    Object? deviceId = freezed,
  }) {
    return _then(_$DeviceInfoImpl(
      os: null == os
          ? _value.os
          : os // ignore: cast_nullable_to_non_nullable
              as String,
      model: null == model
          ? _value.model
          : model // ignore: cast_nullable_to_non_nullable
              as String,
      browser: null == browser
          ? _value.browser
          : browser // ignore: cast_nullable_to_non_nullable
              as String,
      screenResolution: null == screenResolution
          ? _value.screenResolution
          : screenResolution // ignore: cast_nullable_to_non_nullable
              as String,
      timezone: null == timezone
          ? _value.timezone
          : timezone // ignore: cast_nullable_to_non_nullable
              as String,
      deviceId: freezed == deviceId
          ? _value.deviceId
          : deviceId // ignore: cast_nullable_to_non_nullable
              as String?,
    ));
  }
}

/// @nodoc
@JsonSerializable()
class _$DeviceInfoImpl implements _DeviceInfo {
  const _$DeviceInfoImpl(
      {required this.os,
      required this.model,
      required this.browser,
      required this.screenResolution,
      required this.timezone,
      this.deviceId});

  factory _$DeviceInfoImpl.fromJson(Map<String, dynamic> json) =>
      _$$DeviceInfoImplFromJson(json);

  @override
  final String os;
  @override
  final String model;
  @override
  final String browser;
  @override
  final String screenResolution;
  @override
  final String timezone;
  @override
  final String? deviceId;

  @override
  String toString() {
    return 'DeviceInfo(os: $os, model: $model, browser: $browser, screenResolution: $screenResolution, timezone: $timezone, deviceId: $deviceId)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$DeviceInfoImpl &&
            (identical(other.os, os) || other.os == os) &&
            (identical(other.model, model) || other.model == model) &&
            (identical(other.browser, browser) || other.browser == browser) &&
            (identical(other.screenResolution, screenResolution) ||
                other.screenResolution == screenResolution) &&
            (identical(other.timezone, timezone) ||
                other.timezone == timezone) &&
            (identical(other.deviceId, deviceId) ||
                other.deviceId == deviceId));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode => Object.hash(
      runtimeType, os, model, browser, screenResolution, timezone, deviceId);

  /// Create a copy of DeviceInfo
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$DeviceInfoImplCopyWith<_$DeviceInfoImpl> get copyWith =>
      __$$DeviceInfoImplCopyWithImpl<_$DeviceInfoImpl>(this, _$identity);

  @override
  Map<String, dynamic> toJson() {
    return _$$DeviceInfoImplToJson(
      this,
    );
  }
}

abstract class _DeviceInfo implements DeviceInfo {
  const factory _DeviceInfo(
      {required final String os,
      required final String model,
      required final String browser,
      required final String screenResolution,
      required final String timezone,
      final String? deviceId}) = _$DeviceInfoImpl;

  factory _DeviceInfo.fromJson(Map<String, dynamic> json) =
      _$DeviceInfoImpl.fromJson;

  @override
  String get os;
  @override
  String get model;
  @override
  String get browser;
  @override
  String get screenResolution;
  @override
  String get timezone;
  @override
  String? get deviceId;

  /// Create a copy of DeviceInfo
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$DeviceInfoImplCopyWith<_$DeviceInfoImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
