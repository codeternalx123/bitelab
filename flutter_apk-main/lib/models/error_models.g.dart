// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'error_models.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$ApiErrorImpl _$$ApiErrorImplFromJson(Map<String, dynamic> json) =>
    _$ApiErrorImpl(
      code: json['code'] as String,
      message: json['message'] as String,
      target: json['target'] as String?,
      details: (json['details'] as List<dynamic>?)
          ?.map((e) => ApiErrorDetail.fromJson(e as Map<String, dynamic>))
          .toList(),
      metadata: json['metadata'] as Map<String, dynamic>?,
    );

Map<String, dynamic> _$$ApiErrorImplToJson(_$ApiErrorImpl instance) =>
    <String, dynamic>{
      'code': instance.code,
      'message': instance.message,
      'target': instance.target,
      'details': instance.details,
      'metadata': instance.metadata,
    };

_$ApiErrorDetailImpl _$$ApiErrorDetailImplFromJson(Map<String, dynamic> json) =>
    _$ApiErrorDetailImpl(
      code: json['code'] as String,
      message: json['message'] as String,
      target: json['target'] as String?,
      metadata: json['metadata'] as Map<String, dynamic>?,
    );

Map<String, dynamic> _$$ApiErrorDetailImplToJson(
        _$ApiErrorDetailImpl instance) =>
    <String, dynamic>{
      'code': instance.code,
      'message': instance.message,
      'target': instance.target,
      'metadata': instance.metadata,
    };

_$ErrorResponseImpl _$$ErrorResponseImplFromJson(Map<String, dynamic> json) =>
    _$ErrorResponseImpl(
      requestId: json['requestId'] as String,
      timestamp: DateTime.parse(json['timestamp'] as String),
      error: ApiError.fromJson(json['error'] as Map<String, dynamic>),
      debugInfo: json['debugInfo'] as Map<String, dynamic>?,
    );

Map<String, dynamic> _$$ErrorResponseImplToJson(_$ErrorResponseImpl instance) =>
    <String, dynamic>{
      'requestId': instance.requestId,
      'timestamp': instance.timestamp.toIso8601String(),
      'error': instance.error,
      'debugInfo': instance.debugInfo,
    };
