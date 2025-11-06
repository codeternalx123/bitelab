// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'api_error.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

ApiError _$ApiErrorFromJson(Map<String, dynamic> json) => ApiError(
      detail: ErrorDetail.fromJson(json['detail'] as Map<String, dynamic>),
    );

Map<String, dynamic> _$ApiErrorToJson(ApiError instance) => <String, dynamic>{
      'detail': instance.detail,
    };

ErrorDetail _$ErrorDetailFromJson(Map<String, dynamic> json) => ErrorDetail(
      message: json['message'] as String,
      code: json['code'] as String,
      params: json['params'] as Map<String, dynamic>?,
    );

Map<String, dynamic> _$ErrorDetailToJson(ErrorDetail instance) =>
    <String, dynamic>{
      'message': instance.message,
      'code': instance.code,
      'params': instance.params,
    };
