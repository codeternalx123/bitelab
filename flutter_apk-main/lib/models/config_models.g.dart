// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'config_models.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$NotificationSettingsImpl _$$NotificationSettingsImplFromJson(
        Map<String, dynamic> json) =>
    _$NotificationSettingsImpl(
      emailEnabled: json['emailEnabled'] as bool? ?? true,
      pushEnabled: json['pushEnabled'] as bool? ?? true,
      inAppEnabled: json['inAppEnabled'] as bool? ?? true,
      soundEnabled: json['soundEnabled'] as bool? ?? true,
    );

Map<String, dynamic> _$$NotificationSettingsImplToJson(
        _$NotificationSettingsImpl instance) =>
    <String, dynamic>{
      'emailEnabled': instance.emailEnabled,
      'pushEnabled': instance.pushEnabled,
      'inAppEnabled': instance.inAppEnabled,
      'soundEnabled': instance.soundEnabled,
    };

_$PrivacySettingsImpl _$$PrivacySettingsImplFromJson(
        Map<String, dynamic> json) =>
    _$PrivacySettingsImpl(
      locationTrackingEnabled: json['locationTrackingEnabled'] as bool? ?? true,
      dataCollectionEnabled: json['dataCollectionEnabled'] as bool? ?? true,
      personalizationEnabled: json['personalizationEnabled'] as bool? ?? true,
    );

Map<String, dynamic> _$$PrivacySettingsImplToJson(
        _$PrivacySettingsImpl instance) =>
    <String, dynamic>{
      'locationTrackingEnabled': instance.locationTrackingEnabled,
      'dataCollectionEnabled': instance.dataCollectionEnabled,
      'personalizationEnabled': instance.personalizationEnabled,
    };

_$PerformanceSettingsImpl _$$PerformanceSettingsImplFromJson(
        Map<String, dynamic> json) =>
    _$PerformanceSettingsImpl(
      imageOptimizationEnabled:
          json['imageOptimizationEnabled'] as bool? ?? true,
      offlineModeEnabled: json['offlineModeEnabled'] as bool? ?? true,
      backgroundRefreshEnabled:
          json['backgroundRefreshEnabled'] as bool? ?? true,
    );

Map<String, dynamic> _$$PerformanceSettingsImplToJson(
        _$PerformanceSettingsImpl instance) =>
    <String, dynamic>{
      'imageOptimizationEnabled': instance.imageOptimizationEnabled,
      'offlineModeEnabled': instance.offlineModeEnabled,
      'backgroundRefreshEnabled': instance.backgroundRefreshEnabled,
    };
