// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'config_controller.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$ConfigStateImpl _$$ConfigStateImplFromJson(Map<String, dynamic> json) =>
    _$ConfigStateImpl(
      themeMode: $enumDecodeNullable(_$ThemeModeEnumMap, json['themeMode']) ??
          ThemeMode.system,
      locale: json['locale'] as String? ?? 'en_US',
      biometricsEnabled: json['biometricsEnabled'] as bool? ?? false,
      analyticsEnabled: json['analyticsEnabled'] as bool? ?? true,
      crashReportingEnabled: json['crashReportingEnabled'] as bool? ?? true,
      notificationSettings: json['notificationSettings'] == null
          ? const NotificationSettings()
          : NotificationSettings.fromJson(
              json['notificationSettings'] as Map<String, dynamic>),
      privacySettings: json['privacySettings'] == null
          ? const PrivacySettings()
          : PrivacySettings.fromJson(
              json['privacySettings'] as Map<String, dynamic>),
      performanceSettings: json['performanceSettings'] == null
          ? const PerformanceSettings()
          : PerformanceSettings.fromJson(
              json['performanceSettings'] as Map<String, dynamic>),
    );

Map<String, dynamic> _$$ConfigStateImplToJson(_$ConfigStateImpl instance) =>
    <String, dynamic>{
      'themeMode': _$ThemeModeEnumMap[instance.themeMode]!,
      'locale': instance.locale,
      'biometricsEnabled': instance.biometricsEnabled,
      'analyticsEnabled': instance.analyticsEnabled,
      'crashReportingEnabled': instance.crashReportingEnabled,
      'notificationSettings': instance.notificationSettings,
      'privacySettings': instance.privacySettings,
      'performanceSettings': instance.performanceSettings,
    };

const _$ThemeModeEnumMap = {
  ThemeMode.system: 'system',
  ThemeMode.light: 'light',
  ThemeMode.dark: 'dark',
};

_$NotificationSettingsImpl _$$NotificationSettingsImplFromJson(
        Map<String, dynamic> json) =>
    _$NotificationSettingsImpl(
      emailEnabled: json['emailEnabled'] as bool? ?? true,
      pushEnabled: json['pushEnabled'] as bool? ?? true,
      inAppEnabled: json['inAppEnabled'] as bool? ?? true,
    );

Map<String, dynamic> _$$NotificationSettingsImplToJson(
        _$NotificationSettingsImpl instance) =>
    <String, dynamic>{
      'emailEnabled': instance.emailEnabled,
      'pushEnabled': instance.pushEnabled,
      'inAppEnabled': instance.inAppEnabled,
    };

_$PrivacySettingsImpl _$$PrivacySettingsImplFromJson(
        Map<String, dynamic> json) =>
    _$PrivacySettingsImpl(
      locationTrackingEnabled: json['locationTrackingEnabled'] as bool? ?? true,
      dataCollectionEnabled: json['dataCollectionEnabled'] as bool? ?? true,
      thirdPartySharingEnabled:
          json['thirdPartySharingEnabled'] as bool? ?? true,
    );

Map<String, dynamic> _$$PrivacySettingsImplToJson(
        _$PrivacySettingsImpl instance) =>
    <String, dynamic>{
      'locationTrackingEnabled': instance.locationTrackingEnabled,
      'dataCollectionEnabled': instance.dataCollectionEnabled,
      'thirdPartySharingEnabled': instance.thirdPartySharingEnabled,
    };

_$PerformanceSettingsImpl _$$PerformanceSettingsImplFromJson(
        Map<String, dynamic> json) =>
    _$PerformanceSettingsImpl(
      imageCacheEnabled: json['imageCacheEnabled'] as bool? ?? true,
      offlineModeEnabled: json['offlineModeEnabled'] as bool? ?? false,
      backgroundRefreshEnabled:
          json['backgroundRefreshEnabled'] as bool? ?? true,
    );

Map<String, dynamic> _$$PerformanceSettingsImplToJson(
        _$PerformanceSettingsImpl instance) =>
    <String, dynamic>{
      'imageCacheEnabled': instance.imageCacheEnabled,
      'offlineModeEnabled': instance.offlineModeEnabled,
      'backgroundRefreshEnabled': instance.backgroundRefreshEnabled,
    };
