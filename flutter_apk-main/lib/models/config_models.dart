import 'package:freezed_annotation/freezed_annotation.dart';

part 'config_models.freezed.dart';
part 'config_models.g.dart';

@freezed
class NotificationSettings with _$NotificationSettings {
  const factory NotificationSettings({
    @Default(true) bool emailEnabled,
    @Default(true) bool pushEnabled,
    @Default(true) bool inAppEnabled,
    @Default(true) bool soundEnabled,
  }) = _NotificationSettings;

  factory NotificationSettings.fromJson(Map<String, dynamic> json) =>
      _$NotificationSettingsFromJson(json);
}

@freezed
class PrivacySettings with _$PrivacySettings {
  const factory PrivacySettings({
    @Default(true) bool locationTrackingEnabled,
    @Default(true) bool dataCollectionEnabled,
    @Default(true) bool personalizationEnabled,
  }) = _PrivacySettings;

  factory PrivacySettings.fromJson(Map<String, dynamic> json) =>
      _$PrivacySettingsFromJson(json);
}

@freezed
class PerformanceSettings with _$PerformanceSettings {
  const factory PerformanceSettings({
    @Default(true) bool imageOptimizationEnabled,
    @Default(true) bool offlineModeEnabled,
    @Default(true) bool backgroundRefreshEnabled,
  }) = _PerformanceSettings;

  factory PerformanceSettings.fromJson(Map<String, dynamic> json) =>
      _$PerformanceSettingsFromJson(json);
}