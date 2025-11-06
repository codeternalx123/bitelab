import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:freezed_annotation/freezed_annotation.dart';
import '../core/storage/secure_storage_service.dart';
import '../core/di/di.dart';

part 'config_controller.freezed.dart';
part 'config_controller.g.dart';

final secureStorageServiceProvider = Provider((ref) => getIt<SecureStorageService>());

final configProvider =
    StateNotifierProvider<ConfigController, ConfigState>((ref) {
  return ConfigController(
    ref.watch(secureStorageServiceProvider),
  );
});

class ConfigController extends StateNotifier<ConfigState> {
  final SecureStorageService _storage;
  static const _configKey = 'app_config';

  ConfigController(this._storage) : super(const ConfigState()) {
    _loadConfig();
  }

  Future<void> _loadConfig() async {
    final storedConfig = await _storage.read(_configKey);
    if (storedConfig != null) {
      try {
        final config = ConfigState.fromJson(
          Map<String, dynamic>.from(json.decode(storedConfig)),
        );
        state = config;
      } catch (_) {
        // If loading fails, keep default state
      }
    }
  }

  Future<void> _saveConfig() async {
    await _storage.write(
      key: _configKey,
      value: json.encode(state.toJson()),
    );
  }

  Future<void> updateTheme(ThemeMode theme) async {
    state = state.copyWith(themeMode: theme);
    await _saveConfig();
  }

  Future<void> updateLocale(String locale) async {
    state = state.copyWith(locale: locale);
    await _saveConfig();
  }

  Future<void> toggleBiometrics(bool enabled) async {
    state = state.copyWith(biometricsEnabled: enabled);
    await _saveConfig();
  }

  Future<void> updateAnalytics(bool enabled) async {
    state = state.copyWith(analyticsEnabled: enabled);
    await _saveConfig();
  }

  Future<void> updateCrashReporting(bool enabled) async {
    state = state.copyWith(crashReportingEnabled: enabled);
    await _saveConfig();
  }

  Future<void> updateNotifications({
    bool? email,
    bool? push,
    bool? inApp,
  }) async {
    state = state.copyWith(
      notificationSettings: state.notificationSettings.copyWith(
        emailEnabled: email ?? state.notificationSettings.emailEnabled,
        pushEnabled: push ?? state.notificationSettings.pushEnabled,
        inAppEnabled: inApp ?? state.notificationSettings.inAppEnabled,
      ),
    );
    await _saveConfig();
  }

  Future<void> updatePrivacy({
    bool? locationTracking,
    bool? dataCollection,
    bool? thirdPartySharing,
  }) async {
    state = state.copyWith(
      privacySettings: state.privacySettings.copyWith(
        locationTrackingEnabled:
            locationTracking ?? state.privacySettings.locationTrackingEnabled,
        dataCollectionEnabled:
            dataCollection ?? state.privacySettings.dataCollectionEnabled,
        thirdPartySharingEnabled:
            thirdPartySharing ?? state.privacySettings.thirdPartySharingEnabled,
      ),
    );
    await _saveConfig();
  }

  Future<void> updatePerformance({
    bool? imageCache,
    bool? offlineMode,
    bool? backgroundRefresh,
  }) async {
    state = state.copyWith(
      performanceSettings: state.performanceSettings.copyWith(
        imageCacheEnabled:
            imageCache ?? state.performanceSettings.imageCacheEnabled,
        offlineModeEnabled:
            offlineMode ?? state.performanceSettings.offlineModeEnabled,
        backgroundRefreshEnabled: backgroundRefresh ??
            state.performanceSettings.backgroundRefreshEnabled,
      ),
    );
    await _saveConfig();
  }

  Future<void> reset() async {
    state = const ConfigState();
    await _storage.delete(_configKey);
  }
}

enum ThemeMode {
  system,
  light,
  dark,
}

@freezed
class ConfigState with _$ConfigState {
  const factory ConfigState({
    @Default(ThemeMode.system) ThemeMode themeMode,
    @Default('en_US') String locale,
    @Default(false) bool biometricsEnabled,
    @Default(true) bool analyticsEnabled,
    @Default(true) bool crashReportingEnabled,
    @Default(NotificationSettings()) NotificationSettings notificationSettings,
    @Default(PrivacySettings()) PrivacySettings privacySettings,
    @Default(PerformanceSettings()) PerformanceSettings performanceSettings,
  }) = _ConfigState;

  factory ConfigState.fromJson(Map<String, dynamic> json) =>
      _$ConfigStateFromJson(json);
}

@freezed
class NotificationSettings with _$NotificationSettings {
  const factory NotificationSettings({
    @Default(true) bool emailEnabled,
    @Default(true) bool pushEnabled,
    @Default(true) bool inAppEnabled,
  }) = _NotificationSettings;

  factory NotificationSettings.fromJson(Map<String, dynamic> json) =>
      _$NotificationSettingsFromJson(json);
}

@freezed
class PrivacySettings with _$PrivacySettings {
  const factory PrivacySettings({
    @Default(true) bool locationTrackingEnabled,
    @Default(true) bool dataCollectionEnabled,
    @Default(true) bool thirdPartySharingEnabled,
  }) = _PrivacySettings;

  factory PrivacySettings.fromJson(Map<String, dynamic> json) =>
      _$PrivacySettingsFromJson(json);
}

@freezed
class PerformanceSettings with _$PerformanceSettings {
  const factory PerformanceSettings({
    @Default(true) bool imageCacheEnabled,
    @Default(false) bool offlineModeEnabled,
    @Default(true) bool backgroundRefreshEnabled,
  }) = _PerformanceSettings;

  factory PerformanceSettings.fromJson(Map<String, dynamic> json) =>
      _$PerformanceSettingsFromJson(json);
}