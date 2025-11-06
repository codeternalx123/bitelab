// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'config_controller.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
    'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models');

ConfigState _$ConfigStateFromJson(Map<String, dynamic> json) {
  return _ConfigState.fromJson(json);
}

/// @nodoc
mixin _$ConfigState {
  ThemeMode get themeMode => throw _privateConstructorUsedError;
  String get locale => throw _privateConstructorUsedError;
  bool get biometricsEnabled => throw _privateConstructorUsedError;
  bool get analyticsEnabled => throw _privateConstructorUsedError;
  bool get crashReportingEnabled => throw _privateConstructorUsedError;
  NotificationSettings get notificationSettings =>
      throw _privateConstructorUsedError;
  PrivacySettings get privacySettings => throw _privateConstructorUsedError;
  PerformanceSettings get performanceSettings =>
      throw _privateConstructorUsedError;

  /// Serializes this ConfigState to a JSON map.
  Map<String, dynamic> toJson() => throw _privateConstructorUsedError;

  /// Create a copy of ConfigState
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $ConfigStateCopyWith<ConfigState> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $ConfigStateCopyWith<$Res> {
  factory $ConfigStateCopyWith(
          ConfigState value, $Res Function(ConfigState) then) =
      _$ConfigStateCopyWithImpl<$Res, ConfigState>;
  @useResult
  $Res call(
      {ThemeMode themeMode,
      String locale,
      bool biometricsEnabled,
      bool analyticsEnabled,
      bool crashReportingEnabled,
      NotificationSettings notificationSettings,
      PrivacySettings privacySettings,
      PerformanceSettings performanceSettings});

  $NotificationSettingsCopyWith<$Res> get notificationSettings;
  $PrivacySettingsCopyWith<$Res> get privacySettings;
  $PerformanceSettingsCopyWith<$Res> get performanceSettings;
}

/// @nodoc
class _$ConfigStateCopyWithImpl<$Res, $Val extends ConfigState>
    implements $ConfigStateCopyWith<$Res> {
  _$ConfigStateCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of ConfigState
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? themeMode = null,
    Object? locale = null,
    Object? biometricsEnabled = null,
    Object? analyticsEnabled = null,
    Object? crashReportingEnabled = null,
    Object? notificationSettings = null,
    Object? privacySettings = null,
    Object? performanceSettings = null,
  }) {
    return _then(_value.copyWith(
      themeMode: null == themeMode
          ? _value.themeMode
          : themeMode // ignore: cast_nullable_to_non_nullable
              as ThemeMode,
      locale: null == locale
          ? _value.locale
          : locale // ignore: cast_nullable_to_non_nullable
              as String,
      biometricsEnabled: null == biometricsEnabled
          ? _value.biometricsEnabled
          : biometricsEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      analyticsEnabled: null == analyticsEnabled
          ? _value.analyticsEnabled
          : analyticsEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      crashReportingEnabled: null == crashReportingEnabled
          ? _value.crashReportingEnabled
          : crashReportingEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      notificationSettings: null == notificationSettings
          ? _value.notificationSettings
          : notificationSettings // ignore: cast_nullable_to_non_nullable
              as NotificationSettings,
      privacySettings: null == privacySettings
          ? _value.privacySettings
          : privacySettings // ignore: cast_nullable_to_non_nullable
              as PrivacySettings,
      performanceSettings: null == performanceSettings
          ? _value.performanceSettings
          : performanceSettings // ignore: cast_nullable_to_non_nullable
              as PerformanceSettings,
    ) as $Val);
  }

  /// Create a copy of ConfigState
  /// with the given fields replaced by the non-null parameter values.
  @override
  @pragma('vm:prefer-inline')
  $NotificationSettingsCopyWith<$Res> get notificationSettings {
    return $NotificationSettingsCopyWith<$Res>(_value.notificationSettings,
        (value) {
      return _then(_value.copyWith(notificationSettings: value) as $Val);
    });
  }

  /// Create a copy of ConfigState
  /// with the given fields replaced by the non-null parameter values.
  @override
  @pragma('vm:prefer-inline')
  $PrivacySettingsCopyWith<$Res> get privacySettings {
    return $PrivacySettingsCopyWith<$Res>(_value.privacySettings, (value) {
      return _then(_value.copyWith(privacySettings: value) as $Val);
    });
  }

  /// Create a copy of ConfigState
  /// with the given fields replaced by the non-null parameter values.
  @override
  @pragma('vm:prefer-inline')
  $PerformanceSettingsCopyWith<$Res> get performanceSettings {
    return $PerformanceSettingsCopyWith<$Res>(_value.performanceSettings,
        (value) {
      return _then(_value.copyWith(performanceSettings: value) as $Val);
    });
  }
}

/// @nodoc
abstract class _$$ConfigStateImplCopyWith<$Res>
    implements $ConfigStateCopyWith<$Res> {
  factory _$$ConfigStateImplCopyWith(
          _$ConfigStateImpl value, $Res Function(_$ConfigStateImpl) then) =
      __$$ConfigStateImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call(
      {ThemeMode themeMode,
      String locale,
      bool biometricsEnabled,
      bool analyticsEnabled,
      bool crashReportingEnabled,
      NotificationSettings notificationSettings,
      PrivacySettings privacySettings,
      PerformanceSettings performanceSettings});

  @override
  $NotificationSettingsCopyWith<$Res> get notificationSettings;
  @override
  $PrivacySettingsCopyWith<$Res> get privacySettings;
  @override
  $PerformanceSettingsCopyWith<$Res> get performanceSettings;
}

/// @nodoc
class __$$ConfigStateImplCopyWithImpl<$Res>
    extends _$ConfigStateCopyWithImpl<$Res, _$ConfigStateImpl>
    implements _$$ConfigStateImplCopyWith<$Res> {
  __$$ConfigStateImplCopyWithImpl(
      _$ConfigStateImpl _value, $Res Function(_$ConfigStateImpl) _then)
      : super(_value, _then);

  /// Create a copy of ConfigState
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? themeMode = null,
    Object? locale = null,
    Object? biometricsEnabled = null,
    Object? analyticsEnabled = null,
    Object? crashReportingEnabled = null,
    Object? notificationSettings = null,
    Object? privacySettings = null,
    Object? performanceSettings = null,
  }) {
    return _then(_$ConfigStateImpl(
      themeMode: null == themeMode
          ? _value.themeMode
          : themeMode // ignore: cast_nullable_to_non_nullable
              as ThemeMode,
      locale: null == locale
          ? _value.locale
          : locale // ignore: cast_nullable_to_non_nullable
              as String,
      biometricsEnabled: null == biometricsEnabled
          ? _value.biometricsEnabled
          : biometricsEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      analyticsEnabled: null == analyticsEnabled
          ? _value.analyticsEnabled
          : analyticsEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      crashReportingEnabled: null == crashReportingEnabled
          ? _value.crashReportingEnabled
          : crashReportingEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      notificationSettings: null == notificationSettings
          ? _value.notificationSettings
          : notificationSettings // ignore: cast_nullable_to_non_nullable
              as NotificationSettings,
      privacySettings: null == privacySettings
          ? _value.privacySettings
          : privacySettings // ignore: cast_nullable_to_non_nullable
              as PrivacySettings,
      performanceSettings: null == performanceSettings
          ? _value.performanceSettings
          : performanceSettings // ignore: cast_nullable_to_non_nullable
              as PerformanceSettings,
    ));
  }
}

/// @nodoc
@JsonSerializable()
class _$ConfigStateImpl implements _ConfigState {
  const _$ConfigStateImpl(
      {this.themeMode = ThemeMode.system,
      this.locale = 'en_US',
      this.biometricsEnabled = false,
      this.analyticsEnabled = true,
      this.crashReportingEnabled = true,
      this.notificationSettings = const NotificationSettings(),
      this.privacySettings = const PrivacySettings(),
      this.performanceSettings = const PerformanceSettings()});

  factory _$ConfigStateImpl.fromJson(Map<String, dynamic> json) =>
      _$$ConfigStateImplFromJson(json);

  @override
  @JsonKey()
  final ThemeMode themeMode;
  @override
  @JsonKey()
  final String locale;
  @override
  @JsonKey()
  final bool biometricsEnabled;
  @override
  @JsonKey()
  final bool analyticsEnabled;
  @override
  @JsonKey()
  final bool crashReportingEnabled;
  @override
  @JsonKey()
  final NotificationSettings notificationSettings;
  @override
  @JsonKey()
  final PrivacySettings privacySettings;
  @override
  @JsonKey()
  final PerformanceSettings performanceSettings;

  @override
  String toString() {
    return 'ConfigState(themeMode: $themeMode, locale: $locale, biometricsEnabled: $biometricsEnabled, analyticsEnabled: $analyticsEnabled, crashReportingEnabled: $crashReportingEnabled, notificationSettings: $notificationSettings, privacySettings: $privacySettings, performanceSettings: $performanceSettings)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$ConfigStateImpl &&
            (identical(other.themeMode, themeMode) ||
                other.themeMode == themeMode) &&
            (identical(other.locale, locale) || other.locale == locale) &&
            (identical(other.biometricsEnabled, biometricsEnabled) ||
                other.biometricsEnabled == biometricsEnabled) &&
            (identical(other.analyticsEnabled, analyticsEnabled) ||
                other.analyticsEnabled == analyticsEnabled) &&
            (identical(other.crashReportingEnabled, crashReportingEnabled) ||
                other.crashReportingEnabled == crashReportingEnabled) &&
            (identical(other.notificationSettings, notificationSettings) ||
                other.notificationSettings == notificationSettings) &&
            (identical(other.privacySettings, privacySettings) ||
                other.privacySettings == privacySettings) &&
            (identical(other.performanceSettings, performanceSettings) ||
                other.performanceSettings == performanceSettings));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode => Object.hash(
      runtimeType,
      themeMode,
      locale,
      biometricsEnabled,
      analyticsEnabled,
      crashReportingEnabled,
      notificationSettings,
      privacySettings,
      performanceSettings);

  /// Create a copy of ConfigState
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$ConfigStateImplCopyWith<_$ConfigStateImpl> get copyWith =>
      __$$ConfigStateImplCopyWithImpl<_$ConfigStateImpl>(this, _$identity);

  @override
  Map<String, dynamic> toJson() {
    return _$$ConfigStateImplToJson(
      this,
    );
  }
}

abstract class _ConfigState implements ConfigState {
  const factory _ConfigState(
      {final ThemeMode themeMode,
      final String locale,
      final bool biometricsEnabled,
      final bool analyticsEnabled,
      final bool crashReportingEnabled,
      final NotificationSettings notificationSettings,
      final PrivacySettings privacySettings,
      final PerformanceSettings performanceSettings}) = _$ConfigStateImpl;

  factory _ConfigState.fromJson(Map<String, dynamic> json) =
      _$ConfigStateImpl.fromJson;

  @override
  ThemeMode get themeMode;
  @override
  String get locale;
  @override
  bool get biometricsEnabled;
  @override
  bool get analyticsEnabled;
  @override
  bool get crashReportingEnabled;
  @override
  NotificationSettings get notificationSettings;
  @override
  PrivacySettings get privacySettings;
  @override
  PerformanceSettings get performanceSettings;

  /// Create a copy of ConfigState
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$ConfigStateImplCopyWith<_$ConfigStateImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

NotificationSettings _$NotificationSettingsFromJson(Map<String, dynamic> json) {
  return _NotificationSettings.fromJson(json);
}

/// @nodoc
mixin _$NotificationSettings {
  bool get emailEnabled => throw _privateConstructorUsedError;
  bool get pushEnabled => throw _privateConstructorUsedError;
  bool get inAppEnabled => throw _privateConstructorUsedError;

  /// Serializes this NotificationSettings to a JSON map.
  Map<String, dynamic> toJson() => throw _privateConstructorUsedError;

  /// Create a copy of NotificationSettings
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $NotificationSettingsCopyWith<NotificationSettings> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $NotificationSettingsCopyWith<$Res> {
  factory $NotificationSettingsCopyWith(NotificationSettings value,
          $Res Function(NotificationSettings) then) =
      _$NotificationSettingsCopyWithImpl<$Res, NotificationSettings>;
  @useResult
  $Res call({bool emailEnabled, bool pushEnabled, bool inAppEnabled});
}

/// @nodoc
class _$NotificationSettingsCopyWithImpl<$Res,
        $Val extends NotificationSettings>
    implements $NotificationSettingsCopyWith<$Res> {
  _$NotificationSettingsCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of NotificationSettings
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? emailEnabled = null,
    Object? pushEnabled = null,
    Object? inAppEnabled = null,
  }) {
    return _then(_value.copyWith(
      emailEnabled: null == emailEnabled
          ? _value.emailEnabled
          : emailEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      pushEnabled: null == pushEnabled
          ? _value.pushEnabled
          : pushEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      inAppEnabled: null == inAppEnabled
          ? _value.inAppEnabled
          : inAppEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
    ) as $Val);
  }
}

/// @nodoc
abstract class _$$NotificationSettingsImplCopyWith<$Res>
    implements $NotificationSettingsCopyWith<$Res> {
  factory _$$NotificationSettingsImplCopyWith(_$NotificationSettingsImpl value,
          $Res Function(_$NotificationSettingsImpl) then) =
      __$$NotificationSettingsImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({bool emailEnabled, bool pushEnabled, bool inAppEnabled});
}

/// @nodoc
class __$$NotificationSettingsImplCopyWithImpl<$Res>
    extends _$NotificationSettingsCopyWithImpl<$Res, _$NotificationSettingsImpl>
    implements _$$NotificationSettingsImplCopyWith<$Res> {
  __$$NotificationSettingsImplCopyWithImpl(_$NotificationSettingsImpl _value,
      $Res Function(_$NotificationSettingsImpl) _then)
      : super(_value, _then);

  /// Create a copy of NotificationSettings
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? emailEnabled = null,
    Object? pushEnabled = null,
    Object? inAppEnabled = null,
  }) {
    return _then(_$NotificationSettingsImpl(
      emailEnabled: null == emailEnabled
          ? _value.emailEnabled
          : emailEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      pushEnabled: null == pushEnabled
          ? _value.pushEnabled
          : pushEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      inAppEnabled: null == inAppEnabled
          ? _value.inAppEnabled
          : inAppEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
    ));
  }
}

/// @nodoc
@JsonSerializable()
class _$NotificationSettingsImpl implements _NotificationSettings {
  const _$NotificationSettingsImpl(
      {this.emailEnabled = true,
      this.pushEnabled = true,
      this.inAppEnabled = true});

  factory _$NotificationSettingsImpl.fromJson(Map<String, dynamic> json) =>
      _$$NotificationSettingsImplFromJson(json);

  @override
  @JsonKey()
  final bool emailEnabled;
  @override
  @JsonKey()
  final bool pushEnabled;
  @override
  @JsonKey()
  final bool inAppEnabled;

  @override
  String toString() {
    return 'NotificationSettings(emailEnabled: $emailEnabled, pushEnabled: $pushEnabled, inAppEnabled: $inAppEnabled)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$NotificationSettingsImpl &&
            (identical(other.emailEnabled, emailEnabled) ||
                other.emailEnabled == emailEnabled) &&
            (identical(other.pushEnabled, pushEnabled) ||
                other.pushEnabled == pushEnabled) &&
            (identical(other.inAppEnabled, inAppEnabled) ||
                other.inAppEnabled == inAppEnabled));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode =>
      Object.hash(runtimeType, emailEnabled, pushEnabled, inAppEnabled);

  /// Create a copy of NotificationSettings
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$NotificationSettingsImplCopyWith<_$NotificationSettingsImpl>
      get copyWith =>
          __$$NotificationSettingsImplCopyWithImpl<_$NotificationSettingsImpl>(
              this, _$identity);

  @override
  Map<String, dynamic> toJson() {
    return _$$NotificationSettingsImplToJson(
      this,
    );
  }
}

abstract class _NotificationSettings implements NotificationSettings {
  const factory _NotificationSettings(
      {final bool emailEnabled,
      final bool pushEnabled,
      final bool inAppEnabled}) = _$NotificationSettingsImpl;

  factory _NotificationSettings.fromJson(Map<String, dynamic> json) =
      _$NotificationSettingsImpl.fromJson;

  @override
  bool get emailEnabled;
  @override
  bool get pushEnabled;
  @override
  bool get inAppEnabled;

  /// Create a copy of NotificationSettings
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$NotificationSettingsImplCopyWith<_$NotificationSettingsImpl>
      get copyWith => throw _privateConstructorUsedError;
}

PrivacySettings _$PrivacySettingsFromJson(Map<String, dynamic> json) {
  return _PrivacySettings.fromJson(json);
}

/// @nodoc
mixin _$PrivacySettings {
  bool get locationTrackingEnabled => throw _privateConstructorUsedError;
  bool get dataCollectionEnabled => throw _privateConstructorUsedError;
  bool get thirdPartySharingEnabled => throw _privateConstructorUsedError;

  /// Serializes this PrivacySettings to a JSON map.
  Map<String, dynamic> toJson() => throw _privateConstructorUsedError;

  /// Create a copy of PrivacySettings
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $PrivacySettingsCopyWith<PrivacySettings> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $PrivacySettingsCopyWith<$Res> {
  factory $PrivacySettingsCopyWith(
          PrivacySettings value, $Res Function(PrivacySettings) then) =
      _$PrivacySettingsCopyWithImpl<$Res, PrivacySettings>;
  @useResult
  $Res call(
      {bool locationTrackingEnabled,
      bool dataCollectionEnabled,
      bool thirdPartySharingEnabled});
}

/// @nodoc
class _$PrivacySettingsCopyWithImpl<$Res, $Val extends PrivacySettings>
    implements $PrivacySettingsCopyWith<$Res> {
  _$PrivacySettingsCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of PrivacySettings
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? locationTrackingEnabled = null,
    Object? dataCollectionEnabled = null,
    Object? thirdPartySharingEnabled = null,
  }) {
    return _then(_value.copyWith(
      locationTrackingEnabled: null == locationTrackingEnabled
          ? _value.locationTrackingEnabled
          : locationTrackingEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      dataCollectionEnabled: null == dataCollectionEnabled
          ? _value.dataCollectionEnabled
          : dataCollectionEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      thirdPartySharingEnabled: null == thirdPartySharingEnabled
          ? _value.thirdPartySharingEnabled
          : thirdPartySharingEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
    ) as $Val);
  }
}

/// @nodoc
abstract class _$$PrivacySettingsImplCopyWith<$Res>
    implements $PrivacySettingsCopyWith<$Res> {
  factory _$$PrivacySettingsImplCopyWith(_$PrivacySettingsImpl value,
          $Res Function(_$PrivacySettingsImpl) then) =
      __$$PrivacySettingsImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call(
      {bool locationTrackingEnabled,
      bool dataCollectionEnabled,
      bool thirdPartySharingEnabled});
}

/// @nodoc
class __$$PrivacySettingsImplCopyWithImpl<$Res>
    extends _$PrivacySettingsCopyWithImpl<$Res, _$PrivacySettingsImpl>
    implements _$$PrivacySettingsImplCopyWith<$Res> {
  __$$PrivacySettingsImplCopyWithImpl(
      _$PrivacySettingsImpl _value, $Res Function(_$PrivacySettingsImpl) _then)
      : super(_value, _then);

  /// Create a copy of PrivacySettings
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? locationTrackingEnabled = null,
    Object? dataCollectionEnabled = null,
    Object? thirdPartySharingEnabled = null,
  }) {
    return _then(_$PrivacySettingsImpl(
      locationTrackingEnabled: null == locationTrackingEnabled
          ? _value.locationTrackingEnabled
          : locationTrackingEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      dataCollectionEnabled: null == dataCollectionEnabled
          ? _value.dataCollectionEnabled
          : dataCollectionEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      thirdPartySharingEnabled: null == thirdPartySharingEnabled
          ? _value.thirdPartySharingEnabled
          : thirdPartySharingEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
    ));
  }
}

/// @nodoc
@JsonSerializable()
class _$PrivacySettingsImpl implements _PrivacySettings {
  const _$PrivacySettingsImpl(
      {this.locationTrackingEnabled = true,
      this.dataCollectionEnabled = true,
      this.thirdPartySharingEnabled = true});

  factory _$PrivacySettingsImpl.fromJson(Map<String, dynamic> json) =>
      _$$PrivacySettingsImplFromJson(json);

  @override
  @JsonKey()
  final bool locationTrackingEnabled;
  @override
  @JsonKey()
  final bool dataCollectionEnabled;
  @override
  @JsonKey()
  final bool thirdPartySharingEnabled;

  @override
  String toString() {
    return 'PrivacySettings(locationTrackingEnabled: $locationTrackingEnabled, dataCollectionEnabled: $dataCollectionEnabled, thirdPartySharingEnabled: $thirdPartySharingEnabled)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$PrivacySettingsImpl &&
            (identical(
                    other.locationTrackingEnabled, locationTrackingEnabled) ||
                other.locationTrackingEnabled == locationTrackingEnabled) &&
            (identical(other.dataCollectionEnabled, dataCollectionEnabled) ||
                other.dataCollectionEnabled == dataCollectionEnabled) &&
            (identical(
                    other.thirdPartySharingEnabled, thirdPartySharingEnabled) ||
                other.thirdPartySharingEnabled == thirdPartySharingEnabled));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode => Object.hash(runtimeType, locationTrackingEnabled,
      dataCollectionEnabled, thirdPartySharingEnabled);

  /// Create a copy of PrivacySettings
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$PrivacySettingsImplCopyWith<_$PrivacySettingsImpl> get copyWith =>
      __$$PrivacySettingsImplCopyWithImpl<_$PrivacySettingsImpl>(
          this, _$identity);

  @override
  Map<String, dynamic> toJson() {
    return _$$PrivacySettingsImplToJson(
      this,
    );
  }
}

abstract class _PrivacySettings implements PrivacySettings {
  const factory _PrivacySettings(
      {final bool locationTrackingEnabled,
      final bool dataCollectionEnabled,
      final bool thirdPartySharingEnabled}) = _$PrivacySettingsImpl;

  factory _PrivacySettings.fromJson(Map<String, dynamic> json) =
      _$PrivacySettingsImpl.fromJson;

  @override
  bool get locationTrackingEnabled;
  @override
  bool get dataCollectionEnabled;
  @override
  bool get thirdPartySharingEnabled;

  /// Create a copy of PrivacySettings
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$PrivacySettingsImplCopyWith<_$PrivacySettingsImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

PerformanceSettings _$PerformanceSettingsFromJson(Map<String, dynamic> json) {
  return _PerformanceSettings.fromJson(json);
}

/// @nodoc
mixin _$PerformanceSettings {
  bool get imageCacheEnabled => throw _privateConstructorUsedError;
  bool get offlineModeEnabled => throw _privateConstructorUsedError;
  bool get backgroundRefreshEnabled => throw _privateConstructorUsedError;

  /// Serializes this PerformanceSettings to a JSON map.
  Map<String, dynamic> toJson() => throw _privateConstructorUsedError;

  /// Create a copy of PerformanceSettings
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $PerformanceSettingsCopyWith<PerformanceSettings> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $PerformanceSettingsCopyWith<$Res> {
  factory $PerformanceSettingsCopyWith(
          PerformanceSettings value, $Res Function(PerformanceSettings) then) =
      _$PerformanceSettingsCopyWithImpl<$Res, PerformanceSettings>;
  @useResult
  $Res call(
      {bool imageCacheEnabled,
      bool offlineModeEnabled,
      bool backgroundRefreshEnabled});
}

/// @nodoc
class _$PerformanceSettingsCopyWithImpl<$Res, $Val extends PerformanceSettings>
    implements $PerformanceSettingsCopyWith<$Res> {
  _$PerformanceSettingsCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of PerformanceSettings
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? imageCacheEnabled = null,
    Object? offlineModeEnabled = null,
    Object? backgroundRefreshEnabled = null,
  }) {
    return _then(_value.copyWith(
      imageCacheEnabled: null == imageCacheEnabled
          ? _value.imageCacheEnabled
          : imageCacheEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      offlineModeEnabled: null == offlineModeEnabled
          ? _value.offlineModeEnabled
          : offlineModeEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      backgroundRefreshEnabled: null == backgroundRefreshEnabled
          ? _value.backgroundRefreshEnabled
          : backgroundRefreshEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
    ) as $Val);
  }
}

/// @nodoc
abstract class _$$PerformanceSettingsImplCopyWith<$Res>
    implements $PerformanceSettingsCopyWith<$Res> {
  factory _$$PerformanceSettingsImplCopyWith(_$PerformanceSettingsImpl value,
          $Res Function(_$PerformanceSettingsImpl) then) =
      __$$PerformanceSettingsImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call(
      {bool imageCacheEnabled,
      bool offlineModeEnabled,
      bool backgroundRefreshEnabled});
}

/// @nodoc
class __$$PerformanceSettingsImplCopyWithImpl<$Res>
    extends _$PerformanceSettingsCopyWithImpl<$Res, _$PerformanceSettingsImpl>
    implements _$$PerformanceSettingsImplCopyWith<$Res> {
  __$$PerformanceSettingsImplCopyWithImpl(_$PerformanceSettingsImpl _value,
      $Res Function(_$PerformanceSettingsImpl) _then)
      : super(_value, _then);

  /// Create a copy of PerformanceSettings
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? imageCacheEnabled = null,
    Object? offlineModeEnabled = null,
    Object? backgroundRefreshEnabled = null,
  }) {
    return _then(_$PerformanceSettingsImpl(
      imageCacheEnabled: null == imageCacheEnabled
          ? _value.imageCacheEnabled
          : imageCacheEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      offlineModeEnabled: null == offlineModeEnabled
          ? _value.offlineModeEnabled
          : offlineModeEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
      backgroundRefreshEnabled: null == backgroundRefreshEnabled
          ? _value.backgroundRefreshEnabled
          : backgroundRefreshEnabled // ignore: cast_nullable_to_non_nullable
              as bool,
    ));
  }
}

/// @nodoc
@JsonSerializable()
class _$PerformanceSettingsImpl implements _PerformanceSettings {
  const _$PerformanceSettingsImpl(
      {this.imageCacheEnabled = true,
      this.offlineModeEnabled = false,
      this.backgroundRefreshEnabled = true});

  factory _$PerformanceSettingsImpl.fromJson(Map<String, dynamic> json) =>
      _$$PerformanceSettingsImplFromJson(json);

  @override
  @JsonKey()
  final bool imageCacheEnabled;
  @override
  @JsonKey()
  final bool offlineModeEnabled;
  @override
  @JsonKey()
  final bool backgroundRefreshEnabled;

  @override
  String toString() {
    return 'PerformanceSettings(imageCacheEnabled: $imageCacheEnabled, offlineModeEnabled: $offlineModeEnabled, backgroundRefreshEnabled: $backgroundRefreshEnabled)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$PerformanceSettingsImpl &&
            (identical(other.imageCacheEnabled, imageCacheEnabled) ||
                other.imageCacheEnabled == imageCacheEnabled) &&
            (identical(other.offlineModeEnabled, offlineModeEnabled) ||
                other.offlineModeEnabled == offlineModeEnabled) &&
            (identical(
                    other.backgroundRefreshEnabled, backgroundRefreshEnabled) ||
                other.backgroundRefreshEnabled == backgroundRefreshEnabled));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode => Object.hash(runtimeType, imageCacheEnabled,
      offlineModeEnabled, backgroundRefreshEnabled);

  /// Create a copy of PerformanceSettings
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$PerformanceSettingsImplCopyWith<_$PerformanceSettingsImpl> get copyWith =>
      __$$PerformanceSettingsImplCopyWithImpl<_$PerformanceSettingsImpl>(
          this, _$identity);

  @override
  Map<String, dynamic> toJson() {
    return _$$PerformanceSettingsImplToJson(
      this,
    );
  }
}

abstract class _PerformanceSettings implements PerformanceSettings {
  const factory _PerformanceSettings(
      {final bool imageCacheEnabled,
      final bool offlineModeEnabled,
      final bool backgroundRefreshEnabled}) = _$PerformanceSettingsImpl;

  factory _PerformanceSettings.fromJson(Map<String, dynamic> json) =
      _$PerformanceSettingsImpl.fromJson;

  @override
  bool get imageCacheEnabled;
  @override
  bool get offlineModeEnabled;
  @override
  bool get backgroundRefreshEnabled;

  /// Create a copy of PerformanceSettings
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$PerformanceSettingsImplCopyWith<_$PerformanceSettingsImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
