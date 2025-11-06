// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'config_models.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
    'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models');

NotificationSettings _$NotificationSettingsFromJson(Map<String, dynamic> json) {
  return _NotificationSettings.fromJson(json);
}

/// @nodoc
mixin _$NotificationSettings {
  bool get emailEnabled => throw _privateConstructorUsedError;
  bool get pushEnabled => throw _privateConstructorUsedError;
  bool get inAppEnabled => throw _privateConstructorUsedError;
  bool get soundEnabled => throw _privateConstructorUsedError;

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
  $Res call(
      {bool emailEnabled,
      bool pushEnabled,
      bool inAppEnabled,
      bool soundEnabled});
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
    Object? soundEnabled = null,
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
      soundEnabled: null == soundEnabled
          ? _value.soundEnabled
          : soundEnabled // ignore: cast_nullable_to_non_nullable
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
  $Res call(
      {bool emailEnabled,
      bool pushEnabled,
      bool inAppEnabled,
      bool soundEnabled});
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
    Object? soundEnabled = null,
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
      soundEnabled: null == soundEnabled
          ? _value.soundEnabled
          : soundEnabled // ignore: cast_nullable_to_non_nullable
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
      this.inAppEnabled = true,
      this.soundEnabled = true});

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
  @JsonKey()
  final bool soundEnabled;

  @override
  String toString() {
    return 'NotificationSettings(emailEnabled: $emailEnabled, pushEnabled: $pushEnabled, inAppEnabled: $inAppEnabled, soundEnabled: $soundEnabled)';
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
                other.inAppEnabled == inAppEnabled) &&
            (identical(other.soundEnabled, soundEnabled) ||
                other.soundEnabled == soundEnabled));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode => Object.hash(
      runtimeType, emailEnabled, pushEnabled, inAppEnabled, soundEnabled);

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
      final bool inAppEnabled,
      final bool soundEnabled}) = _$NotificationSettingsImpl;

  factory _NotificationSettings.fromJson(Map<String, dynamic> json) =
      _$NotificationSettingsImpl.fromJson;

  @override
  bool get emailEnabled;
  @override
  bool get pushEnabled;
  @override
  bool get inAppEnabled;
  @override
  bool get soundEnabled;

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
  bool get personalizationEnabled => throw _privateConstructorUsedError;

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
      bool personalizationEnabled});
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
    Object? personalizationEnabled = null,
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
      personalizationEnabled: null == personalizationEnabled
          ? _value.personalizationEnabled
          : personalizationEnabled // ignore: cast_nullable_to_non_nullable
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
      bool personalizationEnabled});
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
    Object? personalizationEnabled = null,
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
      personalizationEnabled: null == personalizationEnabled
          ? _value.personalizationEnabled
          : personalizationEnabled // ignore: cast_nullable_to_non_nullable
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
      this.personalizationEnabled = true});

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
  final bool personalizationEnabled;

  @override
  String toString() {
    return 'PrivacySettings(locationTrackingEnabled: $locationTrackingEnabled, dataCollectionEnabled: $dataCollectionEnabled, personalizationEnabled: $personalizationEnabled)';
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
            (identical(other.personalizationEnabled, personalizationEnabled) ||
                other.personalizationEnabled == personalizationEnabled));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode => Object.hash(runtimeType, locationTrackingEnabled,
      dataCollectionEnabled, personalizationEnabled);

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
      final bool personalizationEnabled}) = _$PrivacySettingsImpl;

  factory _PrivacySettings.fromJson(Map<String, dynamic> json) =
      _$PrivacySettingsImpl.fromJson;

  @override
  bool get locationTrackingEnabled;
  @override
  bool get dataCollectionEnabled;
  @override
  bool get personalizationEnabled;

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
  bool get imageOptimizationEnabled => throw _privateConstructorUsedError;
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
      {bool imageOptimizationEnabled,
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
    Object? imageOptimizationEnabled = null,
    Object? offlineModeEnabled = null,
    Object? backgroundRefreshEnabled = null,
  }) {
    return _then(_value.copyWith(
      imageOptimizationEnabled: null == imageOptimizationEnabled
          ? _value.imageOptimizationEnabled
          : imageOptimizationEnabled // ignore: cast_nullable_to_non_nullable
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
      {bool imageOptimizationEnabled,
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
    Object? imageOptimizationEnabled = null,
    Object? offlineModeEnabled = null,
    Object? backgroundRefreshEnabled = null,
  }) {
    return _then(_$PerformanceSettingsImpl(
      imageOptimizationEnabled: null == imageOptimizationEnabled
          ? _value.imageOptimizationEnabled
          : imageOptimizationEnabled // ignore: cast_nullable_to_non_nullable
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
      {this.imageOptimizationEnabled = true,
      this.offlineModeEnabled = true,
      this.backgroundRefreshEnabled = true});

  factory _$PerformanceSettingsImpl.fromJson(Map<String, dynamic> json) =>
      _$$PerformanceSettingsImplFromJson(json);

  @override
  @JsonKey()
  final bool imageOptimizationEnabled;
  @override
  @JsonKey()
  final bool offlineModeEnabled;
  @override
  @JsonKey()
  final bool backgroundRefreshEnabled;

  @override
  String toString() {
    return 'PerformanceSettings(imageOptimizationEnabled: $imageOptimizationEnabled, offlineModeEnabled: $offlineModeEnabled, backgroundRefreshEnabled: $backgroundRefreshEnabled)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$PerformanceSettingsImpl &&
            (identical(
                    other.imageOptimizationEnabled, imageOptimizationEnabled) ||
                other.imageOptimizationEnabled == imageOptimizationEnabled) &&
            (identical(other.offlineModeEnabled, offlineModeEnabled) ||
                other.offlineModeEnabled == offlineModeEnabled) &&
            (identical(
                    other.backgroundRefreshEnabled, backgroundRefreshEnabled) ||
                other.backgroundRefreshEnabled == backgroundRefreshEnabled));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode => Object.hash(runtimeType, imageOptimizationEnabled,
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
      {final bool imageOptimizationEnabled,
      final bool offlineModeEnabled,
      final bool backgroundRefreshEnabled}) = _$PerformanceSettingsImpl;

  factory _PerformanceSettings.fromJson(Map<String, dynamic> json) =
      _$PerformanceSettingsImpl.fromJson;

  @override
  bool get imageOptimizationEnabled;
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
