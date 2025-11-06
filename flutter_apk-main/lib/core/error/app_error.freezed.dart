// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'app_error.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
    'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models');

/// @nodoc
mixin _$AppError {
  String? get message => throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String message, String? endpoint) network,
    required TResult Function(String message, String? code) api,
    required TResult Function(String? message, bool needsReauthentication)
        unauthorized,
    required TResult Function(
            String message, Map<String, List<String>>? fieldErrors)
        validation,
    required TResult Function(String message, String? type) security,
    required TResult Function(String message) quantum,
    required TResult Function(
            String message, Object? error, StackTrace? stackTrace)
        unexpected,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String message, String? endpoint)? network,
    TResult? Function(String message, String? code)? api,
    TResult? Function(String? message, bool needsReauthentication)?
        unauthorized,
    TResult? Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult? Function(String message, String? type)? security,
    TResult? Function(String message)? quantum,
    TResult? Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String message, String? endpoint)? network,
    TResult Function(String message, String? code)? api,
    TResult Function(String? message, bool needsReauthentication)? unauthorized,
    TResult Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult Function(String message, String? type)? security,
    TResult Function(String message)? quantum,
    TResult Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(_NetworkError value) network,
    required TResult Function(_ApiError value) api,
    required TResult Function(_UnauthorizedError value) unauthorized,
    required TResult Function(_ValidationError value) validation,
    required TResult Function(_SecurityError value) security,
    required TResult Function(_QuantumError value) quantum,
    required TResult Function(_UnexpectedError value) unexpected,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(_NetworkError value)? network,
    TResult? Function(_ApiError value)? api,
    TResult? Function(_UnauthorizedError value)? unauthorized,
    TResult? Function(_ValidationError value)? validation,
    TResult? Function(_SecurityError value)? security,
    TResult? Function(_QuantumError value)? quantum,
    TResult? Function(_UnexpectedError value)? unexpected,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(_NetworkError value)? network,
    TResult Function(_ApiError value)? api,
    TResult Function(_UnauthorizedError value)? unauthorized,
    TResult Function(_ValidationError value)? validation,
    TResult Function(_SecurityError value)? security,
    TResult Function(_QuantumError value)? quantum,
    TResult Function(_UnexpectedError value)? unexpected,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $AppErrorCopyWith<AppError> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $AppErrorCopyWith<$Res> {
  factory $AppErrorCopyWith(AppError value, $Res Function(AppError) then) =
      _$AppErrorCopyWithImpl<$Res, AppError>;
  @useResult
  $Res call({String message});
}

/// @nodoc
class _$AppErrorCopyWithImpl<$Res, $Val extends AppError>
    implements $AppErrorCopyWith<$Res> {
  _$AppErrorCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? message = null,
  }) {
    return _then(_value.copyWith(
      message: null == message
          ? _value.message!
          : message // ignore: cast_nullable_to_non_nullable
              as String,
    ) as $Val);
  }
}

/// @nodoc
abstract class _$$NetworkErrorImplCopyWith<$Res>
    implements $AppErrorCopyWith<$Res> {
  factory _$$NetworkErrorImplCopyWith(
          _$NetworkErrorImpl value, $Res Function(_$NetworkErrorImpl) then) =
      __$$NetworkErrorImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({String message, String? endpoint});
}

/// @nodoc
class __$$NetworkErrorImplCopyWithImpl<$Res>
    extends _$AppErrorCopyWithImpl<$Res, _$NetworkErrorImpl>
    implements _$$NetworkErrorImplCopyWith<$Res> {
  __$$NetworkErrorImplCopyWithImpl(
      _$NetworkErrorImpl _value, $Res Function(_$NetworkErrorImpl) _then)
      : super(_value, _then);

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? message = null,
    Object? endpoint = freezed,
  }) {
    return _then(_$NetworkErrorImpl(
      message: null == message
          ? _value.message
          : message // ignore: cast_nullable_to_non_nullable
              as String,
      endpoint: freezed == endpoint
          ? _value.endpoint
          : endpoint // ignore: cast_nullable_to_non_nullable
              as String?,
    ));
  }
}

/// @nodoc

class _$NetworkErrorImpl extends _NetworkError {
  const _$NetworkErrorImpl({required this.message, this.endpoint}) : super._();

  @override
  final String message;
  @override
  final String? endpoint;

  @override
  String toString() {
    return 'AppError.network(message: $message, endpoint: $endpoint)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$NetworkErrorImpl &&
            (identical(other.message, message) || other.message == message) &&
            (identical(other.endpoint, endpoint) ||
                other.endpoint == endpoint));
  }

  @override
  int get hashCode => Object.hash(runtimeType, message, endpoint);

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$NetworkErrorImplCopyWith<_$NetworkErrorImpl> get copyWith =>
      __$$NetworkErrorImplCopyWithImpl<_$NetworkErrorImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String message, String? endpoint) network,
    required TResult Function(String message, String? code) api,
    required TResult Function(String? message, bool needsReauthentication)
        unauthorized,
    required TResult Function(
            String message, Map<String, List<String>>? fieldErrors)
        validation,
    required TResult Function(String message, String? type) security,
    required TResult Function(String message) quantum,
    required TResult Function(
            String message, Object? error, StackTrace? stackTrace)
        unexpected,
  }) {
    return network(message, endpoint);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String message, String? endpoint)? network,
    TResult? Function(String message, String? code)? api,
    TResult? Function(String? message, bool needsReauthentication)?
        unauthorized,
    TResult? Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult? Function(String message, String? type)? security,
    TResult? Function(String message)? quantum,
    TResult? Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
  }) {
    return network?.call(message, endpoint);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String message, String? endpoint)? network,
    TResult Function(String message, String? code)? api,
    TResult Function(String? message, bool needsReauthentication)? unauthorized,
    TResult Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult Function(String message, String? type)? security,
    TResult Function(String message)? quantum,
    TResult Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
    required TResult orElse(),
  }) {
    if (network != null) {
      return network(message, endpoint);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(_NetworkError value) network,
    required TResult Function(_ApiError value) api,
    required TResult Function(_UnauthorizedError value) unauthorized,
    required TResult Function(_ValidationError value) validation,
    required TResult Function(_SecurityError value) security,
    required TResult Function(_QuantumError value) quantum,
    required TResult Function(_UnexpectedError value) unexpected,
  }) {
    return network(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(_NetworkError value)? network,
    TResult? Function(_ApiError value)? api,
    TResult? Function(_UnauthorizedError value)? unauthorized,
    TResult? Function(_ValidationError value)? validation,
    TResult? Function(_SecurityError value)? security,
    TResult? Function(_QuantumError value)? quantum,
    TResult? Function(_UnexpectedError value)? unexpected,
  }) {
    return network?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(_NetworkError value)? network,
    TResult Function(_ApiError value)? api,
    TResult Function(_UnauthorizedError value)? unauthorized,
    TResult Function(_ValidationError value)? validation,
    TResult Function(_SecurityError value)? security,
    TResult Function(_QuantumError value)? quantum,
    TResult Function(_UnexpectedError value)? unexpected,
    required TResult orElse(),
  }) {
    if (network != null) {
      return network(this);
    }
    return orElse();
  }
}

abstract class _NetworkError extends AppError {
  const factory _NetworkError(
      {required final String message,
      final String? endpoint}) = _$NetworkErrorImpl;
  const _NetworkError._() : super._();

  @override
  String get message;
  String? get endpoint;

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$NetworkErrorImplCopyWith<_$NetworkErrorImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$ApiErrorImplCopyWith<$Res>
    implements $AppErrorCopyWith<$Res> {
  factory _$$ApiErrorImplCopyWith(
          _$ApiErrorImpl value, $Res Function(_$ApiErrorImpl) then) =
      __$$ApiErrorImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({String message, String? code});
}

/// @nodoc
class __$$ApiErrorImplCopyWithImpl<$Res>
    extends _$AppErrorCopyWithImpl<$Res, _$ApiErrorImpl>
    implements _$$ApiErrorImplCopyWith<$Res> {
  __$$ApiErrorImplCopyWithImpl(
      _$ApiErrorImpl _value, $Res Function(_$ApiErrorImpl) _then)
      : super(_value, _then);

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? message = null,
    Object? code = freezed,
  }) {
    return _then(_$ApiErrorImpl(
      message: null == message
          ? _value.message
          : message // ignore: cast_nullable_to_non_nullable
              as String,
      code: freezed == code
          ? _value.code
          : code // ignore: cast_nullable_to_non_nullable
              as String?,
    ));
  }
}

/// @nodoc

class _$ApiErrorImpl extends _ApiError {
  const _$ApiErrorImpl({required this.message, this.code}) : super._();

  @override
  final String message;
  @override
  final String? code;

  @override
  String toString() {
    return 'AppError.api(message: $message, code: $code)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$ApiErrorImpl &&
            (identical(other.message, message) || other.message == message) &&
            (identical(other.code, code) || other.code == code));
  }

  @override
  int get hashCode => Object.hash(runtimeType, message, code);

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$ApiErrorImplCopyWith<_$ApiErrorImpl> get copyWith =>
      __$$ApiErrorImplCopyWithImpl<_$ApiErrorImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String message, String? endpoint) network,
    required TResult Function(String message, String? code) api,
    required TResult Function(String? message, bool needsReauthentication)
        unauthorized,
    required TResult Function(
            String message, Map<String, List<String>>? fieldErrors)
        validation,
    required TResult Function(String message, String? type) security,
    required TResult Function(String message) quantum,
    required TResult Function(
            String message, Object? error, StackTrace? stackTrace)
        unexpected,
  }) {
    return api(message, code);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String message, String? endpoint)? network,
    TResult? Function(String message, String? code)? api,
    TResult? Function(String? message, bool needsReauthentication)?
        unauthorized,
    TResult? Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult? Function(String message, String? type)? security,
    TResult? Function(String message)? quantum,
    TResult? Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
  }) {
    return api?.call(message, code);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String message, String? endpoint)? network,
    TResult Function(String message, String? code)? api,
    TResult Function(String? message, bool needsReauthentication)? unauthorized,
    TResult Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult Function(String message, String? type)? security,
    TResult Function(String message)? quantum,
    TResult Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
    required TResult orElse(),
  }) {
    if (api != null) {
      return api(message, code);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(_NetworkError value) network,
    required TResult Function(_ApiError value) api,
    required TResult Function(_UnauthorizedError value) unauthorized,
    required TResult Function(_ValidationError value) validation,
    required TResult Function(_SecurityError value) security,
    required TResult Function(_QuantumError value) quantum,
    required TResult Function(_UnexpectedError value) unexpected,
  }) {
    return api(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(_NetworkError value)? network,
    TResult? Function(_ApiError value)? api,
    TResult? Function(_UnauthorizedError value)? unauthorized,
    TResult? Function(_ValidationError value)? validation,
    TResult? Function(_SecurityError value)? security,
    TResult? Function(_QuantumError value)? quantum,
    TResult? Function(_UnexpectedError value)? unexpected,
  }) {
    return api?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(_NetworkError value)? network,
    TResult Function(_ApiError value)? api,
    TResult Function(_UnauthorizedError value)? unauthorized,
    TResult Function(_ValidationError value)? validation,
    TResult Function(_SecurityError value)? security,
    TResult Function(_QuantumError value)? quantum,
    TResult Function(_UnexpectedError value)? unexpected,
    required TResult orElse(),
  }) {
    if (api != null) {
      return api(this);
    }
    return orElse();
  }
}

abstract class _ApiError extends AppError {
  const factory _ApiError({required final String message, final String? code}) =
      _$ApiErrorImpl;
  const _ApiError._() : super._();

  @override
  String get message;
  String? get code;

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$ApiErrorImplCopyWith<_$ApiErrorImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$UnauthorizedErrorImplCopyWith<$Res>
    implements $AppErrorCopyWith<$Res> {
  factory _$$UnauthorizedErrorImplCopyWith(_$UnauthorizedErrorImpl value,
          $Res Function(_$UnauthorizedErrorImpl) then) =
      __$$UnauthorizedErrorImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({String? message, bool needsReauthentication});
}

/// @nodoc
class __$$UnauthorizedErrorImplCopyWithImpl<$Res>
    extends _$AppErrorCopyWithImpl<$Res, _$UnauthorizedErrorImpl>
    implements _$$UnauthorizedErrorImplCopyWith<$Res> {
  __$$UnauthorizedErrorImplCopyWithImpl(_$UnauthorizedErrorImpl _value,
      $Res Function(_$UnauthorizedErrorImpl) _then)
      : super(_value, _then);

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? message = freezed,
    Object? needsReauthentication = null,
  }) {
    return _then(_$UnauthorizedErrorImpl(
      message: freezed == message
          ? _value.message
          : message // ignore: cast_nullable_to_non_nullable
              as String?,
      needsReauthentication: null == needsReauthentication
          ? _value.needsReauthentication
          : needsReauthentication // ignore: cast_nullable_to_non_nullable
              as bool,
    ));
  }
}

/// @nodoc

class _$UnauthorizedErrorImpl extends _UnauthorizedError {
  const _$UnauthorizedErrorImpl(
      {this.message, this.needsReauthentication = false})
      : super._();

  @override
  final String? message;
  @override
  @JsonKey()
  final bool needsReauthentication;

  @override
  String toString() {
    return 'AppError.unauthorized(message: $message, needsReauthentication: $needsReauthentication)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$UnauthorizedErrorImpl &&
            (identical(other.message, message) || other.message == message) &&
            (identical(other.needsReauthentication, needsReauthentication) ||
                other.needsReauthentication == needsReauthentication));
  }

  @override
  int get hashCode => Object.hash(runtimeType, message, needsReauthentication);

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$UnauthorizedErrorImplCopyWith<_$UnauthorizedErrorImpl> get copyWith =>
      __$$UnauthorizedErrorImplCopyWithImpl<_$UnauthorizedErrorImpl>(
          this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String message, String? endpoint) network,
    required TResult Function(String message, String? code) api,
    required TResult Function(String? message, bool needsReauthentication)
        unauthorized,
    required TResult Function(
            String message, Map<String, List<String>>? fieldErrors)
        validation,
    required TResult Function(String message, String? type) security,
    required TResult Function(String message) quantum,
    required TResult Function(
            String message, Object? error, StackTrace? stackTrace)
        unexpected,
  }) {
    return unauthorized(message, needsReauthentication);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String message, String? endpoint)? network,
    TResult? Function(String message, String? code)? api,
    TResult? Function(String? message, bool needsReauthentication)?
        unauthorized,
    TResult? Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult? Function(String message, String? type)? security,
    TResult? Function(String message)? quantum,
    TResult? Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
  }) {
    return unauthorized?.call(message, needsReauthentication);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String message, String? endpoint)? network,
    TResult Function(String message, String? code)? api,
    TResult Function(String? message, bool needsReauthentication)? unauthorized,
    TResult Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult Function(String message, String? type)? security,
    TResult Function(String message)? quantum,
    TResult Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
    required TResult orElse(),
  }) {
    if (unauthorized != null) {
      return unauthorized(message, needsReauthentication);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(_NetworkError value) network,
    required TResult Function(_ApiError value) api,
    required TResult Function(_UnauthorizedError value) unauthorized,
    required TResult Function(_ValidationError value) validation,
    required TResult Function(_SecurityError value) security,
    required TResult Function(_QuantumError value) quantum,
    required TResult Function(_UnexpectedError value) unexpected,
  }) {
    return unauthorized(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(_NetworkError value)? network,
    TResult? Function(_ApiError value)? api,
    TResult? Function(_UnauthorizedError value)? unauthorized,
    TResult? Function(_ValidationError value)? validation,
    TResult? Function(_SecurityError value)? security,
    TResult? Function(_QuantumError value)? quantum,
    TResult? Function(_UnexpectedError value)? unexpected,
  }) {
    return unauthorized?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(_NetworkError value)? network,
    TResult Function(_ApiError value)? api,
    TResult Function(_UnauthorizedError value)? unauthorized,
    TResult Function(_ValidationError value)? validation,
    TResult Function(_SecurityError value)? security,
    TResult Function(_QuantumError value)? quantum,
    TResult Function(_UnexpectedError value)? unexpected,
    required TResult orElse(),
  }) {
    if (unauthorized != null) {
      return unauthorized(this);
    }
    return orElse();
  }
}

abstract class _UnauthorizedError extends AppError {
  const factory _UnauthorizedError(
      {final String? message,
      final bool needsReauthentication}) = _$UnauthorizedErrorImpl;
  const _UnauthorizedError._() : super._();

  @override
  String? get message;
  bool get needsReauthentication;

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$UnauthorizedErrorImplCopyWith<_$UnauthorizedErrorImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$ValidationErrorImplCopyWith<$Res>
    implements $AppErrorCopyWith<$Res> {
  factory _$$ValidationErrorImplCopyWith(_$ValidationErrorImpl value,
          $Res Function(_$ValidationErrorImpl) then) =
      __$$ValidationErrorImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({String message, Map<String, List<String>>? fieldErrors});
}

/// @nodoc
class __$$ValidationErrorImplCopyWithImpl<$Res>
    extends _$AppErrorCopyWithImpl<$Res, _$ValidationErrorImpl>
    implements _$$ValidationErrorImplCopyWith<$Res> {
  __$$ValidationErrorImplCopyWithImpl(
      _$ValidationErrorImpl _value, $Res Function(_$ValidationErrorImpl) _then)
      : super(_value, _then);

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? message = null,
    Object? fieldErrors = freezed,
  }) {
    return _then(_$ValidationErrorImpl(
      message: null == message
          ? _value.message
          : message // ignore: cast_nullable_to_non_nullable
              as String,
      fieldErrors: freezed == fieldErrors
          ? _value._fieldErrors
          : fieldErrors // ignore: cast_nullable_to_non_nullable
              as Map<String, List<String>>?,
    ));
  }
}

/// @nodoc

class _$ValidationErrorImpl extends _ValidationError {
  const _$ValidationErrorImpl(
      {required this.message, final Map<String, List<String>>? fieldErrors})
      : _fieldErrors = fieldErrors,
        super._();

  @override
  final String message;
  final Map<String, List<String>>? _fieldErrors;
  @override
  Map<String, List<String>>? get fieldErrors {
    final value = _fieldErrors;
    if (value == null) return null;
    if (_fieldErrors is EqualUnmodifiableMapView) return _fieldErrors;
    // ignore: implicit_dynamic_type
    return EqualUnmodifiableMapView(value);
  }

  @override
  String toString() {
    return 'AppError.validation(message: $message, fieldErrors: $fieldErrors)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$ValidationErrorImpl &&
            (identical(other.message, message) || other.message == message) &&
            const DeepCollectionEquality()
                .equals(other._fieldErrors, _fieldErrors));
  }

  @override
  int get hashCode => Object.hash(
      runtimeType, message, const DeepCollectionEquality().hash(_fieldErrors));

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$ValidationErrorImplCopyWith<_$ValidationErrorImpl> get copyWith =>
      __$$ValidationErrorImplCopyWithImpl<_$ValidationErrorImpl>(
          this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String message, String? endpoint) network,
    required TResult Function(String message, String? code) api,
    required TResult Function(String? message, bool needsReauthentication)
        unauthorized,
    required TResult Function(
            String message, Map<String, List<String>>? fieldErrors)
        validation,
    required TResult Function(String message, String? type) security,
    required TResult Function(String message) quantum,
    required TResult Function(
            String message, Object? error, StackTrace? stackTrace)
        unexpected,
  }) {
    return validation(message, fieldErrors);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String message, String? endpoint)? network,
    TResult? Function(String message, String? code)? api,
    TResult? Function(String? message, bool needsReauthentication)?
        unauthorized,
    TResult? Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult? Function(String message, String? type)? security,
    TResult? Function(String message)? quantum,
    TResult? Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
  }) {
    return validation?.call(message, fieldErrors);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String message, String? endpoint)? network,
    TResult Function(String message, String? code)? api,
    TResult Function(String? message, bool needsReauthentication)? unauthorized,
    TResult Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult Function(String message, String? type)? security,
    TResult Function(String message)? quantum,
    TResult Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
    required TResult orElse(),
  }) {
    if (validation != null) {
      return validation(message, fieldErrors);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(_NetworkError value) network,
    required TResult Function(_ApiError value) api,
    required TResult Function(_UnauthorizedError value) unauthorized,
    required TResult Function(_ValidationError value) validation,
    required TResult Function(_SecurityError value) security,
    required TResult Function(_QuantumError value) quantum,
    required TResult Function(_UnexpectedError value) unexpected,
  }) {
    return validation(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(_NetworkError value)? network,
    TResult? Function(_ApiError value)? api,
    TResult? Function(_UnauthorizedError value)? unauthorized,
    TResult? Function(_ValidationError value)? validation,
    TResult? Function(_SecurityError value)? security,
    TResult? Function(_QuantumError value)? quantum,
    TResult? Function(_UnexpectedError value)? unexpected,
  }) {
    return validation?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(_NetworkError value)? network,
    TResult Function(_ApiError value)? api,
    TResult Function(_UnauthorizedError value)? unauthorized,
    TResult Function(_ValidationError value)? validation,
    TResult Function(_SecurityError value)? security,
    TResult Function(_QuantumError value)? quantum,
    TResult Function(_UnexpectedError value)? unexpected,
    required TResult orElse(),
  }) {
    if (validation != null) {
      return validation(this);
    }
    return orElse();
  }
}

abstract class _ValidationError extends AppError {
  const factory _ValidationError(
      {required final String message,
      final Map<String, List<String>>? fieldErrors}) = _$ValidationErrorImpl;
  const _ValidationError._() : super._();

  @override
  String get message;
  Map<String, List<String>>? get fieldErrors;

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$ValidationErrorImplCopyWith<_$ValidationErrorImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$SecurityErrorImplCopyWith<$Res>
    implements $AppErrorCopyWith<$Res> {
  factory _$$SecurityErrorImplCopyWith(
          _$SecurityErrorImpl value, $Res Function(_$SecurityErrorImpl) then) =
      __$$SecurityErrorImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({String message, String? type});
}

/// @nodoc
class __$$SecurityErrorImplCopyWithImpl<$Res>
    extends _$AppErrorCopyWithImpl<$Res, _$SecurityErrorImpl>
    implements _$$SecurityErrorImplCopyWith<$Res> {
  __$$SecurityErrorImplCopyWithImpl(
      _$SecurityErrorImpl _value, $Res Function(_$SecurityErrorImpl) _then)
      : super(_value, _then);

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? message = null,
    Object? type = freezed,
  }) {
    return _then(_$SecurityErrorImpl(
      message: null == message
          ? _value.message
          : message // ignore: cast_nullable_to_non_nullable
              as String,
      type: freezed == type
          ? _value.type
          : type // ignore: cast_nullable_to_non_nullable
              as String?,
    ));
  }
}

/// @nodoc

class _$SecurityErrorImpl extends _SecurityError {
  const _$SecurityErrorImpl({required this.message, this.type}) : super._();

  @override
  final String message;
  @override
  final String? type;

  @override
  String toString() {
    return 'AppError.security(message: $message, type: $type)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$SecurityErrorImpl &&
            (identical(other.message, message) || other.message == message) &&
            (identical(other.type, type) || other.type == type));
  }

  @override
  int get hashCode => Object.hash(runtimeType, message, type);

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$SecurityErrorImplCopyWith<_$SecurityErrorImpl> get copyWith =>
      __$$SecurityErrorImplCopyWithImpl<_$SecurityErrorImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String message, String? endpoint) network,
    required TResult Function(String message, String? code) api,
    required TResult Function(String? message, bool needsReauthentication)
        unauthorized,
    required TResult Function(
            String message, Map<String, List<String>>? fieldErrors)
        validation,
    required TResult Function(String message, String? type) security,
    required TResult Function(String message) quantum,
    required TResult Function(
            String message, Object? error, StackTrace? stackTrace)
        unexpected,
  }) {
    return security(message, type);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String message, String? endpoint)? network,
    TResult? Function(String message, String? code)? api,
    TResult? Function(String? message, bool needsReauthentication)?
        unauthorized,
    TResult? Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult? Function(String message, String? type)? security,
    TResult? Function(String message)? quantum,
    TResult? Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
  }) {
    return security?.call(message, type);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String message, String? endpoint)? network,
    TResult Function(String message, String? code)? api,
    TResult Function(String? message, bool needsReauthentication)? unauthorized,
    TResult Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult Function(String message, String? type)? security,
    TResult Function(String message)? quantum,
    TResult Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
    required TResult orElse(),
  }) {
    if (security != null) {
      return security(message, type);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(_NetworkError value) network,
    required TResult Function(_ApiError value) api,
    required TResult Function(_UnauthorizedError value) unauthorized,
    required TResult Function(_ValidationError value) validation,
    required TResult Function(_SecurityError value) security,
    required TResult Function(_QuantumError value) quantum,
    required TResult Function(_UnexpectedError value) unexpected,
  }) {
    return security(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(_NetworkError value)? network,
    TResult? Function(_ApiError value)? api,
    TResult? Function(_UnauthorizedError value)? unauthorized,
    TResult? Function(_ValidationError value)? validation,
    TResult? Function(_SecurityError value)? security,
    TResult? Function(_QuantumError value)? quantum,
    TResult? Function(_UnexpectedError value)? unexpected,
  }) {
    return security?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(_NetworkError value)? network,
    TResult Function(_ApiError value)? api,
    TResult Function(_UnauthorizedError value)? unauthorized,
    TResult Function(_ValidationError value)? validation,
    TResult Function(_SecurityError value)? security,
    TResult Function(_QuantumError value)? quantum,
    TResult Function(_UnexpectedError value)? unexpected,
    required TResult orElse(),
  }) {
    if (security != null) {
      return security(this);
    }
    return orElse();
  }
}

abstract class _SecurityError extends AppError {
  const factory _SecurityError(
      {required final String message,
      final String? type}) = _$SecurityErrorImpl;
  const _SecurityError._() : super._();

  @override
  String get message;
  String? get type;

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$SecurityErrorImplCopyWith<_$SecurityErrorImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$QuantumErrorImplCopyWith<$Res>
    implements $AppErrorCopyWith<$Res> {
  factory _$$QuantumErrorImplCopyWith(
          _$QuantumErrorImpl value, $Res Function(_$QuantumErrorImpl) then) =
      __$$QuantumErrorImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({String message});
}

/// @nodoc
class __$$QuantumErrorImplCopyWithImpl<$Res>
    extends _$AppErrorCopyWithImpl<$Res, _$QuantumErrorImpl>
    implements _$$QuantumErrorImplCopyWith<$Res> {
  __$$QuantumErrorImplCopyWithImpl(
      _$QuantumErrorImpl _value, $Res Function(_$QuantumErrorImpl) _then)
      : super(_value, _then);

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? message = null,
  }) {
    return _then(_$QuantumErrorImpl(
      message: null == message
          ? _value.message
          : message // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class _$QuantumErrorImpl extends _QuantumError {
  const _$QuantumErrorImpl({required this.message}) : super._();

  @override
  final String message;

  @override
  String toString() {
    return 'AppError.quantum(message: $message)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$QuantumErrorImpl &&
            (identical(other.message, message) || other.message == message));
  }

  @override
  int get hashCode => Object.hash(runtimeType, message);

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$QuantumErrorImplCopyWith<_$QuantumErrorImpl> get copyWith =>
      __$$QuantumErrorImplCopyWithImpl<_$QuantumErrorImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String message, String? endpoint) network,
    required TResult Function(String message, String? code) api,
    required TResult Function(String? message, bool needsReauthentication)
        unauthorized,
    required TResult Function(
            String message, Map<String, List<String>>? fieldErrors)
        validation,
    required TResult Function(String message, String? type) security,
    required TResult Function(String message) quantum,
    required TResult Function(
            String message, Object? error, StackTrace? stackTrace)
        unexpected,
  }) {
    return quantum(message);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String message, String? endpoint)? network,
    TResult? Function(String message, String? code)? api,
    TResult? Function(String? message, bool needsReauthentication)?
        unauthorized,
    TResult? Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult? Function(String message, String? type)? security,
    TResult? Function(String message)? quantum,
    TResult? Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
  }) {
    return quantum?.call(message);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String message, String? endpoint)? network,
    TResult Function(String message, String? code)? api,
    TResult Function(String? message, bool needsReauthentication)? unauthorized,
    TResult Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult Function(String message, String? type)? security,
    TResult Function(String message)? quantum,
    TResult Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
    required TResult orElse(),
  }) {
    if (quantum != null) {
      return quantum(message);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(_NetworkError value) network,
    required TResult Function(_ApiError value) api,
    required TResult Function(_UnauthorizedError value) unauthorized,
    required TResult Function(_ValidationError value) validation,
    required TResult Function(_SecurityError value) security,
    required TResult Function(_QuantumError value) quantum,
    required TResult Function(_UnexpectedError value) unexpected,
  }) {
    return quantum(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(_NetworkError value)? network,
    TResult? Function(_ApiError value)? api,
    TResult? Function(_UnauthorizedError value)? unauthorized,
    TResult? Function(_ValidationError value)? validation,
    TResult? Function(_SecurityError value)? security,
    TResult? Function(_QuantumError value)? quantum,
    TResult? Function(_UnexpectedError value)? unexpected,
  }) {
    return quantum?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(_NetworkError value)? network,
    TResult Function(_ApiError value)? api,
    TResult Function(_UnauthorizedError value)? unauthorized,
    TResult Function(_ValidationError value)? validation,
    TResult Function(_SecurityError value)? security,
    TResult Function(_QuantumError value)? quantum,
    TResult Function(_UnexpectedError value)? unexpected,
    required TResult orElse(),
  }) {
    if (quantum != null) {
      return quantum(this);
    }
    return orElse();
  }
}

abstract class _QuantumError extends AppError {
  const factory _QuantumError({required final String message}) =
      _$QuantumErrorImpl;
  const _QuantumError._() : super._();

  @override
  String get message;

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$QuantumErrorImplCopyWith<_$QuantumErrorImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$UnexpectedErrorImplCopyWith<$Res>
    implements $AppErrorCopyWith<$Res> {
  factory _$$UnexpectedErrorImplCopyWith(_$UnexpectedErrorImpl value,
          $Res Function(_$UnexpectedErrorImpl) then) =
      __$$UnexpectedErrorImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({String message, Object? error, StackTrace? stackTrace});
}

/// @nodoc
class __$$UnexpectedErrorImplCopyWithImpl<$Res>
    extends _$AppErrorCopyWithImpl<$Res, _$UnexpectedErrorImpl>
    implements _$$UnexpectedErrorImplCopyWith<$Res> {
  __$$UnexpectedErrorImplCopyWithImpl(
      _$UnexpectedErrorImpl _value, $Res Function(_$UnexpectedErrorImpl) _then)
      : super(_value, _then);

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? message = null,
    Object? error = freezed,
    Object? stackTrace = freezed,
  }) {
    return _then(_$UnexpectedErrorImpl(
      message: null == message
          ? _value.message
          : message // ignore: cast_nullable_to_non_nullable
              as String,
      error: freezed == error ? _value.error : error,
      stackTrace: freezed == stackTrace
          ? _value.stackTrace
          : stackTrace // ignore: cast_nullable_to_non_nullable
              as StackTrace?,
    ));
  }
}

/// @nodoc

class _$UnexpectedErrorImpl extends _UnexpectedError {
  const _$UnexpectedErrorImpl(
      {required this.message, this.error, this.stackTrace})
      : super._();

  @override
  final String message;
  @override
  final Object? error;
  @override
  final StackTrace? stackTrace;

  @override
  String toString() {
    return 'AppError.unexpected(message: $message, error: $error, stackTrace: $stackTrace)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$UnexpectedErrorImpl &&
            (identical(other.message, message) || other.message == message) &&
            const DeepCollectionEquality().equals(other.error, error) &&
            (identical(other.stackTrace, stackTrace) ||
                other.stackTrace == stackTrace));
  }

  @override
  int get hashCode => Object.hash(runtimeType, message,
      const DeepCollectionEquality().hash(error), stackTrace);

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$UnexpectedErrorImplCopyWith<_$UnexpectedErrorImpl> get copyWith =>
      __$$UnexpectedErrorImplCopyWithImpl<_$UnexpectedErrorImpl>(
          this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String message, String? endpoint) network,
    required TResult Function(String message, String? code) api,
    required TResult Function(String? message, bool needsReauthentication)
        unauthorized,
    required TResult Function(
            String message, Map<String, List<String>>? fieldErrors)
        validation,
    required TResult Function(String message, String? type) security,
    required TResult Function(String message) quantum,
    required TResult Function(
            String message, Object? error, StackTrace? stackTrace)
        unexpected,
  }) {
    return unexpected(message, error, stackTrace);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String message, String? endpoint)? network,
    TResult? Function(String message, String? code)? api,
    TResult? Function(String? message, bool needsReauthentication)?
        unauthorized,
    TResult? Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult? Function(String message, String? type)? security,
    TResult? Function(String message)? quantum,
    TResult? Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
  }) {
    return unexpected?.call(message, error, stackTrace);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String message, String? endpoint)? network,
    TResult Function(String message, String? code)? api,
    TResult Function(String? message, bool needsReauthentication)? unauthorized,
    TResult Function(String message, Map<String, List<String>>? fieldErrors)?
        validation,
    TResult Function(String message, String? type)? security,
    TResult Function(String message)? quantum,
    TResult Function(String message, Object? error, StackTrace? stackTrace)?
        unexpected,
    required TResult orElse(),
  }) {
    if (unexpected != null) {
      return unexpected(message, error, stackTrace);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(_NetworkError value) network,
    required TResult Function(_ApiError value) api,
    required TResult Function(_UnauthorizedError value) unauthorized,
    required TResult Function(_ValidationError value) validation,
    required TResult Function(_SecurityError value) security,
    required TResult Function(_QuantumError value) quantum,
    required TResult Function(_UnexpectedError value) unexpected,
  }) {
    return unexpected(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(_NetworkError value)? network,
    TResult? Function(_ApiError value)? api,
    TResult? Function(_UnauthorizedError value)? unauthorized,
    TResult? Function(_ValidationError value)? validation,
    TResult? Function(_SecurityError value)? security,
    TResult? Function(_QuantumError value)? quantum,
    TResult? Function(_UnexpectedError value)? unexpected,
  }) {
    return unexpected?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(_NetworkError value)? network,
    TResult Function(_ApiError value)? api,
    TResult Function(_UnauthorizedError value)? unauthorized,
    TResult Function(_ValidationError value)? validation,
    TResult Function(_SecurityError value)? security,
    TResult Function(_QuantumError value)? quantum,
    TResult Function(_UnexpectedError value)? unexpected,
    required TResult orElse(),
  }) {
    if (unexpected != null) {
      return unexpected(this);
    }
    return orElse();
  }
}

abstract class _UnexpectedError extends AppError {
  const factory _UnexpectedError(
      {required final String message,
      final Object? error,
      final StackTrace? stackTrace}) = _$UnexpectedErrorImpl;
  const _UnexpectedError._() : super._();

  @override
  String get message;
  Object? get error;
  StackTrace? get stackTrace;

  /// Create a copy of AppError
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$UnexpectedErrorImplCopyWith<_$UnexpectedErrorImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
