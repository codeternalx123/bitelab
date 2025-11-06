// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'user_models.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$UserSettingsImpl _$$UserSettingsImplFromJson(Map<String, dynamic> json) =>
    _$UserSettingsImpl(
      language: json['language'] as String,
      timezone: json['timezone'] as String,
      notifications: NotificationSettings.fromJson(
          json['notifications'] as Map<String, dynamic>),
      privacy:
          PrivacySettings.fromJson(json['privacy'] as Map<String, dynamic>),
      twoFactorEnabled: json['twoFactorEnabled'] as bool? ?? false,
      preferences: json['preferences'] as Map<String, dynamic>?,
    );

Map<String, dynamic> _$$UserSettingsImplToJson(_$UserSettingsImpl instance) =>
    <String, dynamic>{
      'language': instance.language,
      'timezone': instance.timezone,
      'notifications': instance.notifications,
      'privacy': instance.privacy,
      'twoFactorEnabled': instance.twoFactorEnabled,
      'preferences': instance.preferences,
    };

_$NotificationSettingsImpl _$$NotificationSettingsImplFromJson(
        Map<String, dynamic> json) =>
    _$NotificationSettingsImpl(
      email: json['email'] as bool? ?? true,
      push: json['push'] as bool? ?? true,
      sms: json['sms'] as bool? ?? false,
    );

Map<String, dynamic> _$$NotificationSettingsImplToJson(
        _$NotificationSettingsImpl instance) =>
    <String, dynamic>{
      'email': instance.email,
      'push': instance.push,
      'sms': instance.sms,
    };

_$PrivacySettingsImpl _$$PrivacySettingsImplFromJson(
        Map<String, dynamic> json) =>
    _$PrivacySettingsImpl(
      isProfilePublic: json['isProfilePublic'] as bool? ?? false,
      showActivity: json['showActivity'] as bool? ?? true,
      showLocation: json['showLocation'] as bool? ?? true,
      showEmail: json['showEmail'] as bool? ?? false,
      showPhone: json['showPhone'] as bool? ?? false,
    );

Map<String, dynamic> _$$PrivacySettingsImplToJson(
        _$PrivacySettingsImpl instance) =>
    <String, dynamic>{
      'isProfilePublic': instance.isProfilePublic,
      'showActivity': instance.showActivity,
      'showLocation': instance.showLocation,
      'showEmail': instance.showEmail,
      'showPhone': instance.showPhone,
    };

_$SubscriptionImpl _$$SubscriptionImplFromJson(Map<String, dynamic> json) =>
    _$SubscriptionImpl(
      planId: json['planId'] as String,
      status: json['status'] as String,
      isActive: json['isActive'] as bool? ?? false,
      isPremium: json['isPremium'] as bool? ?? false,
    );

Map<String, dynamic> _$$SubscriptionImplToJson(_$SubscriptionImpl instance) =>
    <String, dynamic>{
      'planId': instance.planId,
      'status': instance.status,
      'isActive': instance.isActive,
      'isPremium': instance.isPremium,
    };

_$UserImpl _$$UserImplFromJson(Map<String, dynamic> json) => _$UserImpl(
      id: json['id'] as String,
      email: json['email'] as String,
      name: json['name'] as String,
      isEmailVerified: json['isEmailVerified'] as bool? ?? false,
      roles: (json['roles'] as List<dynamic>).map((e) => e as String).toList(),
      createdAt: DateTime.parse(json['createdAt'] as String),
      lastLoginAt: DateTime.parse(json['lastLoginAt'] as String),
      settings: UserSettings.fromJson(json['settings'] as Map<String, dynamic>),
      subscription: json['subscription'] == null
          ? null
          : Subscription.fromJson(json['subscription'] as Map<String, dynamic>),
      profilePicture: json['profilePicture'] as String?,
      metadata: json['metadata'] as Map<String, dynamic>?,
    );

Map<String, dynamic> _$$UserImplToJson(_$UserImpl instance) =>
    <String, dynamic>{
      'id': instance.id,
      'email': instance.email,
      'name': instance.name,
      'isEmailVerified': instance.isEmailVerified,
      'roles': instance.roles,
      'createdAt': instance.createdAt.toIso8601String(),
      'lastLoginAt': instance.lastLoginAt.toIso8601String(),
      'settings': instance.settings,
      'subscription': instance.subscription,
      'profilePicture': instance.profilePicture,
      'metadata': instance.metadata,
    };
