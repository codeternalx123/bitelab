class NotificationSettings {
  final bool email;
  final bool push;
  final bool sms;

  const NotificationSettings({
    this.email = true,
    this.push = true,
    this.sms = false,
  });

  factory NotificationSettings.fromJson(Map<String, dynamic> json) => NotificationSettings(
    email: json['email'] as bool? ?? true,
    push: json['push'] as bool? ?? true,
    sms: json['sms'] as bool? ?? false,
  );

  Map<String, dynamic> toJson() => {
    'email': email,
    'push': push,
    'sms': sms,
  };
}

class PrivacySettings {
  final bool isProfilePublic;
  final bool showActivity;
  final bool showLocation;
  final bool showEmail;
  final bool showPhone;

  const PrivacySettings({
    this.isProfilePublic = false,
    this.showActivity = true,
    this.showLocation = true,
    this.showEmail = false,
    this.showPhone = false,
  });

  factory PrivacySettings.fromJson(Map<String, dynamic> json) => PrivacySettings(
    isProfilePublic: json['isProfilePublic'] as bool? ?? false,
    showActivity: json['showActivity'] as bool? ?? true,
    showLocation: json['showLocation'] as bool? ?? true,
    showEmail: json['showEmail'] as bool? ?? false,
    showPhone: json['showPhone'] as bool? ?? false,
  );

  Map<String, dynamic> toJson() => {
    'isProfilePublic': isProfilePublic,
    'showActivity': showActivity,
    'showLocation': showLocation,
    'showEmail': showEmail,
    'showPhone': showPhone,
  };
}

class UserSettings {
  final String language;
  final String timezone;
  final NotificationSettings notifications;
  final PrivacySettings privacy;
  final bool twoFactorEnabled;
  final Map<String, dynamic>? preferences;

  const UserSettings({
    required this.language,
    required this.timezone,
    required this.notifications,
    required this.privacy,
    this.twoFactorEnabled = false,
    this.preferences,
  });

  factory UserSettings.fromJson(Map<String, dynamic> json) => UserSettings(
    language: json['language'] as String,
    timezone: json['timezone'] as String,
    notifications: NotificationSettings.fromJson(json['notifications'] as Map<String, dynamic>),
    privacy: PrivacySettings.fromJson(json['privacy'] as Map<String, dynamic>),
    twoFactorEnabled: json['twoFactorEnabled'] as bool? ?? false,
    preferences: json['preferences'] as Map<String, dynamic>?,
  );

  Map<String, dynamic> toJson() => {
    'language': language,
    'timezone': timezone,
    'notifications': notifications.toJson(),
    'privacy': privacy.toJson(),
    'twoFactorEnabled': twoFactorEnabled,
    if (preferences != null) 'preferences': preferences,
  };
}

class Subscription {
  final String planId;
  final String status;
  final bool isActive;
  final bool isPremium;

  const Subscription({
    required this.planId,
    required this.status,
    this.isActive = false,
    this.isPremium = false,
  });

  factory Subscription.fromJson(Map<String, dynamic> json) => Subscription(
    planId: json['planId'] as String,
    status: json['status'] as String,
    isActive: json['isActive'] as bool? ?? false,
    isPremium: json['isPremium'] as bool? ?? false,
  );

  Map<String, dynamic> toJson() => {
    'planId': planId,
    'status': status,
    'isActive': isActive,
    'isPremium': isPremium,
  };
}

class User {
  final String id;
  final String email;
  final String name;
  final bool isEmailVerified;
  final List<String> roles;
  final DateTime createdAt;
  final DateTime lastLoginAt;
  final UserSettings settings;
  final Subscription? subscription;
  final String? profilePicture;
  final Map<String, dynamic>? metadata;

  const User({
    required this.id,
    required this.email,
    required this.name,
    this.isEmailVerified = false,
    required this.roles,
    required this.createdAt,
    required this.lastLoginAt,
    required this.settings,
    this.subscription,
    this.profilePicture,
    this.metadata,
  });

  factory User.fromJson(Map<String, dynamic> json) => User(
    id: json['id'] as String,
    email: json['email'] as String,
    name: json['name'] as String,
    isEmailVerified: json['isEmailVerified'] as bool? ?? false,
    roles: (json['roles'] as List<dynamic>).map((e) => e as String).toList(),
    createdAt: DateTime.parse(json['createdAt'] as String),
    lastLoginAt: DateTime.parse(json['lastLoginAt'] as String),
    settings: UserSettings.fromJson(json['settings'] as Map<String, dynamic>),
    subscription: json['subscription'] != null
        ? Subscription.fromJson(json['subscription'] as Map<String, dynamic>)
        : null,
    profilePicture: json['profilePicture'] as String?,
    metadata: json['metadata'] as Map<String, dynamic>?,
  );

  Map<String, dynamic> toJson() => {
    'id': id,
    'email': email,
    'name': name,
    'isEmailVerified': isEmailVerified,
    'roles': roles,
    'createdAt': createdAt.toIso8601String(),
    'lastLoginAt': lastLoginAt.toIso8601String(),
    'settings': settings.toJson(),
    if (subscription != null) 'subscription': subscription!.toJson(),
    if (profilePicture != null) 'profilePicture': profilePicture,
    if (metadata != null) 'metadata': metadata,
  };
}