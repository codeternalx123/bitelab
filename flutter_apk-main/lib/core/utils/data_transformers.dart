import 'dart:convert';
import 'package:crypto/crypto.dart';
import 'package:intl/intl.dart';
import '../../models/payment_models.dart';
import '../../models/user_models.dart';

class DataTransformers {
  static String hashSensitiveData(Map<String, dynamic> data) {
    final jsonStr = json.encode(data);
    final bytes = utf8.encode(jsonStr);
    return sha256.convert(bytes).toString();
  }

  static Map<String, dynamic> sanitizeUserData(User user) {
    return {
      'id': user.id,
      'name': user.name,
      'email': user.email,
      'isEmailVerified': user.isEmailVerified,
      'roles': user.roles,
      'createdAt': user.createdAt.toIso8601String(),
      'lastLoginAt': user.lastLoginAt.toIso8601String(),
      'settings': {
        'language': user.settings.language,
        'timezone': user.settings.timezone,
        'notifications': {
          'email': user.settings.notifications.email,
          'push': user.settings.notifications.push,
          'sms': user.settings.notifications.sms,
        },
        'privacy': {
          'isProfilePublic': user.settings.privacy.isProfilePublic,
          'showActivity': user.settings.privacy.showActivity,
          'showLocation': user.settings.privacy.showLocation,
        },
      },
      if (user.subscription != null) 'subscription': {
        'planId': user.subscription!.planId,
        'status': user.subscription!.status,
        'isActive': user.subscription!.isActive,
        'isPremium': user.subscription!.isPremium,
      },
      if (user.profilePicture != null) 'profilePicture': user.profilePicture,
      if (user.metadata != null) 'metadata': user.metadata,
    };
  }

  static Map<String, dynamic> preparePaymentData(PaymentRequest request) {
    return {
      'amount': request.amount,
      'currency': request.currency.toUpperCase(),
      'provider': request.provider,
      'paymentMethodId': request.paymentMethodId,
      if (request.subscriptionId != null) 'subscriptionId': request.subscriptionId,
      if (request.metadata != null)
        'metadata': _preparePaymentMetadata(request.metadata!),
      'timestamp': DateTime.now().toIso8601String(),
    };
  }

  static Map<String, dynamic> _preparePaymentMetadata(PaymentMetadata metadata) {
    return {
      'device': {
        'os': metadata.deviceInfo.os,
        'model': metadata.deviceInfo.model,
        'browser': metadata.deviceInfo.browser,
        'screenResolution': metadata.deviceInfo.screenResolution,
        'timezone': metadata.deviceInfo.timezone,
        if (metadata.deviceInfo.deviceId != null)
          'deviceId': metadata.deviceInfo.deviceId,
      },
      'location': {
        'latitude': metadata.location.latitude,
        'longitude': metadata.location.longitude,
        if (metadata.location.country != null)
          'country': metadata.location.country,
        if (metadata.location.city != null) 'city': metadata.location.city,
        if (metadata.location.ip != null) 'ip': metadata.location.ip,
      },
      'network': {
        'ipAddress': metadata.networkInfo.ipAddress,
        'isVpn': metadata.networkInfo.isVpn,
        'isProxy': metadata.networkInfo.isProxy,
        if (metadata.networkInfo.isp != null) 'isp': metadata.networkInfo.isp,
        if (metadata.networkInfo.connectionType != null)
          'connectionType': metadata.networkInfo.connectionType,
      },
      if (metadata.customData != null) 'custom': metadata.customData,
    };
  }

  static Map<String, dynamic> prepareAnalyticsData({
    required User user,
    required PaymentRequest payment,
    Map<String, dynamic>? additionalData,
  }) {
    final now = DateTime.now();
    return {
      'userId': user.id,
      'userMetrics': {
        'accountAge': now.difference(user.createdAt).inDays,
        'isPremium': user.subscription?.isPremium ?? false,
        'lastLoginDays': now.difference(user.lastLoginAt).inDays,
      },
      'paymentMetrics': {
        'amount': payment.amount,
        'currency': payment.currency,
        'requiresRiskAnalysis': payment.requiresRiskAnalysis,
      },
      if (payment.metadata != null) 'deviceMetrics': payment.metadata!.deviceInfo.toJson(),
      if (payment.metadata != null) 'locationMetrics': payment.metadata!.location.toJson(),
      if (payment.metadata != null) 'networkMetrics': payment.metadata!.networkInfo.toJson(),
      if (additionalData != null) ...additionalData,
      'timestamp': now.toIso8601String(),
    };
  }

  static String normalizePhoneNumber(String phone) {
    // Remove all non-digit characters
    final digitsOnly = phone.replaceAll(RegExp(r'\D'), '');

    // Add country code if missing
    if (digitsOnly.length == 10) {
      return '+1$digitsOnly'; // Assuming US/Canada
    }
    return '+$digitsOnly';
  }

  static String maskCardNumber(String cardNumber) {
    // Remove any spaces or dashes
    final cleanNumber = cardNumber.replaceAll(RegExp(r'[\s-]'), '');
    final lastFour = cleanNumber.substring(cleanNumber.length - 4);
    return 'xxxx-xxxx-xxxx-$lastFour';
  }

  static String formatCurrency(double amount, String currency) {
    final formatter = NumberFormat.currency(
      symbol: _getCurrencySymbol(currency),
      decimalDigits: 2,
    );
    return formatter.format(amount);
  }

  static String _getCurrencySymbol(String currency) {
    switch (currency.toUpperCase()) {
      case 'USD':
        return '\$';
      case 'EUR':
        return '€';
      case 'GBP':
        return '£';
      case 'JPY':
        return '¥';
      default:
        return currency;
    }
  }
}

// Extension methods for common transformations
extension StringTransformations on String {
  String get masked => '*' * length;
  
  String get titleCase {
    if (isEmpty) return this;
    return split(' ')
        .map((word) => word.isNotEmpty
            ? word[0].toUpperCase() + word.substring(1).toLowerCase()
            : '')
        .join(' ');
  }

  String get snakeCase {
    return replaceAll(RegExp(r'[^a-zA-Z0-9]'), '_')
        .toLowerCase();
  }

  String maskExcept(int visibleChars, {bool fromEnd = true}) {
    if (length <= visibleChars) return this;
    final visible = fromEnd
        ? substring(length - visibleChars)
        : substring(0, visibleChars);
    final masked = '*' * (length - visibleChars);
    return fromEnd ? masked + visible : visible + masked;
  }
}

extension MapTransformations on Map<String, dynamic> {
  Map<String, dynamic> get sanitized => Map.from(this)
    ..removeWhere((_, value) => value == null)
    ..map((key, value) => MapEntry(key.snakeCase, value));

  Map<String, dynamic> removeNullValues() {
    return Map.from(this)..removeWhere((_, value) => value == null);
  }

  Map<String, dynamic> transformKeys(String Function(String) transform) {
    return map((key, value) => MapEntry(transform(key), value));
  }
}

extension DateTimeTransformations on DateTime {
  String get humanReadable {
    final now = DateTime.now();
    final difference = now.difference(this);

    if (difference.inSeconds < 60) {
      return 'just now';
    } else if (difference.inMinutes < 60) {
      return '${difference.inMinutes}m ago';
    } else if (difference.inHours < 24) {
      return '${difference.inHours}h ago';
    } else if (difference.inDays < 7) {
      return '${difference.inDays}d ago';
    } else if (difference.inDays < 30) {
      return '${(difference.inDays / 7).floor()}w ago';
    } else if (difference.inDays < 365) {
      return '${(difference.inDays / 30).floor()}mo ago';
    } else {
      return '${(difference.inDays / 365).floor()}y ago';
    }
  }

  bool get isToday {
    final now = DateTime.now();
    return year == now.year &&
        month == now.month &&
        day == now.day;
  }

  bool get isYesterday {
    final yesterday = DateTime.now().subtract(const Duration(days: 1));
    return year == yesterday.year &&
        month == yesterday.month &&
        day == yesterday.day;
  }

  bool get isFuture => isAfter(DateTime.now());
  bool get isPast => isBefore(DateTime.now());
}