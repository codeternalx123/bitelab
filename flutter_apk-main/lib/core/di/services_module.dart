import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:injectable/injectable.dart';
import '../network/api_service.dart';
import '../services/auth_service.dart';
import '../services/payment_service.dart';
import '../services/scanner_service.dart';
import '../services/analytics_service.dart';
import '../services/storage_service.dart';

@module
abstract class ServicesModule {
  @singleton
  AuthService getAuthService(
    ApiService apiService,
  ) {
    return AuthService(
      apiService: apiService,
    );
  }

  @singleton
  PaymentService getPaymentService(
    ApiService apiService,
  ) {
    return PaymentService(
      apiService: apiService,
    );
  }

  @singleton
  ScannerService getScannerService(
    ApiService apiService,
  ) {
    return ScannerService(
      apiService: apiService,
    );
  }

  @singleton
  AnalyticsService getAnalyticsService(
    FirebaseAnalytics analytics,
  ) {
    return AnalyticsService(
      analytics: analytics,
    );
  }

  @singleton
  StorageService getStorageService(
    FlutterSecureStorage secureStorage,
  ) {
    return StorageService(
      secureStorage: secureStorage,
    );
  }
}