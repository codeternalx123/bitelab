import 'package:injectable/injectable.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import '../logging/app_logger.dart';
import '../storage/secure_storage_service.dart';
import '../network/network_manager.dart';
import '../cache/cache_manager.dart';
import '../utils/encryption_service.dart';

@module
abstract class CoreModule {
  @singleton
  @preResolve
  Future<Box<String>> get cacheBox async {
    await Hive.initFlutter();
    return await Hive.openBox<String>('app_cache');
  }

  @Named('auth_box')
  @singleton
  @preResolve
  Future<Box<String>> get authBox async {
    return await Hive.openBox<String>('auth_cache');
  }

  @singleton
  @preResolve
  Future<FirebaseApp> get firebaseApp async {
    return await Firebase.initializeApp();
  }

  @singleton
  FirebaseAnalytics getAnalytics(FirebaseApp app) {
    return FirebaseAnalytics.instance;
  }

  @singleton
  FirebaseCrashlytics getCrashlytics(FirebaseApp app) {
    return FirebaseCrashlytics.instance;
  }

  @singleton
  SecureStorageService getSecureStorage(AppLogger logger) {
    return SecureStorageService(const FlutterSecureStorage(), logger);
  }

  @singleton
  NetworkManager getNetworkManager(AppLogger logger) {
    return NetworkManager(Connectivity(), logger);
  }

  @singleton
  CacheManager getCacheManager(
    @Named('cache_box') Box<String> box,
    AppLogger logger,
  ) {
    return CacheManager(box, logger);
  }

  @lazySingleton
  EncryptionService getEncryptionService() {
    return EncryptionService();
  }
}