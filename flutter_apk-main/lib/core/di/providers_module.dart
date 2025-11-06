import 'package:injectable/injectable.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../providers/auth_controller.dart';
import '../../providers/payment_controller.dart';
import '../../providers/scanner_controller.dart';
import '../../providers/config_controller.dart';
import '../../core/services/auth_service.dart';
import '../../core/services/payment_service.dart';
import '../../core/storage/storage.dart';
import '../../core/storage/secure_storage_service.dart';
import '../../core/network/api_service.dart';
import '../logging/app_logger.dart';

@module
abstract class ProvidersModule {
  @singleton
  ProviderContainer getProviderContainer(AppLogger logger) {
    return ProviderContainer(
      observers: [
        ProviderLogger(logger),
      ],
    );
  }

  @singleton
  AuthController getAuthController(
      AuthService authService, ApiService apiService, Storage storage) =>
    AuthController(authService, apiService, storage);

  @singleton
  PaymentController getPaymentController(PaymentService paymentService) =>
    PaymentController(paymentService);

  @singleton
  ScannerController getScannerController(ApiService apiService) =>
    ScannerController(apiService);

  @singleton
  Storage getStorage(SecureStorageService secureStorageService) =>
    Storage(secureStorageService);

  @singleton
  ConfigController getConfigController(SecureStorageService storageService) =>
    ConfigController(storageService);
}

class ProviderLogger extends ProviderObserver {
  final AppLogger _logger;

  ProviderLogger(this._logger);

  @override
  void didUpdateProvider(
    ProviderBase provider,
    Object? previousValue,
    Object? newValue,
    ProviderContainer container,
  ) {
    _logger.debug(
      'Provider ${provider.name ?? provider.runtimeType} updated: $newValue',
    );
  }

  @override
  void didAddProvider(
    ProviderBase provider,
    Object? value,
    ProviderContainer container,
  ) {
    _logger.debug(
      'Provider ${provider.name ?? provider.runtimeType} added: $value',
    );
  }

  @override
  void didDisposeProvider(
    ProviderBase provider,
    ProviderContainer container,
  ) {
    _logger.debug(
      'Provider ${provider.name ?? provider.runtimeType} disposed',
    );
  }
}
