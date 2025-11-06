import 'dart:developer' as developer;
// import 'package:injectable/injectable.dart';
import 'environment_config.dart';

enum LogLevel {
  verbose,
  debug,
  info,
  warning,
  error,
  none;

  bool operator >=(LogLevel other) => index >= other.index;
}

// @singleton
class LoggerConfig {
  final EnvironmentConfig _envConfig;
  late final LogLevel _minLevel;

  LoggerConfig(this._envConfig) {
    _minLevel = _parseLogLevel(_envConfig.logLevel);
  }

  LogLevel _parseLogLevel(String level) {
    switch (level.toLowerCase()) {
      case 'verbose':
        return LogLevel.verbose;
      case 'debug':
        return LogLevel.debug;
      case 'info':
        return LogLevel.info;
      case 'warning':
        return LogLevel.warning;
      case 'error':
        return LogLevel.error;
      default:
        return LogLevel.none;
    }
  }

  bool get isLoggingEnabled => _envConfig.enableLogging;

  bool shouldLog(LogLevel level) {
    if (!isLoggingEnabled) return false;
    return level >= _minLevel;
  }

  void log({
    required String message,
    required LogLevel level,
    String? tag,
    Object? error,
    StackTrace? stackTrace,
  }) {
    if (!shouldLog(level)) return;

    final timestamp = DateTime.now().toIso8601String();
    final tagString = tag != null ? '[$tag] ' : '';
    final logMessage = '$timestamp $tagString$message';

    switch (level) {
      case LogLevel.verbose:
        _logVerbose(logMessage, error, stackTrace);
        break;
      case LogLevel.debug:
        _logDebug(logMessage, error, stackTrace);
        break;
      case LogLevel.info:
        _logInfo(logMessage, error, stackTrace);
        break;
      case LogLevel.warning:
        _logWarning(logMessage, error, stackTrace);
        break;
      case LogLevel.error:
        _logError(logMessage, error, stackTrace);
        break;
      case LogLevel.none:
        break;
    }

    // If running in production, also send logs to remote logging service
    if (_envConfig.isProduction) {
      _sendToRemoteLogging(level, message, error, stackTrace);
    }
  }

  void _logVerbose(String message, Object? error, StackTrace? stackTrace) {
    developer.log(
      message,
      time: DateTime.now(),
      level: 0,
      error: error,
      stackTrace: stackTrace,
    );
  }

  void _logDebug(String message, Object? error, StackTrace? stackTrace) {
    developer.log(
      message,
      time: DateTime.now(),
      level: 500,
      error: error,
      stackTrace: stackTrace,
    );
  }

  void _logInfo(String message, Object? error, StackTrace? stackTrace) {
    developer.log(
      message,
      time: DateTime.now(),
      level: 800,
      error: error,
      stackTrace: stackTrace,
    );
  }

  void _logWarning(String message, Object? error, StackTrace? stackTrace) {
    developer.log(
      message,
      time: DateTime.now(),
      level: 900,
      error: error,
      stackTrace: stackTrace,
    );
  }

  void _logError(String message, Object? error, StackTrace? stackTrace) {
    developer.log(
      message,
      time: DateTime.now(),
      level: 1000,
      error: error,
      stackTrace: stackTrace,
    );
  }

  Future<void> _sendToRemoteLogging(
    LogLevel level,
    String message,
    Object? error,
    StackTrace? stackTrace,
  ) async {
    // TODO: Implement remote logging service integration
    // This could be Firebase Crashlytics, Sentry, or a custom logging service
    
    // Example structure for remote logging payload:
    final payload = {
      'timestamp': DateTime.now().toIso8601String(),
      'level': level.toString(),
      'message': message,
      'error': error?.toString(),
      'stackTrace': stackTrace?.toString(),
      'deviceInfo': await _getDeviceInfo(),
      'appInfo': await _getAppInfo(),
      'userInfo': await _getUserInfo(),
    };

    // await _remoteLoggingService.send(payload);
  }

  Future<Map<String, dynamic>> _getDeviceInfo() async {
    // TODO: Implement device info collection
    return {
      'platform': 'flutter',
      // Add more device info
    };
  }

  Future<Map<String, dynamic>> _getAppInfo() async {
    // TODO: Implement app info collection
    return {
      'version': '1.0.0',
      'buildNumber': '1',
      'flavor': _envConfig.flavor,
      // Add more app info
    };
  }

  Future<Map<String, dynamic>> _getUserInfo() async {
    // TODO: Implement user info collection
    return {
      'isAuthenticated': false,
      'sessionId': null,
      // Add more user info
    };
  }

  // Convenience methods for logging
  void v(String message, {String? tag, Object? error, StackTrace? stackTrace}) {
    log(
      message: message,
      level: LogLevel.verbose,
      tag: tag,
      error: error,
      stackTrace: stackTrace,
    );
  }

  void d(String message, {String? tag, Object? error, StackTrace? stackTrace}) {
    log(
      message: message,
      level: LogLevel.debug,
      tag: tag,
      error: error,
      stackTrace: stackTrace,
    );
  }

  void i(String message, {String? tag, Object? error, StackTrace? stackTrace}) {
    log(
      message: message,
      level: LogLevel.info,
      tag: tag,
      error: error,
      stackTrace: stackTrace,
    );
  }

  void w(String message, {String? tag, Object? error, StackTrace? stackTrace}) {
    log(
      message: message,
      level: LogLevel.warning,
      tag: tag,
      error: error,
      stackTrace: stackTrace,
    );
  }

  void e(String message, {String? tag, Object? error, StackTrace? stackTrace}) {
    log(
      message: message,
      level: LogLevel.error,
      tag: tag,
      error: error,
      stackTrace: stackTrace,
    );
  }
}