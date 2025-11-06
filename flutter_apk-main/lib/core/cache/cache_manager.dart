import 'dart:convert';
// import 'package:hive/hive.dart';
// import 'package:injectable/injectable.dart';
import '../logging/app_logger.dart';

// @singleton
class CacheManager {
  final dynamic _box;
  final AppLogger _logger;
  final Duration _defaultExpiration;

  CacheManager(
    this._box,
    this._logger, [
    this._defaultExpiration = const Duration(hours: 24),
  ]);

  Future<void> put<T>(
    String key,
    T value, {
    Duration? expiration,
  }) async {
    try {
      final cacheEntry = CacheEntry(
        data: value,
        timestamp: DateTime.now(),
        expiration: expiration ?? _defaultExpiration,
      );

      await _box.put(key, jsonEncode(cacheEntry.toJson()));
      _logger.debug('Cached data for key: $key');
    } catch (e, stack) {
      _logger.error('Failed to cache data for key: $key', e, stack);
      rethrow;
    }
  }

  Future<T?> get<T>(String key) async {
    try {
      final value = _box.get(key);
      if (value == null) return null;

      final entry = CacheEntry<T>.fromJson(jsonDecode(value));
      if (entry.isExpired) {
        await _box.delete(key);
        return null;
      }

      return entry.data;
    } catch (e, stack) {
      _logger.error('Failed to retrieve cached data for key: $key', e, stack);
      await _box.delete(key);
      return null;
    }
  }

  Future<void> remove(String key) async {
    try {
      await _box.delete(key);
      _logger.debug('Removed cached data for key: $key');
    } catch (e, stack) {
      _logger.error('Failed to remove cached data for key: $key', e, stack);
      rethrow;
    }
  }

  Future<void> set(String key, dynamic value) async {
    await put(key, value);
  }

  Future<void> clear() async {
    try {
      await _box.clear();
      _logger.info('Cleared all cached data');
    } catch (e, stack) {
      _logger.error('Failed to clear cached data', e, stack);
      rethrow;
    }
  }

  bool hasExpired(String key) {
    try {
      final value = _box.get(key);
      if (value == null) return true;

      final entry = CacheEntry.fromJson(jsonDecode(value));
      return entry.isExpired;
    } catch (e) {
      return true;
    }
  }
}

class CacheEntry<T> {
  final T data;
  final DateTime timestamp;
  final Duration expiration;

  CacheEntry({
    required this.data,
    required this.timestamp,
    required this.expiration,
  });

  bool get isExpired =>
      DateTime.now().difference(timestamp) > expiration;

  Map<String, dynamic> toJson() {
    return {
      'data': data,
      'timestamp': timestamp.toIso8601String(),
      'expiration': expiration.inSeconds,
    };
  }

  factory CacheEntry.fromJson(Map<String, dynamic> json) {
    return CacheEntry(
      data: json['data'] as T,
      timestamp: DateTime.parse(json['timestamp'] as String),
      expiration: Duration(seconds: json['expiration'] as int),
    );
  }
}