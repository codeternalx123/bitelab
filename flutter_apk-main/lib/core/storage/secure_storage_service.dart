import 'dart:convert';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:injectable/injectable.dart';
import '../logging/app_logger.dart';

@singleton
class SecureStorageService {
  final FlutterSecureStorage _storage;
  final AppLogger _logger;

  SecureStorageService(this._storage, this._logger);

  Future<void> write({
    required String key,
    required String value,
    Map<String, String>? metadata,
  }) async {
    try {
      final encryptedValue = await _encryptValue(value);
      final storageValue = _prepareStorageValue(encryptedValue, metadata);
      await _storage.write(
        key: key,
        value: storageValue,
      );
      _logger.debug('Successfully wrote encrypted data for key: $key');
    } catch (e, stack) {
      _logger.error('Failed to write secure storage for key: $key', e, stack);
      rethrow;
    }
  }

  Future<String?> read(String key) async {
    try {
      final storageValue = await _storage.read(key: key);
      if (storageValue == null) return null;

      final parsedValue = _parseStorageValue(storageValue);
      final decryptedValue = await _decryptValue(parsedValue.value);
      
      _logger.debug('Successfully read encrypted data for key: $key');
      return decryptedValue;
    } catch (e, stack) {
      _logger.error('Failed to read secure storage for key: $key', e, stack);
      return null;
    }
  }

  Future<String?> getToken() async {
    return read('access_token');
  }

  Future<Map<String, String>?> readMetadata(String key) async {
    try {
      final storageValue = await _storage.read(key: key);
      if (storageValue == null) return null;

      final parsedValue = _parseStorageValue(storageValue);
      return parsedValue.metadata;
    } catch (e, stack) {
      _logger.error('Failed to read metadata for key: $key', e, stack);
      return null;
    }
  }

  Future<void> delete(String key) async {
    try {
      await _storage.delete(key: key);
      _logger.debug('Successfully deleted data for key: $key');
    } catch (e, stack) {
      _logger.error('Failed to delete secure storage for key: $key', e, stack);
      rethrow;
    }
  }

  Future<void> deleteAll() async {
    try {
      await _storage.deleteAll();
      _logger.info('Successfully deleted all secure storage data');
    } catch (e, stack) {
      _logger.error('Failed to delete all secure storage data', e, stack);
      rethrow;
    }
  }

  Future<bool> containsKey(String key) async {
    try {
      final value = await _storage.containsKey(key: key);
      return value;
    } catch (e, stack) {
      _logger.error('Failed to check key existence: $key', e, stack);
      return false;
    }
  }

  // Private helper methods
  Future<String> _encryptValue(String value) async {
    // TODO: Implement proper encryption
    return base64.encode(utf8.encode(value));
  }

  Future<String> _decryptValue(String value) async {
    // TODO: Implement proper decryption
    return utf8.decode(base64.decode(value));
  }

  String _prepareStorageValue(String value, Map<String, String>? metadata) {
    final storageValue = {
      'value': value,
      'timestamp': DateTime.now().toIso8601String(),
      if (metadata != null) 'metadata': metadata,
    };
    return jsonEncode(storageValue);
  }

  StorageValue _parseStorageValue(String storageValue) {
    final Map<String, dynamic> parsed = jsonDecode(storageValue);
    return StorageValue(
      value: parsed['value'] as String,
      timestamp: DateTime.parse(parsed['timestamp'] as String),
      metadata: (parsed['metadata'] as Map<String, dynamic>?)?.cast<String, String>(),
    );
  }
}

class StorageValue {
  final String value;
  final DateTime timestamp;
  final Map<String, String>? metadata;

  StorageValue({
    required this.value,
    required this.timestamp,
    this.metadata,
  });
}