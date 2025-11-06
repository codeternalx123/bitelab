import 'secure_storage_service.dart';

class Storage {
  final SecureStorageService _storage;

  Storage(this._storage);

  Future<String?> read(String key) async {
    return await _storage.read(key);
  }

  Future<void> write({
    required String key,
    required String value,
    Map<String, dynamic>? metadata,
  }) async {
    await _storage.write(
      key: key,
      value: value,
      metadata: metadata?.cast<String, String>(),
    );
  }

  Future<void> delete(String key) async {
    await _storage.delete(key);
  }
}
