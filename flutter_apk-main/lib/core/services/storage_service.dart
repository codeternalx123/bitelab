import 'dart:convert';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../error/app_error.dart';
import '../result/result.dart';

class StorageService {
  final FlutterSecureStorage _secureStorage;
  
  StorageService({required FlutterSecureStorage secureStorage})
      : _secureStorage = secureStorage;

  Future<Result<String>> getString(String key) async {
    try {
      final value = await _secureStorage.read(key: key);
      if (value == null) {
        return Result.failure(
          AppError.validation(message: 'No value found for key: $key'),
        );
      }
      return Result.success(value);
    } catch (e) {
      return Result.failure(AppError.security(message: e.toString()));
    }
  }

  Future<Result<void>> setString(String key, String value) async {
    try {
      await _secureStorage.write(key: key, value: value);
      return const Result.success(null);
    } catch (e) {
      return Result.failure(AppError.security(message: e.toString()));
    }
  }

  Future<Result<Map<String, dynamic>>> getJson(String key) async {
    try {
      final value = await _secureStorage.read(key: key);
      if (value == null) {
        return Result.failure(
          AppError.validation(message: 'No value found for key: $key'),
        );
      }
      return Result.success(json.decode(value) as Map<String, dynamic>);
    } catch (e) {
      return Result.failure(AppError.security(message: e.toString()));
    }
  }

  Future<Result<void>> setJson(String key, Map<String, dynamic> value) async {
    try {
      final jsonString = json.encode(value);
      await _secureStorage.write(key: key, value: jsonString);
      return const Result.success(null);
    } catch (e) {
      return Result.failure(AppError.security(message: e.toString()));
    }
  }

  Future<Result<void>> delete(String key) async {
    try {
      await _secureStorage.delete(key: key);
      return const Result.success(null);
    } catch (e) {
      return Result.failure(AppError.security(message: e.toString()));
    }
  }

  Future<Result<void>> deleteAll() async {
    try {
      await _secureStorage.deleteAll();
      return const Result.success(null);
    } catch (e) {
      return Result.failure(AppError.security(message: e.toString()));
    }
  }
}