import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../services/secure_storage.dart';
import '../models/user_models.dart';

class AuthRequest {
  final String email;
  final String password;

  AuthRequest({required this.email, required this.password});

  Map<String, dynamic> toJson() => {'email': email, 'password': password};
}

class AuthProvider extends ChangeNotifier {
  final ApiService _api = ApiService();
  final SecureStorageService _storage = SecureStorageService();
  String? _token;
  User? _user;
  bool _loading = false;

  String? get token => _token;
  User? get user => _user;
  bool get loading => _loading;

  Future<bool> register(AuthRequest request) async {
    _loading = true;
    notifyListeners();

    try {
      final resp = await _api.post('/api/v1/auth/register', request.toJson());
      if (resp.statusCode == 201) {
        final accessToken = resp.data['access_token'] as String;
        _token = accessToken;
        await _storage.setToken(accessToken);
        _user = User.fromJson(resp.data['user']);
        _loading = false;
        notifyListeners();
        return true;
      }
      _loading = false;
      notifyListeners();
      return false;
    } catch (e) {
      _loading = false;
      notifyListeners();
      rethrow;
    }
  }

  Future<bool> login(String email, String password) async {
    _loading = true;
    notifyListeners();

    try {
      final request = AuthRequest(email: email, password: password);
      final resp = await _api.post('/api/v1/auth/login', request.toJson());
      if (resp.statusCode == 200) {
        final accessToken = resp.data['access_token'] as String;
        _token = accessToken;
        await _storage.setToken(accessToken);
        _user = User.fromJson(resp.data['user']);
        _loading = false;
        notifyListeners();
        return true;
      }
      _loading = false;
      notifyListeners();
      return false;
    } catch (e) {
      _loading = false;
      notifyListeners();
      rethrow;
    }
  }

  Future<void> logout() async {
    _token = null;
    _user = null;
    await _storage.deleteToken();
    notifyListeners();
  }

  Future<bool> resetPassword(String email) async {
    _loading = true;
    notifyListeners();

    try {
      final resp = await _api.post('/api/v1/auth/reset-password', {'email': email});
      _loading = false;
      notifyListeners();
      return resp.statusCode == 200;
    } catch (e) {
      _loading = false;
      notifyListeners();
      rethrow;
    }
  }

  Future<bool> confirmResetPassword(String token, String newPassword) async {
    _loading = true;
    notifyListeners();

    try {
      final resp = await _api.post('/api/v1/auth/reset-password/confirm', {
        'token': token,
        'new_password': newPassword,
      });
      _loading = false;
      notifyListeners();
      return resp.statusCode == 200;
    } catch (e) {
      _loading = false;
      notifyListeners();
      rethrow;
    }
  }

  Future<bool> refreshToken() async {
    try {
      final resp = await _api.post('/api/v1/auth/refresh', {'token': _token});
      if (resp.statusCode == 200) {
        final accessToken = resp.data['access_token'] as String;
        _token = accessToken;
        await _storage.setToken(accessToken);
        notifyListeners();
        return true;
      }
      return false;
    } catch (e) {
      return false;
    }
  }

  Future<bool> requestOTP(String email, String action) async {
    _loading = true;
    notifyListeners();

    try {
      final resp = await _api.post('/api/v1/auth/otp/request', {
        'email': email,
        'action': action,
      });
      _loading = false;
      notifyListeners();
      return resp.statusCode == 200;
    } catch (e) {
      _loading = false;
      notifyListeners();
      rethrow;
    }
  }

  Future<bool> verifyOTP(String email, String otp, String action) async {
    _loading = true;
    notifyListeners();

    try {
      final resp = await _api.post('/api/v1/auth/otp/verify', {
        'email': email,
        'otp': otp,
        'action': action,
      });
      _loading = false;
      notifyListeners();
      return resp.statusCode == 200;
    } catch (e) {
      _loading = false;
      notifyListeners();
      rethrow;
    }
  }
}
