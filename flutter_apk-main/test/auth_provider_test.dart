import 'package:flutter_test/flutter_test.dart';
import 'package:mockito/mockito.dart';
import 'package:mockito/annotations.dart';
import 'package:dio/dio.dart';
import 'package:tumorheal/services/api_service.dart';
import 'package:tumorheal/services/secure_storage.dart';
import 'package:tumorheal/providers/auth_provider.dart';
import 'package:tumorheal/models/user_models.dart';
import 'package:tumorheal/core/error/exceptions.dart';

import 'auth_provider_test.mocks.dart';

@GenerateMocks([ApiService, SecureStorageService])
void main() {
  late MockApiService mockApiService;
  late MockSecureStorageService mockSecureStorage;
  late AuthProvider authProvider;

  setUp(() {
    mockApiService = MockApiService();
    mockSecureStorage = MockSecureStorageService();
    authProvider = AuthProvider();
  });

  group('AuthProvider', () {
    final mockUser = User(
      id: 'test-id',
      email: 'test@example.com',
      name: 'Test User',
      isEmailVerified: true,
      roles: ['user'],
      createdAt: DateTime.now(),
      lastLoginAt: DateTime.now(),
      settings: const UserSettings(
        language: 'en',
        timezone: 'UTC',
        notifications: NotificationSettings(
          email: true,
          push: true,
          sms: false,
        ),
        privacy: PrivacySettings(
          isProfilePublic: false,
          showActivity: true,
          showLocation: true,
          showEmail: false,
          showPhone: false,
        ),
      ),
    );

    final mockResponse = {
      'access_token': 'test-token',
      'user': {
        'id': mockUser.id,
        'email': mockUser.email,
        'name': mockUser.name,
        'isEmailVerified': mockUser.isEmailVerified,
        'roles': mockUser.roles,
        'createdAt': mockUser.createdAt.toIso8601String(),
        'lastLoginAt': mockUser.lastLoginAt.toIso8601String(),
        'settings': mockUser.settings.toJson(),
      }
    };

    test('login success updates state correctly', () async {
      when(mockApiService.post('/api/v1/auth/login', any)).thenAnswer(
        (_) async => Response(
          requestOptions: RequestOptions(path: '/api/v1/auth/login'),
          statusCode: 200,
          data: mockResponse,
        ),
      );

      when(mockSecureStorage.setToken(any)).thenAnswer(
        (_) async {},
      );

      final result = await authProvider.login('test@example.com', 'password');

      expect(result, true);
      expect(authProvider.user, mockUser);
      expect(authProvider.token, 'test-token');
      expect(authProvider.loading, false);

      verify(mockSecureStorage.setToken('test-token')).called(1);
    });

    test('login failure updates state correctly', () async {
      when(mockApiService.post('/api/v1/auth/login', any)).thenThrow(
        AuthException(
          message: 'Invalid credentials',
          code: 'A003',
        ),
      );

      expect(
        () => authProvider.login('test@example.com', 'password'),
        throwsA(isA<AuthException>()),
      );

      expect(authProvider.user, null);
      expect(authProvider.token, null);
      expect(authProvider.loading, false);

      verifyNever(mockSecureStorage.setToken(any));
    });

    test('logout clears state and storage', () async {
      // First login to set the state
      when(mockApiService.post('/api/v1/auth/login', any)).thenAnswer(
        (_) async => Response(
          requestOptions: RequestOptions(path: '/api/v1/auth/login'),
          statusCode: 200,
          data: mockResponse,
        ),
      );

      when(mockSecureStorage.setToken(any)).thenAnswer(
        (_) async {},
      );

      await authProvider.login('test@example.com', 'password');

      // Now test logout
      when(mockSecureStorage.deleteToken()).thenAnswer(
        (_) async {},
      );

      await authProvider.logout();

      expect(authProvider.user, null);
      expect(authProvider.token, null);
      expect(authProvider.loading, false);

      verify(mockSecureStorage.deleteToken()).called(1);
    });
  });
}