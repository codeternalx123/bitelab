import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/auth_models.dart';
import '../models/user_models.dart';
import '../core/services/auth_service.dart';
import '../core/network/api_service.dart';
import '../core/storage/storage.dart';
import '../core/di/di.dart';

final authServiceProvider = Provider((ref) => getIt<AuthService>());
final apiServiceProvider = Provider((ref) => getIt<ApiService>());
final storageProvider = Provider((ref) => getIt<Storage>());

final authControllerProvider =
    StateNotifierProvider<AuthController, AuthState>((ref) {
  return AuthController(
    ref.watch(authServiceProvider),
    ref.watch(apiServiceProvider),
    ref.watch(storageProvider),
  );
});

class AuthController extends StateNotifier<AuthState> {
  final AuthService _authService;
  final ApiService _apiService;
  final Storage _storage;

  AuthController(
    this._authService,
    this._apiService,
    this._storage)
      : super(const AuthState.initial()) {
    _init();
  }

  Future<void> _init() async {
    state = const AuthState.loading();

    final result = await _authService.getCurrentUser();
    result.when(
      success: (user) {
        state = AuthState.authenticated(user);
      },
      failure: (_) {
        state = const AuthState.unauthenticated();
      },
    );
  }

  Future<void> signIn(SignInRequest request) async {
    if (state is Loading) return;

    state = const AuthState.loading();

    final result = await _authService.signIn(request);
    result.when(
      success: (response) {
        state = AuthState.authenticated(response.user);
        _storage.write(
          key: 'access_token',
          value: response.accessToken,
          metadata: {
            'expires_at': response.expiresAt.toIso8601String(),
          },
        );
        _storage.write(
          key: 'refresh_token',
          value: response.refreshToken,
        );
      },
      failure: (error) {
        state = AuthState.error(error.toJson());
      },
    );
  }

  Future<void> signOut() async {
    if (state is! Authenticated) return;

    state = const AuthState.loading();

    final result = await _apiService.signOut();
    result.when(
      success: (_) async {
        await _storage.delete('access_token');
        await _storage.delete('refresh_token');
        state = const AuthState.unauthenticated();
      },
      failure: (error) {
        state = AuthState.authenticated((state as Authenticated).user);
      },
    );
  }

  Future<void> refreshSession() async {
    if (state is! Authenticated) return;

    final refreshToken = await _storage.read('refresh_token');
    if (refreshToken == null) {
      state = const AuthState.unauthenticated();
      return;
    }

    final result = await _apiService.refreshToken(refreshToken);
    await result.when(
      success: (response) async {
        await _storage.write(
          key: 'access_token',
          value: response['access_token'],
          metadata: {
            'expires_at': response['expires_at'],
          },
        );
        await _storage.write(
          key: 'refresh_token',
          value: response['refresh_token'],
        );
        state = AuthState.authenticated(User.fromJson(response['user']));
      },
      failure: (_) {
        state = const AuthState.unauthenticated();
        _storage.delete('access_token');
        _storage.delete('refresh_token');
      },
    );
  }

  Future<void> updateProfile(User updatedUser) async {
    if (state is! Authenticated) return;

    final currentState = state as Authenticated;
    final result = await _apiService.updateUser(updatedUser.toJson());

    result.when(
      success: (userJson) {
        state = AuthState.authenticated(User.fromJson(userJson));
      },
      failure: (_) {
        state = AuthState.authenticated(currentState.user);
      },
    );
  }

  Future<void> updateSettings(UserSettings settings) async {
    if (state is! Authenticated) return;

    final currentState = state as Authenticated;
    final updatedUser = currentState.user.copyWith(settings: settings);
    
    await updateProfile(updatedUser);
  }

  Future<void> enableTwoFactor() async {
    if (state is! Authenticated) return;

    final currentState = state as Authenticated;
    final currentSettings = currentState.user.settings;
    
    final updatedSettings = currentSettings.copyWith(
      twoFactorEnabled: true,
    );
    
    await updateSettings(updatedSettings);
  }

  Future<void> disableTwoFactor() async {
    if (state is! Authenticated) return;

    final currentState = state as Authenticated;
    final currentSettings = currentState.user.settings;
    
    final updatedSettings = currentSettings.copyWith(
      twoFactorEnabled: false,
    );
    
    await updateSettings(updatedSettings);
  }

  void resetError() {
    if (state is Error) {
      state = const AuthState.unauthenticated();
    }
  }
}

sealed class AuthState {
  const AuthState();

  const factory AuthState.initial() = Initial;
  const factory AuthState.loading() = Loading;
  const factory AuthState.authenticated(User user) = Authenticated;
  const factory AuthState.unauthenticated() = Unauthenticated;
  const factory AuthState.error(Map<String, List<String>> errors) = Error;
}

class Initial extends AuthState {
  const Initial();
}

class Loading extends AuthState {
  const Loading();
}

class Authenticated extends AuthState {
  final User user;
  const Authenticated(this.user);
}

class Unauthenticated extends AuthState {
  const Unauthenticated();
}

class Error extends AuthState {
  final Map<String, List<String>> errors;
  const Error(this.errors);
}