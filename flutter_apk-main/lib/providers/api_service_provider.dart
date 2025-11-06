import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../core/network/api_service.dart';
import '../core/di/di.dart';

final apiServiceProvider = Provider<ApiService>((ref) {
  return getIt<ApiService>();
});