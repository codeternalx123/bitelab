// import 'package:injectable/injectable.dart';

import 'environment_config.dart';
import 'network_config.dart';
import 'logger_config.dart';
import 'cache_config.dart';
import 'security_config.dart';

abstract class ConfigModule {
  EnvironmentConfig get environmentConfig => EnvironmentConfig(
        flavor: const String.fromEnvironment('FLAVOR', defaultValue: 'dev'),
      );

  NetworkConfig networkConfig(EnvironmentConfig envConfig) =>
      NetworkConfig(envConfig);

  LoggerConfig loggerConfig(EnvironmentConfig envConfig) =>
      LoggerConfig(envConfig);

  CacheConfig cacheConfig(EnvironmentConfig envConfig) =>
      CacheConfig(envConfig);

  SecurityConfig securityConfig(EnvironmentConfig envConfig) =>
      SecurityConfig(envConfig);
}