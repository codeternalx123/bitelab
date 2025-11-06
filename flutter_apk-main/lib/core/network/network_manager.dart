import 'dart:async';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:injectable/injectable.dart';
import '../logging/app_logger.dart';

enum NetworkStatus {
  connected,
  disconnected
}

@singleton
class NetworkManager {
  final Connectivity _connectivity;
  final AppLogger _logger;
  final _controller = StreamController<NetworkStatus>.broadcast();

  Stream<NetworkStatus> get status async* {
    yield* _controller.stream;
  }

  NetworkManager(this._connectivity, this._logger) {
    _init();
  }

  void _init() {
    _connectivity.onConnectivityChanged.listen((event) {
      _handleConnectivityChange(event);
    });
  }

  void _handleConnectivityChange(ConnectivityResult result) {
    switch (result) {
      case ConnectivityResult.wifi:
      case ConnectivityResult.mobile:
      case ConnectivityResult.ethernet:
        _controller.add(NetworkStatus.connected);
        _logger.info('Network connected: ${result.toString()}');
        break;
      default:
        _controller.add(NetworkStatus.disconnected);
        _logger.warning('Network disconnected: ${result.toString()}');
    }
  }

  Future<bool> get isConnected async {
    final result = await _connectivity.checkConnectivity();
    return result == ConnectivityResult.wifi ||
           result == ConnectivityResult.mobile ||
           result == ConnectivityResult.ethernet;
  }

  @disposeMethod
  void dispose() {
    _controller.close();
  }
}