import 'package:injectable/injectable.dart';
import '../logging/app_logger.dart';

@singleton
class RequestQueueManager {
  final AppLogger _logger;
  final Map<String, Future<dynamic>> _pendingRequests = {};

  RequestQueueManager(this._logger);

  Future<T> enqueue<T>(String key, Future<T> Function() request) async {
    if (_pendingRequests.containsKey(key)) {
      _logger.debug('Request already in queue: $key');
      return await _pendingRequests[key] as T;
    }

    try {
      _pendingRequests[key] = request();
      final result = await _pendingRequests[key] as T;
      _pendingRequests.remove(key);
      return result;
    } catch (e) {
      _pendingRequests.remove(key);
      rethrow;
    }
  }

  bool isQueued(String key) => _pendingRequests.containsKey(key);

  void clear() {
    _pendingRequests.clear();
  }
}