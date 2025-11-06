import 'dart:async';
import 'package:firebase_performance/firebase_performance.dart';
import 'package:injectable/injectable.dart';
import '../logging/app_logger.dart';

@singleton
class PerformanceMonitor {
  final FirebasePerformance _performance;
  final AppLogger _logger;
  final Map<String, Trace> _activeTraces = {};
  final Map<String, HttpMetric> _activeHttpMetrics = {};

  PerformanceMonitor(this._performance, this._logger);

  Future<T> trackOperation<T>({
    required String name,
    required Future<T> Function() operation,
    Map<String, String>? attributes,
  }) async {
    final trace = _performance.newTrace(name);
    await trace.start();

    if (attributes != null) {
      attributes.forEach((key, value) {
        trace.putAttribute(key, value);
      });
    }

    try {
      final result = await operation();
      await trace.stop();
      return result;
    } catch (e) {
      trace.putAttribute('error', e.toString());
      await trace.stop();
      rethrow;
    }
  }

  Future<void> startTrace(String name) async {
    if (_activeTraces.containsKey(name)) {
      _logger.warning('Trace $name is already running');
      return;
    }

    final trace = _performance.newTrace(name);
    await trace.start();
    _activeTraces[name] = trace;
  }

  Future<void> stopTrace(String name) async {
    final trace = _activeTraces.remove(name);
    if (trace == null) {
      _logger.warning('No active trace found for $name');
      return;
    }

    await trace.stop();
  }

  void setTraceAttribute(String traceName, String attribute, String value) {
    final trace = _activeTraces[traceName];
    if (trace == null) {
      _logger.warning('No active trace found for $traceName');
      return;
    }

    trace.putAttribute(attribute, value);
  }

  void incrementTraceMetric(String traceName, String metric, {int increment = 1}) {
    final trace = _activeTraces[traceName];
    if (trace == null) {
      _logger.warning('No active trace found for $traceName');
      return;
    }

    trace.incrementMetric(metric, increment);
  }

  Future<void> trackHttpRequest({
    required String url,
    required HttpMethod method,
    Map<String, String>? attributes,
  }) async {
    final String metricName = '$method-$url';
    if (_activeHttpMetrics.containsKey(metricName)) {
      _logger.warning('HTTP metric for $metricName is already being tracked');
      return;
    }

    final metric = _performance.newHttpMetric(url, method);
    await metric.start();
    _activeHttpMetrics[metricName] = metric;

    if (attributes != null) {
      attributes.forEach((key, value) {
        metric.putAttribute(key, value);
      });
    }
  }

  Future<void> stopHttpRequest({
    required String url,
    required HttpMethod method,
    required int responseCode,
    int? responseSize,
    String? contentType,
  }) async {
    final String metricName = '$method-$url';
    final metric = _activeHttpMetrics.remove(metricName);
    if (metric == null) {
      _logger.warning('No active HTTP metric found for $metricName');
      return;
    }

    metric.httpResponseCode = responseCode;
    if (responseSize != null) {
      metric.responsePayloadSize = responseSize;
    }
    if (contentType != null) {
      metric.responseContentType = contentType;
    }

    await metric.stop();
  }

  void markAppStart() {
    trackOperation(
      name: 'app_start',
      operation: () async {
        // This trace will capture the time from app start until this point
      },
    );
  }

  Future<void> trackScreenLoad(String screenName) async {
    await trackOperation(
      name: '${screenName}_load',
      operation: () async {
        // Track screen load performance
      },
      attributes: {
        'screen': screenName,
      },
    );
  }

  Future<void> trackWidgetRender(String widgetName) async {
    await trackOperation(
      name: '${widgetName}_render',
      operation: () async {
        // Track widget rendering performance
      },
      attributes: {
        'widget': widgetName,
      },
    );
  }

  Future<void> trackDatabaseOperation({
    required String operation,
    required String table,
    required Future<void> Function() task,
  }) async {
    await trackOperation(
      name: 'db_operation',
      operation: task,
      attributes: {
        'operation': operation,
        'table': table,
      },
    );
  }

  Future<void> trackNetworkCall({
    required String endpoint,
    required Future<void> Function() call,
  }) async {
    await trackOperation(
      name: 'network_call',
      operation: call,
      attributes: {
        'endpoint': endpoint,
      },
    );
  }

  void dispose() {
    // Stop all active traces
    _activeTraces.forEach((name, trace) async {
      await trace.stop();
    });
    _activeTraces.clear();

    // Stop all active HTTP metrics
    _activeHttpMetrics.forEach((name, metric) async {
      await metric.stop();
    });
    _activeHttpMetrics.clear();
  }
}

// Use HttpMethod exported by package:firebase_performance to avoid a type conflict
// (Local enum removed)