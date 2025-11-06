class AnalyticsData {
  final String id;
  final DateTime timestamp;
  final Map<String, dynamic> data;
  final String type;

  AnalyticsData({
    required this.id,
    required this.timestamp,
    required this.data,
    required this.type,
  });
}