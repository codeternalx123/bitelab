class Report {
  final String id;
  final String title;
  final DateTime createdAt;
  final String content;
  final String type;
  final Map<String, dynamic> metadata;

  Report({
    required this.id,
    required this.title,
    required this.createdAt,
    required this.content,
    required this.type,
    required this.metadata,
  });
}