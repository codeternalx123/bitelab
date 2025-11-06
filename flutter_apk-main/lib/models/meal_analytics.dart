class MealAnalytics {
  final DateTime date;
  final double performanceScore;
  final double longevityScore;
  final List<String> flaggedCompounds;

  MealAnalytics({
    required this.date,
    required this.performanceScore,
    required this.longevityScore,
    required this.flaggedCompounds,
  });
}
