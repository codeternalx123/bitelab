import 'package:flutter/material.dart';
import '../../models/meal_analytics.dart';

class MealAnalyticsChart extends StatelessWidget {
  final List<MealAnalytics> data;
  final String period;

  const MealAnalyticsChart({required this.data, required this.period, super.key});

  @override
  Widget build(BuildContext context) {
    // Stub: Replace with chart library for visuals
    return Card(
      child: Column(
        children: [
          Text('Meal Analytics ($period)', style: Theme.of(context).textTheme.titleLarge),
          ...data.map((e) => ListTile(
            title: Text('Date: ${e.date.toLocal()}'),
            subtitle: Text('Performance: ${e.performanceScore}, Longevity: ${e.longevityScore}'),
          ))
        ],
      ),
    );
  }
}
