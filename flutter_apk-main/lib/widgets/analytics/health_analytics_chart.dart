import 'package:flutter/material.dart';
import '../../models/health_analytics.dart';

class HealthAnalyticsChart extends StatelessWidget {
  final List<HealthAnalytics> data;
  final String period;

  const HealthAnalyticsChart({required this.data, required this.period, super.key});

  @override
  Widget build(BuildContext context) {
    // Stub: Replace with chart library for visuals
    return Card(
      child: Column(
        children: [
          Text('Health Analytics ($period)', style: Theme.of(context).textTheme.titleLarge),
          ...data.map((e) => ListTile(
            title: Text('Date: ${e.date.toLocal()}'),
            subtitle: Text('Stress: ${e.stressLevel}, Fatigue: ${e.fatigueLevel}, Sleep: ${e.sleepQuality}'),
          ))
        ],
      ),
    );
  }
}
