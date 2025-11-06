import 'package:flutter/material.dart';
import '../../models/emotion_analytics.dart';

class EmotionAnalyticsChart extends StatelessWidget {
  final List<EmotionAnalytics> data;
  final String period;

  const EmotionAnalyticsChart({required this.data, required this.period, super.key});

  @override
  Widget build(BuildContext context) {
    // Stub: Replace with chart library for visuals
    return Card(
      child: Column(
        children: [
          Text('Emotion Analytics ($period)', style: Theme.of(context).textTheme.titleLarge),
          ...data.map((e) => ListTile(
            title: Text('Date: ${e.date.toLocal()}'),
            subtitle: Text('Happiness: ${e.happiness}, Sadness: ${e.sadness}, Anxiety: ${e.anxiety}'),
          ))
        ],
      ),
    );
  }
}
