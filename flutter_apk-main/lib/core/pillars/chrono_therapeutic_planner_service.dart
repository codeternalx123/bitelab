class ChronoTherapeuticPlannerService {
  String getEatingClock(Map<String, dynamic> userData) {
    String sleepQuality = userData['sleep_quality'] ?? 'good';
    if (sleepQuality == 'poor') {
      return 'Delay breakfast by 1 hour for better metabolic efficiency.';
    }
    return 'Eat between 8am-6pm for optimal results.';
  }
}
