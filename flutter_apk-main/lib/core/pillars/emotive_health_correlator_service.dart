class EmotiveHealthCorrelatorService {
  String getEmotiveHealthAlert(Map<String, dynamic> userData) {
    double stress = userData['stress'] ?? 0.4;
    double fatigue = userData['fatigue'] ?? 0.3;
    if (stress > 0.6) {
      return 'High stress detected. Recommend mindfulness and magnesium-rich snack.';
    } else if (fatigue > 0.5) {
      return 'Fatigue markers detected. Suggest power nap and almonds.';
    }
    return 'All markers normal.';
  }
}
