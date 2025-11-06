enum UserRole {
  cancerPatient,
  nonCancerPatient,
  tumourPatient,
}

class User {
  final String id;
  final String name;
  final String email;
  final UserRole role;
  final SubscriptionType subscriptionType;

  // Non-cancer patient health pillars
  final List<HealthPillar> healthPillars;
  final HealthGoal healthGoal;

  User({
    required this.id,
    required this.name,
    required this.email,
    required this.role,
    required this.subscriptionType,
    this.healthPillars = const [],
    this.healthGoal = HealthGoal.optimization,
  });
}

enum HealthPillar {
  quantumNutritionist,
  emotiveHealthCorrelator,
  chronoTherapeuticPlanner,
  kintsugiCommunity,
}

enum HealthGoal {
  longevity,
  performance,
  optimization,
  wellBeing,
}

enum SubscriptionType {
  none,
  basic10,
  premium20,
}
