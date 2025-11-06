import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../../services/api_service.dart';
import '../../models/user_model.dart';

class FoodScannerScreen extends StatefulWidget {
  const FoodScannerScreen({super.key});

  @override
  State<FoodScannerScreen> createState() => _FoodScannerScreenState();
}

class _FoodScannerScreenState extends State<FoodScannerScreen> {
  // Emotive Health Correlator stub
  String getEmotiveHealthAlert(Map<String, dynamic> userData) {
    // Example: Detect fatigue/stress and recommend intervention
    double stress = userData['stress'] ?? 0.4;
    double fatigue = userData['fatigue'] ?? 0.3;
    if (stress > 0.6) {
      return 'High stress detected. Recommend mindfulness and magnesium-rich snack.';
    } else if (fatigue > 0.5) {
      return 'Fatigue markers detected. Suggest power nap and almonds.';
    }
    return 'All markers normal.';
  }

  // Chrono-Therapeutic Planner stub
  String getEatingClock(Map<String, dynamic> userData) {
    // Example: Suggest optimal eating window
    String sleepQuality = userData['sleep_quality'] ?? 'good';
    if (sleepQuality == 'poor') {
      return 'Delay breakfast by 1 hour for better metabolic efficiency.';
    }
    return 'Eat between 8am-6pm for optimal results.';
  }

  // Kintsugi Community stub
  String getCommunityTip(Map<String, dynamic> userData) {
    // Example: Suggest life hack from community
    return 'Users with similar profiles report 25% more focus after a cold shower and productivity playlist.';
  }
  final ImagePicker _picker = ImagePicker();
  final ApiService _api = ApiService();
  String? result;
  bool loading = false;

  // Example: User role and health goal (should be fetched from provider in real app)
  final bool isNonCancerPatient = true; // Replace with actual user role check
  final HealthGoal userGoal = HealthGoal.performance; // Replace with actual user goal

  // Quantum Nutritionist logic for non-cancer patients
  double calculatePerformanceScore(Map<String, dynamic> mealData) {
    // Quantum optimization stub: score based on synergy, polyphenols, etc.
    double synergy = mealData['synergy'] ?? 0.7;
    double polyphenols = mealData['polyphenols'] ?? 0.5;
    double antiInflammatory = mealData['anti_inflammatory'] ?? 0.6;
    // Dynamic weighting based on userGoal
    double weight = userGoal == HealthGoal.performance ? 1.2 : 1.0;
    return (synergy * 0.4 + polyphenols * 0.3 + antiInflammatory * 0.3) * weight * 100;
  }

  double calculateLongevityScore(Map<String, dynamic> mealData) {
    double antioxidants = mealData['antioxidants'] ?? 0.6;
    double hormoneBalance = mealData['hormone_balance'] ?? 0.5;
    double gutHealth = mealData['gut_health'] ?? 0.7;
    double weight = userGoal == HealthGoal.longevity ? 1.2 : 1.0;
    return (antioxidants * 0.4 + hormoneBalance * 0.3 + gutHealth * 0.3) * weight * 100;
  }

  List<String> flagCompounds(Map<String, dynamic> mealData) {
    List<String> flags = [];
    if ((mealData['fatigue_compounds'] ?? 0) > 0.3) flags.add('Fatigue risk');
    if ((mealData['inflammatory_compounds'] ?? 0) > 0.3) flags.add('Inflammation risk');
    if ((mealData['cardio_compounds'] ?? 0) > 0.3) flags.add('Cardiovascular risk');
    return flags;
  }

  Future<void> _scanFood() async {
    // Example userData for pillar stubs
    final userData = {
      'stress': 0.5,
      'fatigue': 0.2,
      'sleep_quality': 'good',
    };
    final XFile? image = await _picker.pickImage(source: ImageSource.camera);
    if (image == null) return;
    setState(() => loading = true);
    // Upload to backend and get meal analysis
    final resp = await _api.post('/api/v1/food/analyze', {'path': image.path});
    final mealData = resp.data as Map<String, dynamic>? ?? {};

    String displayResult = '';
    if (isNonCancerPatient) {
      double perfScore = calculatePerformanceScore(mealData);
      double longScore = calculateLongevityScore(mealData);
      List<String> flags = flagCompounds(mealData);
      displayResult = 'Performance Score: ${perfScore.toStringAsFixed(1)}\nLongevity Score: ${longScore.toStringAsFixed(1)}';
      if (flags.isNotEmpty) {
        displayResult += '\nFlags: ${flags.join(', ')}';
      }
      // Add pillar outputs
      displayResult += '\n\nEmotive Health: ${getEmotiveHealthAlert(userData)}';
      displayResult += '\nChrono-Therapeutic Planner: ${getEatingClock(userData)}';
      displayResult += '\nKintsugi Community: ${getCommunityTip(userData)}';
    } else {
      displayResult = resp.data.toString();
    }
    setState(() { result = displayResult; loading = false; });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Food Scanner')),
      body: Center(child: loading ? const CircularProgressIndicator() : Column(mainAxisSize: MainAxisSize.min, children: [
        ElevatedButton(onPressed: _scanFood, child: const Text('Scan Food')),
        if (result != null) Padding(padding: const EdgeInsets.all(12), child: Text('Result:\n$result')),
        if (isNonCancerPatient) ...[
          // Placeholder for pillar features
          const Padding(
            padding: EdgeInsets.all(8.0),
            child: Text('Quantum Nutritionist, Emotive Health Correlator, Chrono-Therapeutic Planner, Kintsugi Community features enabled.'),
          ),
        ]
      ])),
    );
  }
}
