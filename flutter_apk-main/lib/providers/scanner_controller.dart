import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../core/network/api_service.dart';
import '../core/di/di.dart';

final apiServiceProvider = Provider<ApiService>((ref) => getIt<ApiService>());

final scannerControllerProvider =
    StateNotifierProvider<ScannerController, ScannerState>((ref) {
  return ScannerController(
    ref.watch(apiServiceProvider),
  );
});

class ScannerController extends StateNotifier<ScannerState> {
  final ApiService _apiService;
  CameraController? _cameraController;
  Timer? _scanTimer;

  ScannerController(this._apiService) : super(const ScannerState.initial());

  Future<void> initialize() async {
    state = const ScannerState.initializing();

    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        state = const ScannerState.error('No cameras available');
        return;
      }

      final camera = cameras.first;
      _cameraController = CameraController(
        camera,
        ResolutionPreset.high,
        enableAudio: false,
      );

      await _cameraController!.initialize();
      state = ScannerState.ready(_cameraController!);
    } catch (e) {
      state = ScannerState.error('Failed to initialize camera: $e');
    }
  }

  Future<void> startScanning() async {
    if (state is! Ready) return;

    state = const ScannerState.scanning();

    _scanTimer = Timer.periodic(
      const Duration(seconds: 1),
      (_) => _scanFrame(),
    );
  }

  Future<void> _scanFrame() async {
    if (_cameraController == null ||
        !_cameraController!.value.isInitialized ||
        state is! Scanning) {
      return;
    }

    try {
      final image = await _cameraController!.takePicture();
      
      // Analyze the image
      final result = await _apiService.analyzeFoodImage(
        imagePath: image.path,
      );

      result.when(
        success: (analysis) {
          if (analysis['food_detected'] == true) {
            _scanTimer?.cancel();
            state = ScannerState.foodDetected(
              foodName: analysis['food_name'],
              nutritionInfo: analysis['nutrition_info'],
              confidence: analysis['confidence'],
            );
          }
        },
        failure: (error) {
          state = ScannerState.error(error.userFriendlyMessage);
          _scanTimer?.cancel();
        },
      );
    } catch (e) {
      state = ScannerState.error('Failed to analyze image: $e');
      _scanTimer?.cancel();
    }
  }

  Future<void> stopScanning() async {
    _scanTimer?.cancel();
    if (state is Scanning) {
      state = ScannerState.ready(_cameraController!);
    }
  }

  void reset() {
    _scanTimer?.cancel();
    state = ScannerState.ready(_cameraController!);
  }

  @override
  void dispose() {
    _scanTimer?.cancel();
    _cameraController?.dispose();
    super.dispose();
  }
}

sealed class ScannerState {
  const ScannerState();

  const factory ScannerState.initial() = Initial;
  const factory ScannerState.initializing() = Initializing;
  const factory ScannerState.ready(CameraController controller) = Ready;
  const factory ScannerState.scanning() = Scanning;
  const factory ScannerState.foodDetected({
    required String foodName,
    required Map<String, dynamic> nutritionInfo,
    required double confidence,
  }) = FoodDetected;
  const factory ScannerState.error(String message) = Error;
}

class Initial extends ScannerState {
  const Initial();
}

class Initializing extends ScannerState {
  const Initializing();
}

class Ready extends ScannerState {
  final CameraController controller;
  const Ready(this.controller);
}

class Scanning extends ScannerState {
  const Scanning();
}

class FoodDetected extends ScannerState {
  final String foodName;
  final Map<String, dynamic> nutritionInfo;
  final double confidence;

  const FoodDetected({
    required this.foodName,
    required this.nutritionInfo,
    required this.confidence,
  });
}

class Error extends ScannerState {
  final String message;
  const Error(this.message);
}
