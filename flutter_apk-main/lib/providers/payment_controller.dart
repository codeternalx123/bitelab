import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../core/services/payment_service.dart';
import '../models/payment_models.dart';
import './payment_service_provider.dart';

Map<String, dynamic> convertPaymentToJson(Payment payment) {
  return payment.toJson();
}

Payment convertJsonToPayment(Map<String, dynamic> json) {
  return Payment.fromJson(json);
}

final paymentControllerProvider =
    StateNotifierProvider<PaymentController, PaymentState>((ref) {
  return PaymentController(
    ref.watch(paymentServiceProvider),
  );
});

class PaymentController extends StateNotifier<PaymentState> {
  final PaymentService _paymentService;

  PaymentController(this._paymentService)
      : super(const PaymentState.initial());

  Future<void> processPayment(PaymentRequest request) async {
    if (state is Processing) return;

    state = const PaymentState.processing();

    // First, analyze the risk
    final riskResult = await _paymentService.analyzePaymentRisk(request);
    await riskResult.when(
      success: (analysis) async {
        if (analysis['risk_analysis']['is_fraudulent'] == true) {
          state = const PaymentState.error({
            'fraud': ['This payment has been flagged as potentially fraudulent'],
          });
          return;
        }

        // If risk is acceptable, process the payment
        final paymentResult = await _paymentService.processPayment(request);

        paymentResult.when(
          success: (payment) {
            if (payment.fraudAnalysis.isFraudulent) {
              state = const PaymentState.error({
                'fraud': ['Payment was flagged as fraudulent during processing'],
              });
              return;
            }

            state = PaymentState.success(payment);
          },
          failure: (error) {
            state = PaymentState.error({
              'payment': [error.userFriendlyMessage],
            });
          },
        );
      },
      failure: (error) {
        state = const PaymentState.error({
          'risk': ['Unable to complete risk analysis'],
        });
      },
    );
  }

  Future<void> verifyPayment(String paymentId) async {
    if (state is! Success) return;

    state = const PaymentState.processing();

    final currentPayment = (state as Success).payment;
    final result = await _paymentService.verifyPayment(
      paymentId,
      currentPayment.securedPayment.toJson(),
    );

    result.when(
      success: (verified) {
        if (verified) {
          state = PaymentState.success(currentPayment);
        } else {
          state = const PaymentState.error({
            'verification': ['Payment verification failed'],
          });
        }
      },
      failure: (error) {
        state = PaymentState.error({
          'verification': [error.userFriendlyMessage],
        });
      },
    );
  }

  Future<void> analyzeFraudRisk(PaymentRequest request) async {
    if (state is Processing) return;

    state = const PaymentState.processing();

    final result = await _paymentService.analyzePaymentRisk(request);
    result.when(
      success: (analysis) {
        if (analysis['risk_analysis']['is_fraudulent'] == true) {
          state = const PaymentState.error({
            'fraud': ['This payment has been flagged as potentially fraudulent'],
          });
        } else {
          state = PaymentState.riskAnalyzed(
            FraudAnalysis(
              isFraudulent: analysis['risk_analysis']['is_fraudulent'],
              fraudProbability: analysis['risk_analysis']['fraud_probability'],
              riskLevel: analysis['risk_analysis']['risk_level'],
              featureImportance: FeatureImportance(
                amount: analysis['feature_importance']['amount'] ?? 0.0,
                frequency: analysis['feature_importance']['frequency'] ?? 0.0,
                timePattern: analysis['feature_importance']['time_pattern'] ?? 0.0,
                locationRisk: analysis['feature_importance']['location_risk'] ?? 0.0,
                deviceRisk: analysis['feature_importance']['device_risk'] ?? 0.0,
              ),
            ),
          );
        }
      },
      failure: (error) {
        state = PaymentState.error({
          'risk': [error.userFriendlyMessage],
        });
      },
    );
  }

  void reset() {
    state = const PaymentState.initial();
  }
}

sealed class PaymentState {
  const PaymentState();

  const factory PaymentState.initial() = Initial;
  const factory PaymentState.processing() = Processing;
  const factory PaymentState.success(Payment payment) = Success;
  const factory PaymentState.error(Map<String, List<String>> errors) = Error;
  const factory PaymentState.riskAnalyzed(FraudAnalysis analysis) = RiskAnalyzed;
}

class Initial extends PaymentState {
  const Initial();
}

class Processing extends PaymentState {
  const Processing();
}

class Success extends PaymentState {
  final Payment payment;
  const Success(this.payment);
}

class Error extends PaymentState {
  final Map<String, List<String>> errors;
  const Error(this.errors);
}

class RiskAnalyzed extends PaymentState {
  final FraudAnalysis analysis;
  const RiskAnalyzed(this.analysis);
}