import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mockito/mockito.dart';
import 'package:mockito/annotations.dart';
import 'package:tumorheal/services/api_service.dart';
import 'package:tumorheal/services/payment_service.dart';
import 'package:tumorheal/models/payment_models.dart';
import 'package:tumorheal/core/error/payment_exception.dart';

import 'payment_service_test.mocks.dart';

@GenerateMocks([ApiService])
class MapMatcher extends Matcher {
  @override
  bool matches(dynamic item, Map matchState) => item is Map<String, dynamic>;
  
  @override
  Description describe(Description description) => description.add('is Map<String, dynamic>');
}

void main() {
  late MockApiService mockApiService;
  late PaymentService paymentService;

  setUp(() {
    mockApiService = MockApiService();
    paymentService = PaymentService(apiService: mockApiService);
  });

  group('PaymentService', () {
    final mockPayment = Payment(
      id: 'test-payment-id',
      status: 'success',
      amount: 100.0,
      currency: 'USD',
      provider: 'stripe',
      createdAt: DateTime.now().toIso8601String(),
      providerPaymentId: 'stripe-payment-id',
      securedPayment: const SecuredPayment(
        encryptedData: 'encrypted',
        encryptedKey: 'key',
        mac: 'mac',
        timestamp: '2023-01-01T00:00:00Z',
      ),
      fraudAnalysis: const FraudAnalysis(
        isFraudulent: false,
        fraudProbability: 0.1,
        riskLevel: 'low',
        featureImportance: FeatureImportance(
          amount: 0.2,
          frequency: 0.1,
          timePattern: 0.3,
          locationRisk: 0.1,
          deviceRisk: 0.1,
        ),
      ),
    );

    test('processPayment success returns payment details', () async {
      when(mockApiService.processPayment({})).thenAnswer(
        (_) async => Payment.fromJson({
          'id': 'test-payment-id',
          'status': 'success',
          'amount': 100.0,
          'currency': 'USD',
          'provider': 'stripe',
          'created_at': DateTime.now().toIso8601String(),
          'provider_payment_id': 'stripe-payment-id',
          'secured_payment': {
            'encrypted_data': 'encrypted',
            'encrypted_key': 'key',
            'mac': 'mac',
            'timestamp': DateTime.now().toIso8601String(),
          },
          'fraud_analysis': {
            'is_fraudulent': false,
            'fraud_probability': 0.1,
            'risk_level': 'low',
            'feature_importance': {
              'amount': 0.2,
              'frequency': 0.1,
              'time_pattern': 0.3,
              'location_risk': 0.1,
              'device_risk': 0.1,
            },
          },
        }),
      );

      when(
        mockApiService.verifyPayment(
          'test-payment-id',
          {},
        ),
      ).thenAnswer(
        (_) async => {'status': 'verified'},
      );

      when(mockApiService.analyzePaymentRisk({})).thenAnswer(
        (_) async => {
          'risk_analysis': {
            'is_fraudulent': false,
            'fraud_probability': 0.1,
            'risk_level': 'low',
          },
        },
      );

      final result = await paymentService.processPayment(
        amount: 100.0,
        currency: 'USD',
        paymentMethodId: 'pm_test',
        provider: 'stripe',
        subscriptionId: 'sub_test',
      );

      expect(result.id, mockPayment.id);
      expect(result.status, 'success');
      expect(result.amount, 100.0);
      expect(result.fraudAnalysis.isFraudulent, false);

      verify(mockApiService.processPayment({})).called(1);
      verify(mockApiService.analyzePaymentRisk({})).called(1);
    });

    test('processPayment handles fraud detection', () async {
      final fraudulentPayment = Payment(
        id: mockPayment.id,
        status: mockPayment.status,
        amount: mockPayment.amount,
        currency: mockPayment.currency,
        provider: mockPayment.provider,
        createdAt: mockPayment.createdAt,
        providerPaymentId: mockPayment.providerPaymentId,
        securedPayment: mockPayment.securedPayment,
        fraudAnalysis: const FraudAnalysis(
          isFraudulent: true,
          fraudProbability: 0.9,
          riskLevel: 'high',
          featureImportance: FeatureImportance(
            amount: 0.8,
            frequency: 0.7,
            timePattern: 0.9,
            locationRisk: 0.8,
            deviceRisk: 0.7,
          ),
        ),
      );

      when(mockApiService.processPayment({})).thenAnswer(
        (_) async => fraudulentPayment,
      );

      expect(
        () => paymentService.processPayment(
          amount: 100.0,
          currency: 'USD',
          paymentMethodId: 'pm_test',
          provider: 'stripe',
        ),
        throwsA(isA<PaymentException>()),
      );
    });

    test('verifyPayment success returns true', () async {
      when(
        mockApiService.verifyPayment(
          'test-payment-id',
          {},
        ),
      ).thenAnswer(
        (_) async => {'status': 'verified'},
      );

      final result = await paymentService.verifyPayment(
        'test-payment-id',
        {'test': 'data'},
      );

      expect(result, true);
      verify(mockApiService.verifyPayment('test-payment-id', {})).called(1);
    });

    test('analyzePaymentRisk returns risk analysis', () async {
      final mockRiskAnalysis = {
        'risk_analysis': {
          'is_fraudulent': false,
          'fraud_probability': 0.1,
          'risk_level': 'low',
        },
        'recommendation': 'proceed',
        'quantum_enhanced': true,
      };

      when(mockApiService.analyzePaymentRisk({})).thenAnswer(
        (_) async => mockRiskAnalysis,
      );

      const request = PaymentRequest(
        amount: 100.0,
        currency: 'USD',
        paymentMethodId: 'pm_test',
        provider: 'stripe',
        metadata: PaymentMetadata(
          deviceInfo: DeviceInfo(
            os: 'iOS',
            model: 'iPhone',
            browser: 'Safari',
            screenResolution: '1920x1080',
            timezone: 'UTC',
          ),
          location: Location(
            latitude: 0.0,
            longitude: 0.0,
          ),
          networkInfo: NetworkInfo(
            ipAddress: '127.0.0.1',
            isVpn: false,
            isProxy: false,
          ),
        ),
      );

      final result = await paymentService.analyzePaymentRisk(request);

      expect(result, mockRiskAnalysis);
      verify(mockApiService.analyzePaymentRisk({})).called(1);
    });
  });
}