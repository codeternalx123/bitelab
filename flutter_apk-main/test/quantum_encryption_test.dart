import 'package:flutter_test/flutter_test.dart';
import 'package:tumorheal/services/quantum_encryption.dart';
import 'package:tumorheal/models/payment_models.dart';

void main() {
  late QuantumEncryption encryption;

  setUp(() {
    encryption = QuantumEncryption();
  });

  group('QuantumEncryption', () {
    final testData = {
      'cardNumber': '4242424242424242',
      'expiryMonth': '12',
      'expiryYear': '2025',
      'cvc': '123',
    };

    test('encryptPaymentData returns valid SecuredPayment', () {
      final result = encryption.encryptPaymentData(testData);

      expect(result, isA<SecuredPayment>());
      expect(result.encryptedData, isNotEmpty);
      expect(result.encryptedKey, isNotEmpty);
      expect(result.mac, isNotEmpty);
      expect(result.timestamp, isNotEmpty);
    });

    test('verifyMAC validates encrypted data integrity', () {
      final securedPayment = encryption.encryptPaymentData(testData);
      
      final isValid = encryption.verifyMAC(
        securedPayment.encryptedData,
        securedPayment.mac,
        securedPayment.timestamp,
      );

      expect(isValid, true);
    });

    test('decryptPaymentData recovers original data', () {
      final securedPayment = encryption.encryptPaymentData(testData);
      
      final decryptedData = encryption.decryptPaymentData(
        securedPayment.encryptedData,
        securedPayment.encryptedKey,
      );

      expect(decryptedData, equals(testData));
    });

    test('generateQuantumSafeKey creates valid key', () {
      final key = encryption.generateQuantumSafeKey();

      expect(key.publicKey, isNotEmpty);
      expect(key.privateKey, isNotEmpty);
      expect(key.algorithm, equals('CRYSTALS-Kyber'));
    });

    test('key exchange simulation works', () {
      final aliceKeys = encryption.generateQuantumSafeKey();
      final bobKeys = encryption.generateQuantumSafeKey();

      final sharedSecretAlice = encryption.performKeyExchange(
        aliceKeys.privateKey,
        bobKeys.publicKey,
      );

      final sharedSecretBob = encryption.performKeyExchange(
        bobKeys.privateKey,
        aliceKeys.publicKey,
      );

      expect(sharedSecretAlice, equals(sharedSecretBob));
    });
  });
}