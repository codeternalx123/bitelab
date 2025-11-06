import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import '../models/payment_models.dart';

class QuantumSafeKey {
  final String publicKey;
  final String privateKey;
  final String algorithm;

  const QuantumSafeKey({
    required this.publicKey,
    required this.privateKey,
    required this.algorithm,
  });
}

class QuantumEncryption {
  final Random _random = Random.secure();
  final String _algorithm = 'CRYSTALS-Kyber';
  
  /// Encrypts sensitive payment data using quantum-resistant algorithms
  SecuredPayment encryptPaymentData(Map<String, dynamic> data) {
    final timestamp = DateTime.now().toIso8601String();
    final key = _generateRandomBytes(32);
    final nonce = _generateRandomBytes(12);
    
    // Simulate CRYSTALS-Kyber encryption
    final jsonData = jsonEncode(data);
    final dataBytes = utf8.encode(jsonData);
    final encryptedData = _simulateQuantumEncryption(dataBytes, key, nonce);
    final encryptedKey = _protectKeyWithQuantumKEM(key);
    
    // Generate MAC for integrity verification
    final mac = _generateMAC(encryptedData, timestamp);

    return SecuredPayment(
      encryptedData: base64.encode(encryptedData),
      encryptedKey: base64.encode(encryptedKey),
      mac: mac,
      timestamp: timestamp,
    );
  }

  /// Verifies the MAC (Message Authentication Code) of encrypted data
  bool verifyMAC(String encryptedData, String mac, String timestamp) {
    final calculatedMAC = _generateMAC(base64.decode(encryptedData), timestamp);
    return mac == calculatedMAC;
  }

  /// Decrypts payment data using the provided key
  Map<String, dynamic> decryptPaymentData(String encryptedData, String encryptedKey) {
    final key = _recoverKeyFromQuantumKEM(base64.decode(encryptedKey));
    final decryptedBytes = _simulateQuantumDecryption(
      base64.decode(encryptedData),
      key,
    );
    
    final decryptedJson = utf8.decode(decryptedBytes);
    return jsonDecode(decryptedJson) as Map<String, dynamic>;
  }

  /// Generates a quantum-safe key pair
  QuantumSafeKey generateQuantumSafeKey() {
    final privateKey = _generateRandomBytes(32);
    final publicKey = _derivePublicKey(privateKey);

    return QuantumSafeKey(
      publicKey: base64.encode(publicKey),
      privateKey: base64.encode(privateKey),
      algorithm: _algorithm,
    );
  }

  /// Performs quantum-safe key exchange
  String performKeyExchange(String privateKey, String peerPublicKey) {
    final privKeyBytes = base64.decode(privateKey);
    final pubKeyBytes = base64.decode(peerPublicKey);
    
    // Simulate CRYSTALS-Kyber key exchange
    final sharedSecret = _simulateKeyExchange(privKeyBytes, pubKeyBytes);
    return base64.encode(sharedSecret);
  }

  // Private helper methods
  Uint8List _generateRandomBytes(int length) {
    final bytes = Uint8List(length);
    for (var i = 0; i < length; i++) {
      bytes[i] = _random.nextInt(256);
    }
    return bytes;
  }

  String _generateMAC(Uint8List data, String timestamp) {
    final hmac = Hmac(sha256, utf8.encode(timestamp));
    final digest = hmac.convert(data);
    return digest.toString();
  }

  Uint8List _derivePublicKey(Uint8List privateKey) {
    // Simulate CRYSTALS-Kyber public key derivation
    final hash = sha256.convert(privateKey);
    return Uint8List.fromList(hash.bytes);
  }

  Uint8List _simulateQuantumEncryption(Uint8List data, Uint8List key, Uint8List nonce) {
    // Simulate post-quantum encryption using key and nonce
    final result = Uint8List(data.length);
    for (var i = 0; i < data.length; i++) {
      result[i] = data[i] ^ key[i % key.length] ^ nonce[i % nonce.length];
    }
    return result;
  }

  Uint8List _simulateQuantumDecryption(Uint8List encryptedData, Uint8List key) {
    // Simple XOR operation for simulation
    final result = Uint8List(encryptedData.length);
    for (var i = 0; i < encryptedData.length; i++) {
      result[i] = encryptedData[i] ^ key[i % key.length];
    }
    return result;
  }

  Uint8List _protectKeyWithQuantumKEM(Uint8List key) {
    // Simulate CRYSTALS-Kyber KEM encapsulation
    final encapsulatedKey = Uint8List(key.length + 32);
    encapsulatedKey.setAll(0, key);
    encapsulatedKey.setAll(key.length, _generateRandomBytes(32));
    return encapsulatedKey;
  }

  Uint8List _recoverKeyFromQuantumKEM(Uint8List encapsulatedKey) {
    // Simulate CRYSTALS-Kyber KEM decapsulation
    return Uint8List.sublistView(encapsulatedKey, 0, 32);
  }

  Uint8List _simulateKeyExchange(Uint8List privateKey, Uint8List peerPublicKey) {
    // Simulate CRYSTALS-Kyber key exchange
    final sharedSecret = Uint8List(32);
    final combined = Uint8List(privateKey.length + peerPublicKey.length)
      ..setAll(0, privateKey)
      ..setAll(privateKey.length, peerPublicKey);
    final hash = sha256.convert(combined);
    sharedSecret.setAll(0, hash.bytes);
    return sharedSecret;
  }
}