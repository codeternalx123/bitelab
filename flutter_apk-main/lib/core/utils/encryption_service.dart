import 'dart:convert';
import 'package:crypto/crypto.dart';

class EncryptionService {
  String hashPassword(String password) {
    final bytes = utf8.encode(password);
    final digest = sha256.convert(bytes);
    return digest.toString();
  }

  bool verifyPassword(String password, String hashedPassword) {
    final hashInput = hashPassword(password);
    return hashInput == hashedPassword;
  }

  String encryptData(String data) {
    // TODO: Implement proper encryption
    // This is a placeholder - in production, use proper encryption algorithms
    final bytes = utf8.encode(data);
    final encrypted = base64.encode(bytes);
    return encrypted;
  }

  String decryptData(String encryptedData) {
    // TODO: Implement proper decryption
    // This is a placeholder - in production, use proper decryption algorithms
    try {
      final bytes = base64.decode(encryptedData);
      return utf8.decode(bytes);
    } catch (e) {
      return '';
    }
  }
}