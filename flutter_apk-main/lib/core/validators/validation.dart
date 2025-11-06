import 'package:freezed_annotation/freezed_annotation.dart';

part 'validation.freezed.dart';

@freezed
class ValidationResult with _$ValidationResult {
  const factory ValidationResult.success() = ValidationSuccess;
  const factory ValidationResult.failure(Map<String, List<String>> errors) = ValidationFailure;
}

abstract class Validatable {
  ValidationResult validate();
}

class Validator {
  static ValidationResult validateEmail(String email) {
    const emailRegex =
        r'^[a-zA-Z0-9.!#$%&*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$';

    if (email.isEmpty) {
      return const ValidationResult.failure({
        'email': ['Email is required'],
      });
    }

    if (!RegExp(emailRegex).hasMatch(email)) {
      return const ValidationResult.failure({
        'email': ['Invalid email format'],
      });
    }

    return const ValidationResult.success();
  }

  static ValidationResult validatePassword(String password) {
    final errors = <String>[];

    if (password.length < 8) {
      errors.add('Password must be at least 8 characters long');
    }

    if (!password.contains(RegExp(r'[A-Z]'))) {
      errors.add('Password must contain at least one uppercase letter');
    }

    if (!password.contains(RegExp(r'[a-z]'))) {
      errors.add('Password must contain at least one lowercase letter');
    }

    if (!password.contains(RegExp(r'[0-9]'))) {
      errors.add('Password must contain at least one number');
    }

    if (!password.contains(RegExp(r'[!@#$%^&*(),.?":{}|<>]'))) {
      errors.add('Password must contain at least one special character');
    }

    return errors.isEmpty
        ? const ValidationResult.success()
        : ValidationResult.failure({'password': errors});
  }

  static ValidationResult validateAmount(double amount, {double? minAmount}) {
    final errors = <String>[];

    if (amount <= 0) {
      errors.add('Amount must be greater than 0');
    }

    if (minAmount != null && amount < minAmount) {
      errors.add('Amount must be at least \$$minAmount');
    }

    return errors.isEmpty
        ? const ValidationResult.success()
        : ValidationResult.failure({'amount': errors});
  }

  static ValidationResult validateCurrency(String currency) {
    if (currency.isEmpty) {
      return const ValidationResult.failure({
        'currency': ['Currency is required'],
      });
    }

    if (currency.length != 3) {
      return const ValidationResult.failure({
        'currency': ['Currency must be a 3-letter code'],
      });
    }

    if (!RegExp(r'^[A-Z]{3}$').hasMatch(currency)) {
      return const ValidationResult.failure({
        'currency': ['Currency must be in uppercase letters'],
      });
    }

    return const ValidationResult.success();
  }

  static ValidationResult validateCardNumber(String cardNumber) {
    // Remove any spaces or dashes
    cardNumber = cardNumber.replaceAll(RegExp(r'[\s-]'), '');

    if (!RegExp(r'^\d{13,19}$').hasMatch(cardNumber)) {
      return const ValidationResult.failure({
        'cardNumber': ['Invalid card number length'],
      });
    }

    // Luhn algorithm
    int sum = 0;
    bool alternate = false;
    for (int i = cardNumber.length - 1; i >= 0; i--) {
      int digit = int.parse(cardNumber[i]);
      if (alternate) {
        digit *= 2;
        if (digit > 9) {
          digit -= 9;
        }
      }
      sum += digit;
      alternate = !alternate;
    }

    if (sum % 10 != 0) {
      return const ValidationResult.failure({
        'cardNumber': ['Invalid card number'],
      });
    }

    return const ValidationResult.success();
  }

  static ValidationResult validateExpiryDate(String month, String year) {
    final errors = <String>[];

    // Parse month and year
    final monthNum = int.tryParse(month);
    final yearNum = int.tryParse(year);

    if (monthNum == null || monthNum < 1 || monthNum > 12) {
      errors.add('Invalid month');
    }

    if (yearNum == null) {
      errors.add('Invalid year');
    } else {
      // Convert 2-digit year to 4-digit
      final fullYear = yearNum < 100 ? 2000 + yearNum : yearNum;
      
      // Get current date
      final now = DateTime.now();
      final expiryDate = DateTime(fullYear, monthNum ?? 0);

      if (expiryDate.isBefore(now)) {
        errors.add('Card has expired');
      }
    }

    return errors.isEmpty
        ? const ValidationResult.success()
        : ValidationResult.failure({'expiryDate': errors});
  }

  static ValidationResult validateCVV(String cvv) {
    if (!RegExp(r'^\d{3,4}$').hasMatch(cvv)) {
      return const ValidationResult.failure({
        'cvv': ['CVV must be 3 or 4 digits'],
      });
    }

    return const ValidationResult.success();
  }

  static ValidationResult validatePhone(String phone) {
    // Remove any spaces, dashes, or parentheses
    phone = phone.replaceAll(RegExp(r'[\s\-\(\)]'), '');

    if (!RegExp(r'^\+?\d{10,15}$').hasMatch(phone)) {
      return const ValidationResult.failure({
        'phone': ['Invalid phone number format'],
      });
    }

    return const ValidationResult.success();
  }

  static ValidationResult validateAddress({
    required String street,
    required String city,
    required String country,
    String? state,
    String? postalCode,
  }) {
    final errors = <String, List<String>>{};

    if (street.isEmpty) {
      errors['street'] = ['Street address is required'];
    }

    if (city.isEmpty) {
      errors['city'] = ['City is required'];
    }

    if (country.isEmpty) {
      errors['country'] = ['Country is required'];
    }

    if (postalCode != null && postalCode.isEmpty) {
      errors['postalCode'] = ['Postal code is required'];
    }

    return errors.isEmpty
        ? const ValidationResult.success()
        : ValidationResult.failure(errors);
  }
}