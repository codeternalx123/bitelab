import 'package:freezed_annotation/freezed_annotation.dart';

part 'validation_result.freezed.dart';

@freezed
abstract class ValidationResult with _$ValidationResult {
  const factory ValidationResult.success() = ValidationSuccess;
  const factory ValidationResult.failure(Map<String, List<String>> errors) = ValidationFailure;
}