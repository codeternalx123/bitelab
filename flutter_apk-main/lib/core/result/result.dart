import 'package:freezed_annotation/freezed_annotation.dart';
import '../error/app_error.dart';

part 'result.freezed.dart';

@freezed
class Result<T> with _$Result<T> {
  const factory Result.success(T data) = Success<T>;
  const factory Result.failure(AppError error) = Failure<T>;
}

extension ResultX<T> on Result<T> {
  bool get isSuccess => this is Success<T>;
  bool get isFailure => this is Failure<T>;

  T? get data => mapOrNull(success: (s) => s.data);
  AppError? get error => mapOrNull(failure: (f) => f.error);

  R when<R>({
    required R Function(T data) success,
    required R Function(AppError error) failure,
  }) {
    return map(
      success: (s) => success(s.data),
      failure: (f) => failure(f.error),
    );
  }

  Result<R> mapResult<R>({
    required R Function(T data) success,
    required R Function(AppError error) failure,
  }) {
    return when(
      success: (data) => Result.success(success(data)),
      failure: (error) => Result.failure(error),
    );
  }

  Future<Result<R>> asyncMap<R>({
    required Future<R> Function(T data) success,
    Future<R> Function(AppError error)? failure,
  }) async {
    return await when(
      success: (data) async => Result.success(await success(data)),
      failure: (error) async => failure != null
          ? Result.success(await failure(error))
          : Result.failure(error),
    );
  }
}