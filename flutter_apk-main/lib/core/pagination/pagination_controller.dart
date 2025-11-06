import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:infinite_scroll_pagination/infinite_scroll_pagination.dart';
import '../cache/cache_manager.dart';
import '../result/result.dart';
import '../../models/analytics_data.dart';
import '../../models/report.dart';

abstract class PaginatedController<T> extends StateNotifier<PaginationState<T>> {
  final CacheManager _cacheManager;
  final PagingController<int, T> pagingController;
  final String cacheKey;
  final Duration cacheDuration;

  PaginatedController(
    this._cacheManager,
    this.cacheKey, {
    this.cacheDuration = const Duration(hours: 1),
  }) : pagingController = PagingController(firstPageKey: 1),
       super(const PaginationState.initial()) {
    pagingController.addPageRequestListener(_fetchPage);
    _init();
  }

  Future<void> _init() async {
    // Try to load cached data first
    final cachedData = await _cacheManager.get<List<T>>(cacheKey);
    if (cachedData != null) {
      pagingController.itemList = cachedData;
      state = PaginationState.data(
        items: cachedData,
        hasMore: true, // We'll verify this with the next API call
        isFromCache: true,
      );
    }
  }

  Future<void> _fetchPage(int pageKey) async {
    try {
      final result = await fetchData(pageKey);
      result.when(
        success: (response) async {
          final items = response.items;
          final isLastPage = !response.hasMore;

          if (isLastPage) {
            pagingController.appendLastPage(items);
          } else {
            pagingController.appendPage(items, pageKey + 1);
          }

          // Cache the complete list
          final currentList = pagingController.itemList ?? [];
          await _cacheManager.put(
            cacheKey,
            currentList,
            expiration: cacheDuration,
          );

          state = PaginationState.data(
            items: currentList,
            hasMore: !isLastPage,
            isFromCache: false,
          );
        },
        failure: (error) {
          pagingController.error = error;
          state = PaginationState.error(error.userFriendlyMessage);
        },
      );
    } catch (e) {
      pagingController.error = e;
      state = PaginationState.error(e.toString());
    }
  }

  Future<Result<PaginatedResponse<T>>> fetchData(int page);

  Future<void> refresh() async {
    pagingController.refresh();
    state = const PaginationState.loading();
  }

  Future<void> clearCache() async {
    await _cacheManager.remove(cacheKey);
    refresh();
  }

  @override
  void dispose() {
    pagingController.dispose();
    super.dispose();
  }
}

class PaginatedResponse<T> {
  final List<T> items;
  final bool hasMore;
  final int totalItems;
  final int currentPage;
  final int totalPages;

  PaginatedResponse({
    required this.items,
    required this.hasMore,
    required this.totalItems,
    required this.currentPage,
    required this.totalPages,
  });
}

sealed class PaginationState<T> {
  const PaginationState();

  const factory PaginationState.initial() = Initial;
  const factory PaginationState.loading() = Loading;
  const factory PaginationState.data({
    required List<T> items,
    required bool hasMore,
    required bool isFromCache,
  }) = Data;
  const factory PaginationState.error(String message) = Error;
}

class Initial<T> extends PaginationState<T> {
  const Initial();
}

class Loading<T> extends PaginationState<T> {
  const Loading();
}

class Data<T> extends PaginationState<T> {
  final List<T> items;
  final bool hasMore;
  final bool isFromCache;

  const Data({
    required this.items,
    required this.hasMore,
    required this.isFromCache,
  });
}

class Error<T> extends PaginationState<T> {
  final String message;
  const Error(this.message);
}

// Example implementation for a specific type
class AnalyticsController extends PaginatedController<AnalyticsData> {
  AnalyticsController(
    CacheManager cacheManager,
  ) : super(
          cacheManager,
          'analytics_data',
          cacheDuration: const Duration(minutes: 30),
        );

  @override
  Future<Result<PaginatedResponse<AnalyticsData>>> fetchData(int page) async {
    // TODO: Implement the specific API call for analytics data
    return Result.success(PaginatedResponse(
      items: [],
      hasMore: false,
      totalItems: 0,
      currentPage: page,
      totalPages: 0,
    ));
  }
}

// Example implementation for reports
class ReportsController extends PaginatedController<Report> {
  final String userId;

  ReportsController(
    CacheManager cacheManager,
    this.userId,
  ) : super(
          cacheManager,
          'user_reports_$userId',
          cacheDuration: const Duration(hours: 2),
        );

  @override
  Future<Result<PaginatedResponse<Report>>> fetchData(int page) async {
    // TODO: Implement the specific API call for user reports
    return Result.success(PaginatedResponse(
      items: [],
      hasMore: false,
      totalItems: 0,
      currentPage: page,
      totalPages: 0,
    ));
  }
}