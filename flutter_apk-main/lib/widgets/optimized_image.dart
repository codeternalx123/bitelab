import 'package:flutter/material.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:shimmer/shimmer.dart';

class OptimizedImage extends StatelessWidget {
  final String imageUrl;
  final double? width;
  final double? height;
  final BoxFit? fit;
  final BorderRadius? borderRadius;
  final Duration? fadeInDuration;
  final Widget? placeholder;
  final Widget? errorWidget;
  final String? cacheKey;
  final Map<String, String>? headers;
  final bool useOldImageOnUrlChange;
  final int? memCacheWidth;
  final int? memCacheHeight;
  final Color? shimmerBaseColor;
  final Color? shimmerHighlightColor;

  const OptimizedImage({
    super.key,
    required this.imageUrl,
    this.width,
    this.height,
    this.fit,
    this.borderRadius,
    this.fadeInDuration,
    this.placeholder,
    this.errorWidget,
    this.cacheKey,
    this.headers,
    this.useOldImageOnUrlChange = true,
    this.memCacheWidth,
    this.memCacheHeight,
    this.shimmerBaseColor,
    this.shimmerHighlightColor,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: borderRadius ?? BorderRadius.zero,
      child: CachedNetworkImage(
        imageUrl: imageUrl,
        width: width,
        height: height,
        fit: fit ?? BoxFit.cover,
        fadeInDuration: fadeInDuration ?? const Duration(milliseconds: 300),
        placeholder: (context, url) => placeholder ?? _buildShimmer(),
        errorWidget: (context, url, error) =>
            errorWidget ?? _buildErrorWidget(error),
        cacheKey: cacheKey,
        httpHeaders: headers,
        useOldImageOnUrlChange: useOldImageOnUrlChange,
        memCacheWidth: memCacheWidth,
        memCacheHeight: memCacheHeight,
        imageBuilder: (context, imageProvider) => Container(
          decoration: BoxDecoration(
            image: DecorationImage(
              image: imageProvider,
              fit: fit ?? BoxFit.cover,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildShimmer() {
    return Shimmer.fromColors(
      baseColor: shimmerBaseColor ?? Colors.grey[300]!,
      highlightColor: shimmerHighlightColor ?? Colors.grey[100]!,
      child: Container(
        width: width,
        height: height,
        color: Colors.white,
      ),
    );
  }

  Widget _buildErrorWidget(dynamic error) {
    return Container(
      width: width,
      height: height,
      color: Colors.grey[200],
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.error_outline,
              color: Colors.grey[400],
              size: 24,
            ),
            const SizedBox(height: 8),
            Text(
              'Failed to load image',
              style: TextStyle(
                color: Colors.grey[600],
                fontSize: 12,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class ImagePreloader {
  static final Map<String, bool> _preloadedImages = {};

  static Future<void> preloadImage(
    BuildContext context,
    String imageUrl, {
    Map<String, String>? headers,
  }) async {
    if (_preloadedImages[imageUrl] == true) return;

    try {
      await precacheImage(
        CachedNetworkImageProvider(
          imageUrl,
          headers: headers,
        ),
        context,
      );
      _preloadedImages[imageUrl] = true;
    } catch (e) {
      _preloadedImages[imageUrl] = false;
    }
  }

  static Future<void> preloadImages(
    BuildContext context,
    List<String> imageUrls, {
    Map<String, String>? headers,
  }) async {
    await Future.wait(
      imageUrls.map(
        (url) => preloadImage(context, url, headers: headers),
      ),
    );
  }

  static void clearPreloadCache() {
    _preloadedImages.clear();
  }

  static bool isPreloaded(String imageUrl) {
    return _preloadedImages[imageUrl] == true;
  }
}

class LazyLoadImageGrid extends StatelessWidget {
  final List<String> imageUrls;
  final int crossAxisCount;
  final double spacing;
  final double childAspectRatio;
  final void Function(int index)? onTap;
  final BorderRadius? borderRadius;
  final Map<String, String>? headers;
  final ScrollController? scrollController;

  const LazyLoadImageGrid({
    super.key,
    required this.imageUrls,
    this.crossAxisCount = 2,
    this.spacing = 8.0,
    this.childAspectRatio = 1.0,
    this.onTap,
    this.borderRadius,
    this.headers,
    this.scrollController,
  });

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      controller: scrollController,
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: crossAxisCount,
        crossAxisSpacing: spacing,
        mainAxisSpacing: spacing,
        childAspectRatio: childAspectRatio,
      ),
      itemCount: imageUrls.length,
      itemBuilder: (context, index) {
        return GestureDetector(
          onTap: onTap != null ? () => onTap!(index) : null,
          child: OptimizedImage(
            imageUrl: imageUrls[index],
            borderRadius: borderRadius,
            headers: headers,
            memCacheWidth: 300, // Adjust based on your needs
            memCacheHeight: 300, // Adjust based on your needs
          ),
        );
      },
    );
  }
}