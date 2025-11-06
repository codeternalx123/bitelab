import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class DebouncedSearchField extends ConsumerStatefulWidget {
  final Duration debounceTime;
  final String hintText;
  final Function(String) onSearch;
  final TextEditingController? controller;
  final bool autofocus;
  final InputDecoration? decoration;
  final TextStyle? style;
  final TextInputType? keyboardType;
  final bool enabled;
  final VoidCallback? onClear;
  final Widget? prefix;
  final Widget? suffix;
  final FocusNode? focusNode;

  const DebouncedSearchField({
    super.key,
    this.debounceTime = const Duration(milliseconds: 500),
    this.hintText = 'Search...',
    required this.onSearch,
    this.controller,
    this.autofocus = false,
    this.decoration,
    this.style,
    this.keyboardType,
    this.enabled = true,
    this.onClear,
    this.prefix,
    this.suffix,
    this.focusNode,
  });

  @override
  DebouncedSearchFieldState createState() => DebouncedSearchFieldState();
}

class DebouncedSearchFieldState extends ConsumerState<DebouncedSearchField> {
  Timer? _debounce;
  late TextEditingController _controller;
  late FocusNode _focusNode;

  @override
  void initState() {
    super.initState();
    _controller = widget.controller ?? TextEditingController();
    _focusNode = widget.focusNode ?? FocusNode();
  }

  @override
  void dispose() {
    _debounce?.cancel();
    if (widget.controller == null) {
      _controller.dispose();
    }
    if (widget.focusNode == null) {
      _focusNode.dispose();
    }
    super.dispose();
  }

  void _onSearchChanged(String query) {
    if (_debounce?.isActive ?? false) {
      _debounce!.cancel();
    }

    _debounce = Timer(widget.debounceTime, () {
      widget.onSearch(query);
    });
  }

  void _clearSearch() {
    _controller.clear();
    widget.onSearch('');
    widget.onClear?.call();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return TextField(
      controller: _controller,
      focusNode: _focusNode,
      onChanged: _onSearchChanged,
      autofocus: widget.autofocus,
      enabled: widget.enabled,
      style: widget.style ?? theme.textTheme.bodyMedium,
      keyboardType: widget.keyboardType,
      decoration: widget.decoration ??
          InputDecoration(
            hintText: widget.hintText,
            filled: true,
            fillColor: theme.cardColor,
            prefixIcon: widget.prefix ??
                Icon(
                  Icons.search,
                  color: theme.hintColor,
                ),
            suffixIcon: _controller.text.isNotEmpty
                ? IconButton(
                    icon: widget.suffix ??
                        Icon(
                          Icons.clear,
                          color: theme.hintColor,
                        ),
                    onPressed: _clearSearch,
                  )
                : null,
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
              borderSide: BorderSide.none,
            ),
            contentPadding: const EdgeInsets.symmetric(
              horizontal: 16,
              vertical: 12,
            ),
          ),
    );
  }
}

class SearchHistory extends ChangeNotifier {
  static const _maxHistoryItems = 10;
  final Set<String> _history = {};

  List<String> get items => _history.toList();

  void add(String query) {
    if (query.trim().isEmpty) return;

    _history.remove(query);
    _history.add(query);

    if (_history.length > _maxHistoryItems) {
      _history.remove(_history.first);
    }

    notifyListeners();
  }

  void remove(String query) {
    _history.remove(query);
    notifyListeners();
  }

  void clear() {
    _history.clear();
    notifyListeners();
  }
}

class SearchSuggestions extends ConsumerStatefulWidget {
  final List<String> searchHistory;
  final List<String> suggestions;
  final Function(String) onSuggestionTap;
  final Function(String)? onHistoryItemTap;
  final Function(String)? onHistoryItemRemove;
  final VoidCallback? onClearHistory;

  const SearchSuggestions({
    super.key,
    required this.searchHistory,
    required this.suggestions,
    required this.onSuggestionTap,
    this.onHistoryItemTap,
    this.onHistoryItemRemove,
    this.onClearHistory,
  });

  @override
  SearchSuggestionsState createState() => SearchSuggestionsState();
}

class SearchSuggestionsState extends ConsumerState<SearchSuggestions> {
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return ListView(
      padding: EdgeInsets.zero,
      shrinkWrap: true,
      physics: const ClampingScrollPhysics(),
      children: [
        if (widget.searchHistory.isNotEmpty) ...[
          _buildSectionHeader(
            context,
            'Recent Searches',
            onClear: widget.onClearHistory,
          ),
          ...widget.searchHistory.map(
            (item) => _buildHistoryItem(
              context,
              item,
              onTap: () => widget.onHistoryItemTap?.call(item),
              onRemove: () => widget.onHistoryItemRemove?.call(item),
            ),
          ),
          const Divider(),
        ],
        if (widget.suggestions.isNotEmpty) ...[
          _buildSectionHeader(context, 'Suggestions'),
          ...widget.suggestions.map(
            (suggestion) => _buildSuggestionItem(
              context,
              suggestion,
              onTap: () => widget.onSuggestionTap(suggestion),
            ),
          ),
        ],
      ],
    );
  }

  Widget _buildSectionHeader(
    BuildContext context,
    String title, {
    VoidCallback? onClear,
  }) {
    final theme = Theme.of(context);

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            title,
            style: theme.textTheme.titleSmall?.copyWith(
              color: theme.hintColor,
            ),
          ),
          if (onClear != null)
            TextButton(
              onPressed: onClear,
              child: Text(
                'Clear',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.primary,
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildHistoryItem(
    BuildContext context,
    String item, {
    VoidCallback? onTap,
    VoidCallback? onRemove,
  }) {
    final theme = Theme.of(context);

    return ListTile(
      leading: const Icon(Icons.history),
      title: Text(item),
      trailing: IconButton(
        icon: const Icon(Icons.close),
        onPressed: onRemove,
      ),
      onTap: onTap,
    );
  }

  Widget _buildSuggestionItem(
    BuildContext context,
    String suggestion, {
    VoidCallback? onTap,
  }) {
    final theme = Theme.of(context);

    return ListTile(
      leading: const Icon(Icons.search),
      title: Text(suggestion),
      onTap: onTap,
    );
  }
}