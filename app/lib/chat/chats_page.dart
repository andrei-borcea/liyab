// Searchable local chat archive; no cloud account or remote history.
import 'dart:async';

import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../state/chat_store.dart';

class ChatsPage extends StatefulWidget {
  const ChatsPage({super.key, required this.app, required this.onOpen});
  final AppState app;
  final VoidCallback onOpen;
  @override
  State<ChatsPage> createState() => _ChatsPageState();
}

class _ChatsPageState extends State<ChatsPage> {
  final _search = TextEditingController();
  late Future<List<SavedChat>> _results;
  Timer? _debounce;
  String _filter = 'all';
  String? _opening;

  @override
  void initState() {
    super.initState();
    widget.app.chats.addListener(_refresh);
    _results = _query();
  }

  Future<List<SavedChat>> _query() => widget.app.chats.list(query: _search.text,
    pinnedOnly: _filter == 'pinned', modelPath: _filter == 'model' ? widget.app.modelPath : null);

  void _refresh() { if (mounted) setState(() { _results = _query(); }); }

  @override
  void dispose() {
    widget.app.chats.removeListener(_refresh);
    _debounce?.cancel();
    _search.dispose();
    super.dispose();
  }

  void _error(Object error) {
    if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$error')));
  }

  Future<void> _open(SavedChat chat) async {
    setState(() => _opening = chat.id);
    try {
      await widget.app.openChat(chat);
      if (mounted) widget.onOpen();
    } catch (error) { _error(error); }
    finally { if (mounted) setState(() => _opening = null); }
  }

  Future<void> _action(SavedChat chat, String action) async {
    try {
      if (action == 'pin') { await widget.app.chats.pin(chat.id, !chat.pinned); return; }
      if (action == 'rename') {
        final input = TextEditingController(text: chat.title);
        final title = await showDialog<String>(context: context, builder: (context) => AlertDialog(
          title: const Text('Rename chat'),
          content: TextField(controller: input, autofocus: true, maxLength: 200,
            onSubmitted: (value) => Navigator.pop(context, value)),
          actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
            FilledButton(onPressed: () => Navigator.pop(context, input.text), child: const Text('Save'))],
        ));
        // Let the dialog's dismissal finish before disposing its text controller.
        if (title != null) await widget.app.chats.rename(chat.id, title);
        await Future<void>.delayed(const Duration(milliseconds: 300));
        input.dispose();
        return;
      }
      final confirmed = await showDialog<bool>(context: context, builder: (context) => AlertDialog(
        title: const Text('Delete chat?'), content: Text('“${chat.title}” will be removed from this device.'),
        actions: [TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Delete'))],
      ));
      if (confirmed == true) {
        if (widget.app.activeChatId == chat.id) widget.app.newChat();
        // New chat queues the final save; delete only after that save finishes.
        await widget.app.deleteChat(chat.id);
      }
    } catch (error) { _error(error); }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Your chats')),
      body: Padding(padding: const EdgeInsets.fromLTRB(24, 12, 24, 0), child: Column(
        crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('Pick up where you left off.', style: theme.textTheme.headlineMedium),
          const SizedBox(height: 6),
          Text('Saved on this device. Search titles and messages.', style: theme.textTheme.labelSmall),
          const SizedBox(height: 24),
          TextField(controller: _search,
            decoration: const InputDecoration(hintText: 'Search conversations', prefixIcon: Icon(Icons.search_rounded)),
            onChanged: (_) { _debounce?.cancel(); _debounce = Timer(const Duration(milliseconds: 200), _refresh); }),
          const SizedBox(height: 12),
          Wrap(spacing: 8, children: [
            for (final entry in {'all': 'All chats', 'pinned': 'Pinned', 'model': 'Current model'}.entries)
              ChoiceChip(label: Text(entry.value), selected: _filter == entry.key,
                onSelected: entry.key == 'model' && widget.app.modelPath == null ? null : (_) {
                  _filter = entry.key; _refresh();
                }),
          ]),
          const SizedBox(height: 20),
          Expanded(child: FutureBuilder<List<SavedChat>>(future: _results, builder: (context, snapshot) {
            if (snapshot.hasError) return Center(child: Text('Could not read chats: ${snapshot.error}'));
            if (!snapshot.hasData) return const Center(child: CircularProgressIndicator());
            final chats = snapshot.data!;
            if (chats.isEmpty) { return Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
              Icon(Icons.forum_outlined, size: 40, color: theme.colorScheme.onSurfaceVariant),
              const SizedBox(height: 12), Text(_search.text.isNotEmpty || _filter != 'all' ? 'No matching conversations' : 'Your first conversation starts here'),
            ])); }
            return ListenableBuilder(listenable: widget.app, builder: (context, _) {
              final busy = widget.app.generating || widget.app.loading || widget.app.switchingChat || _opening != null;
              return ListView.separated(itemCount: chats.length, separatorBuilder: (_, _) => const SizedBox(height: 8),
                itemBuilder: (context, index) {
                  final chat = chats[index];
                  return Material(color: theme.colorScheme.surfaceContainerHighest,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14),
                      side: BorderSide(color: widget.app.activeChatId == chat.id ? theme.colorScheme.primary : theme.colorScheme.outline)),
                    child: ListTile(contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                      leading: Icon(chat.pinned ? Icons.push_pin_rounded : Icons.chat_bubble_outline_rounded,
                        color: chat.pinned ? theme.colorScheme.primary : theme.colorScheme.onSurfaceVariant),
                      title: Text(chat.title, maxLines: 1, overflow: TextOverflow.ellipsis),
                      subtitle: Text('${chat.modelName}\n${MaterialLocalizations.of(context).formatMediumDate(chat.updated.toLocal())}',
                        maxLines: 2, overflow: TextOverflow.ellipsis, style: theme.textTheme.labelSmall),
                      onTap: busy ? null : () => _open(chat),
                      trailing: _opening == chat.id ? const SizedBox.square(dimension: 18, child: CircularProgressIndicator(strokeWidth: 2)) :
                        PopupMenuButton<String>(tooltip: 'Chat actions', enabled: !busy,
                          onSelected: (value) => _action(chat, value), itemBuilder: (_) => [
                            PopupMenuItem(value: 'pin', child: Text(chat.pinned ? 'Unpin' : 'Pin to top')),
                            const PopupMenuItem(value: 'rename', child: Text('Rename')),
                            const PopupMenuItem(value: 'delete', child: Text('Delete')),
                          ]),
                    ),
                  );
                },
              );
            });
          })),
        ],
      )),
    );
  }
}
