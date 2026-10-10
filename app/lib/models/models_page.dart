// Models: the ones on this phone, and getting new ones from Hugging Face.
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../engine/liyab_ffi.dart';
import '../state/app_state.dart';
import '../state/models_store.dart';
import 'downloads.dart';
import 'hugging_face.dart';

class ModelsPage extends StatelessWidget {
  const ModelsPage({super.key, required this.app});
  final AppState app;

  @override
  Widget build(BuildContext context) => DefaultTabController(
        length: 2,
        child: Scaffold(
          appBar: AppBar(
            title: Text('Models', style: Theme.of(context).textTheme.titleLarge),
            bottom: const TabBar(tabs: [Tab(text: 'On this phone'), Tab(text: 'Get models')]),
          ),
          body: TabBarView(children: [_LocalTab(app: app), _HubTab(app: app)]),
        ),
      );
}

// ---------------------------------------------------------------------------
// On this phone

class _LocalTab extends StatefulWidget {
  const _LocalTab({required this.app});
  final AppState app;

  @override
  State<_LocalTab> createState() => _LocalTabState();
}

class _LocalTabState extends State<_LocalTab> {
  late Future<List<LocalModel>> _models = widget.app.localModels();
  String? _importing;

  AppState get app => widget.app;

  void _refresh() => setState(() => _models = app.localModels());

  @override
  void initState() {
    super.initState();
    app.downloads.addListener(_refresh);
  }

  @override
  void dispose() {
    app.downloads.removeListener(_refresh);
    super.dispose();
  }

  /// Brings a GGUF picked anywhere (Downloads, a USB drive…) into app storage.
  /// The system picker grants access to that one file: no storage permission.
  /// On Android the picker hands over a copy in the app's cache, which is
  /// renamed into place (same file system, no second copy).
  Future<void> _import() async {
    final List<PlatformFile> picked;
    try {
      picked = await FilePicker.pickFiles(type: FileType.any);
    } on PlatformException catch (e) {
      if (mounted) _snack('Could not open the file picker: ${e.message ?? e.code}');
      return;
    }
    final path = picked.isEmpty ? null : picked.single.path;
    if (path == null) return;
    if (!path.toLowerCase().endsWith('.gguf')) {
      if (mounted) _snack('Choose a .gguf model file.');
      return;
    }
    final name = path.split('/').last;
    setState(() => _importing = name);
    try {
      final dir = await ModelsStore.modelsDir();
      try {
        await File(path).rename('${dir.path}/$name');
      } on FileSystemException {
        final tmp = await File(path).copy('${dir.path}/$name.importing');
        await tmp.rename('${dir.path}/$name');
      }
      if (mounted) _snack('$name is ready.');
    } on FileSystemException catch (e) {
      if (mounted) _snack('Could not import $name: ${e.message}');
    } finally {
      await FilePicker.clearTemporaryFiles();
      if (mounted) setState(() => _importing = null);
      _refresh();
    }
  }

  void _snack(String text) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));

  Future<void> _delete(LocalModel m) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('Delete this model?'),
        content: Text('${m.name} (${formatBytes(m.bytes)}) will be removed from this phone.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('Keep')),
          FilledButton(onPressed: () => Navigator.pop(c, true), child: const Text('Delete')),
        ],
      ),
    );
    if (ok != true) return;
    if (app.modelPath == m.file.path) await app.unload();
    for (final part in modelParts(m.file)) {
      if (part.existsSync()) part.deleteSync();
    }
    _refresh();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ListenableBuilder(
      listenable: app,
      builder: (context, _) => FutureBuilder<List<LocalModel>>(
        future: _models,
        builder: (context, snap) {
          final models = snap.data ?? const [];
          return RefreshIndicator(
            onRefresh: () async => _refresh(),
            child: ListView(padding: const EdgeInsets.fromLTRB(16, 12, 16, 32), children: [
              if (snap.hasData && models.isEmpty)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 24),
                  child: Text('No models on this phone yet. Get one from Hugging Face, or import a GGUF file.',
                      style: theme.textTheme.bodyLarge),
                ),
              for (final m in models)
                _LocalModelCard(
                  model: m,
                  loaded: app.modelPath == m.file.path,
                  busy: app.loading || app.moving != null,
                  moving: app.moving == m.name,
                  budgetMb: app.device.memoryBudgetMb,
                  onLoad: () {
                    app.load(m.file.path);
                    Navigator.of(context).popUntil((r) => r.isFirst);
                  },
                  onMove: () async {
                    await app.moveToAppStorage(m);
                    _refresh();
                  },
                  onDelete: () => _delete(m),
                ),
              const SizedBox(height: 12),
              OutlinedButton.icon(
                onPressed: _importing != null ? null : _import,
                icon: _importing != null
                    ? const SizedBox.square(dimension: 18, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.file_open_outlined),
                label: Text(_importing != null ? 'Importing $_importing…' : 'Import a GGUF file'),
              ),
            ]),
          );
        },
      ),
    );
  }
}

class _LocalModelCard extends StatelessWidget {
  const _LocalModelCard({
    required this.model,
    required this.loaded,
    required this.busy,
    required this.moving,
    required this.budgetMb,
    required this.onLoad,
    required this.onMove,
    required this.onDelete,
  });

  final LocalModel model;
  final bool loaded, busy, moving;
  final int budgetMb;
  final VoidCallback onLoad, onMove, onDelete;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final streams = model.bytes > budgetMb * 1024 * 1024;
    final notes = [
      formatBytes(model.bytes),
      if (streams) 'streams from storage',
      if (model.shared) 'shared folder: slower streaming',
    ];
    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      elevation: 0,
      color: theme.colorScheme.surfaceContainerHighest,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(18),
        side: BorderSide(color: loaded ? theme.colorScheme.primary : theme.colorScheme.outline),
      ),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 14, 8, 8),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(model.name, style: theme.textTheme.bodyLarge?.copyWith(fontWeight: FontWeight.w600)),
          const SizedBox(height: 4),
          Text(moving ? 'Moving to app storage…' : notes.join(', '), style: theme.textTheme.labelSmall),
          const SizedBox(height: 4),
          Row(children: [
            if (loaded)
              Padding(
                padding: const EdgeInsets.only(right: 8),
                child: Text('In use', style: TextStyle(color: theme.colorScheme.primary, fontWeight: FontWeight.w600)),
              )
            else
              TextButton(onPressed: busy ? null : onLoad, child: const Text('Use')),
            if (model.shared)
              TextButton(onPressed: busy || loaded ? null : onMove, child: const Text('Move to app storage')),
            const Spacer(),
            IconButton(tooltip: 'Delete', onPressed: busy ? null : onDelete, icon: const Icon(Icons.delete_outline)),
          ]),
        ]),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Get models

class _HubTab extends StatefulWidget {
  const _HubTab({required this.app});
  final AppState app;

  @override
  State<_HubTab> createState() => _HubTabState();
}

class _HubTabState extends State<_HubTab> {
  final _query = TextEditingController();
  late Future<List<HubRepo>> _results = HuggingFace.search('');

  @override
  void dispose() {
    _query.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final downloads = widget.app.downloads;
    return ListView(padding: const EdgeInsets.fromLTRB(16, 12, 16, 32), children: [
      ListenableBuilder(
        listenable: downloads,
        builder: (context, _) => Column(children: [
          for (final d in downloads.active.values) _DownloadCard(download: d, downloads: downloads),
        ]),
      ),
      TextField(
        controller: _query,
        textInputAction: TextInputAction.search,
        decoration: InputDecoration(
          hintText: 'Search Hugging Face, e.g. Qwen3.5 4B',
          prefixIcon: const Icon(Icons.search_rounded),
          filled: true,
          fillColor: theme.colorScheme.surfaceContainerHighest,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(28), borderSide: BorderSide.none),
        ),
        onSubmitted: (q) => setState(() => _results = HuggingFace.search(q.trim())),
      ),
      const SizedBox(height: 8),
      Text('Only the search and the model files go to the network.', style: theme.textTheme.labelSmall),
      const SizedBox(height: 8),
      FutureBuilder<List<HubRepo>>(
        future: _results,
        builder: (context, snap) {
          if (snap.hasError) {
            return Padding(
              padding: const EdgeInsets.all(16),
              child: Text('Search failed: ${snap.error}. Check the connection and try again.'),
            );
          }
          if (!snap.hasData) return const Padding(padding: EdgeInsets.all(32), child: Center(child: CircularProgressIndicator()));
          return Column(children: [
            for (final r in snap.data!)
              ListTile(
                contentPadding: EdgeInsets.zero,
                title: Text(r.id),
                subtitle: Text('${_count(r.downloads)} downloads, ${_count(r.likes)} likes', style: theme.textTheme.labelSmall),
                trailing: const Icon(Icons.chevron_right_rounded),
                onTap: () => Navigator.push(
                    context, MaterialPageRoute<void>(builder: (_) => _RepoPage(app: widget.app, repo: r.id))),
              ),
          ]);
        },
      ),
    ]);
  }

  static String _count(int n) => n >= 1e6 ? '${(n / 1e6).toStringAsFixed(1)}M' : n >= 1e3 ? '${(n / 1e3).toStringAsFixed(0)}k' : '$n';
}

class _DownloadCard extends StatelessWidget {
  const _DownloadCard({required this.download, required this.downloads});
  final ModelDownload download;
  final Downloads downloads;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final d = download;
    final status = switch (d.phase) {
      DownloadPhase.queued => 'Waiting to start',
      DownloadPhase.running =>
        '${formatBytes(d.done)} of ${formatBytes(d.totalBytes)}${d.speedMBps > 0 ? ', ${d.speedMBps.toStringAsFixed(1)} MB/s' : ''}',
      DownloadPhase.paused => 'Paused at ${formatBytes(d.done)} of ${formatBytes(d.totalBytes)}',
      DownloadPhase.checking => 'Checking the file',
      DownloadPhase.done => 'Downloaded. Find it under On this phone.',
      DownloadPhase.failed => 'Stopped: ${d.error}',
    };
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      elevation: 0,
      color: theme.colorScheme.surfaceContainerHighest,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 14, 8, 8),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(d.name, style: theme.textTheme.bodyLarge?.copyWith(fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          if (d.phase != DownloadPhase.done)
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: d.phase == DownloadPhase.checking ? null : d.fraction,
                minHeight: 6,
              ),
            ),
          const SizedBox(height: 6),
          Text(status, style: theme.textTheme.labelSmall),
          Row(mainAxisAlignment: MainAxisAlignment.end, children: [
            if (d.phase == DownloadPhase.running) TextButton(onPressed: () => downloads.pause(d), child: const Text('Pause')),
            if (d.phase == DownloadPhase.paused || d.phase == DownloadPhase.failed)
              TextButton(onPressed: () => downloads.resume(d), child: const Text('Resume')),
            if (d.phase != DownloadPhase.done && d.phase != DownloadPhase.checking)
              TextButton(onPressed: () => downloads.cancel(d), child: const Text('Cancel')),
            if (d.phase == DownloadPhase.done) TextButton(onPressed: () => downloads.dismiss(d), child: const Text('Done')),
          ]),
        ]),
      ),
    );
  }
}

class _RepoPage extends StatefulWidget {
  const _RepoPage({required this.app, required this.repo});
  final AppState app;
  final String repo;

  @override
  State<_RepoPage> createState() => _RepoPageState();
}

class _RepoPageState extends State<_RepoPage> {
  late final Future<List<HubModel>> _models = HuggingFace.models(widget.repo);
  final Map<String, Future<String?>> _arch = {};

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final archs = LiyabLib.instance.architectures;
    return Scaffold(
      appBar: AppBar(title: Text(widget.repo, style: theme.textTheme.titleLarge, overflow: TextOverflow.ellipsis)),
      body: FutureBuilder<List<HubModel>>(
        future: _models,
        builder: (context, snap) {
          if (snap.hasError) return Center(child: Padding(padding: const EdgeInsets.all(24), child: Text('${snap.error}')));
          if (!snap.hasData) return const Center(child: CircularProgressIndicator());
          final models = snap.data!;
          if (models.isEmpty) return const Center(child: Text('No GGUF models in this repository.'));
          return ListView.separated(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 32),
            itemCount: models.length,
            separatorBuilder: (_, _) => const Divider(height: 1),
            itemBuilder: (context, i) {
              final m = models[i];
              final arch = _arch.putIfAbsent(m.path, () => HuggingFace.architecture(m));
              return ListenableBuilder(
                listenable: widget.app.downloads,
                builder: (context, _) => FutureBuilder<String?>(
                  future: arch,
                  builder: (context, a) {
                    final known = a.data;
                    final runs = m.supported && (known == null || archs.contains(known));
                    final note = !m.supported
                        ? '${m.quant} is not supported'
                        : known == null
                            ? (a.connectionState == ConnectionState.done ? 'architecture unknown' : 'checking…')
                            : archs.contains(known)
                                ? '$known, runs on Liyab'
                                : '$known is not supported yet';
                    final downloading = widget.app.downloads.active.containsKey('${m.repo}/${m.path}');
                    return ListTile(
                      contentPadding: const EdgeInsets.symmetric(vertical: 4),
                      title: Text('${m.quant}  ${formatBytes(m.size)}'),
                      subtitle: Text(
                          '${m.fileName}${m.parts.length > 1 ? ' (${m.parts.length} parts)' : ''}\n$note',
                          style: theme.textTheme.labelSmall),
                      isThreeLine: true,
                      trailing: FilledButton.tonal(
                        onPressed: !runs || downloading ? null : () => widget.app.downloads.download(m),
                        child: Text(downloading ? 'Downloading' : 'Download'),
                      ),
                    );
                  },
                ),
              );
            },
          );
        },
      ),
    );
  }
}
