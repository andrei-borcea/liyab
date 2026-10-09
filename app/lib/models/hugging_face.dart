// Minimal Hugging Face Hub client: GGUF model search, file listing and a
// metadata peek. Only the public, unauthenticated API is used, and only to
// find and fetch models: nothing about the user is sent.
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

const _hub = 'https://huggingface.co';
const _userAgent = 'Liyab/0.2 (Flutter)';

final _split = RegExp(r'-(\d{5})-of-(\d{5})\.gguf$', caseSensitive: false);
final _quant = RegExp(r'(IQ\d_[A-Z]+|Q\d_K(?:_[SML])?|Q\d_\d|TQ\d_\d|BF16|F16|F32|MXFP4|NVFP4)', caseSensitive: false);

class HubRepo {
  const HubRepo(this.id, this.downloads, this.likes);
  final String id;
  final int downloads;
  final int likes;
}

/// One downloadable model: a single GGUF file, or every part of a split one.
class HubModel {
  HubModel(this.repo, this.parts) : size = parts.fold(0, (s, p) => s + p.size);

  final String repo;
  final List<HubFile> parts;
  final int size;

  String get path => parts.first.path;
  String get fileName => parts.first.fileName;

  /// The quantization named in the file name (the last match: the suffix), or '?'.
  String get quant {
    final matches = _quant.allMatches(path).toList();
    return matches.isEmpty ? '?' : matches.last.group(1)!.toUpperCase();
  }

  /// Liyab decodes every tensor format llama.cpp writes except Q1_0.
  bool get supported => quant != 'Q1_0';
}

class HubFile {
  const HubFile(this.repo, this.path, this.size, this.sha256);
  final String repo;
  final String path;
  final int size;
  final String? sha256; // from the LFS pointer

  String get fileName => path.split('/').last;
  Uri get url => Uri.parse('$_hub/$repo/resolve/main/${path.split('/').map(Uri.encodeComponent).join('/')}');
}

class HubException implements Exception {
  HubException(this.message);
  final String message;
  @override
  String toString() => message;
}

abstract final class HuggingFace {
  static final _client = http.Client();

  static Future<Object?> _json(Uri url) async {
    final r = await _client.get(url, headers: {'User-Agent': _userAgent}).timeout(const Duration(seconds: 20));
    if (r.statusCode != 200) throw HubException('Hugging Face answered ${r.statusCode}');
    return jsonDecode(r.body);
  }

  /// GGUF repositories matching `query` (empty: most downloaded), best first.
  static Future<List<HubRepo>> search(String query) async {
    final url = Uri.parse('$_hub/api/models').replace(queryParameters: {
      'filter': 'gguf',
      'sort': 'downloads',
      'direction': '-1',
      'limit': '40',
      if (query.isNotEmpty) 'search': query,
    });
    final list = await _json(url) as List<Object?>;
    return [
      for (final o in list.cast<Map<String, Object?>>())
        HubRepo(o['id'] as String, (o['downloads'] as num?)?.toInt() ?? 0, (o['likes'] as num?)?.toInt() ?? 0)
    ];
  }

  /// The language models of `repo`, smallest first. Split sets become one entry
  /// once every part is listed; vision projectors, imatrix files and MTP heads
  /// are left out.
  static Future<List<HubModel>> models(String repo) async {
    final list = await _json(Uri.parse('$_hub/api/models/$repo/tree/main?recursive=true')) as List<Object?>;
    final singles = <HubModel>[];
    final sets = <String, List<HubFile>>{};
    for (final o in list.cast<Map<String, Object?>>()) {
      final path = o['path'] as String;
      final lower = path.toLowerCase();
      if (o['type'] != 'file' || !lower.endsWith('.gguf')) continue;
      if (lower.contains('mmproj') || lower.contains('imatrix') || lower.startsWith('mtp/') || lower.contains('/mtp-')) {
        continue;
      }
      final lfs = o['lfs'] as Map<String, Object?>?;
      final size = ((lfs?['size'] ?? o['size']) as num?)?.toInt() ?? 0;
      final file = HubFile(repo, path, size, lfs?['oid'] as String?);
      final m = _split.firstMatch(path);
      if (m == null) {
        singles.add(HubModel(repo, [file]));
      } else {
        sets.putIfAbsent(path.substring(0, m.start), () => []).add(file);
      }
    }
    for (final parts in sets.values) {
      parts.sort((a, b) => a.path.compareTo(b.path));
      final count = int.parse(_split.firstMatch(parts.first.path)!.group(2)!);
      if (parts.length == count) singles.add(HubModel(repo, parts));
    }
    singles.sort((a, b) => a.size.compareTo(b.size));
    return singles;
  }

  /// `general.architecture` of a model, read from the first 64 KiB with a
  /// range request (it is among the first metadata keys), so unsupported
  /// models are flagged before downloading gigabytes. Null when unknown.
  static Future<String?> architecture(HubModel model) async {
    try {
      final r = await _client.get(model.parts.first.url,
          headers: {'User-Agent': _userAgent, 'Range': 'bytes=0-65535'}).timeout(const Duration(seconds: 20));
      if (r.statusCode != 206 && r.statusCode != 200) return null;
      return _readArchitecture(r.bodyBytes);
    } on Exception {
      return null;
    }
  }

  static String? _readArchitecture(Uint8List head) {
    final b = ByteData.sublistView(head);
    var at = 0;
    int u32() => b.getUint32((at += 4) - 4, Endian.little);
    int u64() => b.getUint64((at += 8) - 8, Endian.little);
    String str() {
      final n = u64();
      final s = utf8.decode(head.sublist(at, at + n), allowMalformed: true);
      at += n;
      return s;
    }

    const sizes = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8];
    bool skip(int type) {
      if (type == 8) {
        final n = u64();
        if (at + n > head.length) return false;
        at += n;
        return true;
      }
      if (type == 9) {
        final elem = u32(), count = u64();
        for (var i = 0; i < count; i++) {
          if (at + 8 > head.length || !skip(elem)) return false;
        }
        return true;
      }
      if (type < 0 || type >= sizes.length || sizes[type] == 0 || at + sizes[type] > head.length) return false;
      at += sizes[type];
      return true;
    }

    try {
      if (u32() != 0x46554747) return null; // "GGUF"
      u32(); // version
      u64(); // tensor count
      final kvs = u64();
      for (var i = 0; i < kvs && at + 12 < head.length; i++) {
        final key = str();
        final type = u32();
        if (key == 'general.architecture' && type == 8) return str();
        if (!skip(type)) return null;
      }
    } on RangeError {
      return null;
    }
    return null;
  }
}
