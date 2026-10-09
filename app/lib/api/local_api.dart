// The local API: other apps on this device use the loaded model through the
// OpenAI API (/v1/chat/completions, /v1/completions, /v1/models), the
// Anthropic Messages API (/v1/messages, /v1/messages/count_tokens) or gRPC
// (liyab.v1.Liyab, see app/proto/liyab.proto), as they would a server.
//
// Off by default. The servers listen on the loopback interface only, so
// nothing outside the device can connect, and every call must carry the
// token shown in Settings (Authorization: Bearer <token>, or x-api-key for
// Anthropic clients): other apps on the device, and web pages the browser
// opens, cannot use the model without it. Requests wait in the engine's queue
// like the user's own messages, one generation at a time.
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:grpc/grpc.dart' as grpc;

import 'anthropic_api.dart';
import 'api_core.dart';
import 'grpc_api.dart';
import 'openai_api.dart';

class LocalApi {
  LocalApi(ApiBackend backend, this.token) : core = ApiCore(backend);

  static const httpPort = 8642;
  static const grpcPort = 8643;

  /// Requests bodies larger than this are refused (a long conversation is a few hundred KB).
  static const maxBodyBytes = 8 << 20;

  final ApiCore core;
  final String token;
  HttpServer? _http;
  grpc.Server? _grpc;

  /// A new random token (32 bytes, URL-safe base64).
  static String newToken() {
    final r = Random.secure();
    return 'liyab-${base64Url.encode(List<int>.generate(32, (_) => r.nextInt(256))).replaceAll('=', '')}';
  }

  /// Starts both servers; port 0 picks free ports (tests). Throws if a port is taken.
  Future<void> start({int http = httpPort, int grpcPort = LocalApi.grpcPort}) async {
    final server = await HttpServer.bind(InternetAddress.loopbackIPv4, http);
    _http = server;
    server.listen((request) => unawaited(_handle(request)));
    final g = grpc.Server.create(services: [LiyabGrpcService(core)], interceptors: [_grpcAuth]);
    try {
      await g.serve(address: InternetAddress.loopbackIPv4, port: grpcPort);
    } catch (_) {
      await server.close(force: true);
      _http = null;
      rethrow;
    }
    _grpc = g;
  }

  int? get boundHttpPort => _http?.port;
  int? get boundGrpcPort => _grpc?.port;

  Future<void> stop() async {
    await _http?.close(force: true);
    await _grpc?.shutdown();
    _http = null;
    _grpc = null;
  }

  /// Constant-time comparison: the time a wrong token takes says nothing about the right one.
  bool authorized(String? presented) {
    if (presented == null) return false;
    final a = utf8.encode(presented), b = utf8.encode(token);
    if (a.length != b.length) return false;
    var diff = 0;
    for (var i = 0; i < a.length; ++i) {
      diff |= a[i] ^ b[i];
    }
    return diff == 0;
  }

  static String? _bearer(String? header) =>
      header != null && header.startsWith('Bearer ') ? header.substring('Bearer '.length).trim() : null;

  FutureOr<grpc.GrpcError?> _grpcAuth(grpc.ServiceCall call, grpc.ServiceMethod method) =>
      authorized(_bearer(call.clientMetadata?['authorization']))
          ? null
          : grpc.GrpcError.unauthenticated('missing or wrong token');

  Future<void> _handle(HttpRequest request) async {
    final path = request.uri.path;
    // Anthropic clients send anthropic-version; their errors and model list have their own shapes.
    final anthropic = path.startsWith('/v1/messages') || request.headers.value('anthropic-version') != null;
    Map<String, Object?> error(int status, String message) =>
        anthropic ? AnthropicApi.error(status, message) : OpenAiApi.error(status, message);
    ApiConnection? conn;
    try {
      final presented = _bearer(request.headers.value('authorization')) ?? request.headers.value('x-api-key');
      final open = path == '/health' || authorized(presented);
      final body = open && request.method == 'POST' ? await _body(request) : const <String, Object?>{};
      conn = await ApiConnection.open(request);
      if (path == '/health') return await conn.json(200, const {});
      if (!open) return await conn.json(401, error(401, 'missing or wrong token'));
      final route = '${request.method} $path';
      if (route == 'GET /v1/models') {
        final model = core.backend.modelName;
        return await conn.json(200, anthropic ? AnthropicApi.models(model) : OpenAiApi.models(model));
      }
      switch (route) {
        case 'POST /v1/chat/completions':
          await OpenAiApi(core).chat(body, conn);
        case 'POST /v1/completions':
          await OpenAiApi(core).completion(body, conn);
        case 'POST /v1/messages':
          await AnthropicApi(core).messages(body, conn);
        case 'POST /v1/messages/count_tokens':
          await AnthropicApi(core).countTokens(body, conn);
        default:
          await conn.json(404, error(404, 'no such endpoint: $route'));
      }
    } on ApiError catch (e) {
      await _fail(request, conn, e.status, error(e.status, e.message));
    } on TypeError {
      await _fail(request, conn, 400, error(400, 'a field of the request has the wrong type'));
    } catch (e) {
      await _fail(request, conn, 500, error(500, '$e'));
    }
  }

  /// An error status, unless a stream has started (then the connection just closes).
  static Future<void> _fail(HttpRequest request, ApiConnection? conn, int status, Object body) async {
    try {
      if (conn == null) {
        request.response
          ..statusCode = status
          ..headers.contentType = ContentType.json
          ..write(jsonEncode(body));
        await request.response.close();
      } else if (!conn.started) {
        await conn.json(status, body);
      } else {
        await conn.close();
      }
    } catch (_) {}
  }

  static Future<Map<String, Object?>> _body(HttpRequest request) async {
    final bytes = <int>[];
    await for (final chunk in request) {
      bytes.addAll(chunk);
      if (bytes.length > maxBodyBytes) throw const ApiError(413, 'request body too large');
    }
    try {
      final body = jsonDecode(utf8.decode(bytes));
      if (body is Map<String, Object?>) return body;
    } on FormatException {
      // below
    }
    throw const ApiError(400, 'the body must be a JSON object');
  }
}

/// One response, written straight to the connection's socket.
///
/// dart:io's HttpResponse never reports a client that went away: a flush to
/// a closed connection waits forever, and the generation behind it would run
/// to the end. With the socket detached, its read side ends the moment the
/// client closes, which `gone` reports, and replies stop at once, also while
/// a request still waits in the engine's queue or its prompt is processed.
/// Responses end by closing the connection (Connection: close), which also
/// delimits a stream's body.
class ApiConnection {
  ApiConnection._(this._socket) {
    void lost([Object? _]) {
      if (!_gone.isCompleted) _gone.complete();
    }

    _socket.listen((_) {}, onDone: lost, onError: lost, cancelOnError: true);
    _socket.done.then(lost, onError: lost);
  }

  /// Takes over the connection of `request`, whose body must have been read.
  static Future<ApiConnection> open(HttpRequest request) async =>
      ApiConnection._(await request.response.detachSocket(writeHeaders: false));

  final Socket _socket;
  final _gone = Completer<void>();
  bool started = false; // the status line is out

  /// Completes when the client closes the connection (or the response ends).
  Future<void> get gone => _gone.future;

  void _head(int status, Map<String, String> headers) {
    started = true;
    final h = StringBuffer('HTTP/1.1 $status ${_reasons[status] ?? 'Error'}\r\n');
    headers.forEach((k, v) => h.write('$k: $v\r\n'));
    h.write('connection: close\r\n\r\n');
    _write(h.toString());
  }

  void _write(String text) {
    if (!_gone.isCompleted) _socket.add(utf8.encode(text));
  }

  Future<void> json(int status, Object body) {
    final bytes = utf8.encode(jsonEncode(body));
    _head(status, {'content-type': 'application/json', 'content-length': '${bytes.length}'});
    if (!_gone.isCompleted) _socket.add(bytes);
    return close();
  }

  /// One server-sent event; the first opens the stream. Does nothing once the client is gone.
  Future<void> send(Object? data, {String? event}) async {
    if (!started) _head(200, {'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache'});
    _write('${event != null ? 'event: $event\n' : ''}data: ${data is String ? data : jsonEncode(data)}\n\n');
  }

  Future<void> close() async {
    try {
      await _socket.close();
    } catch (_) {}
  }

  static const _reasons = {
    200: 'OK',
    400: 'Bad Request',
    401: 'Unauthorized',
    404: 'Not Found',
    413: 'Payload Too Large',
    500: 'Internal Server Error',
    503: 'Service Unavailable',
  };
}

/// Runs `r` and waits for its first event, so that a refused request (no
/// model, a prompt too long, a bad role) is answered with an error status
/// before any streamed byte. The run stops when the client goes away.
Future<StreamIterator<ApiEvent>> startRun(ApiCore core, ApiRequest r, ApiConnection conn) async {
  final it = StreamIterator(core.run(r));
  unawaited(conn.gone.then((_) => it.cancel()));
  try {
    if (!await it.moveNext()) throw const ApiError(500, 'the request ended before it started');
  } catch (_) {
    await it.cancel();
    rethrow;
  }
  return it;
}

/// A random id for a response ("chatcmpl-…", "msg_…").
String responseId(String prefix) {
  final r = Random();
  return prefix + List.generate(24, (_) => r.nextInt(16).toRadixString(16)).join();
}

/// The text of a message's content: a string or a list of text parts; other parts are refused.
String textContent(Object? content, {Set<String> skip = const {}}) {
  if (content == null) return '';
  if (content is String) return content;
  if (content is List) {
    final b = StringBuffer();
    for (final part in content) {
      if (part is! Map) throw const ApiError(400, 'unsupported content part');
      final type = part['type'];
      if (skip.contains(type)) continue;
      if (type != 'text') throw ApiError(400, 'unsupported content type: $type (text only)');
      b.write(part['text'] as String? ?? '');
    }
    return b.toString();
  }
  throw const ApiError(400, 'content must be a string or a list of parts');
}

List<String> stopList(Object? stop) => switch (stop) {
      null => const [],
      String s => [s],
      List l => [for (final s in l) '$s'],
      _ => throw const ApiError(400, 'stop must be a string or a list of strings'),
    };

double? numberField(Map<String, Object?> body, String key) => (body[key] as num?)?.toDouble();
int? intField(Map<String, Object?> body, String key) => (body[key] as num?)?.toInt();
