// gRPC service liyab.v1.Liyab (app/proto/liyab.proto) over the local API's
// core. The few messages are encoded by hand with protobuf's wire-format
// reader and writer instead of generated classes: no protoc step in the
// build, and the API's own types (ApiRequest, ApiEvent) go straight through.
// Changes to the messages must follow liyab.proto, which clients compile.
import 'dart:async';
import 'package:grpc/grpc.dart';
import 'package:protobuf/protobuf.dart';

import 'api_core.dart';

class LiyabGrpcService extends Service {
  LiyabGrpcService(this.core) {
    $addMethod(ServiceMethod<void, List<String>>(
        'ListModels', (ServiceCall call, Future<void> _) async => _models(), false, false, (_) {}, encodeModels));
    $addMethod(ServiceMethod<ApiRequest, ApiEvent>(
        'Generate', _generate, false, true, decodeRequest, encodeEvent));
    $addMethod(ServiceMethod<ApiRequest, int>(
        'CountTokens', _countTokens, false, false, decodeRequest, encodeCount));
  }

  final ApiCore core;

  @override
  String get $name => 'liyab.v1.Liyab';

  List<String> _models() => [if (core.backend.modelName.isNotEmpty) core.backend.modelName];

  Stream<ApiEvent> _generate(ServiceCall call, Future<ApiRequest> request) async* {
    try {
      yield* core.run(await request).where((e) => e is! ApiStart);
    } on ApiError catch (e) {
      throw _status(e);
    }
  }

  Future<int> _countTokens(ServiceCall call, Future<ApiRequest> request) async {
    try {
      return await core.countTokens(await request);
    } on ApiError catch (e) {
      throw _status(e);
    }
  }

  static GrpcError _status(ApiError e) => switch (e.status) {
        400 || 413 => GrpcError.invalidArgument(e.message),
        401 => GrpcError.unauthenticated(e.message),
        503 => GrpcError.unavailable(e.message),
        _ => GrpcError.internal(e.message),
      };

  /// GenerateRequest.
  static ApiRequest decodeRequest(List<int> bytes) {
    final r = CodedBufferReader(bytes);
    final messages = <ApiMessage>[], stop = <String>[];
    String? prompt;
    double? temperature, topP;
    int? topK, maxTokens;
    var thinking = false;
    while (!r.isAtEnd()) {
      final tag = r.readTag();
      switch (tag >> 3) {
        case 1:
          messages.add(_decodeMessage(r.readBytes()));
        case 2:
          prompt = r.readString();
        case 3:
          temperature = r.readFloat();
        case 4:
          topP = r.readFloat();
        case 5:
          topK = r.readInt32();
        case 6:
          maxTokens = r.readInt32();
        case 7:
          stop.add(r.readString());
        case 8:
          thinking = r.readBool();
        default:
          r.skipField(tag);
      }
    }
    return ApiRequest(
      messages: messages,
      prompt: prompt,
      temperature: temperature,
      topP: topP,
      topK: topK,
      maxTokens: maxTokens,
      stop: stop,
      thinking: thinking,
    );
  }

  static ApiMessage _decodeMessage(List<int> bytes) {
    final r = CodedBufferReader(bytes);
    var role = '', content = '';
    while (!r.isAtEnd()) {
      final tag = r.readTag();
      switch (tag >> 3) {
        case 1:
          role = r.readString();
        case 2:
          content = r.readString();
        default:
          r.skipField(tag);
      }
    }
    return ApiMessage(role, content);
  }

  /// GenerateResponse (text, reasoning or done).
  static List<int> encodeEvent(ApiEvent e) {
    final w = CodedBufferWriter();
    switch (e) {
      case ApiText(:final text):
        w.writeField(1, PbFieldType.OS, text);
      case ApiReasoning(:final text):
        w.writeField(2, PbFieldType.OS, text);
      case ApiDone d:
        final done = CodedBufferWriter()
          ..writeField(1, PbFieldType.O3, d.reason.index)
          ..writeField(2, PbFieldType.OS, d.stopSequence ?? '')
          ..writeField(3, PbFieldType.O3, d.stats.promptTokens)
          ..writeField(4, PbFieldType.O3, d.completionTokens)
          ..writeField(5, PbFieldType.O3, d.stats.cachedPrefixTokens)
          ..writeField(6, PbFieldType.OF, d.stats.tokensPerSecond);
        w.writeField(3, PbFieldType.OY, done.toBuffer());
      case ApiStart():
        break;
    }
    return w.toBuffer();
  }

  /// ListModelsResponse.
  static List<int> encodeModels(List<String> models) {
    final w = CodedBufferWriter();
    for (final m in models) {
      w.writeField(1, PbFieldType.OS, m);
    }
    return w.toBuffer();
  }

  /// CountTokensResponse.
  static List<int> encodeCount(int tokens) => (CodedBufferWriter()..writeField(1, PbFieldType.O3, tokens)).toBuffer();
}

