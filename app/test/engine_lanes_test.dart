// The engine's lanes (LoadOptions.lanes): a generation on the API's lane runs
// alongside one on the main lane (the engine decodes the two together), each
// with the output it gives alone; stopping one leaves the other alone; and
// the lanes close with the model. Runs the engine built on this host
// (LIYAB_LIBRARY, e.g. build/libliyab.dylib) with a real model
// (LIYAB_TEST_MODEL, a GGUF); skipped without them.
import 'dart:async';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:liyab/engine/engine_service.dart';
import 'package:liyab/engine/liyab_ffi.dart';

final _library = Platform.environment['LIYAB_LIBRARY'];
final _model = Platform.environment['LIYAB_TEST_MODEL'];
final String? _skip = _library == null || !File(_library!).existsSync()
    ? 'LIYAB_LIBRARY does not name the engine built on this host'
    : _model == null || !File(_model!).existsSync()
        ? 'LIYAB_TEST_MODEL does not name a GGUF'
        : null;

const _greedy = SamplingOptions(temperature: 0, topK: 1, topP: 1, maxTokens: 24);

Future<String> _reply(Generation g) async {
  final out = StringBuffer();
  await for (final e in g.events) {
    if (e is TextPiece) out.write(e.text);
  }
  return out.toString();
}

LoadOptions _options({Map<String, int> lanes = const {}}) => LoadOptions(
    modelPath: _model!,
    powerProfile: LiyabPowerProfile.performance,
    memoryBudgetMb: 0,
    contextLength: 2048,
    lanes: lanes);

void main() {
  const chat = 'The capital of France is';
  const api = 'List three colors of the rainbow:';

  test('a generation on the API lane runs alongside the chat, each as alone', () async {
    final service = await EngineService.start();
    // Alone: one after the other on the main lane, a fresh context each.
    await service.load(_options());
    expect(service.hasLane('api'), isFalse);
    final chatAlone = await _reply(service.generate(chat, _greedy));
    await service.unload();
    await service.load(_options());
    final apiAlone = await _reply(service.generate(api, _greedy, lane: 'api')); // no lane: the main one
    // Together: the API's lane open.
    await service.load(_options(lanes: {'api': LiyabSequencePriority.background}));
    expect(service.hasLane('api'), isTrue);
    final results = await Future.wait([
      _reply(service.generate(chat, _greedy)),
      _reply(service.generate(api, _greedy, lane: 'api')),
    ]);
    expect(results[0], chatAlone);
    expect(results[1], apiAlone);
    // Stopping the lane's generation stops it only.
    final long = const SamplingOptions(temperature: 0, topK: 1, topP: 1, maxTokens: 200);
    final apiRun = service.generate(api, long, lane: 'api');
    var pieces = 0;
    final stopped = Completer<void>();
    apiRun.events.listen((e) {
      if (e is TextPiece && ++pieces == 3) apiRun.stop();
      if (e is GenerationDone) {
        expect(e.stats.cancelled, isTrue);
        stopped.complete();
      }
    });
    final chatAgain = await _reply(service.generate(chat, _greedy));
    await stopped.future;
    expect(chatAgain, chatAlone);
    // The lanes close with the model.
    await service.unload();
    expect(service.hasLane('api'), isFalse);
    service.dispose();
  }, skip: _skip, timeout: const Timeout(Duration(minutes: 5)));
}
