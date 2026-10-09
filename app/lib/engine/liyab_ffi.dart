// Dart bindings for Liyab's C ABI (include/liyab/liyab_c_api.h).
//
// Hand-written instead of generated: the API is small, and the struct layouts
// below must match the header field for field (same order, same C types, so
// Dart's natural alignment reproduces the C layout). Keep them in sync when
// the header changes; new fields are only ever appended there.
import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

/// liyab_status: 0 is success.
abstract final class LiyabStatus {
  static const ok = 0;
  static const cancelled = 7;
  static const busy = 9;
}

/// liyab_backend.
abstract final class LiyabBackend {
  static const auto = -1;
  static const cpu = 0;
  static const metal = 1;
  static const vulkan = 2;
}

/// liyab_power_profile.
abstract final class LiyabPowerProfile {
  static const performance = 0;
  static const balanced = 1;
  static const lowPower = 2;
}

final class LiyabEngineConfig extends Struct {
  external Pointer<Utf8> modelPath;
  external Pointer<Utf8> draftModelPath;
  @Int32()
  external int nThreads;
  @Int32()
  external int contextLength;
  @Int32()
  external int slidingWindow;
  @Int32()
  external int kvSinkTokens;
  @Int32()
  external int draftTokens;
  @Int32()
  external int lookupDrafts;
  @Int32()
  external int adaptiveDrafts;
  @Int32()
  external int kvCacheType;
  @Int32()
  external int backend;
  @Int32()
  external int powerProfile;
  @Float()
  external double skinThresholdC;
  @Double()
  external double targetTps;
  @Int32()
  external int thermalPolling;
  @Int32()
  external int tripleBufferLoading;
  @Int64()
  external int expertCacheMb;
  @Int64()
  external int memoryBudgetMb;
  @Int32()
  external int requantBits;
  @Float()
  external double moeExpertMass;
  @Int32()
  external int moeMaxExperts;
  @Int32()
  external int earlyExit;
  @Float()
  external double earlyExitThreshold;
  @Int32()
  external int headPruning;
  @Float()
  external double headKeepRatio;
  @Int32()
  external int egls;
  @Float()
  external double eglsThreshold;
  @Int32()
  external int tdss;
  external Pointer<Utf8> kvDedupDir;
}

final class LiyabSamplingParams extends Struct {
  @Float()
  external double temperature;
  @Int32()
  external int topK;
  @Float()
  external double topP;
  @Uint64()
  external int seed;
  @Int32()
  external int maxTokens;
  @Int32()
  external int addBos;
}

final class LiyabGenerationStats extends Struct {
  @Int32()
  external int promptTokens;
  @Int32()
  external int generatedTokens;
  @Double()
  external double prefillMs;
  @Double()
  external double decodeMs;
  @Double()
  external double tokensPerSecond;
  @Int32()
  external int draftTokensProposed;
  @Int32()
  external int draftTokensAccepted;
  @Int32()
  external int thermalReroutes;
  @Double()
  external double pacedIdleMs;
  @Int32()
  external int cancelled;
  @Int32()
  external int earlyExits;
  @Int32()
  external int earlyExitLayersSkipped;
  @Int32()
  external int headPrunedSteps;
  @Int32()
  external int weightStalls;
  @Double()
  external double weightWaitMs;
  @Uint64()
  external int kvCacheBytes;
  @Int32()
  external int ffnBlocksSkipped;
  @Int32()
  external int sparseFfnSteps;
  @Double()
  external double ttftMs;
  @Int32()
  external int cachedPrefixTokens;
  @Int32()
  external int expertHits;
  @Int32()
  external int expertLate;
  @Int32()
  external int expertMisses;
  @Uint64()
  external int expertBytesRead;
  @Double()
  external double expertStallMs;
  @Int32()
  external int expertUnused;
  @Double()
  external double attentionMs;
  @Double()
  external double deltaNetMs;
  @Double()
  external double routerMs;
  @Double()
  external double expertsMs;
  @Double()
  external double sharedExpertMs;
  @Double()
  external double denseFfnMs;
  @Double()
  external double lmHeadMs;
  @Int32()
  external int expertPredicted;
  @Int32()
  external int expertPredictedUsed;
}

final class LiyabEngineCounters extends Struct {
  @Double()
  external double acceleratorBusyMs;
  @Uint64()
  external int storageBytesRead;
  @Uint64()
  external int tokensGenerated;
}

typedef LiyabTokenCallbackNative = Int32 Function(Pointer<Utf8> piece, Size len, Int32 token, Pointer<Void> user);
typedef LiyabLogCallbackNative = Void Function(Int32 level, Pointer<Utf8> message, Pointer<Void> user);

/// The functions of libliyab. Every call is synchronous; long ones (create,
/// generate, prefill) belong on a background isolate (see EngineService).
final class LiyabLib {
  LiyabLib._(DynamicLibrary lib)
      : version = lib.lookupFunction<Pointer<Utf8> Function(), Pointer<Utf8> Function()>('liyab_version'),
        lastError = lib.lookupFunction<Pointer<Utf8> Function(), Pointer<Utf8> Function()>('liyab_last_error'),
        setLogLevel = lib.lookupFunction<Void Function(Int32), void Function(int)>('liyab_set_log_level'),
        setLogCallback = lib.lookupFunction<
            Void Function(Pointer<NativeFunction<LiyabLogCallbackNative>>, Pointer<Void>),
            void Function(Pointer<NativeFunction<LiyabLogCallbackNative>>, Pointer<Void>)>('liyab_set_log_callback'),
        configDefault = lib.lookupFunction<Void Function(Pointer<LiyabEngineConfig>),
            void Function(Pointer<LiyabEngineConfig>)>('liyab_engine_config_default'),
        samplingDefault = lib.lookupFunction<Void Function(Pointer<LiyabSamplingParams>),
            void Function(Pointer<LiyabSamplingParams>)>('liyab_sampling_params_default'),
        create = lib.lookupFunction<Int32 Function(Pointer<LiyabEngineConfig>, Pointer<Pointer<Void>>),
            int Function(Pointer<LiyabEngineConfig>, Pointer<Pointer<Void>>)>('liyab_engine_create'),
        destroy = lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>('liyab_engine_destroy'),
        generate = lib.lookupFunction<
            Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<LiyabSamplingParams>,
                Pointer<NativeFunction<LiyabTokenCallbackNative>>, Pointer<Void>, Pointer<LiyabGenerationStats>),
            int Function(Pointer<Void>, Pointer<Utf8>, Pointer<LiyabSamplingParams>,
                Pointer<NativeFunction<LiyabTokenCallbackNative>>, Pointer<Void>,
                Pointer<LiyabGenerationStats>)>('liyab_engine_generate'),
        tokenize = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>, Int32, Pointer<Int32>, Int32),
            int Function(Pointer<Void>, Pointer<Utf8>, int, Pointer<Int32>, int)>('liyab_engine_tokenize'),
        cancel = lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>('liyab_engine_cancel'),
        describe = lib.lookupFunction<Size Function(Pointer<Void>, Pointer<Utf8>, Size),
            int Function(Pointer<Void>, Pointer<Utf8>, int)>('liyab_engine_describe'),
        metadata = lib.lookupFunction<Int64 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Size),
            int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, int)>('liyab_engine_metadata'),
        describeDevice = lib.lookupFunction<Size Function(Pointer<Utf8>, Size), int Function(Pointer<Utf8>, int)>(
            'liyab_describe_device'),
        prefill = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>, Int32),
            int Function(Pointer<Void>, Pointer<Utf8>, int)>('liyab_engine_prefill'),
        resetContext =
            lib.lookupFunction<Int32 Function(Pointer<Void>), int Function(Pointer<Void>)>('liyab_engine_reset_context'),
        logBufferEnable = lib.lookupFunction<Void Function(Size), void Function(int)>('liyab_log_buffer_enable'),
        logBufferTake = lib.lookupFunction<Size Function(Pointer<Utf8>, Size), int Function(Pointer<Utf8>, int)>(
            'liyab_log_buffer_take'),
        supportedArchitectures = lib.lookupFunction<Size Function(Pointer<Utf8>, Size), int Function(Pointer<Utf8>, int)>(
            'liyab_supported_architectures'),
        getCounters = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<LiyabEngineCounters>),
            int Function(Pointer<Void>, Pointer<LiyabEngineCounters>)>('liyab_engine_get_counters');

  /// libliyab.so from the APK on Android; linked into the app on iOS.
  static final LiyabLib instance =
      LiyabLib._(Platform.isAndroid ? DynamicLibrary.open('libliyab.so') : DynamicLibrary.process());

  final Pointer<Utf8> Function() version;
  final Pointer<Utf8> Function() lastError;
  final void Function(int) setLogLevel;
  final void Function(Pointer<NativeFunction<LiyabLogCallbackNative>>, Pointer<Void>) setLogCallback;
  final void Function(Pointer<LiyabEngineConfig>) configDefault;
  final void Function(Pointer<LiyabSamplingParams>) samplingDefault;
  final int Function(Pointer<LiyabEngineConfig>, Pointer<Pointer<Void>>) create;
  final void Function(Pointer<Void>) destroy;
  final int Function(Pointer<Void>, Pointer<Utf8>, Pointer<LiyabSamplingParams>,
      Pointer<NativeFunction<LiyabTokenCallbackNative>>, Pointer<Void>, Pointer<LiyabGenerationStats>) generate;
  final int Function(Pointer<Void>, Pointer<Utf8>, int, Pointer<Int32>, int) tokenize;
  final void Function(Pointer<Void>) cancel;
  final int Function(Pointer<Void>, Pointer<Utf8>, int) describe;
  final int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, int) metadata;
  final int Function(Pointer<Utf8>, int) describeDevice;
  final int Function(Pointer<Void>, Pointer<Utf8>, int) prefill;
  final int Function(Pointer<Void>) resetContext;
  final int Function(Pointer<Void>, Pointer<LiyabEngineCounters>) getCounters;
  final int Function(Pointer<Utf8>, int) supportedArchitectures;
  final void Function(int) logBufferEnable;
  final int Function(Pointer<Utf8>, int) logBufferTake;

  /// Engine log lines buffered since the last call ("I message", "W message"…).
  List<String> takeLogLines() {
    final lines = <String>[];
    final buffer = calloc<Uint8>(8192).cast<Utf8>();
    try {
      while (logBufferTake(buffer, 8192) > 0) {
        lines.addAll(buffer.toDartString().split('\n').where((l) => l.isNotEmpty));
      }
    } finally {
      calloc.free(buffer);
    }
    return lines;
  }

  /// GGUF `general.architecture` values this build runs.
  late final Set<String> architectures =
      readText((b, s) => supportedArchitectures(b, s)).split(',').where((a) => a.isNotEmpty).toSet();

  String errorMessage() => lastError().toDartString();

  /// Reads a size-returning text API (describe, metadata) into a Dart string.
  static String readText(int Function(Pointer<Utf8> buffer, int size) call) {
    final needed = call(nullptr, 0);
    if (needed < 0) return '';
    final buffer = calloc<Uint8>(needed + 1).cast<Utf8>();
    try {
      call(buffer, needed + 1);
      return buffer.toDartString();
    } finally {
      calloc.free(buffer);
    }
  }
}
