// Dart bindings for Liyab's C ABI (include/liyab/liyab_c_api.h).
//
// Hand-written instead of generated: the API is small, and the struct layouts
// below must match the header field for field (same order, same C types, so
// Dart's natural alignment reproduces the C layout). Keep them in sync when
// the header changes; new fields are only ever appended there.
import 'dart:convert';
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

/// liyab_sequence_priority: the user's turn first; background work yields.
abstract final class LiyabSequencePriority {
  static const foreground = 0;
  static const background = 1;
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
  external Pointer<Utf8> kvDedupDir; // retired (KV dedup, replaced by the prefix cache's store): kept for the layout
  @Float()
  external double moeSkipSlow;
  @Int64()
  external int prefixCacheMb;
  external Pointer<Utf8> prefixCacheDir;
  @Int64()
  external int prefixCacheDiskMb;
  external Pointer<Utf8> cacheDir;
  @Int32()
  external int suffixDrafts;
  @Int32()
  external int poolingKvCacheType;
  @Int32()
  external int segmentReuse; // segment reuse (lossy, opt-in per request); 0: off
  @Float()
  external double segmentReuseMaxKl;
  @Int32()
  external int prefixCacheLossy;
  @Int64()
  external int adapterMemoryMb; // LoRA adapters: RAM at most for the loaded ones (MiB); 0: off
  @Int32()
  external int draftTree; // suffix drafts verified as a tree (nonzero, the default) or as their best branch
  @Int32()
  external int maxSequences; // sequences kept at once, the default one included (liyab_sequence_open); 1: one
  @Int32()
  external int i8mmTiles; // experimental: Q4_K / Q5_K / Q6_K CPU matmuls over several rows as 2x2 SMMLA tiles
}

/// liyab_adapter_info: one loaded LoRA adapter (liyab_engine_adapter_info).
final class LiyabAdapterInfo extends Struct {
  @Array(128)
  external Array<Uint8> name; // NUL-terminated UTF-8
  @Float()
  external double alpha;
  @Int32()
  external int rank;
  @Int32()
  external int matrices;
  @Uint64()
  external int bytes;
  @Int32()
  external int resident;
  @Int32()
  external int active;
  @Float()
  external double scale;

  String get nameText {
    final bytes = <int>[];
    for (var i = 0; i < 128 && name[i] != 0; ++i) {
      bytes.add(name[i]);
    }
    return utf8.decode(bytes, allowMalformed: true);
  }
}

/// liyab_pooling: what a model does with its final hidden states (liyab_engine_pooling).
abstract final class LiyabPooling {
  static const auto = -1;
  static const none = 0; // a generative model
  static const mean = 1;
  static const cls = 2;
  static const last = 3; // embedding models such as Qwen3-Embedding
  static const rank = 4; // rerankers such as Qwen3-Reranker
}

final class LiyabEmbedParams extends Struct {
  @Int32()
  external int pooling;
  @Int32()
  external int dimensions;
  @Int32()
  external int normalize;
  @Int32()
  external int truncate;
}

final class LiyabRerankParams extends Struct {
  external Pointer<Utf8> promptTemplate;
  external Pointer<Pointer<Utf8>> labels;
  @Size()
  external int nLabels;
}

final class LiyabPoolingStats extends Struct {
  @Int32()
  external int tokens;
  @Int32()
  external int reusedTokens;
  @Double()
  external double ms;
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
  @Int32()
  external int expertDropped;
  @Int32()
  external int expertSkipped;
  @Int32()
  external int forcedTokens;
  @Double()
  external double forcedMs;
  @Double()
  external double contextMs;
  @Double()
  external double ttftStallMs;
  @Int32()
  external int segmentReusedTokens;
  @Int32()
  external int segmentRecomputedTokens;
  @Double()
  external double segmentMs;
}

final class LiyabMemoryPlan extends Struct {
  @Uint64()
  external int residentBytes;
  @Uint64()
  external int expertBytes;
  @Uint64()
  external int expertCacheBytes;
  @Uint64()
  external int recommendedBytes;
  @Int32()
  external int requantBits;
}

final class LiyabWeightLayout extends Struct {
  @Uint64()
  external int fileBytes;
  @Uint64()
  external int streamedBytes;
  @Int32()
  external int streamedBlocks;
  @Int32()
  external int totalBlocks;
  @Int32()
  external int mmapStreaming;
}

final class LiyabResourceSnapshot extends Struct {
  @Uint64()
  external int availableBytes;
  @Uint64()
  external int totalBytes;
  @Float()
  external double cpuFraction;
  @Float()
  external double thermalPressure;
  @Float()
  external double slowdown;
  @Int32()
  external int throttled;
}

final class LiyabEngineCounters extends Struct {
  @Double()
  external double acceleratorBusyMs;
  @Uint64()
  external int storageBytesRead;
  @Uint64()
  external int tokensGenerated;
}

/// liyab_grammar_stats: the cost of constrained decoding in one generation.
final class LiyabGrammarStats extends Struct {
  @Int32()
  external int constrainedTokens;
  @Double()
  external double grammarMs;
}

typedef LiyabTokenCallbackNative = Int32 Function(Pointer<Utf8> piece, Size len, Int32 token, Pointer<Void> user);
typedef LiyabContinuationCallbackNative = Size Function(Pointer<Utf8> generated, Size len, Pointer<Uint8> out, Size cap,
    Pointer<Pointer<Void>> grammar, Pointer<Void> user);
typedef LiyabForceCallbackNative = Size Function(
    Pointer<Utf8> generated, Size len, Pointer<Uint8> out, Size cap, Pointer<Void> user);
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
        generateForced = lib.lookupFunction<
            Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<LiyabSamplingParams>,
                Pointer<NativeFunction<LiyabTokenCallbackNative>>, Pointer<NativeFunction<LiyabForceCallbackNative>>,
                Pointer<Void>, Pointer<LiyabGenerationStats>),
            int Function(Pointer<Void>, Pointer<Utf8>, Pointer<LiyabSamplingParams>,
                Pointer<NativeFunction<LiyabTokenCallbackNative>>, Pointer<NativeFunction<LiyabForceCallbackNative>>,
                Pointer<Void>, Pointer<LiyabGenerationStats>)>('liyab_engine_generate_forced'),
        tokenize = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>, Int32, Pointer<Int32>, Int32),
            int Function(Pointer<Void>, Pointer<Utf8>, int, Pointer<Int32>, int)>('liyab_engine_tokenize'),
        cancel = lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>('liyab_engine_cancel'),
        cancelPrefill = lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>(
            'liyab_engine_cancel_prefill'),
        setPowerProfile = lib.lookupFunction<Int32 Function(Pointer<Void>, Int32), int Function(Pointer<Void>, int)>(
            'liyab_engine_set_power_profile'),
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
        saveState = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>), int Function(Pointer<Void>, Pointer<Utf8>)>(
            'liyab_engine_save_state'),
        saveStateOf = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Uint8>, Size),
            int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Uint8>, int)>('liyab_engine_save_state_of'),
        loadState = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>),
            int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>)>('liyab_engine_load_state'),
        trimMemory = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Uint64>),
            int Function(Pointer<Void>, Pointer<Uint64>)>('liyab_engine_trim_memory'),
        warmMemory = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Uint32>),
            int Function(Pointer<Void>, Pointer<Uint32>)>('liyab_engine_warm_memory'),
        saveExpertProfile = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>),
            int Function(Pointer<Void>, Pointer<Utf8>)>('liyab_engine_save_expert_profile'),
        loadExpertProfile = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>),
            int Function(Pointer<Void>, Pointer<Utf8>)>('liyab_engine_load_expert_profile'),
        contextLength = lib.lookupFunction<Int32 Function(Pointer<Void>), int Function(Pointer<Void>)>(
            'liyab_engine_context_length'),
        memoryPlan = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<LiyabMemoryPlan>),
            int Function(Pointer<Void>, Pointer<LiyabMemoryPlan>)>('liyab_engine_memory_plan'),
        weightLayout = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<LiyabWeightLayout>),
            int Function(Pointer<Void>, Pointer<LiyabWeightLayout>)>('liyab_engine_weight_layout'),
        supportedArchitectures = lib.lookupFunction<Size Function(Pointer<Utf8>, Size), int Function(Pointer<Utf8>, int)>(
            'liyab_supported_architectures'),
        getResourceSnapshot = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<LiyabResourceSnapshot>), int Function(Pointer<Void>, Pointer<LiyabResourceSnapshot>)>('liyab_get_resource_snapshot'),
        getCounters = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<LiyabEngineCounters>),
            int Function(Pointer<Void>, Pointer<LiyabEngineCounters>)>('liyab_engine_get_counters'),
        renderChat = lib.lookupFunction<Int64 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Size),
            int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, int)>('liyab_engine_render_chat'),
        chatFormat = lib.lookupFunction<Int64 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Size),
            int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, int)>('liyab_engine_chat_format'),
        grammarFromJsonSchema = lib.lookupFunction<Int32 Function(Pointer<Utf8>, Int32, Pointer<Pointer<Void>>),
            int Function(Pointer<Utf8>, int, Pointer<Pointer<Void>>)>('liyab_grammar_from_json_schema'),
        grammarParse = lib.lookupFunction<Int32 Function(Pointer<Utf8>, Pointer<Pointer<Void>>),
            int Function(Pointer<Utf8>, Pointer<Pointer<Void>>)>('liyab_grammar_parse'),
        grammarDestroy =
            lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>('liyab_grammar_destroy'),
        grammarMatches = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>, Size),
            int Function(Pointer<Void>, Pointer<Utf8>, int)>('liyab_grammar_matches'),
        generateStructured = lib.lookupFunction<
            Int32 Function(
                Pointer<Void>,
                Pointer<Utf8>,
                Pointer<LiyabSamplingParams>,
                Pointer<NativeFunction<LiyabTokenCallbackNative>>,
                Pointer<Void>,
                Pointer<NativeFunction<LiyabContinuationCallbackNative>>,
                Pointer<Void>,
                Pointer<LiyabGenerationStats>,
                Pointer<LiyabGrammarStats>),
            int Function(
                Pointer<Void>,
                Pointer<Utf8>,
                Pointer<LiyabSamplingParams>,
                Pointer<NativeFunction<LiyabTokenCallbackNative>>,
                Pointer<Void>,
                Pointer<NativeFunction<LiyabContinuationCallbackNative>>,
                Pointer<Void>,
                Pointer<LiyabGenerationStats>,
                Pointer<LiyabGrammarStats>)>('liyab_engine_generate_structured'),
        embedParamsDefault = lib.lookupFunction<Void Function(Pointer<LiyabEmbedParams>),
            void Function(Pointer<LiyabEmbedParams>)>('liyab_embed_params_default'),
        rerankParamsDefault = lib.lookupFunction<Void Function(Pointer<LiyabRerankParams>),
            void Function(Pointer<LiyabRerankParams>)>('liyab_rerank_params_default'),
        pooling = lib.lookupFunction<Int32 Function(Pointer<Void>), int Function(Pointer<Void>)>('liyab_engine_pooling'),
        embeddingSize = lib.lookupFunction<Int32 Function(Pointer<Void>), int Function(Pointer<Void>)>(
            'liyab_engine_embedding_size'),
        embed = lib.lookupFunction<
            Int32 Function(Pointer<Void>, Pointer<Pointer<Utf8>>, Pointer<Size>, Size, Pointer<LiyabEmbedParams>,
                Pointer<Float>, Size, Pointer<LiyabPoolingStats>),
            int Function(Pointer<Void>, Pointer<Pointer<Utf8>>, Pointer<Size>, int, Pointer<LiyabEmbedParams>,
                Pointer<Float>, int, Pointer<LiyabPoolingStats>)>('liyab_engine_embed'),
        rerank = lib.lookupFunction<
            Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Pointer<Utf8>>, Pointer<Size>, Size,
                Pointer<LiyabRerankParams>, Pointer<Float>, Pointer<LiyabPoolingStats>),
            int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Pointer<Utf8>>, Pointer<Size>, int,
                Pointer<LiyabRerankParams>, Pointer<Float>, Pointer<LiyabPoolingStats>)>('liyab_engine_rerank'),
        loadAdapter = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
            int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>)>('liyab_engine_load_adapter'),
        unloadAdapter = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>),
            int Function(Pointer<Void>, Pointer<Utf8>)>('liyab_engine_unload_adapter'),
        setAdapter = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>, Float),
            int Function(Pointer<Void>, Pointer<Utf8>, double)>('liyab_engine_set_adapter'),
        adapterCount = lib.lookupFunction<Int32 Function(Pointer<Void>), int Function(Pointer<Void>)>(
            'liyab_engine_adapter_count'),
        adapterInfo = lib.lookupFunction<Int32 Function(Pointer<Void>, Int32, Pointer<LiyabAdapterInfo>),
            int Function(Pointer<Void>, int, Pointer<LiyabAdapterInfo>)>('liyab_engine_adapter_info'),
        sequenceOpen = lib.lookupFunction<Int32 Function(Pointer<Void>, Int32, Pointer<Pointer<Void>>),
            int Function(Pointer<Void>, int, Pointer<Pointer<Void>>)>('liyab_sequence_open'),
        sequenceClose =
            lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>('liyab_sequence_close'),
        sequenceCancel =
            lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>('liyab_sequence_cancel'),
        sequenceSetPriority = lib.lookupFunction<Int32 Function(Pointer<Void>, Int32), int Function(Pointer<Void>, int)>(
            'liyab_sequence_set_priority'),
        sequenceSetAdapter = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>, Float),
            int Function(Pointer<Void>, Pointer<Utf8>, double)>('liyab_sequence_set_adapter'),
        sequenceGenerate = lib.lookupFunction<
            Int32 Function(
                Pointer<Void>,
                Pointer<Utf8>,
                Pointer<LiyabSamplingParams>,
                Pointer<NativeFunction<LiyabTokenCallbackNative>>,
                Pointer<NativeFunction<LiyabForceCallbackNative>>,
                Pointer<Void>,
                Pointer<NativeFunction<LiyabContinuationCallbackNative>>,
                Pointer<Void>,
                Pointer<LiyabGenerationStats>,
                Pointer<LiyabGrammarStats>),
            int Function(
                Pointer<Void>,
                Pointer<Utf8>,
                Pointer<LiyabSamplingParams>,
                Pointer<NativeFunction<LiyabTokenCallbackNative>>,
                Pointer<NativeFunction<LiyabForceCallbackNative>>,
                Pointer<Void>,
                Pointer<NativeFunction<LiyabContinuationCallbackNative>>,
                Pointer<Void>,
                Pointer<LiyabGenerationStats>,
                Pointer<LiyabGrammarStats>)>('liyab_sequence_generate');

  /// libliyab.so from the APK on Android; linked into the app on iOS and macOS; the library LIYAB_LIBRARY names
  /// when set (tests on the host: build/libliyab.dylib).
  static final LiyabLib instance = LiyabLib._(Platform.environment['LIYAB_LIBRARY'] != null
      ? DynamicLibrary.open(Platform.environment['LIYAB_LIBRARY']!)
      : Platform.isAndroid
          ? DynamicLibrary.open('libliyab.so')
          : Platform.isWindows ? DynamicLibrary.open('liyab.dll')
          : Platform.isLinux ? DynamicLibrary.open('libliyab.so')
          : DynamicLibrary.process());

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
  final int Function(Pointer<Void>, Pointer<Utf8>, Pointer<LiyabSamplingParams>,
      Pointer<NativeFunction<LiyabTokenCallbackNative>>, Pointer<NativeFunction<LiyabForceCallbackNative>>,
      Pointer<Void>, Pointer<LiyabGenerationStats>) generateForced;
  final int Function(Pointer<Void>, Pointer<Utf8>, int, Pointer<Int32>, int) tokenize;
  final void Function(Pointer<Void>) cancel;
  final void Function(Pointer<Void>) cancelPrefill;
  final int Function(Pointer<Void>, int) setPowerProfile;
  final int Function(Pointer<Void>, Pointer<Utf8>, int) describe;
  final int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, int) metadata;
  final int Function(Pointer<Utf8>, int) describeDevice;
  final int Function(Pointer<Void>, Pointer<Utf8>, int) prefill;
  final int Function(Pointer<Void>) resetContext;
  final int Function(Pointer<Void>, Pointer<LiyabResourceSnapshot>) getResourceSnapshot;
  final int Function(Pointer<Void>, Pointer<LiyabEngineCounters>) getCounters;
  final int Function(Pointer<Utf8>, int) supportedArchitectures;
  final int Function(Pointer<Void>, Pointer<Utf8>) saveState;
  final int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Uint8>, int) saveStateOf;
  final int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>) loadState;
  final int Function(Pointer<Void>, Pointer<Uint64>) trimMemory;
  final int Function(Pointer<Void>, Pointer<LiyabMemoryPlan>) memoryPlan;
  final int Function(Pointer<Void>, Pointer<LiyabWeightLayout>) weightLayout;
  final int Function(Pointer<Void>) contextLength;
  final int Function(Pointer<Void>, Pointer<Uint32>) warmMemory;
  final int Function(Pointer<Void>, Pointer<Utf8>) saveExpertProfile;
  final int Function(Pointer<Void>, Pointer<Utf8>) loadExpertProfile;
  final void Function(int) logBufferEnable;

  /// The model's chat template: a conversation rendered (request JSON in), and
  /// its layout of turns (probe JSON in, ChatFormat JSON out). -1 on failure.
  final int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, int) renderChat;
  final int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, int) chatFormat;

  /// Constrained decoding (liyab/grammar.h): grammars from a JSON Schema or
  /// grammar text (an opaque liyab_grammar*, destroyed with grammarDestroy),
  /// and generation constrained by a whole-output grammar or by spans a
  /// continuation callback opens.
  final int Function(Pointer<Utf8>, int, Pointer<Pointer<Void>>) grammarFromJsonSchema;
  final int Function(Pointer<Utf8>, Pointer<Pointer<Void>>) grammarParse;
  final void Function(Pointer<Void>) grammarDestroy;
  final int Function(Pointer<Void>, Pointer<Utf8>, int) grammarMatches;
  final int Function(
      Pointer<Void>,
      Pointer<Utf8>,
      Pointer<LiyabSamplingParams>,
      Pointer<NativeFunction<LiyabTokenCallbackNative>>,
      Pointer<Void>,
      Pointer<NativeFunction<LiyabContinuationCallbackNative>>,
      Pointer<Void>,
      Pointer<LiyabGenerationStats>,
      Pointer<LiyabGrammarStats>) generateStructured;

  /// Pooling models (embeddings, reranking): parameter defaults, what the loaded model pools (LiyabPooling) and
  /// its vector size, and the blocking calls (texts as UTF-8 pointers with byte lengths; vectors written one after
  /// another, scores one per document). See liyab_engine_embed / liyab_engine_rerank.
  final void Function(Pointer<LiyabEmbedParams>) embedParamsDefault;
  final void Function(Pointer<LiyabRerankParams>) rerankParamsDefault;
  final int Function(Pointer<Void>) pooling;
  final int Function(Pointer<Void>) embeddingSize;
  final int Function(Pointer<Void>, Pointer<Pointer<Utf8>>, Pointer<Size>, int, Pointer<LiyabEmbedParams>,
      Pointer<Float>, int, Pointer<LiyabPoolingStats>) embed;
  final int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Pointer<Utf8>>, Pointer<Size>, int,
      Pointer<LiyabRerankParams>, Pointer<Float>, Pointer<LiyabPoolingStats>) rerank;

  /// LoRA adapters (liyab_engine_load_adapter and the rest): load one under a name, forget it, select the one
  /// later calls apply (name null or '' for none, at a scale), and list them (count, then each by index).
  final int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>) loadAdapter;
  final int Function(Pointer<Void>, Pointer<Utf8>) unloadAdapter;
  final int Function(Pointer<Void>, Pointer<Utf8>, double) setAdapter;
  final int Function(Pointer<Void>) adapterCount;
  final int Function(Pointer<Void>, int, Pointer<LiyabAdapterInfo>) adapterInfo;

  /// Sequences (liyab_sequence_open and the rest): another conversation of the engine, with a context of its own,
  /// whose generations run alongside the default sequence's (the engine decodes them together). Open (an opaque
  /// liyab_sequence*, closed with sequenceClose before the engine is destroyed), cancel its generation (thread-safe),
  /// its priority (LiyabSequencePriority) and adapter, and a generation on it: at most one of the force callback,
  /// the grammar and the continuation callback.
  final int Function(Pointer<Void>, int, Pointer<Pointer<Void>>) sequenceOpen;
  final void Function(Pointer<Void>) sequenceClose;
  final void Function(Pointer<Void>) sequenceCancel;
  final int Function(Pointer<Void>, int) sequenceSetPriority;
  final int Function(Pointer<Void>, Pointer<Utf8>, double) sequenceSetAdapter;
  final int Function(
      Pointer<Void>,
      Pointer<Utf8>,
      Pointer<LiyabSamplingParams>,
      Pointer<NativeFunction<LiyabTokenCallbackNative>>,
      Pointer<NativeFunction<LiyabForceCallbackNative>>,
      Pointer<Void>,
      Pointer<NativeFunction<LiyabContinuationCallbackNative>>,
      Pointer<Void>,
      Pointer<LiyabGenerationStats>,
      Pointer<LiyabGrammarStats>) sequenceGenerate;

  /// Why grammar text does not compile (null when it does).
  String? grammarError(String text) => using((arena) {
        final out = arena<Pointer<Void>>();
        if (grammarParse(text.toNativeUtf8(allocator: arena), out) != LiyabStatus.ok) return errorMessage();
        grammarDestroy(out.value);
        return null;
      });
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

  /// Reads a size-returning text API that returns -1 on failure; null then.
  static String? readTextOrNull(int Function(Pointer<Utf8> buffer, int size) call) {
    final needed = call(nullptr, 0);
    if (needed < 0) return null;
    final buffer = calloc<Uint8>(needed + 1).cast<Utf8>();
    try {
      return call(buffer, needed + 1) < 0 ? null : buffer.toDartString(length: needed);
    } finally {
      calloc.free(buffer);
    }
  }

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
