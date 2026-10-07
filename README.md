# 🔥 Liyab — on-device LLM inference for Android and iOS

[![Platform](https://img.shields.io/badge/Platform-Android_%7C_iOS_%7C_macOS-brightgreen.svg)]()
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)]()

**Liyab** (*Tagalog for "flame"*) is a C++20 inference engine for running GGUF language models directly on phones,
with no cloud dependency. Weights are memory-mapped and never copied, work is routed per device
(NPU → GPU → CPU), and a power manager paces token output and reacts to thermal pressure.

This README describes what the code does today. Anything not implemented is listed under
[Limitations and roadmap](#limitations-and-roadmap), not in the feature list.

---

## Contents

* [Status at a glance](#status-at-a-glance)
* [Architecture](#architecture)
* [Supported models](#supported-models)
* [Building](#building)
* [Testing](#testing)
* [Using Liyab](#using-liyab)
* [Measured results](#measured-results)
* [Experimental modules](#experimental-modules)
* [Limitations and roadmap](#limitations-and-roadmap)
* [Project layout](#project-layout)

---

## Status at a glance

| Area | Status |
| :--- | :--- |
| GGUF loader (`mmap`, zero-copy, validated against malformed files) | ✅ implemented |
| Streaming mode for models larger than RAM (triple-window `madvise` prefetch) | ✅ implemented |
| Triple-buffered block loader (fetch → prepare → execute) | ✅ implemented, opt-in (models larger than RAM) |
| Transformer: llama / mistral / qwen2 / qwen3 layouts, GQA, RoPE (normal & NeoX), QKV bias, Q/K norm | ✅ implemented, cross-checked against llama.cpp |
| Mixed precision per tensor: F32, F16, Q8_0, Q4_0, Q4_1 | ✅ implemented |
| Paged KV cache, F16 / Q8_0 / INT4 (Q4_0 symmetric, Q4_1 asymmetric) | ✅ implemented |
| Sliding-window attention with attention sinks | ✅ implemented |
| Speculative decoding (draft model, batched verification) | ✅ implemented |
| CPU backend: ARM NEON + dot-product (SDOT), multithreaded | ✅ implemented |
| Apple GPU backend: Metal, zero-copy weights on unified memory | ✅ implemented |
| Android GPU backend: Vulkan compute (Adreno / Mali), weights repacked in GPU memory | ✅ implemented, +25% decode vs CPU on Adreno 830 |
| Power manager: duty-cycle pacing, thermal polling, throttle routing | ✅ implemented |
| SoC detection (Snapdragon, Dimensity, Tensor, Exynos, Apple) and backend ranking | ✅ implemented |
| Qualcomm QNN (Hexagon NPU), MediaTek NeuroPilot | 🟡 runtime detection only. Compute falls back to the next backend |
| Text encoding for byte-level BPE vocabularies (Llama 3, Qwen) | 🟡 decode only. Pass token ids instead |
| Experimental: early exit, head pruning, `O_DIRECT` / io_uring loader | 🧪 behind `LIYAB_ENABLE_EXPERIMENTAL` |

---

## Architecture

```
           Kotlin (JNI)  /  Swift  /  C++
                        │
              liyab_c_api.h  (stable C ABI)
                        │
   ┌────────────────────▼─────────────────────────────────────────────┐
   │ Engine                                                           │
   │  ├─ DeviceDetect ── SoC + accelerator probes → backend ranking   │
   │  ├─ PowerManager ── thermal polling, pacing, throttle policy     │
   │  ├─ Tokenizer, Sampler, SpeculativeDecoder                       │
   │  └─ Transformer ── paged KV cache (F16/Q8_0/Q4_0/Q4_1, sinks)   │
   └───────┬──────────────────────────────────┬───────────────────────┘
           │ weights                          │ matmuls (Route)
   ┌───────▼──────────────────────┐   ┌───────▼─────────────────────────┐
   │ MmapLoader (zero-copy GGUF)  │   │ attention → GPU, FFN → NPU,     │
   │  or TripleBufferLoader       │   │ fallback chain NPU → GPU → CPU  │
   │  (fetch | prepare | execute) │   │  Metal ✅ Vulkan ✅ CPU NEON ✅  │
   └──────────────────────────────┘   │  QNN / NeuroPilot 🟡            │
                                      └─────────────────────────────────┘
```

* **Zero-copy weights.** `MmapLoader` maps the GGUF file read-only. Every tensor is a view into the mapping, and
  the CPU kernels read Q4_0/Q4_1/Q8_0 blocks in place. On Apple Silicon each tensor is wrapped in an `MTLBuffer`
  with `newBufferWithBytesNoCopy`, so the GPU reads the page cache directly.
* **Streaming mode.** When the file is larger than 80% of available memory, the mapping is advised
  `MADV_SEQUENTIAL` and a prefetch thread keeps a three-block window resident: the block being computed and the
  next two being faulted in. Blocks behind the cursor are released with `MADV_DONTNEED`.
* **Triple-buffered loader** (`EngineConfig::triple_buffer_loading`, for models larger than RAM). Three page-aligned slots rotate through a
  fetch thread (stage 1), a transform thread (stage 2), and the executing layer (stage 3). Weight memory is bounded
  at 3 × the largest block. Stage 2 is a pass-through in the engine because the kernels consume packed INT4
  directly; an INT4→INT8 pass would double the bytes read. Stalls are counted in `GenerationStats::weight_stalls`.
* **Vulkan backend (Android GPUs).** Mobile drivers lack `VK_EXT_external_memory_host` (Adreno 830 included), so
  each tensor is copied once into host-visible GPU memory on first use and repacked into an aligned layout: float
  scales plus 16-byte quant words. Kernels (`src/backends/vulkan/shaders/matvec.comp`, one SPIR-V module per weight
  type, compiled with the NDK's `glslc` and embedded at build time) compute 4 rows per 64-lane workgroup with
  16-byte loads and `vec4` dot products. Matmuls that share an input (Q/K/V, gate/up) go out in one submission, and
  completion is polled briefly before the driver wait. A per-call dispatch costs 0.055 ms.
* **Heterogeneous routing.** Attention projections go to the GPU, while the FFN and output head go to the NPU.
  Any backend that reports `Unsupported` falls back to the CPU per matmul.
* **Paged KV cache.** Fixed 64-token pages (all layers per page) come from a pool on demand through a page table.
  Memory grows with the tokens actually cached, freed pages are reused exactly (no fragmentation), and sliding
  windows keep the first `kv_sink_tokens` positions (attention sinks, default 8) plus the last *N* tokens.
* **Power manager.** `pace_token()` caps output at the profile's rate (Balanced 12 tok/s, LowPower 6 tok/s) so the
  SoC idles between tokens. A polling thread reads the OS thermal status (Android `AThermal_*`, Apple
  `NSProcessInfo.thermalState`) and the sysfs skin/SoC zones. At ≥ 40 °C skin temperature, or at a severe OS
  status, it reroutes GPU work to the NPU/CPU, halves the active threads and halves the token rate.
* **Speculative decoding.** A small draft model proposes *k* tokens, and the target verifies them in one batched
  pass, reading its weights once for *k*+1 tokens. Acceptance follows Leviathan et al. (2023), so output matches the
  target's distribution. With greedy sampling it is token-for-token identical to plain decoding (tested).

---

## Supported models

* **Format:** GGUF v2/v3.
* **Architectures:** `llama` (Llama 1/2, Mistral, TinyLlama…), `mistral`, `qwen2`, `qwen3` (dense).
  Linear RoPE scaling and Llama 3.x `rope_freqs` are supported. YaRN is rejected with a clear error.
* **Tensor types:** F32, F16, Q8_0, Q4_0, Q4_1, mixed freely per tensor. K-quants and IQ-quants (Q4_K_M, IQ3_M,
  …) are rejected at load with a message. Convert them with llama.cpp:
  `llama-quantize --pure model.gguf model-q4_0.gguf Q4_0`
* **Tokenizers:** SentencePiece (`tokenizer.ggml.model = llama`) encodes and decodes. Byte-level BPE (`gpt2`:
  Llama 3, Qwen) decodes only, so pass token ids via `generate_tokens` / `liyab_engine_generate_tokens`.
* Prompts are used as given: apply the model's chat template in the app.

---

## Building

Requirements: CMake ≥ 3.22, a C++20 compiler (Clang 17+ / Apple Clang 15+), Ninja recommended.
Android: NDK r26+. iOS: Xcode 15+.

### Host (macOS / Linux)

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

### Android (`arm64-v8a`)

```bash
scripts/build_android.sh                 # libliyab.so, liyab-cli and tests in build/android/arm64-v8a
scripts/build_android.sh --experimental  # also builds the experimental modules
scripts/build_android.sh --help          # --abi, --api, --ndk, --no-vulkan, --no-qnn, --no-neuropilot, ...
```

The NDK is located from `--ndk`, `$ANDROID_NDK_HOME`, or the newest NDK in the usual SDK locations.
The `.so` is linked with 16 KB page alignment, which Android 15+ devices with 16 KB pages require.

### iOS (`.xcframework`)

```bash
scripts/build_ios.sh   # build/ios/Liyab.xcframework: device arm64 + simulator arm64/x86_64
```

The framework holds a static library, the headers and a module map, so Swift can `import Liyab`. The module map
auto-links Metal, Foundation and libc++. If `xcodebuild` reports missing components, run
`sudo xcodebuild -runFirstLaunch` once.

### CMake options

| Option | Default | Effect |
| :--- | :--- | :--- |
| `LIYAB_USE_NEON` | ON on arm64 | NEON intrinsics kernels |
| `LIYAB_ARM_DOTPROD` | ON on arm64 | `-march=armv8.2-a+dotprod+fp16` on Android/iOS/Linux (SDOT). AArch64 needs no `-mfpu`: NEON is part of the base ISA |
| `LIYAB_USE_METAL` | ON on Apple | Metal GPU backend |
| `LIYAB_USE_VULKAN` | ON on Android | Vulkan GPU backend + probe (loader opened with `dlopen`; shaders need `glslc` from the NDK or Vulkan SDK) |
| `LIYAB_USE_QNN` | ON on Android | QNN HTP runtime probe (no SDK needed) |
| `LIYAB_USE_NEUROPILOT` | ON on Android | NeuroPilot runtime probe |
| `LIYAB_ENABLE_EXPERIMENTAL` | OFF | Early exit, head pruning, direct-I/O loader, `test_experimental` |
| `LIYAB_BUILD_SHARED` | ON | Shared (`.so`/`.dylib`) or static library |
| `LIYAB_BUILD_TESTS` / `LIYAB_BUILD_CLI` | ON | Unit tests / `liyab-cli` |

---

## Testing

The tests depend on nothing beyond the compiler. They generate small GGUF models on the fly, so no download is
needed.

| Suite | Covers |
| :--- | :--- |
| `test_mmap` | mapping, `madvise` hints, NEON INT4→INT8 unpacking, GGUF parsing and malformed-file rejection, prefetcher, triple-buffer pipeline (ordering, resync, unpack stage, stall accounting, errors) |
| `test_device_detect` | SoC classification, backend ranking, live detection, sysfs thermal parsing, power policy, pacing |
| `test_engine` | quant kernels, tokenizer, transformer vs an independent float reference, batching/rollback, sliding window + sinks, Metal vs CPU, triple-buffer vs mmap equivalence, speculative decoding invariants, cancellation, pacing, C API |
| `test_kv_cache` | memory per token, on-demand paging and reuse, sinks and page recycling over 5000 positions, quantization accuracy, engine KV memory |
| `test_backends` | per-matmul latency of each backend on a model's shapes, grouped submissions, dispatch overhead |
| `test_experimental` | early exit, head pruning, direct I/O correctness, plus throughput and I/O benchmarks (`LIYAB_BENCH=0` skips them, `LIYAB_BENCH_MB=N` sizes the I/O file) |

On a device (USB debugging enabled):

```bash
scripts/build_android.sh --experimental
adb shell mkdir -p /data/local/tmp/liyab
adb push build/android/arm64-v8a/{libliyab.so,liyab-cli,test_*} /data/local/tmp/liyab/
adb shell 'cd /data/local/tmp/liyab && export LD_LIBRARY_PATH=. TMPDIR=/data/local/tmp/liyab &&
           for t in test_*; do ./$t || exit 1; done'
adb shell 'cd /data/local/tmp/liyab && LD_LIBRARY_PATH=. ./liyab-cli --device'
```

---

## Using Liyab

### C++

```cpp
#include "liyab/liyab.h"
#include <iostream>

int main() {
    liyab::EngineConfig config;
    config.model_path = "/data/local/tmp/model-q4_0.gguf";
    config.power.profile = liyab::PowerProfile::Balanced;  // paced to ~12 tok/s, throttles at 40 °C skin
    config.kv_cache_type = liyab::KvCacheType::Q8_0;       // or Q4_0 / Q4_1 for long contexts

    auto engine = liyab::Engine::create(config);
    if (!engine) {
        std::cerr << engine.status().to_string() << "\n";
        return 1;
    }
    std::cerr << engine.value()->describe();

    liyab::SamplingParams params;
    params.max_tokens = 128;
    auto stats = engine.value()->generate("Explain unified memory in one paragraph.", params,
                                          [](std::string_view piece, int32_t /*token*/) {
                                              std::cout << piece << std::flush;
                                              return true;  // false stops generation
                                          });
    if (stats) std::cerr << "\n" << stats->tokens_per_second << " tok/s\n";
}
```

`Engine::cancel()` is thread-safe and stops generation at the next token boundary.

### C ABI (`liyab_c_api.h`)

```c
liyab_engine_config config;
liyab_engine_config_default(&config);
config.model_path = path;

liyab_engine* engine = NULL;
if (liyab_engine_create(&config, &engine) != LIYAB_OK) { fprintf(stderr, "%s\n", liyab_last_error()); }

liyab_sampling_params params;
liyab_sampling_params_default(&params);
liyab_generation_stats stats;
liyab_engine_generate(engine, "Hello", &params, on_piece /* (piece, len, token, user) -> continue? */, user, &stats);
liyab_engine_destroy(engine);
```

No C++ exception crosses the ABI. Errors are returned as `liyab_status`, with the message available from
`liyab_last_error()` (thread-local).

### Android (Kotlin)

The library exposes the C ABI. The JNI glue lives in your app, because its symbol names depend on your Kotlin
package:

```cpp
// app/src/main/cpp/liyab_jni.cpp
#include <jni.h>
#include "liyab/liyab_c_api.h"

struct Ctx { JNIEnv* env; jobject cb; jmethodID onToken; };

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_ai_Liyab_nativeCreate(JNIEnv* env, jclass, jstring path) {
    liyab_engine_config c; liyab_engine_config_default(&c);
    const char* p = env->GetStringUTFChars(path, nullptr);
    c.model_path = p;
    liyab_engine* e = nullptr;
    liyab_engine_create(&c, &e);
    env->ReleaseStringUTFChars(path, p);
    return reinterpret_cast<jlong>(e);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_ai_Liyab_nativeGenerate(JNIEnv* env, jclass, jlong h, jstring prompt, jobject cb) {
    Ctx ctx{env, cb, env->GetMethodID(env->GetObjectClass(cb), "onToken", "([B)Z")};
    liyab_sampling_params sp; liyab_sampling_params_default(&sp);
    const char* p = env->GetStringUTFChars(prompt, nullptr);
    liyab_engine_generate(reinterpret_cast<liyab_engine*>(h), p, &sp,
        [](const char* s, size_t n, int32_t, void* u) -> int32_t {
            auto* c = static_cast<Ctx*>(u);
            jbyteArray bytes = c->env->NewByteArray(static_cast<jsize>(n));
            c->env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(n), reinterpret_cast<const jbyte*>(s));
            const jboolean go = c->env->CallBooleanMethod(c->cb, c->onToken, bytes);
            c->env->DeleteLocalRef(bytes);
            return go ? 1 : 0;
        }, &ctx, nullptr);
    env->ReleaseStringUTFChars(prompt, p);
}
```

```kotlin
object Liyab {
    init { System.loadLibrary("liyab"); System.loadLibrary("app_jni") }
    external fun nativeCreate(modelPath: String): Long
    external fun nativeGenerate(handle: Long, prompt: String, cb: TokenCallback)
    fun interface TokenCallback { fun onToken(utf8: ByteArray): Boolean }
}
// Pieces are complete UTF-8, so String(bytes, Charsets.UTF_8) is safe.
```

Run generation off the main thread. `liyab_engine_cancel` can be called from any thread.

### iOS (Swift)

```swift
import Liyab

final class LiyabEngine {
    private var engine: OpaquePointer?

    init(modelPath: String) throws {
        var config = liyab_engine_config()
        liyab_engine_config_default(&config)
        let status = modelPath.withCString { path -> liyab_status in
            config.model_path = path
            return liyab_engine_create(&config, &engine)
        }
        guard status == LIYAB_OK else { throw NSError(domain: String(cString: liyab_last_error()), code: Int(status.rawValue)) }
    }

    func generate(_ prompt: String, onPiece: @escaping (String) -> Bool) {
        var params = liyab_sampling_params()
        liyab_sampling_params_default(&params)
        let box = Unmanaged.passRetained(onPiece as AnyObject)
        defer { box.release() }
        liyab_engine_generate(engine, prompt, &params, { piece, len, _, user in
            let cb = Unmanaged<AnyObject>.fromOpaque(user!).takeUnretainedValue() as! (String) -> Bool
            let data = Data(bytes: piece!, count: len)
            return cb(String(decoding: data, as: UTF8.self)) ? 1 : 0
        }, box.toOpaque(), nil)
    }

    deinit { liyab_engine_destroy(engine) }
}
```

### Demo app: Liyab Chat (Android)

`android/chat` is a single-screen chat app in plain Java (no Gradle, no AndroidX) over the C ABI:

* **CPU / GPU switch.** Reloads the model on the CPU (NEON) or the GPU (Vulkan). In GPU mode the weights are
  uploaded at load time, not on the first message.
* **Model picker.** Lists the `*.gguf` files in the app folder, or picks any file with the system picker. Picked
  files are mapped through `/proc/self/fd`, so there is no copy and no storage permission. Loading a model first
  unloads the previous one.
* **Hugging Face downloads.** Searches GGUF repositories (sorted by downloads) and lists each file with its size,
  quantization and a compatibility badge (Q8_0/F16 run today; Q4_0/Q4_1 presets usually keep a Q6_K output head).
  Downloads use 4 parallel HTTP range connections. Per-segment progress is persisted, so a paused, killed or
  disconnected download resumes where it stopped. Each segment retries with exponential backoff, and the file is
  checked against the SHA-256 published by Hugging Face before it is loaded. Verified on the phone: a download
  killed mid-way, then resumed, matches the published SHA-256. The live loader shows progress, GB, smoothed
  MB/s, connections, elapsed time, ETA and retries, with Pause and Cancel.
* **Model management.** Lists downloaded and partial models with their disk usage. Any of them can be deleted,
  except the one currently loaded (it is memory-mapped).
* **Debug panel.** Shows timestamped load steps plus Liyab's own log lines (via `liyab_set_log_callback`), with
  per-message stats: tok/s, time to first token, reused prompt tokens.
* **Streaming and multi-turn.** Replies stream token by token, with Stop to cancel. Earlier turns are reused through
  the KV dedup prefix cache.

```bash
scripts/build_android_app.sh --install                            # builds build/android-app/liyab-chat.apk
adb push tinyllama-q4_0.gguf /sdcard/Android/data/com.liyab.chat/files/   # optional, after the first launch
# scripted download (tests / automation):
adb shell am start -n com.liyab.chat/.MainActivity --es download "'ggml-org/tiny-llamas|stories15M.gguf'"
```

The app uses the Zephyr / TinyLlama chat template. Launch it once before pushing, so that Android creates the
folder with the app as its owner.

### Command line

```bash
liyab-cli --device                                    # SoC, accelerators, backend ranking
liyab-cli -m model.gguf -p "Once upon a time" -n 128 --temp 0.8
liyab-cli -m model.gguf -p "..." --draft draft.gguf   # speculative decoding
liyab-cli -m model.gguf -p "..." --window 1024 --sinks 8 --kv q4_1
liyab-cli -m model.gguf -p "..." --profile low_power --skin-threshold 45
liyab-cli -m big-model.gguf -p "..." --triple-buffer      # only for models larger than RAM
liyab-cli -m model.gguf -p "..." --tokenize               # print token ids
liyab-cli --help
```

---

## Measured results

### Correctness against llama.cpp

The same GGUF files were evaluated by Liyab and by llama.cpp (via `llama-cpp-python`), and the logits were
compared over a 12-token sequence:

| Model layout | max abs logit diff | relative | argmax agreement |
| :--- | ---: | ---: | ---: |
| llama, F32 (normal RoPE, GQA) | 0.0015 | 0.04% | 100% |
| qwen2, F32 (NeoX RoPE, QKV bias) | 0.0015 | 0.04% | 100% |
| llama, tied embeddings, MHA | 0.012 | 0.03% | 100% |
| llama, mixed Q8_0 attention / Q4_0 FFN | 0.048 | 1.5% (activation quantization) | 100% |

### Real model: TinyLlama-1.1B-Chat

`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`, used as Q8_0 and re-quantized with llama.cpp to pure Q4_0 (621 MB).
With greedy decoding, the prompt *"What is the capital of France? Answer in one sentence."* yields
*"The capital of France is Paris."* on both Mac and phone.

Snapdragon 8 Elite phone, CPU backend (NEON + SDOT), 37-token prompt, 128 generated tokens, model in page cache,
throttle threshold raised to 60 °C so the comparison is not distorted by thermal rerouting:

| Configuration | Decode |
| :--- | ---: |
| Q4_0 weights, Q8_0 KV, 8 threads | 30.4 tok/s |
| Q4_0 weights, Q8_0 KV, 4 threads | 32.1 tok/s |
| Q4_0 weights, Q4_1 (INT4) KV | 30.4 tok/s, 1.3 MiB KV vs 2.2 MiB |
| Q8_0 weights, Q8_0 KV | 27.5 tok/s |
| Q4_0, `balanced` profile (12 tok/s pacing) | 12.05 tok/s, SoC idle 49% of the decode time |
| Q4_0, `--triple-buffer` (`O_DIRECT`) | 5.5 tok/s, 2814 weight stalls |

Same model and prompt, Vulkan GPU vs CPU (`--backend vulkan|cpu`, 4 threads):

| Backend | Decode | Prefill (37 tokens) |
| :--- | ---: | ---: |
| CPU NEON + SDOT | 31.0 tok/s | 0.76 s |
| Vulkan, Adreno 830 | **38.6 tok/s (+25%)** | 1.00 s |

Per-matmul latencies come from `test_backends` (`LIYAB_BENCH_MODEL=...`), after a 0.5 s warm-up. Mobile GPU clocks
ramp up under load, and cold measurements vary by up to 10×.

Reading the table:
* **Thermal guard.** With the default 40 °C threshold, this device's board thermistor (`xo-therm`, no dedicated skin
  sensor; the phone was charging over USB) crossed 40 °C under sustained load. The guard then rerouted, halved
  threads and capped the rate, as designed. Pass `skin_threshold_c` (C API) or `--skin-threshold` to tune it.
* **Triple buffering** re-reads every block from flash for every token (bounded memory, no page cache). That is
  the right trade only for models larger than RAM: here 560 MB/token ÷ 2.8 GB/s ≈ 5 tok/s, which matches the
  measurement. The engine logs a warning when it is enabled for a model that fits in memory.
* **8 threads slower than 4** points at the fork/join cost of the condition-variable thread pool. Prefill
  (~74 tok/s) still uses the per-row matvec kernel; a tiled GEMM for batches is the next CPU optimization.
* Apple M4 Pro, same prompt: 49 tok/s (CPU, Q8_0) and 40 tok/s (Metal, Q4_0). Metal is limited by one command
  buffer per matmul.


### Phone: Xiaomi 25010PN30G, Snapdragon 8 Elite (SM8750), 15 GB RAM

All test suites pass on the device. Detection output:

```
SoC:        Snapdragon 8 Elite [Qualcomm, SM8750]
NPU:        Hexagon v79
CPU:        8 cores (8 performance), neon=1 dotprod=1 i8mm=1 fp16=1
QNN HTP:    yes (libQnnHtp.so, 1 interface provider(s))
Vulkan:     yes (Adreno (TM) 830, Vulkan 1.3, no host-memory import)
Backends:   qnn -> vulkan -> cpu
```

Weight streaming from UFS (256 MiB, 1 MiB chunks, page cache dropped before each run):

| Method | Throughput |
| :--- | ---: |
| `mmap` + page touch | 2278 MB/s |
| `pread` through the page cache | 2218 MB/s |
| `pread` + `O_DIRECT` (io_uring denied by SELinux, see below) | 2796 MB/s (+23% vs mmap) |

KV cache memory (32 layers, 8 KV heads, head_dim 128 — Llama-3-8B class):

| Format | Per token | vs F16 |
| :--- | ---: | ---: |
| F16 | 128 KiB | — |
| Q8_0 | 68 KiB | −46.9% |
| Q4_0 (symmetric INT4) | 36 KiB | −71.9% |
| Q4_1 (asymmetric INT4) | 40 KiB | −68.8% |

INT4 needs a scale per 32-value block, so more than −72% vs F16 is not reachable at 4 bits; against F32 the
savings are 86% / 84%. On key vectors with a channel offset, Q4_1's relative RMS error is 1.5% vs 5.0% for Q4_0.

---

## Experimental modules

Build with `-DLIYAB_ENABLE_EXPERIMENTAL=ON` (or `scripts/build_android.sh --experimental`). They are enabled through
`EngineConfig::experimental` / the C config / `liyab-cli --early-exit P --prune-heads R --egls T --tdss hot|always --kv-dedup DIR`. They currently apply to
non-speculative decoding. The numbers below come from `test_experimental` on the Snapdragon 8 Elite: synthetic
12-block, d=512 model with random weights, or real TinyLlama when `LIYAB_BENCH_MODEL` points to a GGUF.

| Module | What it does | Measured | Caveat |
| :--- | :--- | :--- | :--- |
| **Early exit** (`early_exit.h`) | After probed blocks, projects the hidden state through the final norm + LM head. If the top-token probability > 0.98, it skips the remaining blocks and writes their K/V from the exited state (state propagation) | forced exit at block 6: 159 → 243 tok/s (+53%) | Lossy for models not trained for early exit. Random weights never reach 0.98, so real gains depend on the model. Each probe costs one LM-head matmul |
| **Head pruning** (`head_pruner.h`) | Ranks query heads by the Frobenius norm of their `Wo` columns. While throttled (≥ 40 °C) or in LowPower, skips the QK·softmax·V work of the least important heads | 50% heads: +12% at 1536 context, +3% at 64 | Lossy. K/V and the Q/O projections still run at full width |
| **EGLS: entropy-guided layer skipping** (`egls.h`) | Per decode step and block, computes the change dH in normalized energy entropy of the residual stream across attention. If dH < threshold, skips the block's FFN (identity shortcut; first/last 2 blocks protected) | TinyLlama-1.1B Q4_0 on the phone: 34.3 → 36.2 tok/s skipping 12% of FFNs, 50.6 tok/s at 64% | **Not viable without a model trained for it**: even at 12% skip the greedy output diverges from the full model at token 4 (9% of tokens match). A learned low-rank substitute would need offline calibration |
| **JIT tensor unpacker** (`jit_unpacker.h`) | Emits a fully unrolled, branch-free AArch64 NEON routine for a fixed size and layout (interleaved INT4, or GGML Q4_0 blocks with the scales skipped), mapped W^X (`mprotect` / `MAP_JIT`) | Phone: 78.8 vs 57.5 GB/s for a 4096-element row (+37%), +12% at 11008. 1M elements: 25 vs 63 GB/s (I-cache thrash). Mac M4: 4–10% slower than intrinsics | Helps only for row-sized routines (≤ ~12 KiB of code). Not on Liyab's hot path today (kernels read packed INT4 directly). Unavailable on iOS (no runtime codegen) |
| **TDSS: thermal-driven 2:4 sparsity** (`tdss.h`) | While throttled (≥ 40 °C), FFN projections switch to a 2:4 magnitude-pruned copy (3.5 bits/weight: INT4 values + 2-bit positions). The NEON kernel gathers activations with one `TBL` and does one `SDOT` per 32 columns | TinyLlama FFN: 408 → 318 MiB. Decode **32.3 → 14.8 tok/s on the phone (−54%)**, −57% on M4. Output diverges at token 4 (14% of tokens match) | **Counterproductive on current mobile hardware**: no 2:4 sparse units (an NVIDIA Ampere+ feature), so index decode + gather cost more than the halved MACs. Watts not measured (needs a battery-powered device and a power monitor) |
| **KV dedup: persistent prefix cache** (`kv_dedup.h`) | After prefill, whole KV pages of the prompt are saved to a file keyed by xxHash64(model fingerprint, KV layout, prefix tokens). A new session with the same prefix (system prompt) mmaps the file and attaches the pages read-only, without copying, to the paged KV cache. Only the rest is prefilled | 447-token system prompt, TinyLlama: **TTFT 6.4 s → 1.06 s on the phone (6×)**, 1.8 s → 0.27 s on M4. Restored KV is bit-identical (same logits, same output). 9.5 MiB on disk | Full attention only, no speculative decoding, no directory eviction yet. TTFT < 10 ms only when the whole prompt is cached; the new user turn is still prefilled |
| **Direct I/O loader** (`io_uring_loader.h`) | `O_DIRECT` reads into page-aligned buffers via raw io_uring syscalls (no liburing), with fallbacks. `read_into()` is the triple buffer's DMA stage | `O_DIRECT` +23% vs mmap (table above) | Android blocks `io_uring_setup` for apps *and* for `adb shell` (`EACCES`), so phones use the `O_DIRECT` pread fallback. macOS uses `F_NOCACHE`. DMA-BUF heaps are not app-accessible, so buffers are `posix_memalign` |

---

## Limitations and roadmap

* **NPU compute.** QNN and NeuroPilot are detected at runtime and ranked, but no compute kernels exist yet, so
  their work runs on the GPU/CPU. Next step: QNN HTP graphs for the FFN with rpcmem-registered weights.
* **Vulkan.** Batched prefill runs the matvec kernel once per token (no weight reuse across the batch); a tiled
  GEMM kernel is the next step. Weights are copied into GPU memory, so models must fit in RAM twice over; zero-copy
  needs `VK_EXT_external_memory_host` or AHardwareBuffer imports.
* **Core ML / Apple Neural Engine.** Not used: the ANE is only reachable through compiled Core ML models, not
  per-layer kernels over mmap'd weights. Metal is the Apple accelerator path.
* **Quantization formats.** K-quants (Q4_K, Q6_K…) and IQ formats are not implemented yet.
* **Tokenizer.** Byte-level BPE (Llama 3, Qwen) text encoding is missing; pass token ids.
* **Architectures.** MoE models (e.g. Qwen3-30B-A3B), Gemma, Phi and YaRN RoPE scaling are not supported.
* **Dense 30–35B models on phones** are bandwidth-bound. Each generated token reads every weight once: a 32B model
  at ~4.5 bits is ~18 GB, so at ~77 GB/s LPDDR5X the ceiling is ~4 tok/s even fully resident, and far less when
  streamed from flash (~2.8 GB/s measured). Speculative decoding, MoE models, and smaller dense models are the
  practical routes to interactive speeds.
* **CPU threading / prefill.** The thread pool synchronizes with condition variables (4 threads beat 8 on the
  phone), and prefill reuses the matvec kernel. A spinning pool and a tiled GEMM are the next CPU steps.
* **Metal dispatch** submits one command buffer per matmul. Batching a whole layer per command buffer is the next
  optimization.
* **Draft model** runs on the CPU, sequentially before verification, not concurrently.

---

## Project layout

```
include/liyab/          public headers (C++ API, C ABI, experimental/)
src/core/               engine, loader, KV cache, triple buffer, transformer, tokenizer, sampling, power, detection
src/backends/           cpu/ (NEON), metal/ (Metal), vulkan/ (Vulkan compute + shaders/), qnn/, neuropilot/ (runtime probes)
src/experimental/       early exit, head pruning, direct-I/O loader
src/c_api/              C ABI implementation
tools/liyab_cli.cpp     command-line front end over the C ABI
android/chat/           demo chat app (Java + JNI), built by scripts/build_android_app.sh
tests/                  self-contained unit tests and benchmarks
scripts/                build_android.sh, build_ios.sh
```

## License

Apache 2.0 (see `LICENSE`).
